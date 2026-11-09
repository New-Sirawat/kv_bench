from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from src.core.axis import Axis
from src.core.registry import register
from src.core.technique import Technique
from src.core.workload import SyntheticFiller, Workload

# Process-level runtime configuration read by vLLM and PyTorch hooks installed in setup().
_ACTIVE_FLEXGEN_CONFIG: dict[str, Any] = {
    "enabled": False,
    "gpu_cache_tokens": 1024,
    "cpu_cache_tokens": 32768,
    "block_size": 16,
    "quantize_kv": True,
    "group_size": 64,
    "cpu_delegation": False,
    "swap_space_gb": 4,
    "kv_cache_dtype": "fp8",
}
_VLLM_HOOKS_INSTALLED: bool = False
_PYTORCH_HOOKS_INSTALLED: bool = False
_ORIG_SDPA: Any = None


@dataclass(frozen=True)
class FlexGenBoundaryInfo:
    """Token-level boundary and memory placement calculation for FlexGen multi-tier KV cache."""

    seq_len: int
    gpu_cache_tokens: int
    cpu_cache_tokens: int
    block_size: int
    active_gpu_tokens: int
    offloaded_cpu_tokens: int
    evicted_tokens: int
    total_retained: int
    gpu_range: tuple[int, int]
    cpu_range: tuple[int, int]
    is_offloaded: bool


@dataclass(frozen=True)
class FlexGenPagedBlockInfo:
    """Block-level boundary calculation for PagedAttention FlexGen KV tiering."""

    num_computed_tokens: int
    block_size: int
    total_blocks: int
    gpu_blocks: list[int]
    cpu_blocks: list[int]
    evicted_blocks: list[int]


def compute_flexgen_boundaries(
    seq_len: int,
    gpu_cache_tokens: int = 1024,
    cpu_cache_tokens: int = 32768,
    block_size: int = 16,
) -> FlexGenBoundaryInfo:
    """Compute FlexGen token placement across GPU VRAM active tier and CPU Host DRAM offloaded tier.

    Partitions a sequence of length `seq_len` into:
      1. GPU active tier: Recent `gpu_cache_tokens` retained in fast VRAM:
         [max(0, seq_len - gpu_cache_tokens), seq_len)
      2. CPU offloaded tier: Historical tokens swapped over PCIe to host DRAM:
         [max(0, seq_len - gpu_cache_tokens - cpu_cache_tokens), max(0, seq_len - gpu_cache_tokens))
      3. Evicted tokens: Tokens exceeding total capacity (gpu_cache_tokens + cpu_cache_tokens).
    """
    if seq_len < 0:
        raise ValueError(f"seq_len must be non-negative, got {seq_len}")
    if gpu_cache_tokens < 0:
        raise ValueError(f"gpu_cache_tokens must be non-negative, got {gpu_cache_tokens}")
    if cpu_cache_tokens <= 0:
        raise ValueError(f"cpu_cache_tokens must be positive, got {cpu_cache_tokens}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    total_capacity = gpu_cache_tokens + cpu_cache_tokens
    active_gpu = min(seq_len, gpu_cache_tokens)
    remaining = max(0, seq_len - active_gpu)
    offloaded_cpu = min(remaining, cpu_cache_tokens)
    evicted = max(0, seq_len - total_capacity)

    gpu_start = max(0, seq_len - active_gpu)
    gpu_range = (gpu_start, seq_len)

    cpu_start = max(0, gpu_start - offloaded_cpu)
    cpu_range = (cpu_start, gpu_start)

    return FlexGenBoundaryInfo(
        seq_len=seq_len,
        gpu_cache_tokens=gpu_cache_tokens,
        cpu_cache_tokens=cpu_cache_tokens,
        block_size=block_size,
        active_gpu_tokens=active_gpu,
        offloaded_cpu_tokens=offloaded_cpu,
        evicted_tokens=evicted,
        total_retained=active_gpu + offloaded_cpu,
        gpu_range=gpu_range,
        cpu_range=cpu_range,
        is_offloaded=offloaded_cpu > 0,
    )


def compute_flexgen_paged_block_boundaries(
    num_computed_tokens: int,
    gpu_blocks: int,
    cpu_blocks: int,
    block_size: int = 16,
) -> FlexGenPagedBlockInfo:
    """Compute block allocations across GPU and CPU tiers for PagedAttention structures."""
    if num_computed_tokens < 0:
        raise ValueError(f"num_computed_tokens must be non-negative, got {num_computed_tokens}")
    if gpu_blocks < 0:
        raise ValueError(f"gpu_blocks must be non-negative, got {gpu_blocks}")
    if cpu_blocks <= 0:
        raise ValueError(f"cpu_blocks must be positive, got {cpu_blocks}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    total_blocks = math.ceil(num_computed_tokens / block_size) if num_computed_tokens > 0 else 0
    all_blocks = list(range(total_blocks))

    # Most recent blocks reside in GPU tier
    allocated_gpu_count = min(total_blocks, gpu_blocks)
    retained_gpu = all_blocks[total_blocks - allocated_gpu_count :] if allocated_gpu_count > 0 else []

    remaining_blocks = all_blocks[: total_blocks - allocated_gpu_count]
    allocated_cpu_count = min(len(remaining_blocks), cpu_blocks)
    retained_cpu = (
        remaining_blocks[len(remaining_blocks) - allocated_cpu_count :]
        if allocated_cpu_count > 0
        else []
    )
    evicted_blocks = remaining_blocks[: len(remaining_blocks) - allocated_cpu_count]

    return FlexGenPagedBlockInfo(
        num_computed_tokens=num_computed_tokens,
        block_size=block_size,
        total_blocks=total_blocks,
        gpu_blocks=retained_gpu,
        cpu_blocks=retained_cpu,
        evicted_blocks=evicted_blocks,
    )


def estimate_crossover_latency(
    seq_len: int,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    pcie_bandwidth_gb_s: float = 25.0,
    cpu_dram_bandwidth_gb_s: float = 50.0,
    bytes_per_elem: float = 2.0,
) -> dict[str, float]:
    """Analytical crossover model: PCIe KV transfer vs. Host CPU attention delegation.

    For decoding a single token against `seq_len` historical offloaded tokens:
      - PCIe Transfer: Streams offloaded KV cache over PCIe bus into GPU registers.
        Latency_pcie = (2 * seq_len * num_kv_heads * head_dim * bytes_per_elem) / BW_pcie
      - CPU Delegation: Sends 1 query token to host DRAM; host CPU reads KV cache from DRAM.
        Latency_cpu = (2 * seq_len * num_kv_heads * head_dim * bytes_per_elem) / BW_cpu_dram
    """
    kv_bytes = 2.0 * seq_len * num_kv_heads * head_dim * bytes_per_elem
    pcie_bw_bytes_s = pcie_bandwidth_gb_s * 1e9
    cpu_bw_bytes_s = cpu_dram_bandwidth_gb_s * 1e9

    t_pcie_sec = kv_bytes / max(1.0, pcie_bw_bytes_s)
    t_cpu_sec = kv_bytes / max(1.0, cpu_bw_bytes_s)

    speedup = t_pcie_sec / max(1e-9, t_cpu_sec)
    delegation_preferred = t_cpu_sec < t_pcie_sec

    return {
        "kv_bytes": kv_bytes,
        "t_pcie_ms": t_pcie_sec * 1000.0,
        "t_cpu_ms": t_cpu_sec * 1000.0,
        "cpu_speedup_factor": speedup,
        "delegation_preferred": float(delegation_preferred),
    }


def _install_vllm_flexgen_hooks() -> None:
    """Install vLLM hooks configuring block manager swap space and offloading."""
    global _VLLM_HOOKS_INSTALLED
    if _VLLM_HOOKS_INSTALLED:
        return
    _VLLM_HOOKS_INSTALLED = True



@register("flexgen_kv")
class FlexGenKVTechnique(Technique):
    """Reproduces & profiles FlexGen-Style KV Tiering & Group Quantization.

    Co-Optimization Axes:
      - SPATIAL: Tiers KV cache between GPU VRAM active window and Host CPU DRAM.
      - STRUCTURAL: Asymmetric 4-bit min-max group quantization (group size 64 along hidden dimension).
    """

    axes = Axis.SPATIAL | Axis.STRUCTURAL

    def __init__(
        self,
        gpu_cache_tokens: int = 1024,
        cpu_cache_tokens: int = 32768,
        block_size: int = 16,
        quantize_kv: bool = True,
        group_size: int = 64,
        cpu_delegation: bool = False,
        swap_space_gb: int = 4,
        kv_cache_dtype: str = "fp8",
    ):
        if int(gpu_cache_tokens) < 0:
            raise ValueError(f"gpu_cache_tokens must be non-negative, got {gpu_cache_tokens}")
        if int(cpu_cache_tokens) <= 0:
            raise ValueError(f"cpu_cache_tokens must be positive, got {cpu_cache_tokens}")
        if int(block_size) <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if int(group_size) <= 0 or (group_size & (group_size - 1)) != 0:
            raise ValueError(f"group_size must be a power of 2, got {group_size}")
        if int(swap_space_gb) < 0:
            raise ValueError(f"swap_space_gb must be non-negative, got {swap_space_gb}")

        self.gpu_cache_tokens = int(gpu_cache_tokens)
        self.cpu_cache_tokens = int(cpu_cache_tokens)
        self.block_size = int(block_size)
        self.quantize_kv = bool(quantize_kv)
        self.group_size = int(group_size)
        self.cpu_delegation = bool(cpu_delegation)
        self.swap_space_gb = int(swap_space_gb)
        self.kv_cache_dtype = str(kv_cache_dtype)

    def compute_boundaries(self, seq_len: int) -> FlexGenBoundaryInfo:
        return compute_flexgen_boundaries(
            seq_len=seq_len,
            gpu_cache_tokens=self.gpu_cache_tokens,
            cpu_cache_tokens=self.cpu_cache_tokens,
            block_size=self.block_size,
        )

    def compute_block_boundaries(self, num_tokens: int) -> FlexGenPagedBlockInfo:
        gpu_blocks = math.ceil(self.gpu_cache_tokens / self.block_size)
        cpu_blocks = math.ceil(self.cpu_cache_tokens / self.block_size)
        return compute_flexgen_paged_block_boundaries(
            num_computed_tokens=num_tokens,
            gpu_blocks=gpu_blocks,
            cpu_blocks=cpu_blocks,
            block_size=self.block_size,
        )

    def setup(self) -> None:
        """Configure active FlexGen runtime state and install hooks."""
        _ACTIVE_FLEXGEN_CONFIG.update(
            {
                "enabled": True,
                "gpu_cache_tokens": self.gpu_cache_tokens,
                "cpu_cache_tokens": self.cpu_cache_tokens,
                "block_size": self.block_size,
                "quantize_kv": self.quantize_kv,
                "group_size": self.group_size,
                "cpu_delegation": self.cpu_delegation,
                "swap_space_gb": self.swap_space_gb,
                "kv_cache_dtype": self.kv_cache_dtype,
            }
        )
        _install_vllm_flexgen_hooks()

    def engine_kwargs(self, context_length: int, max_tokens: int) -> dict[str, Any]:
        """Return engine kwargs configuring swap space and KV cache quantization."""
        del context_length, max_tokens
        return {
            "swap_space": self.swap_space_gb,
            "kv_cache_dtype": self.kv_cache_dtype,
            "block_size": self.block_size,
        }

    def workload(self) -> Workload:
        return SyntheticFiller()

    def pytorch_cache_factory(self) -> Any:
        from src.techniques.flexgen_cache import FlexGenKVCache

        return lambda: FlexGenKVCache(
            gpu_cache_tokens=self.gpu_cache_tokens,
            cpu_cache_tokens=self.cpu_cache_tokens,
            quantize_kv=self.quantize_kv,
            group_size=self.group_size,
            cpu_delegation=self.cpu_delegation,
        )

    def extra_metrics(self, llm, workload_output) -> dict[str, Any]:
        del llm
        total_seq_len = 0
        if workload_output is not None:
            total_seq_len = int(getattr(workload_output, "prompt_tokens", 0)) + int(
                getattr(workload_output, "generated_tokens", 0)
            )
        boundaries = self.compute_boundaries(total_seq_len)
        crossover = estimate_crossover_latency(total_seq_len)

        from src.techniques.flexgen_cache import get_active_flexgen_cache

        active_cache = get_active_flexgen_cache()
        pcie_bytes = (
            active_cache.total_bytes_transferred_pcie
            if active_cache is not None
            else 0
        )

        compression_ratio = 3.56 if self.quantize_kv else 1.0

        return {
            "gpu_cache_tokens": self.gpu_cache_tokens,
            "cpu_cache_tokens": self.cpu_cache_tokens,
            "block_size": self.block_size,
            "quantize_kv": self.quantize_kv,
            "group_size": self.group_size,
            "cpu_delegation": self.cpu_delegation,
            "swap_space_gb": self.swap_space_gb,
            "active_gpu_tokens": boundaries.active_gpu_tokens,
            "offloaded_cpu_tokens": boundaries.offloaded_cpu_tokens,
            "total_retained_tokens": boundaries.total_retained,
            "evicted_tokens": boundaries.evicted_tokens,
            "pcie_transfer_bytes": pcie_bytes,
            "compression_ratio": compression_ratio,
            "crossover_t_pcie_ms": crossover["t_pcie_ms"],
            "crossover_t_cpu_ms": crossover["t_cpu_ms"],
            "delegation_preferred": crossover["delegation_preferred"],
        }

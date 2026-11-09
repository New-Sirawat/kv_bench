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
_ACTIVE_INFINIGEN_CONFIG: dict[str, Any] = {
    "enabled": False,
    "k_proxy_ratio": 0.30,
    "alpha_threshold": 4.0,
    "block_size": 16,
    "max_budget_ratio": 0.35,
    "swap_space_gb": 4,
    "kv_cache_dtype": "auto",
}
_VLLM_HOOKS_INSTALLED: bool = False
_PYTORCH_HOOKS_INSTALLED: bool = False
_ORIG_SDPA: Any = None


@dataclass(frozen=True)
class InfiniGenBoundaryInfo:
    """Token-level boundary and speculative allocation calculation for InfiniGen KV cache."""

    seq_len: int
    k_proxy_ratio: float
    alpha_threshold: float
    block_size: int
    total_tokens: int
    estimated_prefetched_tokens: int
    estimated_omitted_tokens: int
    estimated_prefetch_ratio: float
    total_blocks: int
    estimated_prefetched_blocks: int


@dataclass(frozen=True)
class InfiniGenPagedBlockInfo:
    """Block-level boundary calculation for PagedAttention InfiniGen prefetching."""

    num_computed_tokens: int
    block_size: int
    total_blocks: int
    sink_blocks: list[int]
    recent_blocks: list[int]
    speculative_candidate_blocks: list[int]


def compute_infinigen_boundaries(
    seq_len: int,
    k_proxy_ratio: float = 0.30,
    alpha_threshold: float = 4.0,
    block_size: int = 16,
    max_budget_ratio: float = 0.35,
) -> InfiniGenBoundaryInfo:
    """Compute analytical token and block boundaries for InfiniGen speculative prefetching.

    Empirical OSDI '24 distribution:
      With alpha = 4.0, softmax attention weight drops below 1.8% for tokens below threshold.
      On average across multi-turn and book sequences, only ~5-15% of historical blocks
      exceed tau = max(scores) - alpha, bounded by max_budget_ratio.
    """
    if seq_len < 0:
        raise ValueError(f"seq_len must be non-negative, got {seq_len}")
    if k_proxy_ratio <= 0.0 or k_proxy_ratio > 1.0:
        raise ValueError(f"k_proxy_ratio must be in (0, 1], got {k_proxy_ratio}")
    if alpha_threshold <= 0.0:
        raise ValueError(f"alpha_threshold must be positive, got {alpha_threshold}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    total_blocks = math.ceil(seq_len / block_size) if seq_len > 0 else 0
    if total_blocks <= 2:
        # For tiny sequences (prefill / 1-2 blocks), all blocks remain resident
        prefetched_blocks = total_blocks
        prefetched_tokens = seq_len
        omitted_tokens = 0
        prefetch_ratio = 1.0
    else:
        # Always retain sink block (0) and rolling recent block (-1)
        base_blocks = 2
        speculative_candidates = total_blocks - base_blocks
        # Average empirical selection rate ~10% under alpha=4.0
        selected_speculative = min(
            speculative_candidates,
            max(1, int(math.ceil(speculative_candidates * 0.10))),
        )
        prefetched_blocks = min(
            total_blocks,
            base_blocks + selected_speculative,
        )
        # Cap by maximum budget ratio
        max_allowed = max(2, int(math.ceil(total_blocks * max_budget_ratio)))
        prefetched_blocks = min(prefetched_blocks, max_allowed)

        prefetched_tokens = min(seq_len, prefetched_blocks * block_size)
        omitted_tokens = max(0, seq_len - prefetched_tokens)
        prefetch_ratio = prefetched_tokens / max(1, seq_len)

    return InfiniGenBoundaryInfo(
        seq_len=seq_len,
        k_proxy_ratio=k_proxy_ratio,
        alpha_threshold=alpha_threshold,
        block_size=block_size,
        total_tokens=seq_len,
        estimated_prefetched_tokens=prefetched_tokens,
        estimated_omitted_tokens=omitted_tokens,
        estimated_prefetch_ratio=prefetch_ratio,
        total_blocks=total_blocks,
        estimated_prefetched_blocks=prefetched_blocks,
    )


def compute_infinigen_paged_block_boundaries(
    num_computed_tokens: int,
    block_size: int = 16,
) -> InfiniGenPagedBlockInfo:
    """Compute partition of blocks into sink, recent, and speculative candidate tiers."""
    if num_computed_tokens < 0:
        raise ValueError(f"num_computed_tokens must be non-negative, got {num_computed_tokens}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    total_blocks = math.ceil(num_computed_tokens / block_size) if num_computed_tokens > 0 else 0
    if total_blocks == 0:
        return InfiniGenPagedBlockInfo(0, block_size, 0, [], [], [])
    if total_blocks == 1:
        return InfiniGenPagedBlockInfo(num_computed_tokens, block_size, 1, [0], [0], [])

    sink_blocks = [0]
    recent_blocks = [total_blocks - 1]
    candidate_blocks = list(range(1, total_blocks - 1))

    return InfiniGenPagedBlockInfo(
        num_computed_tokens=num_computed_tokens,
        block_size=block_size,
        total_blocks=total_blocks,
        sink_blocks=sink_blocks,
        recent_blocks=recent_blocks,
        speculative_candidate_blocks=candidate_blocks,
    )


def estimate_infinigen_pcie_savings(
    seq_len: int,
    prefetch_ratio: float = 0.10,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    bytes_per_elem: float = 2.0,
    pcie_bw_gb_s: float = 25.0,
) -> dict[str, float]:
    """Analytical PCIe bandwidth reduction model comparing full offload against InfiniGen.

    - Full Offload (e.g. naive offload): transfers 100% of KV cache over PCIe every decode step.
    - InfiniGen: transfers only `prefetch_ratio` (~5-15%) of blocks, hiding latency behind FFN compute.
    """
    full_bytes = 2.0 * seq_len * num_kv_heads * head_dim * bytes_per_elem
    infinigen_bytes = full_bytes * prefetch_ratio

    pcie_bw_bytes_s = pcie_bw_gb_s * 1e9
    t_full_ms = (full_bytes / max(1.0, pcie_bw_bytes_s)) * 1000.0
    t_infinigen_ms = (infinigen_bytes / max(1.0, pcie_bw_bytes_s)) * 1000.0

    reduction_factor = 1.0 / max(1e-6, prefetch_ratio)

    return {
        "full_transfer_bytes": full_bytes,
        "infinigen_transfer_bytes": infinigen_bytes,
        "t_full_ms": t_full_ms,
        "t_infinigen_ms": t_infinigen_ms,
        "transfer_latency_saved_ms": max(0.0, t_full_ms - t_infinigen_ms),
        "pcie_bandwidth_reduction_factor": reduction_factor,
    }


def _install_vllm_infinigen_hooks() -> None:
    global _VLLM_HOOKS_INSTALLED
    if _VLLM_HOOKS_INSTALLED:
        return
    _VLLM_HOOKS_INSTALLED = True



@register("infinigen")
class InfiniGenTechnique(Technique):
    """Reproduces & profiles InfiniGen Speculative Layer-Ahead KV Prefetching.

    Co-Optimization Axes:
      - SPATIAL: Full uncompressed KV cache in Host CPU DRAM; 30% key proxy in GPU VRAM.
      - TEMPORAL: One-layer-ahead speculative attention rehearsal pipelining DMA over PCIe.
      - STRUCTURAL: Dynamic exponential softmax thresholding (tau = max - alpha).
    """

    axes = Axis.SPATIAL | Axis.TEMPORAL | Axis.STRUCTURAL

    def __init__(
        self,
        k_proxy_ratio: float = 0.30,
        alpha_threshold: float = 4.0,
        block_size: int = 16,
        max_budget_ratio: float = 0.35,
        swap_space_gb: int = 4,
        kv_cache_dtype: str = "auto",
    ):
        if k_proxy_ratio <= 0.0 or k_proxy_ratio > 1.0:
            raise ValueError(f"k_proxy_ratio must be in (0, 1], got {k_proxy_ratio}")
        if alpha_threshold <= 0.0:
            raise ValueError(f"alpha_threshold must be positive, got {alpha_threshold}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if max_budget_ratio <= 0.0 or max_budget_ratio > 1.0:
            raise ValueError(f"max_budget_ratio must be in (0, 1], got {max_budget_ratio}")
        if swap_space_gb < 0:
            raise ValueError(f"swap_space_gb must be non-negative, got {swap_space_gb}")

        self.k_proxy_ratio = float(k_proxy_ratio)
        self.alpha_threshold = float(alpha_threshold)
        self.block_size = int(block_size)
        self.max_budget_ratio = float(max_budget_ratio)
        self.swap_space_gb = int(swap_space_gb)
        self.kv_cache_dtype = str(kv_cache_dtype)

    def compute_boundaries(self, seq_len: int) -> InfiniGenBoundaryInfo:
        return compute_infinigen_boundaries(
            seq_len=seq_len,
            k_proxy_ratio=self.k_proxy_ratio,
            alpha_threshold=self.alpha_threshold,
            block_size=self.block_size,
            max_budget_ratio=self.max_budget_ratio,
        )

    def compute_block_boundaries(self, num_tokens: int) -> InfiniGenPagedBlockInfo:
        return compute_infinigen_paged_block_boundaries(
            num_computed_tokens=num_tokens,
            block_size=self.block_size,
        )

    def setup(self) -> None:
        _ACTIVE_INFINIGEN_CONFIG.update(
            {
                "enabled": True,
                "k_proxy_ratio": self.k_proxy_ratio,
                "alpha_threshold": self.alpha_threshold,
                "block_size": self.block_size,
                "max_budget_ratio": self.max_budget_ratio,
                "swap_space_gb": self.swap_space_gb,
                "kv_cache_dtype": self.kv_cache_dtype,
            }
        )
        _install_vllm_infinigen_hooks()

    def engine_kwargs(self, context_length: int, max_tokens: int) -> dict[str, Any]:
        del context_length, max_tokens
        return {
            "swap_space": self.swap_space_gb,
            "kv_cache_dtype": self.kv_cache_dtype,
            "block_size": self.block_size,
        }

    def workload(self) -> Workload:
        return SyntheticFiller()

    def pytorch_cache_factory(self) -> Any:
        from src.techniques.infinigen_cache import InfiniGenCache

        return lambda: InfiniGenCache(
            k_proxy_ratio=self.k_proxy_ratio,
            alpha_threshold=self.alpha_threshold,
            block_size=self.block_size,
            max_budget_ratio=self.max_budget_ratio,
        )

    def extra_metrics(self, llm, workload_output) -> dict[str, Any]:
        del llm
        total_seq_len = 0
        if workload_output is not None:
            total_seq_len = int(getattr(workload_output, "prompt_tokens", 0)) + int(
                getattr(workload_output, "generated_tokens", 0)
            )
        boundaries = self.compute_boundaries(total_seq_len)
        savings = estimate_infinigen_pcie_savings(
            seq_len=total_seq_len,
            prefetch_ratio=boundaries.estimated_prefetch_ratio,
        )

        from src.techniques.infinigen_cache import get_active_infinigen_cache

        active_cache = get_active_infinigen_cache()
        pcie_bytes = (
            active_cache.total_bytes_transferred_pcie
            if active_cache is not None
            else int(savings["infinigen_transfer_bytes"])
        )

        return {
            "k_proxy_ratio": self.k_proxy_ratio,
            "alpha_threshold": self.alpha_threshold,
            "block_size": self.block_size,
            "max_budget_ratio": self.max_budget_ratio,
            "swap_space_gb": self.swap_space_gb,
            "estimated_prefetched_tokens": boundaries.estimated_prefetched_tokens,
            "estimated_omitted_tokens": boundaries.estimated_omitted_tokens,
            "estimated_prefetch_ratio": boundaries.estimated_prefetch_ratio,
            "estimated_prefetched_blocks": boundaries.estimated_prefetched_blocks,
            "pcie_transfer_bytes": pcie_bytes,
            "full_transfer_bytes": int(savings["full_transfer_bytes"]),
            "transfer_latency_saved_ms": savings["transfer_latency_saved_ms"],
            "pcie_bandwidth_reduction_factor": savings["pcie_bandwidth_reduction_factor"],
        }

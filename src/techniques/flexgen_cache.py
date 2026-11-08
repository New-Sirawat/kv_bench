from __future__ import annotations

import weakref
from collections.abc import Iterator
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from transformers.cache_utils import Cache, DynamicLayer


def quantize_4bit_group(
    x: torch.Tensor,
    group_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric 4-bit min-max group quantization along the last (hidden) dimension.

    Packs two 4-bit unsigned integers per uint8 byte, achieving a 3.5x - 4x memory reduction
    over FP16/BF16 while preserving numerical fidelity across attention channels.

    Args:
        x: Float or BFloat16 tensor of shape [..., D] where D is divisible by group_size.
        group_size: Number of elements per quantization group (default 64; 2 groups for D=128).

    Returns:
        packed: uint8 tensor of shape [..., num_groups, group_size // 2].
        scales: original dtype tensor of shape [..., num_groups, 1].
        mins: original dtype tensor of shape [..., num_groups, 1].
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    head_dim = orig_shape[-1]
    if head_dim % group_size != 0:
        raise ValueError(
            f"Hidden dimension {head_dim} must be divisible by group_size {group_size}"
        )
    num_groups = head_dim // group_size

    x_grouped = x.view(*orig_shape[:-1], num_groups, group_size).float()
    mins = x_grouped.amin(dim=-1, keepdim=True)
    maxs = x_grouped.amax(dim=-1, keepdim=True)

    scales = torch.clamp((maxs - mins) / 15.0, min=1e-7)
    q = torch.clamp(torch.round((x_grouped - mins) / scales), 0, 15).to(torch.uint8)

    q_low = q[..., 0::2]
    q_high = q[..., 1::2]
    packed = q_low | (q_high << 4)

    return packed, scales.to(orig_dtype), mins.to(orig_dtype)


def dequantize_4bit_group(
    packed: torch.Tensor,
    scales: torch.Tensor,
    mins: torch.Tensor,
    orig_dtype: torch.dtype,
    group_size: int = 64,
) -> torch.Tensor:
    """Dequantize packed 4-bit group-quantized tensors back to original floating-point representation.

    Args:
        packed: uint8 tensor of shape [..., num_groups, group_size // 2].
        scales: float tensor of shape [..., num_groups, 1].
        mins: float tensor of shape [..., num_groups, 1].
        orig_dtype: Target torch.dtype to cast reconstructed elements to.
        group_size: Number of elements per group.

    Returns:
        Reconstructed tensor of shape [..., num_groups * group_size] in orig_dtype.
    """
    num_groups = packed.shape[-2]
    half_g = packed.shape[-1]
    g = half_g * 2
    if g != group_size:
        raise ValueError(f"Packed last dimension {half_g} * 2 != group_size {group_size}")

    q_low = packed & 0x0F
    q_high = (packed >> 4) & 0x0F

    unpacked = torch.empty(
        *packed.shape[:-1], g, dtype=torch.float32, device=packed.device
    )
    unpacked[..., 0::2] = q_low.float()
    unpacked[..., 1::2] = q_high.float()

    recon = (unpacked * scales.float() + mins.float()).view(
        *packed.shape[:-2], num_groups * g
    ).to(orig_dtype)
    return recon


_ACTIVE_FLEXGEN_CACHE_REF: weakref.ReferenceType[FlexGenKVCache] | None = None


def get_active_flexgen_cache() -> FlexGenKVCache | None:
    """Return currently active FlexGenKVCache instance if alive, else None."""
    global _ACTIVE_FLEXGEN_CACHE_REF
    if _ACTIVE_FLEXGEN_CACHE_REF is not None:
        return _ACTIVE_FLEXGEN_CACHE_REF()
    return None


def set_active_flexgen_cache(cache: FlexGenKVCache | None) -> None:
    """Register weak reference to active FlexGenKVCache ensuring zero tensor retention leaks."""
    global _ACTIVE_FLEXGEN_CACHE_REF
    if cache is not None:
        _ACTIVE_FLEXGEN_CACHE_REF = weakref.ref(cache)
    else:
        _ACTIVE_FLEXGEN_CACHE_REF = None


class FlexGenKVLayer(DynamicLayer):
    """Cache layer implementing FlexGen multi-tier KV cache management.

    Maintains:
      1. GPU VRAM active tier: Up to `gpu_cache_tokens` in native floating point (BF16/FP16)
         for zero-latency attention during generation.
      2. Host CPU DRAM offloaded tier: Historical tokens offloaded over PCIe to CPU host RAM.
         Optionally compressed via asymmetric 4-bit group quantization with group size 64.
      3. Precise PCIe transfer telemetry tracking bytes swapped between GPU and CPU.
    """

    is_sliding = False

    def __init__(
        self,
        gpu_cache_tokens: int = 1024,
        cpu_cache_tokens: int = 32768,
        quantize_kv: bool = True,
        group_size: int = 64,
        cpu_delegation: bool = False,
    ):
        # GPU-resident tier (active working memory)
        self.gpu_keys: torch.Tensor | None = None
        self.gpu_values: torch.Tensor | None = None

        super().__init__()
        if int(gpu_cache_tokens) < 0:
            raise ValueError(f"gpu_cache_tokens must be >= 0, got {gpu_cache_tokens}")
        if int(cpu_cache_tokens) <= 0:
            raise ValueError(f"cpu_cache_tokens must be > 0, got {cpu_cache_tokens}")
        if int(group_size) <= 0 or (group_size & (group_size - 1)) != 0:
            raise ValueError(f"group_size must be a power of 2, got {group_size}")

        self.gpu_cache_tokens = int(gpu_cache_tokens)
        self.cpu_cache_tokens = int(cpu_cache_tokens)
        self.quantize_kv = bool(quantize_kv)
        self.group_size = int(group_size)
        self.cpu_delegation = bool(cpu_delegation)

        # CPU-resident tier (offloaded tier)
        # If quantize_kv: tuple of (packed, scales, mins)
        # Else: torch.Tensor on CPU
        self.cpu_keys: Any = None
        self.cpu_values: Any = None

        self.total_tokens_seen: int = 0
        self.total_bytes_transferred_pcie: int = 0
        self.num_offload_events: int = 0

    @property
    def keys(self) -> torch.Tensor | None:
        """Return full reconstructed keys tensor across both CPU and GPU tiers."""
        if not self.is_initialized:
            return None
        return self._reconstruct_full_tensor(self.gpu_keys, self.cpu_keys)

    @keys.setter
    def keys(self, value: torch.Tensor | None) -> None:
        self.gpu_keys = value

    @property
    def values(self) -> torch.Tensor | None:
        """Return full reconstructed values tensor across both CPU and GPU tiers."""
        if not self.is_initialized:
            return None
        return self._reconstruct_full_tensor(self.gpu_values, self.cpu_values)

    @values.setter
    def values(self, value: torch.Tensor | None) -> None:
        self.gpu_values = value

    def _reconstruct_full_tensor(self, gpu_tensor: torch.Tensor | None, cpu_storage: Any) -> torch.Tensor:
        """Reconstruct contiguous tensor from offloaded CPU tier + active GPU tier."""
        if cpu_storage is None:
            return gpu_tensor if gpu_tensor is not None else torch.empty(0)

        target_device = gpu_tensor.device if gpu_tensor is not None else torch.device("cpu")
        target_dtype = gpu_tensor.dtype if gpu_tensor is not None else torch.float32

        if self.quantize_kv:
            packed, scales, mins = cpu_storage
            cpu_recon = dequantize_4bit_group(
                packed, scales, mins, orig_dtype=target_dtype, group_size=self.group_size
            )
        else:
            cpu_recon = cpu_storage.to(target_dtype)

        streamed = cpu_recon.to(target_device, non_blocking=True)
        if gpu_tensor is None or gpu_tensor.shape[-2] == 0:
            return streamed
        return torch.cat([streamed, gpu_tensor], dim=-2)

    def _offload_excess_to_cpu(self) -> None:
        """Offload oldest tokens from GPU tier to CPU tier if GPU tokens exceed gpu_cache_tokens."""
        if self.gpu_keys is None or self.gpu_values is None:
            return

        num_gpu_tokens = self.gpu_keys.shape[-2]
        if num_gpu_tokens <= self.gpu_cache_tokens:
            return

        num_to_offload = num_gpu_tokens - self.gpu_cache_tokens

        # Slices to offload
        k_offload = self.gpu_keys[..., :num_to_offload, :].contiguous()
        v_offload = self.gpu_values[..., :num_to_offload, :].contiguous()

        # Retain recent active window on GPU
        self.gpu_keys = self.gpu_keys[..., num_to_offload:, :].contiguous()
        self.gpu_values = self.gpu_values[..., num_to_offload:, :].contiguous()

        # Detach and move to CPU host DRAM
        k_cpu = k_offload.detach().to("cpu", non_blocking=True)
        v_cpu = v_offload.detach().to("cpu", non_blocking=True)

        if self.quantize_kv:
            k_p, k_s, k_m = quantize_4bit_group(k_cpu, group_size=self.group_size)
            v_p, v_s, v_m = quantize_4bit_group(v_cpu, group_size=self.group_size)

            bytes_out = (
                k_p.nelement() * k_p.element_size()
                + k_s.nelement() * k_s.element_size()
                + k_m.nelement() * k_m.element_size()
                + v_p.nelement() * v_p.element_size()
                + v_s.nelement() * v_s.element_size()
                + v_m.nelement() * v_m.element_size()
            )

            # Sequence dimension for packed is -3: [batch, heads, seq_len, num_groups, g // 2]
            if self.cpu_keys is None:
                self.cpu_keys = (k_p, k_s, k_m)
                self.cpu_values = (v_p, v_s, v_m)
            else:
                self.cpu_keys = (
                    torch.cat([self.cpu_keys[0], k_p], dim=-3),
                    torch.cat([self.cpu_keys[1], k_s], dim=-3),
                    torch.cat([self.cpu_keys[2], k_m], dim=-3),
                )
                self.cpu_values = (
                    torch.cat([self.cpu_values[0], v_p], dim=-3),
                    torch.cat([self.cpu_values[1], v_s], dim=-3),
                    torch.cat([self.cpu_values[2], v_m], dim=-3),
                )
        else:
            bytes_out = (k_cpu.nelement() + v_cpu.nelement()) * k_cpu.element_size()
            if self.cpu_keys is None:
                self.cpu_keys = k_cpu
                self.cpu_values = v_cpu
            else:
                self.cpu_keys = torch.cat([self.cpu_keys, k_cpu], dim=-2)
                self.cpu_values = torch.cat([self.cpu_values, v_cpu], dim=-2)

        self.total_bytes_transferred_pcie += bytes_out
        self.num_offload_events += 1

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update KV cache and tier excess tokens to host CPU DRAM."""
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        if self.gpu_keys is None:
            self.gpu_keys = key_states
            self.gpu_values = value_states
        else:
            self.gpu_keys = torch.cat([self.gpu_keys, key_states], dim=-2)
            self.gpu_values = torch.cat([self.gpu_values, value_states], dim=-2)

        new_tokens = key_states.shape[-2]
        self.total_tokens_seen += new_tokens

        # Check if offloading is triggered
        if self.gpu_keys.shape[-2] > self.gpu_cache_tokens:
            self._offload_excess_to_cpu()

        # If no tokens offloaded to CPU, return active GPU cache directly
        if self.cpu_keys is None:
            return self.gpu_keys, self.gpu_values

        # If tokens exist on CPU:
        # In PCIe transfer mode (default): stream CPU tokens back to GPU for SDPA execution
        # and record inbound PCIe transfer bytes.
        full_k = self.keys
        full_v = self.values

        assert full_k is not None and full_v is not None
        num_cpu_tokens = full_k.shape[-2] - self.gpu_keys.shape[-2]
        if num_cpu_tokens > 0:
            if self.quantize_kv:
                k_p, k_s, k_m = self.cpu_keys
                v_p, v_s, v_m = self.cpu_values
                bytes_in = (
                    k_p.nelement() * k_p.element_size()
                    + k_s.nelement() * k_s.element_size()
                    + k_m.nelement() * k_m.element_size()
                    + v_p.nelement() * v_p.element_size()
                    + v_s.nelement() * v_s.element_size()
                    + v_m.nelement() * v_m.element_size()
                )
            else:
                bytes_in = (self.cpu_keys.nelement() + self.cpu_values.nelement()) * self.cpu_keys.element_size()
            self.total_bytes_transferred_pcie += bytes_in

        return full_k, full_v

    def prune(self) -> None:
        """Explicitly offload excess tokens to CPU after chunked prefill."""
        self._offload_excess_to_cpu()

    def get_seq_length(self) -> int:
        return self.total_tokens_seen

    def get_max_length(self) -> int:
        return self.gpu_cache_tokens + self.cpu_cache_tokens


class FlexGenKVCache(Cache):
    """Cache implementation for FlexGen multi-tier KV offloading with 4-bit group quantization."""

    is_flexgen = True

    def __init__(
        self,
        gpu_cache_tokens: int = 1024,
        cpu_cache_tokens: int = 32768,
        quantize_kv: bool = True,
        group_size: int = 64,
        cpu_delegation: bool = False,
    ):
        if int(gpu_cache_tokens) < 0:
            raise ValueError(f"gpu_cache_tokens must be >= 0, got {gpu_cache_tokens}")
        if int(cpu_cache_tokens) <= 0:
            raise ValueError(f"cpu_cache_tokens must be > 0, got {cpu_cache_tokens}")
        if int(group_size) <= 0:
            raise ValueError(f"group_size must be > 0, got {group_size}")

        self.gpu_cache_tokens = int(gpu_cache_tokens)
        self.cpu_cache_tokens = int(cpu_cache_tokens)
        self.quantize_kv = bool(quantize_kv)
        self.group_size = int(group_size)
        self.cpu_delegation = bool(cpu_delegation)

        super().__init__(
            layer_class_to_replicate=lambda: FlexGenKVLayer(
                gpu_cache_tokens=self.gpu_cache_tokens,
                cpu_cache_tokens=self.cpu_cache_tokens,
                quantize_kv=self.quantize_kv,
                group_size=self.group_size,
                cpu_delegation=self.cpu_delegation,
            )
        )
        set_active_flexgen_cache(self)

    @property
    def total_bytes_transferred_pcie(self) -> int:
        """Total bytes moved over PCIe bus across all initialized layers."""
        return sum(getattr(layer, "total_bytes_transferred_pcie", 0) for layer in self.layers)

    @property
    def active_gpu_tokens(self) -> int:
        if not self.layers or self.layers[0].gpu_keys is None:
            return 0
        return self.layers[0].gpu_keys.shape[-2]

    @property
    def offloaded_cpu_tokens(self) -> int:
        if not self.layers:
            return 0
        total = self.layers[0].get_seq_length()
        gpu = self.active_gpu_tokens
        return max(0, total - gpu)

    def prune_after_chunk(self) -> None:
        """Trigger explicit offload across all layers following prefill."""
        for layer in self.layers:
            if hasattr(layer, "prune"):
                layer.prune()

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_max_length(self) -> int:
        return self.gpu_cache_tokens + self.cpu_cache_tokens

    def __iter__(self) -> Iterator[tuple[Any, Any, Any]]:
        for layer in self.layers:
            yield getattr(layer, "keys", None), getattr(layer, "values", None), None

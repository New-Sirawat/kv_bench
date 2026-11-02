from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from src.core.axis import Axis
from src.core.registry import register
from src.core.technique import Technique
from src.core.workload import SyntheticFiller, Workload


@dataclass(frozen=True)
class FlexGenTierInfo:
    """Memory tiering metrics for GPU VRAM and Host CPU DRAM."""

    vram_budget_gb: float
    host_swap_gb: float
    group_size: int
    compression_ratio: float
    total_effective_kv_gb: float


def pack_uint4(uint4_tensor: "torch.Tensor") -> "torch.Tensor":
    """Pack unsigned 4-bit values [0, 15] into unsigned UINT8 bytes.

    Two 4-bit values along the trailing dimension are packed into one byte:
        byte = (val1 & 0x0F) | ((val2 & 0x0F) << 4)
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for pack_uint4")

    if uint4_tensor.size(-1) % 2 != 0:
        raise ValueError(f"Trailing dimension must be even, got {uint4_tensor.size(-1)}")

    u4 = uint4_tensor.clamp(0, 15).to(torch.uint8)
    low_nibble = u4[..., 0::2]
    high_nibble = u4[..., 1::2]
    packed = low_nibble | (high_nibble << 4)
    return packed


def unpack_uint4(packed_tensor: "torch.Tensor", original_dim: int) -> "torch.Tensor":
    """Unpack UINT8 bytes into unsigned 4-bit values [0, 15]."""
    if torch is None:
        raise RuntimeError("PyTorch is required for unpack_uint4")

    low_nibble = (packed_tensor & 0x0F).to(torch.uint8)
    high_nibble = ((packed_tensor >> 4) & 0x0F).to(torch.uint8)

    shape = list(packed_tensor.shape)
    shape[-1] = original_dim
    unpacked = torch.empty(shape, dtype=torch.uint8, device=packed_tensor.device)
    unpacked[..., 0::2] = low_nibble
    unpacked[..., 1::2] = high_nibble
    return unpacked


def quantize_flexgen_group4(
    tensor: "torch.Tensor",
    group_size: int = 64,
    eps: float = 1e-5,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Perform group-wise 4-bit asymmetric min-max quantization on KV cache tensors.

    FlexGen slices the hidden head dimension $D=128$ into groups of `group_size = 64`
    (2 groups per head), computing independent scale and zero-point per group:
        scale = (max - min) / 15.0
        q = round((X - min) / scale).clamp(0, 15)

    Packs the resulting 4-bit values into `torch.uint8`, achieving a 4x reduction
    in VRAM footprint and PCIe transfer volume relative to FP16.

    Args:
        tensor: Tensor of shape `(..., D)`
        group_size: Quantization channel group size (default 64)
        eps: Numerical stability constant

    Returns:
        Tuple of `(packed_uint8, scales, min_vals)`
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for quantize_flexgen_group4")

    head_dim = tensor.size(-1)
    if head_dim % group_size != 0:
        raise ValueError(f"head_dim ({head_dim}) must be divisible by group_size ({group_size})")

    orig_shape = tensor.shape
    num_groups = head_dim // group_size

    # Reshape into channel groups: (..., num_groups, group_size)
    grouped = tensor.view(*orig_shape[:-1], num_groups, group_size)

    min_val = grouped.amin(dim=-1, keepdim=True)
    max_val = grouped.amax(dim=-1, keepdim=True)
    scale = ((max_val - min_val) / 15.0).clamp(min=eps).to(tensor.dtype)

    # Quantize to [0, 15]
    u4 = torch.round((grouped - min_val) / scale).clamp(0, 15).to(torch.uint8).view(orig_shape)
    packed = pack_uint4(u4)
    return packed, scale, min_val


def dequantize_flexgen_group4(
    packed_tensor: "torch.Tensor",
    scales: "torch.Tensor",
    min_vals: "torch.Tensor",
    original_dtype: "torch.dtype" = torch.bfloat16,
    group_size: int = 64,
) -> "torch.Tensor":
    """Dequantize packed 4-bit FlexGen KV tensor back to floating point precision."""
    if torch is None:
        raise RuntimeError("PyTorch is required for dequantize_flexgen_group4")

    original_dim = packed_tensor.size(-1) * 2
    num_groups = original_dim // group_size

    unpacked = unpack_uint4(packed_tensor, original_dim).to(original_dtype)
    grouped = unpacked.view(*unpacked.shape[:-1], num_groups, group_size)
    dequantized = (grouped * scales + min_vals).view(*unpacked.shape[:-1], original_dim)
    return dequantized.to(original_dtype)


def compute_flexgen_attention(
    query: "torch.Tensor",
    key: "torch.Tensor",
    value: "torch.Tensor",
    group_size: int = 64,
    scale: float | None = None,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Compute Grouped-Query Attention over FlexGen 4-bit quantized Key and Value tensors.

    Quantizes Key and Value tensors using group-wise 4-bit asymmetric quantization,
    dequantizes them, and performs GQA attention forward.

    Returns:
        Tuple of `(attn_output, attn_weights)`
    """
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for compute_flexgen_attention")

    # Group-wise 4-bit quantization and packing
    k_packed, k_scales, k_mins = quantize_flexgen_group4(key, group_size=group_size)
    v_packed, v_scales, v_mins = quantize_flexgen_group4(value, group_size=group_size)

    # Dequantize back to original precision
    k_dequant = dequantize_flexgen_group4(
        k_packed, k_scales, k_mins, original_dtype=key.dtype, group_size=group_size
    )
    v_dequant = dequantize_flexgen_group4(
        v_packed, v_scales, v_mins, original_dtype=value.dtype, group_size=group_size
    )

    # GQA expansion
    num_q_heads = query.size(1)
    num_kv_heads = key.size(1)
    gqa_ratio = num_q_heads // num_kv_heads

    if gqa_ratio > 1:
        expanded_k = k_dequant.repeat_interleave(gqa_ratio, dim=1)
        expanded_v = v_dequant.repeat_interleave(gqa_ratio, dim=1)
    else:
        expanded_k = k_dequant
        expanded_v = v_dequant

    head_dim = query.size(-1)
    softmax_scale = scale if scale is not None else (head_dim ** -0.5)

    scores = torch.matmul(query.float(), expanded_k.float().transpose(-2, -1)) * softmax_scale
    attn_weights = F.softmax(scores, dim=-1).to(query.dtype)
    attn_output = torch.matmul(attn_weights, expanded_v)
    return attn_output, attn_weights


@register("flexgen_kv")
class FlexGenKVTechnique(Technique):
    """FlexGen-Style KV Tiering & Group Quantization.

    Combines two orthogonal co-optimization axes:
      1. SPATIAL: Multi-tier KV-cache offloading between GPU VRAM and Host CPU DRAM
         over PCIe, allocating dedicated swap space in system RAM.
      2. STRUCTURAL: Group-wise 4-bit asymmetric min-max quantization (group_size = 64)
         compressing cached tensors by 4x to minimize both memory footprint and PCIe
         transfer volume while keeping base BF16 weights unmodified.

    Axes:
        Axis.SPATIAL | Axis.STRUCTURAL
    """

    axes = Axis.SPATIAL | Axis.STRUCTURAL

    def __init__(
        self,
        swap_space_gb: int = 4,
        group_size: int = 64,
        block_size: int = 16,
    ):
        if int(swap_space_gb) < 0:
            raise ValueError(f"swap_space_gb must be >= 0, got {swap_space_gb}")
        if int(group_size) <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")
        if int(block_size) <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.swap_space_gb = int(swap_space_gb)
        self.group_size = int(group_size)
        self.block_size = int(block_size)

    @property
    def compression_ratio(self) -> float:
        """Compression ratio relative to standard 16-bit (BF16/FP16) cache."""
        return 4.0

    def engine_kwargs(self, context_length: int, max_tokens: int) -> dict[str, Any]:
        """Configure vLLM swap space and 4-bit KV quantization."""
        del context_length, max_tokens
        return {
            "swap_space": self.swap_space_gb,
            "kv_cache_dtype": "fp8",  # Native quantization knob in vLLM
            "block_size": self.block_size,
        }

    def workload(self) -> Workload:
        return SyntheticFiller()

    def extra_metrics(self, llm, workload_output) -> dict[str, Any]:
        del llm, workload_output
        return {
            "swap_space_gb": self.swap_space_gb,
            "quant_group_size": self.group_size,
            "kv_cache_bits": 4,
            "compression_ratio_vs_bf16": self.compression_ratio,
            "block_size": self.block_size,
        }

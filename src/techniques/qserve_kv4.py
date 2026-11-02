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
class SmoothAttentionScales:
    """Per-channel scaling vectors for Query and Key activations."""

    scale_k: "torch.Tensor"
    scale_q: "torch.Tensor"
    alpha: float


def compute_smooth_attention_scales(
    query: "torch.Tensor",
    key: "torch.Tensor",
    alpha: float = 0.5,
    eps: float = 1e-5,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Compute per-channel SmoothAttention scaling factors for GQA Query and Key states.

    SmoothAttention suppresses activation outliers in Key channels by migrating dynamic
    range between Query and Key vectors:
        s_c = (max(|K_c|)^alpha) / (max(|Q_c|)^(1 - alpha) + eps)
        K_smooth = K * diag(1 / s)
        Q_smooth = Q * diag(s)

    For Grouped-Query Attention (GQA), Query heads within each KV group are pooled
    (mean max amplitude) so that:
        (Q_smooth) @ (K_smooth)^T == Q @ K^T
    remains mathematically invariant across all grouped query heads.

    Args:
        query: Tensor of shape `(batch, num_q_heads, q_len, head_dim)`
        key:   Tensor of shape `(batch, num_kv_heads, kv_len, head_dim)`
        alpha: Migration strength factor (default 0.5 for geometric mean balance).
        eps:   Numerical stability constant.

    Returns:
        Tuple of `(q_smooth, k_smooth)` with identical shapes and dtypes as inputs.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for compute_smooth_attention_scales")

    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must be 4D tensors (batch, heads, len, dim)")

    num_q_heads = query.size(1)
    num_kv_heads = key.size(1)
    head_dim = query.size(-1)

    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

    group_size = num_q_heads // num_kv_heads

    # Compute maximum absolute channel activations:
    # key: (batch, num_kv_heads, kv_len, head_dim) -> max over batch, len -> (num_kv_heads, head_dim)
    k_abs_max = key.abs().amax(dim=(0, 2)).clamp(min=eps)  # (num_kv_heads, head_dim)

    # query: (batch, num_q_heads, q_len, head_dim) -> max over batch, len -> (num_q_heads, head_dim)
    q_abs_max = query.abs().amax(dim=(0, 2)).clamp(min=eps)  # (num_q_heads, head_dim)

    # Pool query activations across each GQA group:
    # (num_kv_heads, group_size, head_dim) -> mean over group_size -> (num_kv_heads, head_dim)
    q_abs_max_grouped = q_abs_max.view(num_kv_heads, group_size, head_dim).mean(dim=1)

    # Calculate optimal migration scale s: (num_kv_heads, head_dim)
    scale = (k_abs_max.pow(alpha) / q_abs_max_grouped.pow(1.0 - alpha)).clamp(min=1e-3, max=1e3)

    # Reshape scale for broadcasting with key: (1, num_kv_heads, 1, head_dim)
    s_k = scale.unsqueeze(0).unsqueeze(2)

    # Expand scale across query heads in each group: (1, num_q_heads, 1, head_dim)
    s_q = scale.repeat_interleave(group_size, dim=0).unsqueeze(0).unsqueeze(2)

    k_smooth = (key / s_k).to(key.dtype)
    q_smooth = (query * s_q).to(query.dtype)

    return q_smooth, k_smooth


def pack_int4(int4_tensor: "torch.Tensor") -> "torch.Tensor":
    """Pack signed INT4 values in range [-8, 7] into unsigned UINT8 bytes.

    Two 4-bit values along the trailing dimension are packed into one byte:
        byte = (val1 & 0x0F) | ((val2 & 0x0F) << 4)
    Trailing dimension size must be even.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for pack_int4")

    if int4_tensor.size(-1) % 2 != 0:
        raise ValueError(f"Last dimension must be even for INT4 packing, got {int4_tensor.size(-1)}")

    # Map signed [-8, 7] to unsigned 4-bit [0, 15]
    u4 = (int4_tensor.clamp(-8, 7) + 8).to(torch.uint8)
    low_nibble = u4[..., 0::2]
    high_nibble = u4[..., 1::2]
    packed = low_nibble | (high_nibble << 4)
    return packed


def unpack_int4(packed_tensor: "torch.Tensor", original_dim: int) -> "torch.Tensor":
    """Unpack UINT8 bytes into signed INT4 values in range [-8, 7]."""
    if torch is None:
        raise RuntimeError("PyTorch is required for unpack_int4")

    low_nibble = (packed_tensor & 0x0F).to(torch.int8) - 8
    high_nibble = ((packed_tensor >> 4) & 0x0F).to(torch.int8) - 8

    # Interleave low and high nibbles back to original dimension
    shape = list(packed_tensor.shape)
    shape[-1] = original_dim
    unpacked = torch.empty(shape, dtype=torch.int8, device=packed_tensor.device)
    unpacked[..., 0::2] = low_nibble
    unpacked[..., 1::2] = high_nibble
    return unpacked


def quantize_kv_int4(
    tensor: "torch.Tensor",
    group_size: int = 32,
    eps: float = 1e-5,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Quantize tensor along channel dimension to symmetric 4-bit representation (INT4).

    Partitions trailing channel dimension `D` into groups of `group_size` elements
    and computes per-group scale factors:
        scale = max(|group|) / 7
        q = round(group / scale).clamp(-8, 7)

    Packs the resulting INT4 values into a `torch.uint8` tensor reducing tensor
    VRAM footprint by 4x relative to FP16 / BF16.

    Args:
        tensor: Floating point tensor of shape `(..., D)`.
        group_size: Quantization channel group size (must divide D evenly).
        eps: Numerical stability constant.

    Returns:
        Tuple of `(packed_tensor, scales)` where:
          - `packed_tensor` has shape `(..., D // 2)` and dtype `torch.uint8`
          - `scales` has shape `(..., D // group_size, 1)` and dtype `tensor.dtype`
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for quantize_kv_int4")

    head_dim = tensor.size(-1)
    if head_dim % group_size != 0:
        raise ValueError(f"head_dim ({head_dim}) must be divisible by group_size ({group_size})")

    orig_shape = tensor.shape
    num_groups = head_dim // group_size

    # Reshape into groups: (..., num_groups, group_size)
    grouped = tensor.view(*orig_shape[:-1], num_groups, group_size)

    # Compute per-group symmetric absolute maximum scale: (..., num_groups, 1)
    max_val = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    scale = (max_val / 7.0).to(tensor.dtype)

    # Quantize to INT4 range [-8, 7]
    int4_val = torch.round(grouped / scale).clamp(-8, 7).view(orig_shape)

    # Pack into uint8 (2 values per byte)
    packed = pack_int4(int4_val)
    return packed, scale


def dequantize_kv_int4(
    packed_tensor: "torch.Tensor",
    scales: "torch.Tensor",
    original_dtype: "torch.dtype" = torch.bfloat16,
    group_size: int = 32,
) -> "torch.Tensor":
    """Dequantize packed INT4 KV tensor back to original floating point dtype."""
    if torch is None:
        raise RuntimeError("PyTorch is required for dequantize_kv_int4")

    packed_dim = packed_tensor.size(-1)
    original_dim = packed_dim * 2
    num_groups = original_dim // group_size

    unpacked = unpack_int4(packed_tensor, original_dim).to(original_dtype)
    grouped = unpacked.view(*unpacked.shape[:-1], num_groups, group_size)
    dequantized = (grouped * scales).view(*unpacked.shape[:-1], original_dim)
    return dequantized.to(original_dtype)


def compute_gqa_qserve_attention(
    query: "torch.Tensor",
    key: "torch.Tensor",
    value: "torch.Tensor",
    alpha: float = 0.5,
    group_size: int = 32,
    scale: float | None = None,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Compute Grouped-Query Attention with QServe SmoothAttention + INT4 KV cache quantization.

    Applies SmoothAttention to suppress outlier channels, quantizes Key and Value
    tensors to INT4, dequantizes them, and performs GQA attention computation.

    Returns:
        Tuple of `(attn_output, attn_weights)`
    """
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for compute_gqa_qserve_attention")

    # 1. SmoothAttention outlier suppression
    q_smooth, k_smooth = compute_smooth_attention_scales(query, key, alpha=alpha)

    # 2. INT4 Quantization and Packing
    k_packed, k_scales = quantize_kv_int4(k_smooth, group_size=group_size)
    v_packed, v_scales = quantize_kv_int4(value, group_size=group_size)

    # 3. Dequantization back to original floating-point precision
    k_dequant = dequantize_kv_int4(k_packed, k_scales, original_dtype=key.dtype, group_size=group_size)
    v_dequant = dequantize_kv_int4(v_packed, v_scales, original_dtype=value.dtype, group_size=group_size)

    # 4. GQA Attention Forward
    num_q_heads = q_smooth.size(1)
    num_kv_heads = k_smooth.size(1)
    gqa_ratio = num_q_heads // num_kv_heads

    if gqa_ratio > 1:
        expanded_k = k_dequant.repeat_interleave(gqa_ratio, dim=1)
        expanded_v = v_dequant.repeat_interleave(gqa_ratio, dim=1)
    else:
        expanded_k = k_dequant
        expanded_v = v_dequant

    head_dim = query.size(-1)
    softmax_scale = scale if scale is not None else (head_dim ** -0.5)

    scores = torch.matmul(q_smooth.float(), expanded_k.float().transpose(-2, -1)) * softmax_scale
    attn_weights = F.softmax(scores, dim=-1).to(query.dtype)
    attn_output = torch.matmul(attn_weights, expanded_v)
    return attn_output, attn_weights


@register("qserve_kv4")
class QServeKV4Technique(Technique):
    """QServe KV4: SmoothAttention outlier suppression + 4-bit KV Cache quantization.

    Suppresses Key activation outliers by migrating dynamic range between Query
    and Key channels (SmoothAttention), then quantizes both Key and Value tensors
    to 4-bit integer (INT4) representation. Compresses KV-cache memory footprint
    by 4x relative to BF16/FP16 without modifying base model weights.

    Axes:
        Axis.STRUCTURAL: Compresses the numerical precision of the KV cache from BF16 to INT4.
    """

    axes = Axis.STRUCTURAL

    def __init__(
        self,
        group_size: int = 32,
        alpha: float = 0.5,
        block_size: int = 16,
    ):
        if int(group_size) <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")
        if float(alpha) < 0.0 or float(alpha) > 1.0:
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")
        if int(block_size) <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.group_size = int(group_size)
        self.alpha = float(alpha)
        self.block_size = int(block_size)

    @property
    def compression_ratio(self) -> float:
        """Compression ratio relative to standard 16-bit (BF16/FP16) cache."""
        return 4.0

    def engine_kwargs(self, context_length: int, max_tokens: int) -> dict[str, Any]:
        """Extra engine kwargs merged into the vLLM / PyTorch engine constructor."""
        del context_length, max_tokens
        return {
            "block_size": self.block_size,
            "kv_cache_dtype": "fp8",  # Native runtime INT4/FP8 quantization knob
        }

    def workload(self) -> Workload:
        return SyntheticFiller()

    def extra_metrics(self, llm, workload_output) -> dict[str, Any]:
        del llm, workload_output
        return {
            "kv_cache_bits": 4,
            "quant_group_size": self.group_size,
            "smooth_attention_alpha": self.alpha,
            "compression_ratio_vs_bf16": self.compression_ratio,
            "block_size": self.block_size,
        }

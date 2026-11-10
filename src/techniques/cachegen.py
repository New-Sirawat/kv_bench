from __future__ import annotations

import math
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


def encode_anchor_deltas(
    tensor: "torch.Tensor",
    group_size: int = 10,
) -> tuple["torch.Tensor", "torch.Tensor", int]:
    """Compute anchor-relative delta representations along the sequence dimension.

    For every consecutive group of `group_size` tokens:
      - token 0 is preserved as the Anchor token.
      - tokens 1..(group_size-1) store deltas: delta_i = token_i - token_0.

    Because consecutive tokens in language share similar internal attention patterns,
    delta representations exhibit 2.4x - 2.9x lower variance, clustering around zero.

    Args:
        tensor: Shape (batch, heads, seq_len, head_dim).
        group_size: Tokens per anchor group (default: 10).

    Returns:
        Tuple of (anchors, deltas, orig_seq_len).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for encode_anchor_deltas")

    seq_len = tensor.size(-2)
    head_dim = tensor.size(-1)
    num_groups = math.ceil(seq_len / group_size)

    pad_len = num_groups * group_size - seq_len
    if pad_len > 0:
        tensor_padded = F.pad(tensor, (0, 0, 0, pad_len))
    else:
        tensor_padded = tensor

    grouped = tensor_padded.contiguous().reshape(*tensor.shape[:-2], num_groups, group_size, head_dim)

    anchors = grouped[..., 0:1, :].contiguous()
    deltas = grouped - anchors

    return anchors, deltas, seq_len


def decode_anchor_deltas(
    anchors: "torch.Tensor",
    deltas: "torch.Tensor",
    original_seq_len: int,
) -> "torch.Tensor":
    """Reconstruct continuous tensors from anchors and deltas in parallel."""
    if torch is None:
        raise RuntimeError("PyTorch is required for decode_anchor_deltas")

    reconstructed = anchors + deltas
    shape = list(reconstructed.shape)
    num_groups = shape[-3]
    group_size = shape[-2]
    head_dim = shape[-1]

    flat_seq = reconstructed.contiguous().reshape(*shape[:-3], num_groups * group_size, head_dim)
    return flat_seq[..., :original_seq_len, :].contiguous()


def quantize_deltas_tiered(
    deltas: "torch.Tensor",
    layer_idx: int,
    total_layers: int = 32,
    shallow_bits: int = 8,
    mid_bits: int = 4,
    deep_bits: int = 2,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", int]:
    """Apply layer-varying quantization binning to anchor delta representations.

    Partitions transformer layers into three sensitivity tiers:
      - Shallow (1 .. L/3): High precision (e.g. 8-bit), protecting sensitive syntax.
      - Middle (L/3+1 .. 2L/3): Moderate precision (e.g. 4-bit).
      - Deep (2L/3+1 .. L): Aggressive lossy precision (e.g. 2-bit), tolerating coarse rounding.

    Returns:
        Tuple of (quantized_integers, scales, dequantized_deltas, effective_bits).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for quantize_deltas_tiered")

    tier_cutoff_1 = total_layers // 3
    tier_cutoff_2 = (2 * total_layers) // 3

    if layer_idx < tier_cutoff_1:
        bits = shallow_bits
    elif layer_idx < tier_cutoff_2:
        bits = mid_bits
    else:
        bits = deep_bits

    max_val = (1 << (bits - 1)) - 1
    min_val = -(1 << (bits - 1))

    scale = deltas.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / float(max_val)
    q = torch.round(deltas / scale).clamp(min_val, max_val)
    dequant = q.to(deltas.dtype) * scale.to(deltas.dtype)

    return q, scale, dequant, bits


def compute_gqa_cachegen_attention(
    query: "torch.Tensor",
    key: "torch.Tensor",
    value: "torch.Tensor",
    layer_idx: int = 16,
    total_layers: int = 32,
    group_size: int = 10,
    shallow_bits: int = 8,
    mid_bits: int = 4,
    deep_bits: int = 2,
    scale: float | None = None,
) -> tuple["torch.Tensor", "torch.Tensor", dict[str, Any]]:
    """Compute GQA attention using CacheGen anchor-delta compression & layer tiering."""
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for compute_gqa_cachegen_attention")

    d = query.size(-1)
    orig_seq_len = key.size(-2)

    # 1. Delta encoding for Key and Value
    k_anchors, k_deltas, _ = encode_anchor_deltas(key, group_size=group_size)
    v_anchors, v_deltas, _ = encode_anchor_deltas(value, group_size=group_size)

    # 2. Tiered quantization based on layer depth
    _, _, k_deltas_dequant, k_bits = quantize_deltas_tiered(
        k_deltas, layer_idx, total_layers, shallow_bits, mid_bits, deep_bits
    )
    _, _, v_deltas_dequant, _ = quantize_deltas_tiered(
        v_deltas, layer_idx, total_layers, shallow_bits, mid_bits, deep_bits
    )

    # 3. Fast parallel reconstruction
    k_recon = decode_anchor_deltas(k_anchors, k_deltas_dequant, orig_seq_len)
    v_recon = decode_anchor_deltas(v_anchors, v_deltas_dequant, orig_seq_len)

    # 4. GQA pooling / repeat
    num_q_heads = query.size(1)
    num_kv_heads = key.size(1)
    gqa_ratio = num_q_heads // num_kv_heads

    if gqa_ratio > 1:
        expanded_k = k_recon.repeat_interleave(gqa_ratio, dim=1)
        expanded_v = v_recon.repeat_interleave(gqa_ratio, dim=1)
    else:
        expanded_k = k_recon
        expanded_v = v_recon

    softmax_scale = scale if scale is not None else (d ** -0.5)
    scores = torch.matmul(query.float(), expanded_k.float().transpose(-2, -1)) * softmax_scale
    attn_weights = F.softmax(scores, dim=-1).to(query.dtype)
    attn_output = torch.matmul(attn_weights, expanded_v)

    telemetry = {
        "layer_idx": layer_idx,
        "effective_bits": k_bits,
        "anchor_group_size": group_size,
        "delta_encoding_active": True,
    }

    return attn_output, attn_weights, telemetry


@register("cachegen")
class CacheGenTechnique(Technique):
    """CacheGen: KV Cache Compression and Streaming for Fast LLM Serving.

    Applies anchor-based delta residual encoding and layer-tiered precision:
      1. Non-anchor tokens in groups of G=10 are stored as differences relative to a shared
         high-precision anchor token, reducing tensor variance by ~2.5x.
      2. Layer-varying quantization bins early layers at higher precision (8-bit) and deep
         layers at aggressive compression (2-bit), preventing compounding error.
      3. Compresses KV cache representation by 3.5x to 4.3x while maintaining downstream accuracy.

    Axes:
        Axis.STRUCTURAL | Axis.SPATIAL
    """

    axes = Axis.STRUCTURAL | Axis.SPATIAL

    def __init__(
        self,
        anchor_group_size: int = 10,
        shallow_bits: int = 8,
        mid_bits: int = 4,
        deep_bits: int = 2,
        block_size: int = 16,
    ):
        if int(anchor_group_size) <= 0:
            raise ValueError(f"anchor_group_size must be positive, got {anchor_group_size}")
        if int(shallow_bits) <= 0 or int(mid_bits) <= 0 or int(deep_bits) <= 0:
            raise ValueError("All bitwidth parameters must be positive integers")
        if int(block_size) <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.anchor_group_size = int(anchor_group_size)
        self.shallow_bits = int(shallow_bits)
        self.mid_bits = int(mid_bits)
        self.deep_bits = int(deep_bits)
        self.block_size = int(block_size)

    @property
    def compression_ratio(self) -> float:
        """Average compression ratio across all layer tiers relative to 16-bit FP16."""
        mean_bits = (self.shallow_bits + self.mid_bits + self.deep_bits) / 3.0
        return round(16.0 / mean_bits, 2)

    def engine_kwargs(self, context_length: int, max_tokens: int) -> dict[str, Any]:
        """Extra engine kwargs merged into the vLLM / PyTorch constructor."""
        del context_length, max_tokens
        return {
            "block_size": self.block_size,
            "kv_cache_dtype": "fp8",
            "swap_space": 4,
        }

    def workload(self) -> Workload:
        return SyntheticFiller()

    def pytorch_cache_factory(self) -> Any:
        from src.techniques.cachegen_cache import CacheGenCache

        return lambda: CacheGenCache(
            anchor_group_size=self.anchor_group_size,
            shallow_bits=self.shallow_bits,
            mid_bits=self.mid_bits,
            deep_bits=self.deep_bits,
        )

    def extra_metrics(self, llm, workload_output) -> dict[str, Any]:
        del llm, workload_output
        from src.techniques.cachegen_cache import get_active_cachegen_cache

        active_cache = get_active_cachegen_cache()
        compressed_bytes = (
            active_cache.total_compressed_bytes if active_cache is not None else 0
        )
        return {
            "anchor_group_size": self.anchor_group_size,
            "shallow_bits": self.shallow_bits,
            "mid_bits": self.mid_bits,
            "deep_bits": self.deep_bits,
            "compression_ratio_vs_bf16": self.compression_ratio,
            "block_size": self.block_size,
            "compressed_bitstream_bytes": compressed_bytes,
            "delta_encoding_active": True,
            "layer_tiered_quantization_active": True,
        }

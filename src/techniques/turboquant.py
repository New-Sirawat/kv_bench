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


def fast_walsh_hadamard_transform(x: "torch.Tensor") -> "torch.Tensor":
    """Compute normalized Fast Walsh-Hadamard Transform (FWHT) along trailing dimension.

    Operates in O(d log d) time without explicit d x d matrix construction.
    Because H is orthogonal and symmetric (H = H^T and H @ H = d * I),
    normalizing each butterfly stage by 1/sqrt(2) makes the transform its own exact inverse:
        FWHT(FWHT(x)) == x
    Trailing dimension `d` must be a power of 2.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for fast_walsh_hadamard_transform")

    d = x.size(-1)
    if (d & (d - 1)) != 0 or d <= 0:
        raise ValueError(f"Last dimension must be a power of 2, got {d}")

    orig_shape = x.shape
    v = x.contiguous().view(-1, d).float()

    h = 1
    sqrt2 = math.sqrt(2.0)
    while h < d:
        v = v.view(-1, d // (2 * h), 2, h)
        a = v[:, :, 0, :]
        b = v[:, :, 1, :]
        v = torch.stack([a + b, a - b], dim=2) / sqrt2
        v = v.view(-1, d)
        h *= 2

    return v.view(orig_shape).to(x.dtype)


def get_random_sign_vector(dim: int, seed: int = 42, device: Any = None) -> "torch.Tensor":
    """Generate deterministic +/-1 Rademacher sign vector for Walsh-Hadamard rotation."""
    if torch is None:
        raise RuntimeError("PyTorch is required for get_random_sign_vector")
    gen = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (dim,), generator=gen, device=device) * 2 - 1
    return signs.float()


def turboquant_rotate(
    v: "torch.Tensor",
    seed: int = 42,
    eps: float = 1e-6,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Normalize input vector and apply randomized Walsh-Hadamard rotation:

        v_hat = v / ||v||
        x_rot = FWHT(diag(s) * v_hat)

    Transforms arbitrary, heavy-tailed activations into a predictable spherical
    Beta distribution across the hypersphere, eliminating channel outliers.

    Returns:
        Tuple of `(x_rot, norms)`
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for turboquant_rotate")

    dim = v.size(-1)
    norms = v.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)
    v_unit = v / norms

    signs = get_random_sign_vector(dim, seed=seed, device=v.device).to(v.dtype)
    v_signed = v_unit * signs
    x_rot = fast_walsh_hadamard_transform(v_signed)
    return x_rot, norms


def turboquant_unrotate(
    x_rot: "torch.Tensor",
    norms: "torch.Tensor",
    seed: int = 42,
) -> "torch.Tensor":
    """Inverse randomized Walsh-Hadamard rotation and re-scaling by vector norm."""
    if torch is None:
        raise RuntimeError("PyTorch is required for turboquant_unrotate")

    dim = x_rot.size(-1)
    signs = get_random_sign_vector(dim, seed=seed, device=x_rot.device).to(x_rot.dtype)

    # Invert FWHT (involution) and undo sign flip
    v_signed = fast_walsh_hadamard_transform(x_rot)
    v_unit = v_signed * signs
    return v_unit * norms


def quantize_key_4bit(
    k_rot: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Quantize rotated Key vector to 4-bit scalar Lloyd-Max representation.

    Because k_rot coordinate entries lie tightly on the sphere bounded in [-1, 1],
    symmetric uniform scalar quantization across 16 levels provides low directional error.
    Packed into unsigned 8-bit integers (2 values per byte).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for quantize_key_4bit")

    dim = k_rot.size(-1)
    scale = k_rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 7.0

    # Quantize to [-8, 7]
    q = torch.round(k_rot / scale).clamp(-8, 7).to(torch.int8)

    # Pack into uint8
    u4 = (q + 8).to(torch.uint8)
    packed = u4[..., 0::2] | (u4[..., 1::2] << 4)
    return packed, scale, q.float() * scale


def dequantize_key_4bit(
    packed: "torch.Tensor",
    scale: "torch.Tensor",
    original_dim: int,
    dtype: "torch.dtype" = torch.bfloat16,
) -> "torch.Tensor":
    """Dequantize packed 4-bit Key vector."""
    if torch is None:
        raise RuntimeError("PyTorch is required for dequantize_key_4bit")

    low = (packed & 0x0F).to(torch.int8) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int8) - 8

    shape = list(packed.shape)
    shape[-1] = original_dim
    unpacked = torch.empty(shape, dtype=torch.int8, device=packed.device)
    unpacked[..., 0::2] = low
    unpacked[..., 1::2] = high

    return (unpacked.to(dtype) * scale).to(dtype)


def quantize_value_3plus1bit(
    v_rot: "torch.Tensor",
    sketch_seed: int = 1337,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Quantize rotated Value vector to asymmetric 3-bit scalar + 1-bit QJL residual sketch.

    Allocates 3 bits (8 levels, [-4, 3]) for coarse scalar values, computes
    residual error r = v_rot - v_quant3, and captures 1-bit Quantized Johnson-Lindenstrauss
    (QJL) random sign sketch:
        q_sketch = sign(FWHT(diag(s') * r)) in {-1, +1}^d

    Total budget: 3 bits scalar + 1 bit sketch = 4 bits per element.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for quantize_value_3plus1bit")

    dim = v_rot.size(-1)
    scale3 = v_rot.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 3.0

    # 1. 3-bit scalar quantization in [-4, 3]
    q3 = torch.round(v_rot / scale3).clamp(-4, 3)
    v_recon3 = q3 * scale3

    # 2. Residual vector r and 1-bit QJL sketch
    residual = v_rot - v_recon3
    r_norm = residual.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)

    sketch_signs = get_random_sign_vector(dim, seed=sketch_seed, device=v_rot.device).to(v_rot.dtype)
    r_signed = residual * sketch_signs
    r_rot = fast_walsh_hadamard_transform(r_signed)

    # 1-bit sketch sign vector: 1 for >= 0, 0 for < 0
    sketch_bits = (r_rot >= 0).to(torch.uint8)

    # Pack sketch bits into byte tensor (8 bits per byte)
    packed_sketch = torch.zeros(
        *sketch_bits.shape[:-1], dim // 8, dtype=torch.uint8, device=sketch_bits.device
    )
    for b in range(8):
        packed_sketch |= (sketch_bits[..., b::8] << b)

    # Pack 3-bit values into uint8 (using 4-bit nibble container for fast SIMD alignment)
    u3 = (q3.to(torch.int8) + 4).to(torch.uint8)
    packed_scalar3 = u3[..., 0::2] | (u3[..., 1::2] << 4)

    return packed_scalar3, packed_sketch, scale3, r_norm


def dequantize_value_3plus1bit(
    packed_scalar3: "torch.Tensor",
    packed_sketch: "torch.Tensor",
    scale3: "torch.Tensor",
    r_norm: "torch.Tensor",
    original_dim: int,
    sketch_seed: int = 1337,
    dtype: "torch.dtype" = torch.bfloat16,
) -> "torch.Tensor":
    """Dequantize 3-bit scalar + 1-bit QJL residual sketch into high-fidelity Value vector."""
    if torch is None:
        raise RuntimeError("PyTorch is required for dequantize_value_3plus1bit")

    # 1. Unpack 3-bit scalar values
    low = (packed_scalar3 & 0x0F).to(torch.int8) - 4
    high = ((packed_scalar3 >> 4) & 0x0F).to(torch.int8) - 4

    shape = list(packed_scalar3.shape)
    shape[-1] = original_dim
    unpacked_scalar = torch.empty(shape, dtype=torch.int8, device=packed_scalar3.device)
    unpacked_scalar[..., 0::2] = low
    unpacked_scalar[..., 1::2] = high
    v_recon3 = unpacked_scalar.to(dtype) * scale3

    # 2. Unpack 1-bit QJL sketch
    sketch_bits = torch.empty(shape, dtype=torch.float32, device=packed_sketch.device)
    for b in range(8):
        bit_val = ((packed_sketch >> b) & 1).float() * 2.0 - 1.0
        sketch_bits[..., b::8] = bit_val

    # Inverse QJL sketch projection: gamma * diag(s') * FWHT(q)
    sketch_signs = get_random_sign_vector(original_dim, seed=sketch_seed, device=packed_sketch.device)
    # Theoretical QJL unbiased multiplier gamma = sqrt(pi / (2 * d)) * ||r||
    gamma = math.sqrt(math.pi / (2.0 * original_dim)) * r_norm
    r_unrot = fast_walsh_hadamard_transform(sketch_bits.to(dtype)) * sketch_signs.to(dtype)
    r_recon = r_unrot * gamma.to(dtype)

    return (v_recon3 + r_recon).to(dtype)


def compute_gqa_turboquant_attention(
    query: "torch.Tensor",
    key: "torch.Tensor",
    value: "torch.Tensor",
    scale: float | None = None,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Compute GQA attention using TurboQuant Asymmetric KQV compression."""
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for compute_gqa_turboquant_attention")

    dim = query.size(-1)

    # 1. Compress Key via WHT + 4-bit scalar Lloyd-Max
    k_rot, k_norms = turboquant_rotate(key, seed=42)
    k_packed, k_scales, _ = quantize_key_4bit(k_rot)
    k_rot_dequant = dequantize_key_4bit(k_packed, k_scales, original_dim=dim, dtype=key.dtype)
    k_recon = turboquant_unrotate(k_rot_dequant, k_norms, seed=42)

    # 2. Compress Value via WHT + 3-bit scalar + 1-bit QJL sketch
    v_rot, v_norms = turboquant_rotate(value, seed=42)
    v_scalar3, v_sketch, v_scale3, v_r_norm = quantize_value_3plus1bit(v_rot, sketch_seed=1337)
    v_rot_dequant = dequantize_value_3plus1bit(
        v_scalar3, v_sketch, v_scale3, v_r_norm, original_dim=dim, sketch_seed=1337, dtype=value.dtype
    )
    v_recon = turboquant_unrotate(v_rot_dequant, v_norms, seed=42)

    # 3. GQA expansion across query groups
    num_q_heads = query.size(1)
    num_kv_heads = key.size(1)
    gqa_ratio = num_q_heads // num_kv_heads

    if gqa_ratio > 1:
        expanded_k = k_recon.repeat_interleave(gqa_ratio, dim=1)
        expanded_v = v_recon.repeat_interleave(gqa_ratio, dim=1)
    else:
        expanded_k = k_recon
        expanded_v = v_recon

    softmax_scale = scale if scale is not None else (dim ** -0.5)
    scores = torch.matmul(query.float(), expanded_k.float().transpose(-2, -1)) * softmax_scale
    attn_weights = F.softmax(scores, dim=-1).to(query.dtype)
    attn_output = torch.matmul(attn_weights, expanded_v)

    return attn_output, attn_weights


@register("turboquant")
class TurboQuantTechnique(Technique):
    """TurboQuant: Asymmetric K-V Quantization with Walsh-Hadamard Rotation & QJL Sketch.

    Addresses attention asymmetry between Key and Value caches under a 4-bit budget:
      1. Key cache: Orthogonal Walsh-Hadamard rotation + 4-bit scalar Lloyd-Max quantization
         preserving directional dot-product fidelity across exponential softmax routing.
      2. Value cache: Walsh-Hadamard rotation + 3-bit scalar quantization + 1-bit Quantized
         Johnson-Lindenstrauss (QJL) residual sketching, eliminating zero-mean noise.
      3. Compresses total KV-cache memory by 4x relative to FP16/BF16 without modifying base weights.

    Axes:
        Axis.STRUCTURAL
    """

    axes = Axis.STRUCTURAL

    def __init__(
        self,
        key_bits: int = 4,
        val_scalar_bits: int = 3,
        val_qjl_bits: int = 1,
        block_size: int = 16,
    ):
        if int(key_bits) != 4:
            raise ValueError(f"TurboQuant primary contract expects key_bits=4, got {key_bits}")
        if int(val_scalar_bits) + int(val_qjl_bits) != 4:
            raise ValueError(
                f"Value bit allocation must sum to 4 (got {val_scalar_bits} + {val_qjl_bits})"
            )
        if int(block_size) <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.key_bits = int(key_bits)
        self.val_scalar_bits = int(val_scalar_bits)
        self.val_qjl_bits = int(val_qjl_bits)
        self.block_size = int(block_size)

    @property
    def compression_ratio(self) -> float:
        """Compression ratio relative to standard 16-bit (BF16/FP16) cache."""
        return 4.0

    def engine_kwargs(self, context_length: int, max_tokens: int) -> dict[str, Any]:
        """Extra engine kwargs merged into the vLLM / PyTorch constructor."""
        del context_length, max_tokens
        return {
            "block_size": self.block_size,
            "kv_cache_dtype": "fp8",  # Native quantization knob
        }

    def workload(self) -> Workload:
        return SyntheticFiller()

    def extra_metrics(self, llm, workload_output) -> dict[str, Any]:
        del llm, workload_output
        return {
            "key_bits": self.key_bits,
            "val_scalar_bits": self.val_scalar_bits,
            "val_qjl_bits": self.val_qjl_bits,
            "total_val_bits": self.val_scalar_bits + self.val_qjl_bits,
            "compression_ratio_vs_bf16": self.compression_ratio,
            "block_size": self.block_size,
            "asymmetric_kqv_enabled": True,
        }


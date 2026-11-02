import unittest

import torch
import torch.nn.functional as F

from src.core.axis import Axis
from src.core.registry import get_technique, list_techniques
from src.techniques.flexgen_kv import (
    compute_flexgen_attention,
    dequantize_flexgen_group4,
    pack_uint4,
    quantize_flexgen_group4,
    unpack_uint4,
)


class TestFlexGenKVTechnique(unittest.TestCase):
    def test_registration_and_axes(self):
        self.assertIn("flexgen_kv", list_techniques())
        technique = get_technique("flexgen_kv")
        self.assertEqual(technique.axes, Axis.SPATIAL | Axis.STRUCTURAL)

    def test_defaults_and_custom_overrides(self):
        technique = get_technique("flexgen_kv")
        self.assertEqual(technique.swap_space_gb, 4)
        self.assertEqual(technique.group_size, 64)
        self.assertEqual(technique.block_size, 16)
        self.assertEqual(technique.compression_ratio, 4.0)

        custom = get_technique("flexgen_kv", swap_space_gb=8, group_size=32, block_size=32)
        self.assertEqual(custom.swap_space_gb, 8)
        self.assertEqual(custom.group_size, 32)
        self.assertEqual(custom.block_size, 32)

        metrics = custom.extra_metrics(None, None)
        self.assertEqual(metrics["swap_space_gb"], 8)
        self.assertEqual(metrics["quant_group_size"], 32)
        self.assertEqual(metrics["kv_cache_bits"], 4)
        self.assertEqual(metrics["compression_ratio_vs_bf16"], 4.0)

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            get_technique("flexgen_kv", swap_space_gb=-1)
        with self.assertRaises(ValueError):
            get_technique("flexgen_kv", group_size=0)
        with self.assertRaises(ValueError):
            get_technique("flexgen_kv", block_size=0)

    def test_pack_and_unpack_uint4(self):
        """Verify lossless 4-bit packing into uint8 bytes and unpacking."""
        torch.manual_seed(42)
        values = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=torch.uint8)
        packed = pack_uint4(values)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.numel(), len(values) // 2)

        unpacked = unpack_uint4(packed, original_dim=len(values))
        self.assertTrue(torch.equal(values, unpacked))

    def test_quantize_and_dequantize_group4_fidelity(self):
        """Verify group-wise 4-bit asymmetric quantization fidelity (cosine sim > 0.99)."""
        torch.manual_seed(42)
        shape = (2, 8, 64, 128)  # D=128, group_size=64 -> 2 groups per head
        tensor = torch.randn(shape, dtype=torch.float32)

        packed, scales, min_vals = quantize_flexgen_group4(tensor, group_size=64)
        # Check packed shape: last dimension halved in uint8
        self.assertEqual(packed.shape, (2, 8, 64, 64))
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(scales.shape, (2, 8, 64, 2, 1))
        self.assertEqual(min_vals.shape, (2, 8, 64, 2, 1))

        # Dequantize back
        dequant = dequantize_flexgen_group4(
            packed, scales, min_vals, original_dtype=torch.float32, group_size=64
        )
        self.assertEqual(dequant.shape, shape)
        self.assertEqual(dequant.dtype, torch.float32)

        cos_sim = F.cosine_similarity(tensor.flatten(), dequant.flatten(), dim=0).item()
        self.assertGreater(cos_sim, 0.99)

    def test_gqa_flexgen_attention_forward(self):
        """Verify full forward pass with FlexGen group-wise 4-bit KV quantization."""
        torch.manual_seed(42)
        batch = 1
        num_q_heads = 32   # Qwen3-4B
        num_kv_heads = 8   # 4:1 ratio
        q_len = 1
        kv_len = 128
        head_dim = 128

        q = torch.randn(batch, num_q_heads, q_len, head_dim)
        k = torch.randn(batch, num_kv_heads, kv_len, head_dim)
        v = torch.randn(batch, num_kv_heads, kv_len, head_dim)

        out, weights = compute_flexgen_attention(q, k, v, group_size=64)
        self.assertEqual(out.shape, (batch, num_q_heads, q_len, head_dim))
        self.assertEqual(weights.shape, (batch, num_q_heads, q_len, kv_len))

        # Softmax weights must sum to 1.0 across all heads
        weight_sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5))

        # Compute full unquantized reference output
        gqa_ratio = num_q_heads // num_kv_heads
        k_exp = k.repeat_interleave(gqa_ratio, dim=1)
        v_exp = v.repeat_interleave(gqa_ratio, dim=1)
        scores_ref = torch.matmul(q.float(), k_exp.float().transpose(-2, -1)) * (head_dim ** -0.5)
        weights_ref = F.softmax(scores_ref, dim=-1)
        out_ref = torch.matmul(weights_ref, v_exp)

        # Output cosine similarity must exceed 0.98
        cos_sim = F.cosine_similarity(out.flatten(), out_ref.flatten(), dim=0).item()
        self.assertGreater(cos_sim, 0.98)

    def test_engine_kwargs(self):
        technique = get_technique("flexgen_kv", swap_space_gb=6, group_size=64, block_size=16)
        kwargs = technique.engine_kwargs(context_length=4096, max_tokens=128)
        self.assertEqual(kwargs["swap_space"], 6)
        self.assertEqual(kwargs["kv_cache_dtype"], "fp8")
        self.assertEqual(kwargs["block_size"], 16)


if __name__ == "__main__":
    unittest.main()

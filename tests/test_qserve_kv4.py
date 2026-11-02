import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.core.axis import Axis
from src.core.registry import get_technique, list_techniques
from src.techniques.qserve_kv4 import (
    compute_gqa_qserve_attention,
    compute_smooth_attention_scales,
    dequantize_kv_int4,
    pack_int4,
    quantize_kv_int4,
    unpack_int4,
)


class TestQServeKV4Technique(unittest.TestCase):
    def test_registration_and_axis(self):
        self.assertIn("qserve_kv4", list_techniques())
        technique = get_technique("qserve_kv4")
        self.assertEqual(technique.axes, Axis.STRUCTURAL)

    def test_defaults_and_custom_overrides(self):
        technique = get_technique("qserve_kv4")
        self.assertEqual(technique.group_size, 32)
        self.assertEqual(technique.alpha, 0.5)
        self.assertEqual(technique.block_size, 16)
        self.assertEqual(technique.compression_ratio, 4.0)

        custom = get_technique("qserve_kv4", group_size=64, alpha=0.75, block_size=32)
        self.assertEqual(custom.group_size, 64)
        self.assertEqual(custom.alpha, 0.75)
        self.assertEqual(custom.block_size, 32)

        metrics = custom.extra_metrics(None, None)
        self.assertEqual(metrics["kv_cache_bits"], 4)
        self.assertEqual(metrics["quant_group_size"], 64)
        self.assertEqual(metrics["smooth_attention_alpha"], 0.75)
        self.assertEqual(metrics["compression_ratio_vs_bf16"], 4.0)

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            get_technique("qserve_kv4", group_size=0)
        with self.assertRaises(ValueError):
            get_technique("qserve_kv4", alpha=-0.1)
        with self.assertRaises(ValueError):
            get_technique("qserve_kv4", alpha=1.5)
        with self.assertRaises(ValueError):
            get_technique("qserve_kv4", block_size=0)

    def test_smooth_attention_mathematical_invariance(self):
        """Verify SmoothAttention scales preserve Q @ K^T while attenuating Key channel outliers."""
        torch.manual_seed(42)
        batch = 2
        num_q_heads = 16   # 4:1 GQA group ratio
        num_kv_heads = 4
        q_len = 8
        kv_len = 32
        head_dim = 64

        q = torch.randn(batch, num_q_heads, q_len, head_dim)
        k = torch.randn(batch, num_kv_heads, kv_len, head_dim)

        # Inject severe outlier into Key channel 5 (50x amplitude)
        k[:, :, :, 5] *= 50.0

        q_smooth, k_smooth = compute_smooth_attention_scales(q, k, alpha=0.5)

        # 1. Verify attention logits Q @ K^T are mathematically invariant across all GQA heads
        group_size = num_q_heads // num_kv_heads
        k_exp = k.repeat_interleave(group_size, dim=1)
        k_smooth_exp = k_smooth.repeat_interleave(group_size, dim=1)

        orig_logits = torch.matmul(q, k_exp.transpose(-2, -1))
        smooth_logits = torch.matmul(q_smooth, k_smooth_exp.transpose(-2, -1))

        # Relative difference must be within float precision tolerance
        self.assertTrue(torch.allclose(orig_logits, smooth_logits, atol=1e-4, rtol=1e-4))

        # 2. Key channel outlier amplitude must be significantly attenuated in k_smooth
        orig_max = k[:, :, :, 5].abs().max().item()
        smooth_max = k_smooth[:, :, :, 5].abs().max().item()
        self.assertLess(smooth_max, orig_max * 0.5)

    def test_pack_and_unpack_int4(self):
        """Verify lossless bit packing and unpacking of INT4 values in [-8, 7]."""
        torch.manual_seed(42)
        # Create tensor with all possible signed 4-bit values [-8 .. 7]
        values = torch.tensor([-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int8)
        packed = pack_int4(values)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.numel(), len(values) // 2)

        unpacked = unpack_int4(packed, original_dim=len(values))
        self.assertTrue(torch.equal(values, unpacked))

    def test_quantize_and_dequantize_kv_int4_fidelity(self):
        """Verify 4-bit quantization compresses memory by 4x and retains >0.99 cosine similarity."""
        torch.manual_seed(42)
        shape = (2, 8, 64, 128)  # D=128
        tensor = torch.randn(shape, dtype=torch.float32)

        packed, scales = quantize_kv_int4(tensor, group_size=32)
        # Trailing dimension must be halved in uint8 (2 values per byte)
        self.assertEqual(packed.shape, (2, 8, 64, 64))
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(scales.shape, (2, 8, 64, 4, 1))

        # Dequantize back
        dequant = dequantize_kv_int4(packed, scales, original_dtype=torch.float32, group_size=32)
        self.assertEqual(dequant.shape, shape)
        self.assertEqual(dequant.dtype, torch.float32)

        # Measure Cosine Similarity across reconstructed vectors
        cos_sim = F.cosine_similarity(tensor.flatten(), dequant.flatten(), dim=0).item()
        self.assertGreater(cos_sim, 0.99)

    def test_gqa_qserve_attention_end_to_end(self):
        """Verify full forward attention pass with QServe SmoothAttention + INT4 KV cache."""
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

        out, weights = compute_gqa_qserve_attention(q, k, v, alpha=0.5, group_size=32)
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

        # Cosine similarity between QServe KV4 and reference output must exceed 0.98
        cos_sim = F.cosine_similarity(out.flatten(), out_ref.flatten(), dim=0).item()
        self.assertGreater(cos_sim, 0.98)

    def test_engine_kwargs(self):
        technique = get_technique("qserve_kv4", group_size=32, block_size=16)
        kwargs = technique.engine_kwargs(context_length=4096, max_tokens=128)
        self.assertEqual(kwargs["block_size"], 16)
        self.assertEqual(kwargs["kv_cache_dtype"], "fp8")


if __name__ == "__main__":
    unittest.main()

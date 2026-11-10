import unittest
from types import SimpleNamespace

import torch

from src.core.axis import Axis
from src.core.registry import get_technique, list_techniques
from src.techniques.headq import (
    HeadQTechnique,
    compute_gqa_headq_attention,
    compute_key_sidecode,
    compute_query_pca_basis,
    dequantize_key_2bit,
    dequantize_key_sidecode,
    dequantize_value_4bit,
    quantize_key_2bit,
    quantize_value_4bit,
)


class TestHeadQTechnique(unittest.TestCase):
    def test_registration_and_axis(self):
        self.assertIn("headq", list_techniques())
        technique = get_technique("headq")
        self.assertIsInstance(technique, HeadQTechnique)
        self.assertEqual(technique.axes, Axis.STRUCTURAL)
        self.assertGreater(technique.compression_ratio, 4.0)

    def test_defaults_and_custom_parameters(self):
        technique = get_technique("headq")
        self.assertEqual(technique.key_bits, 2)
        self.assertEqual(technique.val_bits, 4)
        self.assertEqual(technique.rank, 4)
        self.assertEqual(technique.sidecode_bits, 4)
        self.assertEqual(technique.block_size, 16)

        mock_out = SimpleNamespace(prompt_tokens=1024, generated_tokens=64)
        metrics = technique.extra_metrics(None, mock_out)
        self.assertEqual(metrics["key_bits"], 2)
        self.assertEqual(metrics["val_bits"], 4)
        self.assertEqual(metrics["pca_rank"], 4)
        self.assertTrue(metrics["model_visible_distortion_aware"])

        kwargs = technique.engine_kwargs(context_length=2048, max_tokens=128)
        self.assertEqual(kwargs["block_size"], 16)
        self.assertEqual(kwargs["kv_cache_dtype"], "fp8")

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            HeadQTechnique(key_bits=4)  # Must be 2-bit primary contract
        with self.assertRaises(ValueError):
            HeadQTechnique(val_bits=0)
        with self.assertRaises(ValueError):
            HeadQTechnique(rank=0)
        with self.assertRaises(ValueError):
            HeadQTechnique(block_size=-1)

    def test_query_pca_basis_orthonormality(self):
        torch.manual_seed(42)
        dim = 64
        rank = 4
        queries = torch.randn(100, dim, dtype=torch.float32)
        basis = compute_query_pca_basis(queries, rank=rank)

        self.assertEqual(basis.shape, (dim, rank))
        # Check U^T @ U == I_rank
        identity = torch.eye(rank, dtype=torch.float32)
        inner = torch.matmul(basis.transpose(0, 1), basis)
        self.assertTrue(torch.allclose(inner, identity, atol=1e-5))

    def test_key_2bit_quantization_roundtrip(self):
        torch.manual_seed(100)
        dim = 64
        k = torch.randn(2, 4, 16, dim, dtype=torch.float32)

        packed, scale, dequant = quantize_key_2bit(k)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.shape[-1], dim // 4)  # 4 values per byte

        unpacked = dequantize_key_2bit(packed, scale, original_dim=dim, dtype=torch.float32)
        self.assertEqual(unpacked.shape, k.shape)
        self.assertTrue(torch.allclose(dequant, unpacked, atol=1e-5))

    def test_key_sidecode_projection_roundtrip(self):
        torch.manual_seed(200)
        dim = 64
        rank = 4
        queries = torch.randn(50, dim, dtype=torch.float32)
        basis = compute_query_pca_basis(queries, rank=rank)

        k_residual = torch.randn(2, 4, 16, dim, dtype=torch.float32) * 0.1
        packed_code, scale, dequant_code = compute_key_sidecode(k_residual, basis, sidecode_bits=4)

        self.assertEqual(packed_code.dtype, torch.uint8)
        self.assertEqual(packed_code.shape[-1], rank // 2)

        unpacked_code = dequantize_key_sidecode(packed_code, scale, rank=rank, dtype=torch.float32)
        self.assertEqual(unpacked_code.shape, (2, 4, 16, rank))
        self.assertTrue(torch.allclose(dequant_code, unpacked_code, atol=1e-5))

    def test_value_4bit_quantization_roundtrip(self):
        torch.manual_seed(300)
        dim = 64
        v = torch.randn(2, 4, 16, dim, dtype=torch.float32)

        packed, scale, dequant = quantize_value_4bit(v)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.shape[-1], dim // 2)

        unpacked = dequantize_value_4bit(packed, scale, original_dim=dim, dtype=torch.float32)
        self.assertEqual(unpacked.shape, v.shape)
        self.assertTrue(torch.allclose(dequant, unpacked, atol=1e-5))

    def test_gqa_headq_attention_and_kl_reduction(self):
        batch, seq_len, head_dim = 1, 32, 64
        num_q_heads, num_kv_heads = 8, 2  # 4:1 GQA ratio

        torch.manual_seed(400)
        q = torch.randn(batch, num_q_heads, seq_len, head_dim, dtype=torch.float32)
        k = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.float32)
        v = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.float32)

        attn_out, attn_weights, metrics = compute_gqa_headq_attention(q, k, v, rank=4)

        self.assertEqual(attn_out.shape, (batch, num_q_heads, seq_len, head_dim))
        self.assertEqual(attn_weights.shape, (batch, num_q_heads, seq_len, seq_len))
        self.assertTrue(torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)), atol=1e-4))

        # Additive logit correction should reduce attention KL divergence compared to uncorrected 2-bit
        self.assertIn("attention_kl_corrected", metrics)
        self.assertIn("attention_kl_coarse", metrics)
        self.assertIn("kl_reduction_ratio", metrics)
        self.assertLess(metrics["attention_kl_corrected"], metrics["attention_kl_coarse"])
        self.assertGreater(metrics["kl_reduction_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()

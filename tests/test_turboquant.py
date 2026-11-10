import unittest
from types import SimpleNamespace

import torch

from src.core.axis import Axis
from src.core.registry import get_technique, list_techniques
from src.techniques.turboquant import (
    TurboQuantTechnique,
    compute_gqa_turboquant_attention,
    dequantize_key_4bit,
    dequantize_value_3plus1bit,
    fast_walsh_hadamard_transform,
    get_random_sign_vector,
    quantize_key_4bit,
    quantize_value_3plus1bit,
    turboquant_rotate,
    turboquant_unrotate,
)


class TestTurboQuantTechnique(unittest.TestCase):
    def test_registration_and_axis(self):
        self.assertIn("turboquant", list_techniques())
        technique = get_technique("turboquant")
        self.assertIsInstance(technique, TurboQuantTechnique)
        self.assertEqual(technique.axes, Axis.STRUCTURAL)
        self.assertEqual(technique.compression_ratio, 4.0)

    def test_defaults_and_custom_parameters(self):
        technique = get_technique("turboquant")
        self.assertEqual(technique.key_bits, 4)
        self.assertEqual(technique.val_scalar_bits, 3)
        self.assertEqual(technique.val_qjl_bits, 1)
        self.assertEqual(technique.block_size, 16)

        mock_out = SimpleNamespace(prompt_tokens=2048, generated_tokens=128)
        metrics = technique.extra_metrics(None, mock_out)
        self.assertEqual(metrics["key_bits"], 4)
        self.assertEqual(metrics["val_scalar_bits"], 3)
        self.assertEqual(metrics["val_qjl_bits"], 1)
        self.assertEqual(metrics["total_val_bits"], 4)
        self.assertEqual(metrics["compression_ratio_vs_bf16"], 4.0)
        self.assertTrue(metrics["asymmetric_kqv_enabled"])

        engine_kwargs = technique.engine_kwargs(context_length=4096, max_tokens=256)
        self.assertEqual(engine_kwargs["block_size"], 16)
        self.assertEqual(engine_kwargs["kv_cache_dtype"], "fp8")

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            TurboQuantTechnique(key_bits=8)
        with self.assertRaises(ValueError):
            TurboQuantTechnique(val_scalar_bits=2, val_qjl_bits=1)  # sum != 4
        with self.assertRaises(ValueError):
            TurboQuantTechnique(block_size=0)

    def test_fast_walsh_hadamard_transform_involution(self):
        # FWHT must be its own exact inverse: FWHT(FWHT(x)) == x
        torch.manual_seed(42)
        x = torch.randn(2, 4, 64, dtype=torch.float32)
        x_fwht = fast_walsh_hadamard_transform(x)
        x_recon = fast_walsh_hadamard_transform(x_fwht)
        self.assertTrue(torch.allclose(x, x_recon, atol=1e-5))

        # Check non-power-of-2 dimension raises ValueError
        x_bad = torch.randn(2, 4, 50)
        with self.assertRaises(ValueError):
            fast_walsh_hadamard_transform(x_bad)

    def test_turboquant_rotate_unrotate(self):
        torch.manual_seed(100)
        v = torch.randn(2, 8, 32, 64, dtype=torch.float32)
        v_rot, v_norms = turboquant_rotate(v, seed=42)

        # Norms should be positive and vector should be spherical
        self.assertEqual(v_rot.shape, v.shape)
        self.assertTrue((v_norms > 0).all())

        # Unrotate should accurately reconstruct input
        v_recon = turboquant_unrotate(v_rot, v_norms, seed=42)
        self.assertTrue(torch.allclose(v, v_recon, atol=1e-4))

    def test_key_4bit_quantization_roundtrip(self):
        torch.manual_seed(200)
        dim = 64
        k_rot = torch.randn(2, 4, 16, dim, dtype=torch.float32)
        # Unit spherical norm
        k_rot = k_rot / k_rot.norm(dim=-1, keepdim=True)

        packed, scale, _ = quantize_key_4bit(k_rot)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.shape[-1], dim // 2)

        k_dequant = dequantize_key_4bit(packed, scale, original_dim=dim, dtype=torch.float32)
        self.assertEqual(k_dequant.shape, k_rot.shape)

        # Check cosine similarity is very high (> 0.97 for 4-bit spherical)
        cos_sim = torch.cosine_similarity(k_rot.view(-1, dim), k_dequant.view(-1, dim), dim=-1)
        self.assertGreater(cos_sim.mean().item(), 0.97)

    def test_value_3plus1bit_quantization_roundtrip(self):
        torch.manual_seed(300)
        dim = 64
        v_rot = torch.randn(2, 4, 16, dim, dtype=torch.float32)
        v_rot = v_rot / v_rot.norm(dim=-1, keepdim=True)

        packed_scalar, packed_sketch, scale3, r_norm = quantize_value_3plus1bit(v_rot, sketch_seed=1337)
        self.assertEqual(packed_scalar.dtype, torch.uint8)
        self.assertEqual(packed_sketch.dtype, torch.uint8)
        self.assertEqual(packed_scalar.shape[-1], dim // 2)
        self.assertEqual(packed_sketch.shape[-1], dim // 8)

        v_dequant = dequantize_value_3plus1bit(
            packed_scalar, packed_sketch, scale3, r_norm, original_dim=dim, sketch_seed=1337, dtype=torch.float32
        )
        self.assertEqual(v_dequant.shape, v_rot.shape)

        # Cosine similarity should be high (> 0.94)
        cos_sim = torch.cosine_similarity(v_rot.view(-1, dim), v_dequant.view(-1, dim), dim=-1)
        self.assertGreater(cos_sim.mean().item(), 0.94)

    def test_compute_gqa_turboquant_attention(self):
        batch, seq_len, head_dim = 1, 16, 64
        num_q_heads, num_kv_heads = 8, 2  # 4:1 GQA ratio

        torch.manual_seed(400)
        q = torch.randn(batch, num_q_heads, seq_len, head_dim, dtype=torch.float32)
        k = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.float32)
        v = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.float32)

        attn_out, attn_weights = compute_gqa_turboquant_attention(q, k, v)

        self.assertEqual(attn_out.shape, (batch, num_q_heads, seq_len, head_dim))
        self.assertEqual(attn_weights.shape, (batch, num_q_heads, seq_len, seq_len))
        self.assertTrue(torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)), atol=1e-4))


if __name__ == "__main__":
    unittest.main()


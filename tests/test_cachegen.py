import unittest
from types import SimpleNamespace

import torch

from src.core.axis import Axis
from src.core.diagnostic import inspect_sample_diagnostic
from src.core.pytorch_runner import PyTorchRunner
from src.core.registry import get_technique, list_techniques
from src.techniques.cachegen import (
    CacheGenTechnique,
    compute_gqa_cachegen_attention,
    decode_anchor_deltas,
    encode_anchor_deltas,
    quantize_deltas_tiered,
)
from src.techniques.cachegen_cache import (
    CacheGenCache,
    CacheGenLayer,
    get_active_cachegen_cache,
)


class TestCacheGenTechnique(unittest.TestCase):
    def test_registration_and_axes(self):
        self.assertIn("cachegen", list_techniques())
        technique = get_technique("cachegen")
        self.assertIsInstance(technique, CacheGenTechnique)
        self.assertEqual(technique.axes, Axis.STRUCTURAL | Axis.SPATIAL)
        self.assertGreater(technique.compression_ratio, 3.0)

    def test_defaults_and_custom_parameters(self):
        technique = get_technique("cachegen")
        self.assertEqual(technique.anchor_group_size, 10)
        self.assertEqual(technique.shallow_bits, 8)
        self.assertEqual(technique.mid_bits, 4)
        self.assertEqual(technique.deep_bits, 2)
        self.assertEqual(technique.block_size, 16)

        mock_out = SimpleNamespace(prompt_tokens=2048, generated_tokens=128)
        metrics = technique.extra_metrics(None, mock_out)
        self.assertEqual(metrics["anchor_group_size"], 10)
        self.assertTrue(metrics["delta_encoding_active"])
        self.assertTrue(metrics["layer_tiered_quantization_active"])

        kwargs = technique.engine_kwargs(context_length=4096, max_tokens=128)
        self.assertEqual(kwargs["block_size"], 16)
        self.assertEqual(kwargs["kv_cache_dtype"], "fp8")
        self.assertEqual(kwargs["swap_space"], 4)

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            CacheGenTechnique(anchor_group_size=0)
        with self.assertRaises(ValueError):
            CacheGenTechnique(shallow_bits=-1)
        with self.assertRaises(ValueError):
            CacheGenTechnique(block_size=0)

    def test_anchor_delta_exact_roundtrip_unquantized(self):
        torch.manual_seed(42)
        batch, heads, seq_len, dim = 2, 4, 37, 64
        group_size = 10
        tensor = torch.randn(batch, heads, seq_len, dim, dtype=torch.float32)

        anchors, deltas, orig_len = encode_anchor_deltas(tensor, group_size=group_size)
        self.assertEqual(orig_len, seq_len)
        self.assertEqual(anchors.shape[-2], 1)
        self.assertTrue(torch.allclose(deltas[..., 0, :], torch.zeros_like(deltas[..., 0, :])))

        reconstructed = decode_anchor_deltas(anchors, deltas, original_seq_len=seq_len)
        self.assertEqual(reconstructed.shape, tensor.shape)
        self.assertTrue(torch.allclose(tensor, reconstructed, atol=1e-6))

    def test_layer_tiered_quantization(self):
        deltas = torch.randn(1, 1, 10, 64, dtype=torch.float32)

        _, _, _, bits_shallow = quantize_deltas_tiered(deltas, layer_idx=5, total_layers=30)
        self.assertEqual(bits_shallow, 8)

        _, _, _, bits_mid = quantize_deltas_tiered(deltas, layer_idx=15, total_layers=30)
        self.assertEqual(bits_mid, 4)

        _, _, _, bits_deep = quantize_deltas_tiered(deltas, layer_idx=25, total_layers=30)
        self.assertEqual(bits_deep, 2)

    def test_compute_gqa_cachegen_attention(self):
        batch, seq_len, head_dim = 1, 32, 64
        num_q_heads, num_kv_heads = 8, 2

        torch.manual_seed(100)
        q = torch.randn(batch, num_q_heads, seq_len, head_dim, dtype=torch.float32)
        k = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.float32)
        v = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.float32)

        attn_out, attn_weights, telemetry = compute_gqa_cachegen_attention(
            q, k, v, layer_idx=10, total_layers=32, group_size=8
        )

        self.assertEqual(attn_out.shape, (batch, num_q_heads, seq_len, head_dim))
        self.assertEqual(attn_weights.shape, (batch, num_q_heads, seq_len, seq_len))
        self.assertTrue(torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)), atol=1e-4))
        self.assertEqual(telemetry["layer_idx"], 10)
        self.assertTrue(telemetry["delta_encoding_active"])

    def test_cachegen_layer_unaligned_chunks_and_decode(self):
        torch.manual_seed(66)
        layer = CacheGenLayer(layer_idx=18, total_layers=36, anchor_group_size=10)
        batch, heads, dim = 1, 8, 64

        # Chunk 1 of length 17 (unaligned with group_size=10)
        k1 = torch.randn(batch, 17, heads, dim, dtype=torch.bfloat16).transpose(1, 2)
        v1 = torch.randn(batch, 17, heads, dim, dtype=torch.bfloat16).transpose(1, 2)
        out_k1, out_v1 = layer.update(k1, v1)
        self.assertEqual(out_k1.shape, (batch, heads, 17, dim))
        self.assertEqual(out_v1.shape, (batch, heads, 17, dim))

        # Decode across group boundary (tokens 17..22)
        for _ in range(5):
            k_s = torch.randn(batch, heads, 1, dim, dtype=torch.bfloat16)
            v_s = torch.randn(batch, heads, 1, dim, dtype=torch.bfloat16)
            out_k, out_v = layer.update(k_s, v_s)

        self.assertEqual(layer.get_seq_length(), 22)
        self.assertEqual(out_k.shape, (batch, heads, 22, dim))
        self.assertEqual(layer.original_positions, list(range(22)))

        bf16_bytes = 2 * batch * heads * 22 * dim * 2
        self.assertLess(layer.compressed_bytes, bf16_bytes * 0.5)

    def test_cachegen_cache_factory_and_pytorch_runner(self):
        technique = get_technique("cachegen")
        cache_factory = technique.pytorch_cache_factory()
        self.assertIsNotNone(cache_factory)

        cache = cache_factory()
        self.assertIsInstance(cache, CacheGenCache)
        self.assertEqual(get_active_cachegen_cache(), cache)

        k1 = torch.randn(1, 8, 25, 64, dtype=torch.float32)
        v1 = torch.randn(1, 8, 25, 64, dtype=torch.float32)
        cache.update(k1, v1, layer_idx=0)

        diag = inspect_sample_diagnostic(
            cache=cache,
            prompt_text="Test CacheGen diagnostic",
            prompt_token_ids=list(range(25)),
            sample_meta={"id": "cg_diag_0", "task_type": "unit"},
            is_correct=True,
            baseline_correct=True,
            bits_per_scalar=8.0,
        )
        self.assertEqual(diag.overall_retention_ratio, 1.0)
        self.assertEqual(diag.overall_effective_bitwidth, 8.0)

        runner = PyTorchRunner(
            model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
            revision="cdbee75f17c01a7cc42f958dc650907174af0554",
            cache_factory=cache_factory,
            dry_run=True,
        )
        outputs = runner.generate(["Test CacheGen prompt."], max_tokens=16)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(outputs[0].outputs[0].token_ids), 16)


if __name__ == "__main__":
    unittest.main()

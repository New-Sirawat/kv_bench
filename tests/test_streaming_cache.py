from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None

from src.core.pytorch_runner import PyTorchRunner
from src.techniques.streaming_cache import StreamingCache, StreamingLayer
from src.techniques.streamingllm import StreamingLLMTechnique


class TestStreamingCache(unittest.TestCase):
    def setUp(self):
        if torch is None:
            self.skipTest("PyTorch is required for TestStreamingCache")

    def test_init_validation(self):
        with self.assertRaises(ValueError):
            StreamingCache(sink_tokens=-1, window_size=100)
        with self.assertRaises(ValueError):
            StreamingCache(sink_tokens=4, window_size=0)
        with self.assertRaises(ValueError):
            StreamingLayer(sink_tokens=-1, window_size=100)
        with self.assertRaises(ValueError):
            StreamingLayer(sink_tokens=4, window_size=-10)

    def test_layer_within_capacity(self):
        layer = StreamingLayer(sink_tokens=4, window_size=16)
        k = torch.randn(1, 2, 10, 32)
        v = torch.randn(1, 2, 10, 32)

        out_k, out_v = layer.update(k, v)
        self.assertEqual(out_k.shape, (1, 2, 10, 32))
        self.assertEqual(out_v.shape, (1, 2, 10, 32))
        self.assertEqual(layer.get_seq_length(), 10)

    def test_layer_prune_over_capacity(self):
        layer = StreamingLayer(sink_tokens=4, window_size=16)
        # Prefill 50 tokens
        k = torch.randn(1, 2, 50, 32)
        v = torch.randn(1, 2, 50, 32)

        layer.update(k, v)
        self.assertEqual(layer.get_seq_length(), 50)

        layer.prune()
        # Active capacity should be 4 + 16 = 20
        self.assertEqual(layer.get_seq_length(), 20)
        self.assertEqual(layer.keys.shape, (1, 2, 20, 32))
        self.assertEqual(layer.values.shape, (1, 2, 20, 32))

        # Check prefix match (first 4 tokens) and suffix match (last 16 tokens)
        self.assertTrue(torch.equal(layer.keys[:, :, :4, :], k[:, :, :4, :]))
        self.assertTrue(torch.equal(layer.keys[:, :, 4:, :], k[:, :, -16:, :]))

    def test_layer_decode_immediate_eviction(self):
        layer = StreamingLayer(sink_tokens=2, window_size=4)
        # Prefill 6 tokens (at capacity)
        k_init = torch.randn(1, 2, 6, 16)
        v_init = torch.randn(1, 2, 6, 16)
        layer.update(k_init, v_init)
        self.assertEqual(layer.get_seq_length(), 6)

        # Decode step 1 (1 token)
        k_dec = torch.randn(1, 2, 1, 16)
        v_dec = torch.randn(1, 2, 1, 16)
        out_k, out_v = layer.update(k_dec, v_dec)

        # Clamped at capacity: 2 + 4 = 6
        self.assertEqual(out_k.shape, (1, 2, 6, 16))
        self.assertEqual(out_v.shape, (1, 2, 6, 16))
        self.assertEqual(layer.get_seq_length(), 6)

        # First 2 should still be initial sink tokens
        self.assertTrue(torch.equal(out_k[:, :, :2, :], k_init[:, :, :2, :]))
        # Last token should be the newly decoded token
        self.assertTrue(torch.equal(out_k[:, :, -1:, :], k_dec))

    def test_streaming_cache_multilyer_prune(self):
        cache = StreamingCache(sink_tokens=4, window_size=8)
        k1 = torch.randn(1, 2, 30, 16)
        v1 = torch.randn(1, 2, 30, 16)
        k2 = torch.randn(1, 2, 30, 16)
        v2 = torch.randn(1, 2, 30, 16)

        cache.update(k1, v1, layer_idx=0)
        cache.update(k2, v2, layer_idx=1)

        self.assertEqual(len(cache.layers), 2)
        self.assertEqual(cache.get_seq_length(0), 30)
        self.assertEqual(cache.get_seq_length(1), 30)

        cache.prune_after_chunk()
        # Capacity = 4 + 8 = 12
        self.assertEqual(cache.get_seq_length(0), 12)
        self.assertEqual(cache.get_seq_length(1), 12)
        self.assertEqual(cache.get_max_length(), 12)

    def test_dtype_preservation(self):
        layer = StreamingLayer(sink_tokens=2, window_size=4)
        k_bf16 = torch.randn(1, 1, 10, 16, dtype=torch.bfloat16)
        v_bf16 = torch.randn(1, 1, 10, 16, dtype=torch.bfloat16)

        layer.update(k_bf16, v_bf16)
        layer.prune()

        self.assertEqual(layer.keys.dtype, torch.bfloat16)
        self.assertEqual(layer.values.dtype, torch.bfloat16)

    def test_technique_pytorch_cache_factory(self):
        tech = StreamingLLMTechnique(sink_tokens=8, window_size=1024)
        factory = tech.pytorch_cache_factory()
        self.assertIsNotNone(factory)
        cache = factory()
        self.assertIsInstance(cache, StreamingCache)
        self.assertEqual(cache.sink_tokens, 8)
        self.assertEqual(cache.window_size, 1024)
        self.assertEqual(cache.get_max_length(), 1032)

    def test_pytorch_runner_dry_run_with_streaming_cache(self):
        tech = StreamingLLMTechnique(sink_tokens=4, window_size=2048)
        runner = PyTorchRunner(
            model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
            revision="cdbee75f17c01a7cc42f958dc650907174af0554",
            cache_factory=tech.pytorch_cache_factory(),
            dry_run=True,
        )
        res = runner.generate(["Test prompt with streaming cache"], max_tokens=16)
        self.assertEqual(len(res), 1)
        self.assertEqual(len(res[0].outputs[0].token_ids), 16)

    def test_streaming_cache_diagnostic_inspection(self):
        from src.core.diagnostic import inspect_sample_diagnostic

        # Capacity: 4 sink tokens + 16 window tokens = 20 tokens
        cache = StreamingCache(sink_tokens=4, window_size=16)
        batch, num_kv_heads, head_dim = 1, 2, 32

        # Feed 40 tokens across 2 sessions (20 tokens in S0, 20 tokens in S1)
        k = torch.randn(batch, num_kv_heads, 40, head_dim, dtype=torch.bfloat16)
        v = torch.randn(batch, num_kv_heads, 40, head_dim, dtype=torch.bfloat16)
        cache.update(k, v, layer_idx=0)
        cache.prune_after_chunk()

        self.assertEqual(cache.get_seq_length(0), 20)
        self.assertEqual(len(cache.layers[0].original_positions), 20)
        # Positions should be first 4 sinks [0, 1, 2, 3] + last 16 [24 ... 39]
        expected_positions = list(range(4)) + list(range(24, 40))
        self.assertEqual(cache.layers[0].original_positions, expected_positions)

        prompt_token_ids = list(range(40))
        sample_meta = {"id": "streaming_sample_01", "task_type": "cross_session_reasoning"}

        res = inspect_sample_diagnostic(
            cache=cache,
            prompt_text="User: Past premise\n\n[Session Break]\n\nUser: Recent question",
            prompt_token_ids=prompt_token_ids,
            sample_meta=sample_meta,
            is_correct=False,
            baseline_correct=True,
            tokenizer=None,
            bits_per_scalar=16.0,
        )

        self.assertEqual(res.causal_verdict, "QUALITY_REGRESSION")
        self.assertEqual(res.overall_retention_ratio, 0.5)
        self.assertEqual(res.overall_effective_bitwidth, 8.0)
        self.assertIsNotNone(res.session_retention_ratios)


if __name__ == "__main__":
    unittest.main()

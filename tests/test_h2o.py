import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.core.axis import Axis
from src.core.registry import get_technique, list_techniques
from src.techniques.h2o import (
    _ACTIVE_H2O_CONFIG,
    compute_gqa_h2o_attention,
    compute_h2o_boundaries,
    compute_h2o_paged_block_boundaries,
    evict_h2o_kv_cache,
)


class TestH2OTechnique(unittest.TestCase):
    def test_registration_and_axis(self):
        self.assertIn("h2o", list_techniques())
        technique = get_technique("h2o")
        self.assertEqual(technique.axes, Axis.STRUCTURAL)

    def test_defaults_and_custom_overrides(self):
        technique = get_technique("h2o")
        self.assertEqual(technique.heavy_hitter_tokens, 1024)
        self.assertEqual(technique.recent_window_tokens, 1024)
        self.assertEqual(technique.block_size, 16)
        self.assertEqual(technique.effective_cache_tokens, 2048)
        self.assertEqual(technique.num_hh_blocks, 64)
        self.assertEqual(technique.num_recent_blocks, 64)
        self.assertEqual(technique.effective_cache_blocks, 128)

        custom = get_technique("h2o", heavy_hitter_tokens=512, recent_window_tokens=256, block_size=16)
        self.assertEqual(custom.heavy_hitter_tokens, 512)
        self.assertEqual(custom.recent_window_tokens, 256)
        self.assertEqual(custom.effective_cache_tokens, 768)
        self.assertEqual(custom.num_hh_blocks, 32)
        self.assertEqual(custom.num_recent_blocks, 16)
        self.assertEqual(custom.effective_cache_blocks, 48)

        mock_out = SimpleNamespace(prompt_tokens=4096, generated_tokens=64)
        metrics = custom.extra_metrics(None, mock_out)
        self.assertEqual(metrics["heavy_hitter_tokens"], 512)
        self.assertEqual(metrics["recent_window_tokens"], 256)
        self.assertEqual(metrics["effective_cache_tokens"], 768)
        self.assertEqual(metrics["active_kv_tokens"], 768)
        self.assertEqual(metrics["evicted_tokens"], 4160 - 768)
        self.assertGreater(metrics["evicted_blocks"], 0)

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            get_technique("h2o", heavy_hitter_tokens=-1)
        with self.assertRaises(ValueError):
            get_technique("h2o", recent_window_tokens=0)
        with self.assertRaises(ValueError):
            get_technique("h2o", block_size=0)

    def test_h2o_boundary_calculations(self):
        # 1. Sequence shorter than capacity (512 <= 1024 + 1024)
        short_info = compute_h2o_boundaries(
            seq_len=512, heavy_hitter_tokens=1024, recent_window_tokens=1024
        )
        self.assertEqual(short_info.num_evicted, 0)
        self.assertEqual(short_info.active_tokens, 512)
        self.assertEqual(short_info.retained_indices, list(range(512)))
        self.assertEqual(short_info.cache_positions, list(range(512)))
        self.assertEqual(short_info.next_query_position, 512)

        # 2. Sequence at exact capacity (2048 == 1024 + 1024)
        exact_info = compute_h2o_boundaries(
            seq_len=2048, heavy_hitter_tokens=1024, recent_window_tokens=1024
        )
        self.assertEqual(exact_info.num_evicted, 0)
        self.assertEqual(exact_info.active_tokens, 2048)
        self.assertEqual(len(exact_info.retained_recent_indices), 1024)
        self.assertEqual(len(exact_info.retained_hh_indices), 1024)

        # 3. Long sequence with custom cumulative scores
        seq_len = 100
        hh_k = 10
        rec_k = 20
        # Candidate region: [0, 80). Recent region: [80, 100).
        # Construct synthetic scores where token 5, 12, 19, 25, 33, 42, 50, 61, 70, 77 have highest values
        scores = [0.01] * seq_len
        target_hh = [5, 12, 19, 25, 33, 42, 50, 61, 70, 77]
        for idx, t in enumerate(target_hh):
            scores[t] = 10.0 + idx

        info = compute_h2o_boundaries(
            seq_len=seq_len,
            cumulative_scores=scores,
            heavy_hitter_tokens=hh_k,
            recent_window_tokens=rec_k,
        )
        self.assertEqual(info.active_tokens, 30)
        self.assertEqual(info.num_evicted, 70)
        self.assertEqual(info.retained_recent_indices, list(range(80, 100)))
        self.assertEqual(info.retained_hh_indices, sorted(target_hh))
        # Retained indices must be strictly ascending
        self.assertEqual(info.retained_indices, sorted(target_hh + list(range(80, 100))))
        self.assertEqual(info.cache_positions, list(range(30)))
        self.assertEqual(info.next_query_position, 30)

    def test_paged_block_boundaries(self):
        blk_info = compute_h2o_paged_block_boundaries(
            num_computed_tokens=4096,
            heavy_hitter_tokens=512,
            recent_window_tokens=512,
            block_size=16,
        )
        self.assertEqual(blk_info.num_hh_blocks, 32)
        self.assertEqual(blk_info.num_recent_blocks, 32)
        self.assertEqual(blk_info.effective_cache_blocks, 64)
        self.assertEqual(blk_info.total_blocks, 256)
        self.assertEqual(len(blk_info.retained_blocks), 64)
        self.assertEqual(len(blk_info.evicted_blocks), 256 - 64)
        # Recent blocks are the last 32 blocks [224 .. 255]
        self.assertEqual(blk_info.retained_recent_blocks, list(range(224, 256)))
        # Retained blocks must include all retained recent blocks
        for b in blk_info.retained_recent_blocks:
            self.assertIn(b, blk_info.retained_blocks)

    def test_evict_h2o_kv_cache_preserves_bf16_and_shapes(self):
        torch.manual_seed(42)
        batch, num_kv_heads, seq_len, head_dim = 2, 8, 256, 64
        k = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16)
        v = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16)

        retained_indices = [2, 10, 15, 50] + list(range(200, 256))
        k_pruned, v_pruned = evict_h2o_kv_cache(k, v, retained_indices=retained_indices, seq_dim=2)

        self.assertEqual(k_pruned.dtype, torch.bfloat16)
        self.assertEqual(v_pruned.dtype, torch.bfloat16)
        self.assertEqual(k_pruned.shape, (batch, num_kv_heads, len(retained_indices), head_dim))
        self.assertEqual(v_pruned.shape, (batch, num_kv_heads, len(retained_indices), head_dim))

        # Check exact value preservation
        for out_idx, orig_idx in enumerate(retained_indices):
            self.assertTrue(torch.equal(k_pruned[:, :, out_idx, :], k[:, :, orig_idx, :]))
            self.assertTrue(torch.equal(v_pruned[:, :, out_idx, :], v[:, :, orig_idx, :]))

    def test_gqa_h2o_attention_and_score_tracking(self):
        """Verify Grouped-Query Attention head pooling and cumulative score accumulation."""
        torch.manual_seed(42)
        batch = 1
        num_q_heads = 32   # Qwen3-4B query heads
        num_kv_heads = 8   # Qwen3-4B KV heads (4:1 GQA ratio)
        kv_len = 128
        head_dim = 64
        hh_tokens = 16
        recent_tokens = 16
        capacity = hh_tokens + recent_tokens  # 32

        q = torch.randn(batch, num_q_heads, 1, head_dim)
        k = torch.randn(batch, num_kv_heads, kv_len, head_dim)
        v = torch.randn(batch, num_kv_heads, kv_len, head_dim)

        # 1. First step without prior scores
        out1, weights1, scores1 = compute_gqa_h2o_attention(
            q, k, v, cumulative_scores=None,
            heavy_hitter_tokens=hh_tokens,
            recent_window_tokens=recent_tokens,
        )
        self.assertEqual(out1.shape, (batch, num_q_heads, 1, head_dim))
        self.assertEqual(weights1.shape, (batch, num_q_heads, 1, kv_len))
        # Weights across all GQA heads must sum to 1
        sums = weights1.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

        # Output score length is pruned to capacity (32)
        self.assertEqual(scores1.shape, (batch, num_kv_heads, capacity))

        # 2. Cumulative score updates across subsequent steps
        q2 = torch.randn(batch, num_q_heads, 1, head_dim)
        # Sliced K, V to current capacity
        k_active = k[:, :, :capacity, :]
        v_active = v[:, :, :capacity, :]
        out2, weights2, scores2 = compute_gqa_h2o_attention(
            q2, k_active, v_active, cumulative_scores=scores1,
            heavy_hitter_tokens=hh_tokens,
            recent_window_tokens=recent_tokens,
        )
        self.assertEqual(out2.shape, (batch, num_q_heads, 1, head_dim))
        self.assertEqual(scores2.shape, (batch, num_kv_heads, capacity))
        # Cumulative scores strictly increase
        self.assertTrue((scores2 >= scores1).all())

    def test_engine_kwargs_and_setup(self):
        technique = get_technique("h2o", heavy_hitter_tokens=1024, recent_window_tokens=1024, block_size=16)
        technique.setup()
        self.assertTrue(_ACTIVE_H2O_CONFIG["enabled"])
        self.assertEqual(_ACTIVE_H2O_CONFIG["heavy_hitter_tokens"], 1024)
        self.assertEqual(_ACTIVE_H2O_CONFIG["recent_window_tokens"], 1024)

        kwargs = technique.engine_kwargs(context_length=8192, max_tokens=128)
        self.assertEqual(kwargs["block_size"], 16)
        self.assertFalse(kwargs["enable_prefix_caching"])
        self.assertIn("hf_overrides", kwargs)
        self.assertEqual(kwargs["hf_overrides"]["sliding_window"], 2048)

        # Test EngineArgs only if vllm is installed in this environment
        try:
            from vllm import EngineArgs
            engine_args = EngineArgs(model="dummy", **kwargs)
            self.assertEqual(engine_args.block_size, 16)
        except ImportError:
            pass

        _ACTIVE_H2O_CONFIG["enabled"] = False

    def test_pytorch_cache_factory_and_h2o_cache_lifecycle(self):
        from src.techniques.h2o_cache import H2OCache, H2OLayer, record_h2o_attention_scores

        technique = get_technique("h2o", heavy_hitter_tokens=16, recent_window_tokens=16)
        factory = technique.pytorch_cache_factory()
        self.assertIsNotNone(factory)

        cache = factory()
        self.assertIsInstance(cache, H2OCache)
        self.assertEqual(cache.get_max_length(), 32)
        self.assertEqual(cache.heavy_hitter_tokens, 16)
        self.assertEqual(cache.recent_window_tokens, 16)

        # Feed initial prompt of 32 tokens
        batch, num_kv_heads, num_q_heads, head_dim = 1, 4, 16, 64
        layer = H2OLayer(heavy_hitter_tokens=16, recent_window_tokens=16)

        k1 = torch.randn(batch, num_kv_heads, 32, head_dim, dtype=torch.bfloat16)
        v1 = torch.randn(batch, num_kv_heads, 32, head_dim, dtype=torch.bfloat16)
        k_out, v_out = layer.update(k1, v1)
        self.assertEqual(k_out.shape, (batch, num_kv_heads, 32, head_dim))
        self.assertEqual(layer.cumulative_scores.shape, (batch, num_kv_heads, 32))
        self.assertTrue((layer.cumulative_scores == 0.0).all())

        # Give tokens 2 and 5 high synthetic attention
        step_scores = torch.zeros(batch, num_kv_heads, 32)
        step_scores[:, :, 2] = 100.0
        step_scores[:, :, 5] = 100.0
        layer.record_scores(step_scores)
        self.assertEqual(layer.cumulative_scores[:, :, 2].mean().item(), 100.0)

        # Append 1 new token during decode (seq_len becomes 33 > capacity 32)
        k2 = torch.randn(batch, num_kv_heads, 1, head_dim, dtype=torch.bfloat16)
        v2 = torch.randn(batch, num_kv_heads, 1, head_dim, dtype=torch.bfloat16)
        layer.update(k2, v2)
        self.assertEqual(layer.get_seq_length(), 33)

        # Run decode attention scoring
        q_dec = torch.randn(batch, num_q_heads, 1, head_dim, dtype=torch.bfloat16)
        record_h2o_attention_scores(layer, q_dec, layer.keys)
        self.assertEqual(layer.cumulative_scores.shape, (batch, num_kv_heads, 33))

        # Prune: must evict 1 middle token while retaining token 2, 5, and the recent 16 tokens [17..32]
        layer.prune()
        self.assertEqual(layer.get_seq_length(), 32)
        self.assertEqual(layer.keys.shape, (batch, num_kv_heads, 32, head_dim))
        self.assertEqual(layer.values.shape, (batch, num_kv_heads, 32, head_dim))
        self.assertEqual(layer.cumulative_scores.shape, (batch, num_kv_heads, 32))

    def test_h2o_cache_prune_after_chunk(self):
        from src.techniques.h2o_cache import H2OCache

        cache = H2OCache(heavy_hitter_tokens=16, recent_window_tokens=16)
        batch, num_kv_heads, head_dim = 1, 4, 64

        # Chunk 1: 32 tokens (fills capacity)
        k_chunk1 = torch.randn(batch, num_kv_heads, 32, head_dim, dtype=torch.bfloat16)
        v_chunk1 = torch.randn(batch, num_kv_heads, 32, head_dim, dtype=torch.bfloat16)
        cache.update(k_chunk1, v_chunk1, layer_idx=0)
        cache.prune_after_chunk()
        self.assertEqual(cache.get_seq_length(0), 32)

        # Chunk 2: 16 more tokens (now 48 tokens > 32)
        k_chunk2 = torch.randn(batch, num_kv_heads, 16, head_dim, dtype=torch.bfloat16)
        v_chunk2 = torch.randn(batch, num_kv_heads, 16, head_dim, dtype=torch.bfloat16)
        cache.update(k_chunk2, v_chunk2, layer_idx=0)
        self.assertEqual(cache.get_seq_length(0), 48)

        # prune_after_chunk must clamp cache back to 32 tokens
        cache.prune_after_chunk()
        self.assertEqual(cache.get_seq_length(0), 32)


if __name__ == "__main__":
    unittest.main()

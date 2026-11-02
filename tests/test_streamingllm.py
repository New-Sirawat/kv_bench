import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.core.axis import Axis
from src.core.registry import get_technique, list_techniques
from src.techniques.streamingllm import (
    _ACTIVE_STREAMING_CONFIG,
    compute_cache_boundaries,
    compute_gqa_streaming_attention,
    compute_paged_block_boundaries,
    evict_streaming_kv_cache,
)


class TestStreamingLLMTechnique(unittest.TestCase):
    def test_registration_and_axis(self):
        self.assertIn("streamingllm", list_techniques())
        technique = get_technique("streamingllm")
        self.assertEqual(technique.axes, Axis.STRUCTURAL)

    def test_defaults_and_custom_overrides(self):
        technique = get_technique("streamingllm")
        self.assertEqual(technique.sink_tokens, 4)
        self.assertEqual(technique.window_size, 2048)
        self.assertEqual(technique.block_size, 16)
        self.assertEqual(technique.num_sink_blocks, 1)
        self.assertEqual(technique.effective_cache_tokens, 2052)
        self.assertEqual(technique.effective_sliding_window, 2064)

        custom = get_technique("streamingllm", sink_tokens=20, window_size=1024, block_size=16)
        self.assertEqual(custom.sink_tokens, 20)
        self.assertEqual(custom.window_size, 1024)
        self.assertEqual(custom.num_sink_blocks, 2)
        self.assertEqual(custom.effective_cache_tokens, 1044)
        self.assertEqual(custom.effective_sliding_window, 1056)

        mock_out = SimpleNamespace(prompt_tokens=4096, generated_tokens=64)
        metrics = custom.extra_metrics(None, mock_out)
        self.assertEqual(metrics["sink_tokens"], 20)
        self.assertEqual(metrics["window_size"], 1024)
        self.assertEqual(metrics["num_sink_blocks"], 2)
        self.assertEqual(metrics["active_kv_tokens"], 1044)
        self.assertEqual(metrics["evicted_middle_tokens"], 4160 - 1044)

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            get_technique("streamingllm", sink_tokens=-1)
        with self.assertRaises(ValueError):
            get_technique("streamingllm", window_size=0)
        with self.assertRaises(ValueError):
            get_technique("streamingllm", block_size=0)

    def test_cache_boundary_calculations(self):
        # 1. Sequence shorter than capacity (512 <= 4 + 2048)
        short_info = compute_cache_boundaries(seq_len=512, sink_tokens=4, window_size=2048)
        self.assertEqual(short_info.sink_range, (0, 4))
        self.assertEqual(short_info.evicted_range, (4, 4))
        self.assertEqual(short_info.window_range, (4, 512))
        self.assertEqual(short_info.num_evicted, 0)
        self.assertEqual(short_info.active_tokens, 512)
        self.assertEqual(short_info.retained_indices, list(range(512)))
        self.assertEqual(short_info.cache_positions, list(range(512)))
        self.assertEqual(short_info.next_query_position, 512)

        # 2. Sequence at exact capacity (2052 == 4 + 2048)
        exact_info = compute_cache_boundaries(seq_len=2052, sink_tokens=4, window_size=2048)
        self.assertEqual(exact_info.num_evicted, 0)
        self.assertEqual(exact_info.active_tokens, 2052)
        self.assertEqual(exact_info.sink_range, (0, 4))
        self.assertEqual(exact_info.window_range, (4, 2052))

        # 3. Long sequence exceeding capacity (8192 > 4 + 2048)
        long_info = compute_cache_boundaries(seq_len=8192, sink_tokens=4, window_size=2048)
        self.assertEqual(long_info.sink_range, (0, 4))
        self.assertEqual(long_info.evicted_range, (4, 6144))
        self.assertEqual(long_info.window_range, (6144, 8192))
        self.assertEqual(long_info.num_evicted, 6140)
        self.assertEqual(long_info.active_tokens, 2052)
        self.assertEqual(long_info.retained_indices[:4], [0, 1, 2, 3])
        self.assertEqual(long_info.retained_indices[4:], list(range(6144, 8192)))
        # Cache-centric RoPE positions must be contiguous [0 .. 2051] with no index gap
        self.assertEqual(long_info.cache_positions, list(range(2052)))
        self.assertEqual(long_info.next_query_position, 2052)

    def test_paged_block_boundary_calculations(self):
        blk_info = compute_paged_block_boundaries(
            num_computed_tokens=8192,
            sink_tokens=4,
            window_size=2048,
            block_size=16,
        )
        self.assertEqual(blk_info.num_sink_blocks, 1)
        self.assertEqual(blk_info.num_window_blocks, 128)
        self.assertEqual(blk_info.effective_sliding_window, 2064)
        self.assertEqual(blk_info.total_blocks, 512)
        # First block (block 0) is pinned as sink; middle blocks [1, 383) are evicted
        self.assertEqual(blk_info.retained_sink_blocks, [0])
        self.assertEqual(blk_info.evict_block_range[0], 1)
        self.assertGreater(blk_info.evict_block_range[1], 380)
        self.assertNotIn(0, range(*blk_info.evict_block_range))

    def test_evict_streaming_kv_cache_preserves_bf16_and_boundaries(self):
        torch.manual_seed(42)
        batch, num_kv_heads, seq_len, head_dim = 2, 8, 256, 64
        k = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16)
        v = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16)

        k_pruned, v_pruned = evict_streaming_kv_cache(k, v, sink_tokens=4, window_size=64, seq_dim=2)
        self.assertEqual(k_pruned.dtype, torch.bfloat16)
        self.assertEqual(v_pruned.dtype, torch.bfloat16)
        self.assertEqual(k_pruned.shape, (batch, num_kv_heads, 68, head_dim))
        self.assertEqual(v_pruned.shape, (batch, num_kv_heads, 68, head_dim))

        # Verify first 4 sink tokens and last 64 recent tokens are exact
        self.assertTrue(torch.equal(k_pruned[:, :, :4, :], k[:, :, :4, :]))
        self.assertTrue(torch.equal(k_pruned[:, :, 4:, :], k[:, :, -64:, :]))
        self.assertTrue(torch.equal(v_pruned[:, :, :4, :], v[:, :, :4, :]))
        self.assertTrue(torch.equal(v_pruned[:, :, 4:, :], v[:, :, -64:, :]))

    def test_gqa_head_stability_with_attention_sinks(self):
        """Verify that pinning sink tokens stabilizes all grouped query heads in GQA."""
        torch.manual_seed(7)
        batch = 1
        num_q_heads = 32   # Qwen3-4B query heads
        num_kv_heads = 8   # Qwen3-4B KV heads (4:1 GQA ratio)
        group_size = num_q_heads // num_kv_heads
        seq_len = 256
        head_dim = 64
        sink_tokens = 4
        window_size = 64

        # Construct synthetic Q, K, V where initial 4 tokens act as strong attention sinks
        sink_dir = F.normalize(torch.ones(1, 1, 1, head_dim), dim=-1)
        q = sink_dir.expand(batch, num_q_heads, 1, head_dim) * 8.0 + 0.05 * torch.randn(
            batch, num_q_heads, 1, head_dim
        )

        k = 0.05 * torch.randn(batch, num_kv_heads, seq_len, head_dim)
        v = torch.randn(batch, num_kv_heads, seq_len, head_dim)

        # Initial 4 sink tokens strongly align with query direction (attention sink)
        k[:, :, :sink_tokens, :] = sink_dir * 10.0
        # Sink values are near zero (serving as a softmax denominator dump site)
        v[:, :, :sink_tokens, :] = 0.0
        # Middle tokens [4 : seq_len - window_size] are unaligned (negligible attention)
        k[:, :, sink_tokens : seq_len - window_size, :] = -sink_dir * 8.0

        # 1. Full Reference Attention across all 32 GQA heads
        k_full_exp = k.repeat_interleave(group_size, dim=1)
        v_full_exp = v.repeat_interleave(group_size, dim=1)
        scores_full = torch.matmul(q, k_full_exp.transpose(-2, -1)) * (head_dim ** -0.5)
        weights_full = F.softmax(scores_full, dim=-1)
        out_full = torch.matmul(weights_full, v_full_exp)

        # 2. StreamingLLM (sink_tokens=4, window_size=64)
        out_streaming, weights_streaming = compute_gqa_streaming_attention(
            q, k, v, sink_tokens=sink_tokens, window_size=window_size
        )

        # 3. Naive Sliding Window without Sinks (sink_tokens=0, window_size=64)
        out_naive, _ = compute_gqa_streaming_attention(
            q, k, v, sink_tokens=0, window_size=window_size
        )

        # Check shape and probability conservation across all 32 GQA heads
        self.assertEqual(out_streaming.shape, (batch, num_q_heads, 1, head_dim))
        self.assertEqual(weights_streaming.shape, (batch, num_q_heads, 1, sink_tokens + window_size))
        weight_sums = weights_streaming.sum(dim=-1)
        self.assertTrue(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5))

        # StreamingLLM preserves the softmax denominator across all 32 grouped heads;
        # naive window attention without sinks collapses and diverges sharply.
        err_streaming = (out_streaming - out_full).abs().max().item()
        err_naive = (out_naive - out_full).abs().max().item()
        self.assertLess(err_streaming, 1e-3)
        self.assertGreater(err_naive, 0.1)

    def test_engine_kwargs_and_vllm_setup_hooks(self):
        from vllm import EngineArgs
        import vllm.v1.core.single_type_kv_cache_manager as vllm_kv_mgr

        technique = get_technique("streamingllm", sink_tokens=4, window_size=2048, block_size=16)
        technique.setup()
        self.assertTrue(_ACTIVE_STREAMING_CONFIG["enabled"])

        kwargs = technique.engine_kwargs(context_length=8192, max_tokens=128)
        self.assertEqual(kwargs["block_size"], 16)
        self.assertIn("hf_overrides", kwargs)
        self.assertEqual(kwargs["hf_overrides"]["sliding_window"], 2064)

        # Verify EngineArgs accepts all returned kwargs without TypeError
        engine_args = EngineArgs(model="dummy", **kwargs)
        self.assertEqual(engine_args.block_size, 16)

        # Verify that the patched SlidingWindowManager.remove_skipped_blocks
        # pins block 0 (sink block) and only frees middle blocks [1, num_skipped_blocks)
        freed_ranges: list[tuple[int, int]] = []

        class DummySWManager:
            block_size = 16
            req_to_blocks = {"req-1": [f"blk-{i}" for i in range(200)]}

            def get_num_skipped_tokens(self, processed_computed_tokens: int) -> int:
                return max(0, processed_computed_tokens - 2064 + 1)

            def _remove_blocks_in_range(self, request_id: str, start: int, end: int) -> None:
                freed_ranges.append((start, end))

        dummy_mgr = DummySWManager()
        vllm_kv_mgr.SlidingWindowManager.remove_skipped_blocks(
            dummy_mgr, "req-1", processed_computed_tokens=4096
        )
        self.assertEqual(len(freed_ranges), 1)
        start_blk, end_blk = freed_ranges[0]
        # Block 0 MUST be preserved (start_blk == 1), while middle blocks up to 127 are freed
        self.assertEqual(start_blk, 1)
        self.assertEqual(end_blk, (4096 - 2064 + 1) // 16)

        _ACTIVE_STREAMING_CONFIG["enabled"] = False


if __name__ == "__main__":
    unittest.main()

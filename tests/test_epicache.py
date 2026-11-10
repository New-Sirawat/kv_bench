import unittest
from types import SimpleNamespace

import torch

from src.core.axis import Axis
from src.core.diagnostic import inspect_sample_diagnostic
from src.core.pytorch_runner import PyTorchRunner
from src.core.registry import get_technique, list_techniques
from src.techniques.epicache import (
    EpiCacheTechnique,
    compute_gqa_epicache_attention,
    compute_layer_sensitivity_budgets,
    evict_block_prefill,
    route_query_to_episode,
)
from src.techniques.epicache_cache import (
    EpiCacheCache,
    EpiCacheLayer,
    get_active_epicache_cache,
)


class TestEpiCacheTechnique(unittest.TestCase):
    def test_registration_and_axes(self):
        self.assertIn("epicache", list_techniques())
        technique = get_technique("epicache")
        self.assertIsInstance(technique, EpiCacheTechnique)
        self.assertEqual(technique.axes, Axis.STRUCTURAL | Axis.SPATIAL)
        self.assertEqual(technique.compression_ratio, 3.8)

    def test_defaults_and_custom_parameters(self):
        technique = get_technique("epicache")
        self.assertEqual(technique.num_episodes, 4)
        self.assertEqual(technique.episode_budget_tokens, 1024)
        self.assertEqual(technique.layer_sensitivity_alpha, 2.0)
        self.assertEqual(technique.block_size, 16)

        mock_out = SimpleNamespace(prompt_tokens=4096, generated_tokens=64)
        metrics = technique.extra_metrics(None, mock_out)
        self.assertEqual(metrics["num_episodes"], 4)
        self.assertTrue(metrics["block_wise_prefill_active"])
        self.assertTrue(metrics["episodic_routing_active"])

        kwargs = technique.engine_kwargs(context_length=8192, max_tokens=128)
        self.assertEqual(kwargs["block_size"], 16)
        self.assertEqual(kwargs["swap_space"], 8)

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            EpiCacheTechnique(num_episodes=0)
        with self.assertRaises(ValueError):
            EpiCacheTechnique(episode_budget_tokens=-1)
        with self.assertRaises(ValueError):
            EpiCacheTechnique(layer_sensitivity_alpha=0.0)
        with self.assertRaises(ValueError):
            EpiCacheTechnique(block_size=0)

    def test_layer_sensitivity_budgets(self):
        budgets = compute_layer_sensitivity_budgets(total_layers=16, base_budget_per_layer=512, alpha=2.0)
        self.assertEqual(len(budgets), 16)
        for l, b in budgets.items():
            self.assertGreaterEqual(b, 64)
        self.assertAlmostEqual(sum(budgets.values()) / (16 * 512), 1.0, delta=0.05)

    def test_route_query_to_episode(self):
        dim = 32
        num_episodes = 4

        torch.manual_seed(42)
        centroids = torch.randn(num_episodes, dim, dtype=torch.float32)
        query_emb = centroids[2] + 0.01 * torch.randn(dim)

        best_ep = route_query_to_episode(query_emb, centroids)
        self.assertEqual(best_ep, 2)

    def test_evict_block_prefill(self):
        batch, heads, seq_len, dim = 1, 2, 64, 32
        budget = 16

        torch.manual_seed(100)
        k = torch.randn(batch, heads, seq_len, dim, dtype=torch.float32)
        v = torch.randn(batch, heads, seq_len, dim, dtype=torch.float32)

        k_ret, v_ret, indices = evict_block_prefill(k, v, budget=budget)
        self.assertEqual(k_ret.shape[-2], budget)
        self.assertEqual(v_ret.shape[-2], budget)
        self.assertIn(0, indices.tolist())

    def test_compute_gqa_epicache_attention(self):
        batch, seq_len_q, head_dim = 1, 1, 32
        num_q_heads, num_kv_heads = 8, 2
        seq_len_kv = 64

        torch.manual_seed(200)
        q = torch.randn(batch, num_q_heads, seq_len_q, head_dim, dtype=torch.float32)
        k = torch.randn(batch, num_kv_heads, seq_len_kv, head_dim, dtype=torch.float32)
        v = torch.randn(batch, num_kv_heads, seq_len_kv, head_dim, dtype=torch.float32)

        attn_out, attn_weights, telemetry = compute_gqa_epicache_attention(q, k, v, budget=16)

        self.assertEqual(attn_out.shape, (batch, num_q_heads, seq_len_q, head_dim))
        self.assertTrue(torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)), atol=1e-4))
        self.assertEqual(telemetry["active_tokens"], 16)
        self.assertTrue(telemetry["block_prefill_capped"])

    def test_epicache_layer_chunk_pruning_and_decode(self):
        torch.manual_seed(123)
        cache = EpiCacheCache(total_layers=4, episode_budget_tokens=16)
        self.assertEqual(get_active_epicache_cache(), cache)

        batch, heads, dim = 1, 4, 64
        k_chunk = torch.randn(batch, 32, heads, dim, dtype=torch.bfloat16).transpose(1, 2)
        v_chunk = torch.randn(batch, 32, heads, dim, dtype=torch.bfloat16).transpose(1, 2)

        cache.update(k_chunk, v_chunk, layer_idx=0)
        cache.prune_after_chunk()
        self.assertEqual(cache.get_seq_length(0), 16)
        self.assertIn(0, cache.original_positions)

        # Decode 1 token beyond budget -> automatically re-prunes to budget 16
        k_dec = torch.randn(batch, heads, 1, dim, dtype=torch.bfloat16)
        v_dec = torch.randn(batch, heads, 1, dim, dtype=torch.bfloat16)
        out_k, _ = cache.update(k_dec, v_dec, layer_idx=0)
        self.assertEqual(out_k.shape[-2], 16)

    def test_epicache_factory_diagnostic_and_pytorch_runner(self):
        technique = get_technique("epicache", episode_budget_tokens=16)
        cache_factory = technique.pytorch_cache_factory()
        self.assertIsNotNone(cache_factory)

        cache = cache_factory()
        self.assertIsInstance(cache, EpiCacheCache)

        k1 = torch.randn(1, 4, 40, 64, dtype=torch.float32)
        v1 = torch.randn(1, 4, 40, 64, dtype=torch.float32)
        cache.update(k1, v1, layer_idx=0)
        cache.prune_after_chunk()

        diag = inspect_sample_diagnostic(
            cache=cache,
            prompt_text="Test EpiCache diagnostic",
            prompt_token_ids=list(range(40)),
            sample_meta={"id": "ec_diag_0", "task_type": "unit"},
            is_correct=True,
            baseline_correct=True,
            bits_per_scalar=16.0,
        )
        self.assertAlmostEqual(diag.overall_retention_ratio, 16 / 40, places=3)

        runner = PyTorchRunner(
            model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
            revision="cdbee75f17c01a7cc42f958dc650907174af0554",
            cache_factory=cache_factory,
            dry_run=True,
        )
        outputs = runner.generate(["Test EpiCache prompt."], max_tokens=16)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(outputs[0].outputs[0].token_ids), 16)


if __name__ == "__main__":
    unittest.main()

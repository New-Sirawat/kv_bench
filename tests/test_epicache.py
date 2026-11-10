import unittest
from types import SimpleNamespace

import torch

from src.core.axis import Axis
from src.core.registry import get_technique, list_techniques
from src.techniques.epicache import (
    EpiCacheTechnique,
    compute_gqa_epicache_attention,
    compute_layer_sensitivity_budgets,
    evict_block_prefill,
    route_query_to_episode,
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
        # All layer budgets must be positive and non-trivial
        for l, b in budgets.items():
            self.assertGreaterEqual(b, 64)
        # Sum of budgets should be within reasonable bounds of total allocation
        self.assertAlmostEqual(sum(budgets.values()) / (16 * 512), 1.0, delta=0.05)

    def test_route_query_to_episode(self):
        dim = 32
        num_episodes = 4

        torch.manual_seed(42)
        centroids = torch.randn(num_episodes, dim, dtype=torch.float32)
        # Query closely aligned with episode 2
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
        # Attention sink (token 0) must always be preserved
        self.assertIn(0, indices.tolist())

    def test_compute_gqa_epicache_attention(self):
        batch, seq_len_q, head_dim = 1, 1, 32
        num_q_heads, num_kv_heads = 8, 2  # 4:1 GQA ratio
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


if __name__ == "__main__":
    unittest.main()

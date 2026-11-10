import unittest
from types import SimpleNamespace

import torch

from src.core.axis import Axis
from src.core.registry import get_technique, list_techniques
from src.techniques.arkvale import (
    ArkValeTechnique,
    compute_gqa_arkvale_attention,
    compute_page_aabb_digests,
    score_page_digests_gqa,
    select_and_recall_pages,
)


class TestArkValeTechnique(unittest.TestCase):
    def test_registration_and_axes(self):
        self.assertIn("arkvale", list_techniques())
        technique = get_technique("arkvale")
        self.assertIsInstance(technique, ArkValeTechnique)
        self.assertEqual(technique.axes, Axis.STRUCTURAL | Axis.SPATIAL)
        self.assertEqual(technique.compression_ratio, 4.5)

    def test_defaults_and_custom_parameters(self):
        technique = get_technique("arkvale")
        self.assertEqual(technique.page_size, 16)
        self.assertEqual(technique.active_pages_budget, 16)
        self.assertEqual(technique.exempt_early_layers, 2)
        self.assertEqual(technique.block_size, 16)

        mock_out = SimpleNamespace(prompt_tokens=4096, generated_tokens=64)
        metrics = technique.extra_metrics(None, mock_out)
        self.assertEqual(metrics["page_size"], 16)
        self.assertTrue(metrics["aabb_digests_active"])
        self.assertTrue(metrics["page_recall_supported"])

        kwargs = technique.engine_kwargs(context_length=8192, max_tokens=128)
        self.assertEqual(kwargs["block_size"], 16)
        self.assertEqual(kwargs["swap_space"], 8)

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            ArkValeTechnique(page_size=0)
        with self.assertRaises(ValueError):
            ArkValeTechnique(active_pages_budget=-1)
        with self.assertRaises(ValueError):
            ArkValeTechnique(exempt_early_layers=-1)
        with self.assertRaises(ValueError):
            ArkValeTechnique(block_size=0)

    def test_page_aabb_digests_properties(self):
        torch.manual_seed(42)
        batch, heads, seq_len, dim = 2, 4, 48, 64
        page_size = 16
        k = torch.randn(batch, heads, seq_len, dim, dtype=torch.float32)

        b_min, b_max, num_pages = compute_page_aabb_digests(k, page_size=page_size)
        self.assertEqual(num_pages, 3)
        self.assertEqual(b_min.shape, (batch, heads, 3, dim))
        self.assertEqual(b_max.shape, (batch, heads, 3, dim))
        # Upper bounds must be strictly >= lower bounds
        self.assertTrue((b_max >= b_min).all())

    def test_score_page_digests_gqa_and_select(self):
        batch, seq_len_q, dim = 1, 1, 64
        num_q_heads, num_kv_heads = 8, 2
        seq_len_kv = 64
        page_size = 16

        torch.manual_seed(100)
        q = torch.randn(batch, num_q_heads, seq_len_q, dim, dtype=torch.float32)
        k = torch.randn(batch, num_kv_heads, seq_len_kv, dim, dtype=torch.float32)

        b_min, b_max, num_pages = compute_page_aabb_digests(k, page_size=page_size)
        kv_scores = score_page_digests_gqa(q, b_min, b_max, gqa_ratio=4)

        self.assertEqual(kv_scores.shape, (batch, num_kv_heads, num_pages))

        active_pages, telemetry = select_and_recall_pages(kv_scores, active_pages_budget=2)
        # Page 0 must always be preserved
        self.assertIn(0, active_pages.tolist())
        self.assertLessEqual(active_pages.numel(), 2)
        self.assertIn("page_sparsity_ratio", telemetry)

    def test_compute_gqa_arkvale_attention(self):
        batch, seq_len_q, head_dim = 1, 1, 64
        num_q_heads, num_kv_heads = 8, 2  # 4:1 GQA ratio
        seq_len_kv = 64

        torch.manual_seed(200)
        q = torch.randn(batch, num_q_heads, seq_len_q, head_dim, dtype=torch.float32)
        k = torch.randn(batch, num_kv_heads, seq_len_kv, head_dim, dtype=torch.float32)
        v = torch.randn(batch, num_kv_heads, seq_len_kv, head_dim, dtype=torch.float32)

        attn_out, attn_weights, telemetry = compute_gqa_arkvale_attention(
            q, k, v, page_size=16, active_pages_budget=3
        )

        self.assertEqual(attn_out.shape, (batch, num_q_heads, seq_len_q, head_dim))
        self.assertTrue(torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)), atol=1e-4))
        self.assertIn("active_tokens", telemetry)
        self.assertTrue(telemetry["zero_copy_eviction_active"])


if __name__ == "__main__":
    unittest.main()

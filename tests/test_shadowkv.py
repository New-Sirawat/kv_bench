import unittest
from types import SimpleNamespace

import torch

from src.core.axis import Axis
from src.core.registry import get_technique, list_techniques
from src.techniques.shadowkv import (
    ShadowKVTechnique,
    compute_chunk_landmarks,
    compute_gqa_shadowkv_attention,
    compute_pre_rope_svd,
    identify_outlier_chunks,
    select_topk_chunks_gqa,
)


class TestShadowKVTechnique(unittest.TestCase):
    def test_registration_and_axes(self):
        self.assertIn("shadowkv", list_techniques())
        technique = get_technique("shadowkv")
        self.assertIsInstance(technique, ShadowKVTechnique)
        self.assertEqual(
            technique.axes,
            Axis.SPATIAL | Axis.STRUCTURAL | Axis.TEMPORAL,
        )
        self.assertEqual(technique.compression_ratio, 6.5)

    def test_defaults_and_custom_parameters(self):
        technique = get_technique("shadowkv")
        self.assertEqual(technique.rank, 16)
        self.assertEqual(technique.chunk_size, 8)
        self.assertEqual(technique.outlier_ratio, 0.05)
        self.assertEqual(technique.sparse_budget_chunks, 16)
        self.assertEqual(technique.temporal_reuse_ratio, 0.60)
        self.assertEqual(technique.block_size, 16)

        mock_out = SimpleNamespace(prompt_tokens=4096, generated_tokens=128)
        metrics = technique.extra_metrics(None, mock_out)
        self.assertEqual(metrics["svd_rank"], 16)
        self.assertEqual(metrics["chunk_size"], 8)
        self.assertTrue(metrics["spatial_offload_active"])
        self.assertTrue(metrics["temporal_caching_active"])
        self.assertTrue(metrics["dual_stream_overlap_supported"])

        kwargs = technique.engine_kwargs(context_length=8192, max_tokens=256)
        self.assertEqual(kwargs["block_size"], 16)
        self.assertEqual(kwargs["swap_space"], 8)

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            ShadowKVTechnique(rank=0)
        with self.assertRaises(ValueError):
            ShadowKVTechnique(chunk_size=-1)
        with self.assertRaises(ValueError):
            ShadowKVTechnique(outlier_ratio=1.5)
        with self.assertRaises(ValueError):
            ShadowKVTechnique(temporal_reuse_ratio=-0.1)

    def test_pre_rope_svd_reconstruction(self):
        torch.manual_seed(42)
        batch, heads, seq_len, dim = 2, 4, 32, 64
        rank = 16
        k = torch.randn(batch, heads, seq_len, dim, dtype=torch.float32)

        A, B = compute_pre_rope_svd(k, rank=rank)
        self.assertEqual(A.shape, (batch, heads, seq_len, rank))
        self.assertEqual(B.shape, (batch, heads, rank, dim))

        k_recon = torch.matmul(A, B)
        self.assertEqual(k_recon.shape, k.shape)
        # Cosine similarity should be high (> 0.85 for rank 16 on dim 64)
        cos_sim = torch.cosine_similarity(k.view(-1, dim), k_recon.view(-1, dim), dim=-1)
        self.assertGreater(cos_sim.mean().item(), 0.80)

    def test_chunk_landmarks_and_outliers(self):
        torch.manual_seed(100)
        batch, heads, seq_len, dim = 1, 2, 64, 32
        chunk_size = 8
        k = torch.randn(batch, heads, seq_len, dim, dtype=torch.float32)

        landmarks, num_chunks = compute_chunk_landmarks(k, chunk_size=chunk_size)
        self.assertEqual(num_chunks, 8)
        self.assertEqual(landmarks.shape, (batch, heads, 8, dim))

        outliers = identify_outlier_chunks(k, landmarks, chunk_size=chunk_size, outlier_ratio=0.15)
        # Chunk 0 (attention sink) must always be retained
        self.assertIn(0, outliers.flatten().tolist())

    def test_select_topk_chunks_gqa(self):
        batch, seq_len_q, dim = 1, 1, 32
        num_q_heads, num_kv_heads = 8, 2
        num_chunks = 16
        gqa_ratio = 4

        torch.manual_seed(200)
        q = torch.randn(batch, num_q_heads, seq_len_q, dim, dtype=torch.float32)
        landmarks = torch.randn(batch, num_kv_heads, num_chunks, dim, dtype=torch.float32)

        top_chunks = select_topk_chunks_gqa(q, landmarks, k=4, gqa_ratio=gqa_ratio)
        self.assertEqual(top_chunks.shape, (batch, num_kv_heads, 4))
        self.assertTrue((top_chunks < num_chunks).all())

    def test_compute_gqa_shadowkv_attention(self):
        batch, seq_len_q, head_dim = 1, 1, 32
        num_q_heads, num_kv_heads = 8, 2  # 4:1 GQA ratio
        seq_len_kv = 64

        torch.manual_seed(300)
        q = torch.randn(batch, num_q_heads, seq_len_q, head_dim, dtype=torch.float32)
        k_pre = torch.randn(batch, num_kv_heads, seq_len_kv, head_dim, dtype=torch.float32)
        k_post = torch.randn(batch, num_kv_heads, seq_len_kv, head_dim, dtype=torch.float32)
        v = torch.randn(batch, num_kv_heads, seq_len_kv, head_dim, dtype=torch.float32)

        attn_out, attn_weights, telemetry = compute_gqa_shadowkv_attention(
            q, k_pre, k_post, v, rank=8, chunk_size=8, sparse_chunks_budget=3, outlier_ratio=0.1
        )

        self.assertEqual(attn_out.shape, (batch, num_q_heads, seq_len_q, head_dim))
        self.assertTrue(torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)), atol=1e-4))
        self.assertIn("active_tokens", telemetry)
        self.assertIn("sparsity_ratio", telemetry)
        self.assertGreater(telemetry["sparsity_ratio"], 0.0)
        self.assertTrue(telemetry["dual_stream_overlapped"])


if __name__ == "__main__":
    unittest.main()

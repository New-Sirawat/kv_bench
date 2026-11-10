import unittest
from types import SimpleNamespace

import torch

from src.core.axis import Axis
from src.core.diagnostic import inspect_sample_diagnostic
from src.core.pytorch_runner import PyTorchRunner
from src.core.registry import get_technique, list_techniques
from src.techniques.shadowkv import (
    ShadowKVTechnique,
    compute_chunk_landmarks,
    compute_gqa_shadowkv_attention,
    compute_pre_rope_svd,
    identify_outlier_chunks,
    select_topk_chunks_gqa,
)
from src.techniques.shadowkv_cache import (
    ShadowKVCache,
    ShadowKVLayer,
    get_active_shadowkv_cache,
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
        num_q_heads, num_kv_heads = 8, 2
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
        self.assertGreater(telemetry["sparsity_ratio"], 0.0)
        self.assertTrue(telemetry["dual_stream_overlapped"])

    def test_shadowkv_layer_offload_and_sparse_decode(self):
        torch.manual_seed(99)
        layer = ShadowKVLayer(rank=8, chunk_size=8, sparse_budget_chunks=2)
        batch, heads, dim = 1, 4, 64

        # Prefill 64 tokens -> 8 chunks (> budget of 2*8=16 tokens)
        k_pre = torch.randn(batch, 64, heads, dim, dtype=torch.bfloat16).transpose(1, 2)
        v_pre = torch.randn(batch, 64, heads, dim, dtype=torch.bfloat16).transpose(1, 2)
        out_k, out_v = layer.update(k_pre, v_pre)
        self.assertEqual(out_k.shape, (batch, heads, 64, dim))
        self.assertEqual(out_v.shape, (batch, heads, 64, dim))
        self.assertEqual(layer.get_seq_length(), 64)

        # Decode 1 token -> triggers sparse landmark retrieval (< 64 tokens returned)
        k_dec = torch.randn(batch, heads, 1, dim, dtype=torch.bfloat16)
        v_dec = torch.randn(batch, heads, 1, dim, dtype=torch.bfloat16)
        out_k2, out_v2 = layer.update(k_dec, v_dec)
        self.assertEqual(layer.get_seq_length(), 65)
        self.assertLess(out_k2.shape[-2], 65)
        self.assertEqual(out_k2.shape[-2], out_v2.shape[-2])
        # Chunk 0 (sink) must remain in active_positions
        self.assertIn(0, layer.original_positions)
        self.assertGreater(layer.total_bytes_transferred_pcie, 0)

    def test_shadowkv_cache_factory_and_pytorch_runner(self):
        technique = get_technique("shadowkv", rank=8, chunk_size=8, sparse_budget_chunks=2)
        cache_factory = technique.pytorch_cache_factory()
        self.assertIsNotNone(cache_factory)

        cache = cache_factory()
        self.assertIsInstance(cache, ShadowKVCache)
        self.assertEqual(get_active_shadowkv_cache(), cache)

        k1 = torch.randn(1, 4, 32, 64, dtype=torch.float32)
        v1 = torch.randn(1, 4, 32, 64, dtype=torch.float32)
        cache.update(k1, v1, layer_idx=0)
        cache.update(torch.randn(1, 4, 1, 64), torch.randn(1, 4, 1, 64), layer_idx=0)

        diag = inspect_sample_diagnostic(
            cache=cache,
            prompt_text="Test ShadowKV diagnostic",
            prompt_token_ids=list(range(32)),
            sample_meta={"id": "skv_diag_0", "task_type": "unit"},
            is_correct=True,
            baseline_correct=True,
            bits_per_scalar=16.0,
        )
        self.assertGreater(diag.overall_retention_ratio, 0.0)
        self.assertLess(diag.overall_retention_ratio, 1.0)

        runner = PyTorchRunner(
            model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
            revision="cdbee75f17c01a7cc42f958dc650907174af0554",
            cache_factory=cache_factory,
            dry_run=True,
        )
        outputs = runner.generate(["Test ShadowKV prompt."], max_tokens=16)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(outputs[0].outputs[0].token_ids), 16)


if __name__ == "__main__":
    unittest.main()

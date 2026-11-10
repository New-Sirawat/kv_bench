import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.core.axis import Axis
from src.core.pytorch_runner import PyTorchRunner
from src.core.registry import get_technique, list_techniques
from src.techniques.infinigen_cache import (
    InfiniGenCache,
    InfiniGenLayer,
    compute_speculative_block_mask,
    generate_orthogonal_matrix,
    get_active_infinigen_cache,
)
from src.techniques.infinigen import (
    _ACTIVE_INFINIGEN_CONFIG,
    compute_infinigen_boundaries,
    compute_infinigen_paged_block_boundaries,
    estimate_infinigen_pcie_savings,
)


class TestInfiniGenTechnique(unittest.TestCase):
    def test_registration_and_axes(self):
        self.assertIn("infinigen", list_techniques())
        technique = get_technique("infinigen")
        self.assertEqual(technique.axes, Axis.SPATIAL | Axis.TEMPORAL | Axis.STRUCTURAL)

    def test_defaults_and_custom_overrides(self):
        technique = get_technique("infinigen")
        self.assertEqual(technique.k_proxy_ratio, 0.30)
        self.assertEqual(technique.alpha_threshold, 4.0)
        self.assertEqual(technique.block_size, 16)
        self.assertEqual(technique.max_budget_ratio, 0.35)
        self.assertEqual(technique.swap_space_gb, 4)
        self.assertEqual(technique.kv_cache_dtype, "auto")

        custom = get_technique(
            "infinigen",
            k_proxy_ratio=0.25,
            alpha_threshold=5.0,
            block_size=32,
            max_budget_ratio=0.20,
            swap_space_gb=8,
            kv_cache_dtype="fp8",
        )
        self.assertEqual(custom.k_proxy_ratio, 0.25)
        self.assertEqual(custom.alpha_threshold, 5.0)
        self.assertEqual(custom.block_size, 32)
        self.assertEqual(custom.max_budget_ratio, 0.20)
        self.assertEqual(custom.swap_space_gb, 8)
        self.assertEqual(custom.kv_cache_dtype, "fp8")

        kwargs = custom.engine_kwargs(context_length=2048, max_tokens=64)
        self.assertEqual(kwargs["swap_space"], 8)
        self.assertEqual(kwargs["kv_cache_dtype"], "fp8")
        self.assertEqual(kwargs["block_size"], 32)

        mock_out = SimpleNamespace(prompt_tokens=2048, generated_tokens=64)
        metrics = custom.extra_metrics(None, mock_out)
        self.assertEqual(metrics["k_proxy_ratio"], 0.25)
        self.assertEqual(metrics["alpha_threshold"], 5.0)
        self.assertIn("pcie_transfer_bytes", metrics)
        self.assertIn("transfer_latency_saved_ms", metrics)
        self.assertGreater(metrics["pcie_bandwidth_reduction_factor"], 1.0)

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            get_technique("infinigen", k_proxy_ratio=0.0)
        with self.assertRaises(ValueError):
            get_technique("infinigen", k_proxy_ratio=1.5)
        with self.assertRaises(ValueError):
            get_technique("infinigen", alpha_threshold=0.0)
        with self.assertRaises(ValueError):
            get_technique("infinigen", block_size=0)
        with self.assertRaises(ValueError):
            get_technique("infinigen", max_budget_ratio=0.0)
        with self.assertRaises(ValueError):
            get_technique("infinigen", swap_space_gb=-1)

    def test_orthogonal_rotation_mathematical_invariance(self):
        torch.manual_seed(42)
        D = 128
        A = generate_orthogonal_matrix(D)

        # Check orthogonality: A @ A.T = I
        ident = torch.matmul(A, A.T)
        self.assertTrue(torch.allclose(ident, torch.eye(D), atol=1e-5))

        q = torch.randn(1, 4, 1, D)
        k = torch.randn(1, 4, 32, D)

        scores_orig = torch.matmul(q, k.transpose(-2, -1))
        q_rot = torch.matmul(q, A)
        k_rot = torch.matmul(k, A)
        scores_rot = torch.matmul(q_rot, k_rot.transpose(-2, -1))

        # Dot product is mathematically preserved
        max_diff = (scores_orig - scores_rot).abs().max().item()
        self.assertLess(max_diff, 1e-4)

    def test_speculative_scoring_and_softmax_mass_retention(self):
        torch.manual_seed(42)
        B, N_q, N_kv, S, D_proxy = 1, 8, 2, 64, 38
        q_proxy = torch.randn(B, N_q, 1, D_proxy)
        k_proxy = torch.randn(B, N_kv, S, D_proxy)

        token_mask, prefetched_blocks, ratio = compute_speculative_block_mask(
            query=q_proxy,
            proxy_keys=k_proxy,
            alpha=4.0,
            block_size=16,
            max_budget_ratio=0.35,
        )

        self.assertIsInstance(token_mask, torch.Tensor)
        self.assertEqual(token_mask.shape, (B, N_kv, S))
        self.assertIsInstance(prefetched_blocks, list)
        self.assertGreater(len(prefetched_blocks), 0)
        self.assertLessEqual(len(prefetched_blocks), 2)
        self.assertLessEqual(ratio, 0.50)

    def test_boundary_calculations(self):
        # 1. Short sequence fits into 2 base blocks (sink + recent)
        b1 = compute_infinigen_boundaries(seq_len=32, block_size=16)
        self.assertEqual(b1.total_blocks, 2)
        self.assertEqual(b1.estimated_prefetched_blocks, 2)
        self.assertEqual(b1.estimated_prefetch_ratio, 1.0)

        # 2. Longer sequence (2048 tokens -> 128 blocks)
        b2 = compute_infinigen_boundaries(seq_len=2048, block_size=16, max_budget_ratio=0.35)
        self.assertEqual(b2.total_blocks, 128)
        self.assertLess(b2.estimated_prefetched_blocks, 128)
        self.assertLessEqual(b2.estimated_prefetch_ratio, 0.35)
        self.assertGreater(b2.estimated_omitted_tokens, 0)

    def test_paged_block_boundaries(self):
        pb = compute_infinigen_paged_block_boundaries(num_computed_tokens=160, block_size=16)
        self.assertEqual(pb.total_blocks, 10)
        self.assertEqual(pb.sink_blocks, [0])
        self.assertEqual(pb.recent_blocks, [9])
        self.assertEqual(pb.speculative_candidate_blocks, list(range(1, 9)))

    def test_pcie_savings_model(self):
        savings = estimate_infinigen_pcie_savings(
            seq_len=4096,
            prefetch_ratio=0.10,
            num_kv_heads=4,
            head_dim=128,
            pcie_bw_gb_s=25.0,
        )
        self.assertEqual(savings["full_transfer_bytes"], 2 * 4096 * 4 * 128 * 2)
        self.assertAlmostEqual(savings["infinigen_transfer_bytes"], savings["full_transfer_bytes"] * 0.10)
        self.assertGreater(savings["transfer_latency_saved_ms"], 0.0)
        self.assertAlmostEqual(savings["pcie_bandwidth_reduction_factor"], 10.0, places=1)

    def test_infinigen_layer_prefill_and_decode(self):
        layer = InfiniGenLayer(k_proxy_ratio=0.30, alpha_threshold=4.0, block_size=16)

        # Prefill 32 tokens
        k_init = torch.randn(1, 2, 32, 128, dtype=torch.bfloat16)
        v_init = torch.randn(1, 2, 32, 128, dtype=torch.bfloat16)
        out_k, out_v = layer.update(k_init, v_init)

        self.assertEqual(out_k.shape, (1, 2, 32, 128))
        self.assertEqual(layer.get_seq_length(), 32)
        self.assertIsNotNone(layer.cpu_keys)
        self.assertIsNotNone(layer.proxy_keys)
        self.assertEqual(layer.proxy_keys.shape[-1], int(128 * 0.30))

        # Autoregressive Decode step 1 (1 token)
        k_dec = torch.randn(1, 2, 1, 128, dtype=torch.bfloat16)
        v_dec = torch.randn(1, 2, 1, 128, dtype=torch.bfloat16)
        out_k_dec, out_v_dec = layer.update(k_dec, v_dec)

        self.assertEqual(layer.get_seq_length(), 33)
        self.assertGreater(layer.total_prefetched_blocks, 0)
        self.assertGreater(layer.total_bytes_transferred_pcie, 0)
        self.assertLessEqual(out_k_dec.shape[-2], 33)

    def test_infinigen_cache_lifecycle(self):
        cache = InfiniGenCache(k_proxy_ratio=0.30, alpha_threshold=4.0, block_size=16)
        self.assertEqual(get_active_infinigen_cache(), cache)

        k1 = torch.randn(1, 2, 32, 128, dtype=torch.bfloat16)
        v1 = torch.randn(1, 2, 32, 128, dtype=torch.bfloat16)
        cache.update(k1, v1, layer_idx=0)

        k2 = torch.randn(1, 2, 1, 128, dtype=torch.bfloat16)
        v2 = torch.randn(1, 2, 1, 128, dtype=torch.bfloat16)
        cache.update(k2, v2, layer_idx=0)

        self.assertGreater(cache.total_bytes_transferred_pcie, 0)
        self.assertGreater(cache.total_prefetched_blocks, 0)
        self.assertEqual(cache.get_seq_length(0), 33)

    def test_pytorch_runner_dry_run_with_infinigen(self):
        technique = get_technique("infinigen", k_proxy_ratio=0.30)
        cache_factory = technique.pytorch_cache_factory()

        runner = PyTorchRunner(
            model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
            revision="cdbee75f17c01a7cc42f958dc650907174af0554",
            cache_factory=cache_factory,
            dry_run=True,
        )
        prompts = ["Test prompt for InfiniGen speculative prefetching."]
        outputs = runner.generate(prompts, max_tokens=16)

        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(outputs[0].outputs[0].token_ids), 16)


if __name__ == "__main__":
    unittest.main()

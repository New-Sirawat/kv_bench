import unittest
from types import SimpleNamespace

import torch

from src.core.axis import Axis
from src.core.pytorch_runner import PyTorchRunner
from src.core.registry import get_technique, list_techniques
from src.techniques.flexgen_cache import (
    FlexGenKVCache,
    FlexGenKVLayer,
    dequantize_4bit_group,
    get_active_flexgen_cache,
    quantize_4bit_group,
)
from src.techniques.flexgen_kv import (
    _ACTIVE_FLEXGEN_CONFIG,
    compute_flexgen_boundaries,
    compute_flexgen_paged_block_boundaries,
    estimate_crossover_latency,
)


class TestFlexGenKVTechnique(unittest.TestCase):
    def test_registration_and_axes(self):
        self.assertIn("flexgen_kv", list_techniques())
        technique = get_technique("flexgen_kv")
        self.assertEqual(technique.axes, Axis.SPATIAL | Axis.STRUCTURAL)

    def test_defaults_and_custom_overrides(self):
        technique = get_technique("flexgen_kv")
        self.assertEqual(technique.gpu_cache_tokens, 1024)
        self.assertEqual(technique.cpu_cache_tokens, 32768)
        self.assertEqual(technique.block_size, 16)
        self.assertTrue(technique.quantize_kv)
        self.assertEqual(technique.group_size, 64)
        self.assertFalse(technique.cpu_delegation)
        self.assertEqual(technique.swap_space_gb, 4)
        self.assertEqual(technique.kv_cache_dtype, "fp8")

        custom = get_technique(
            "flexgen_kv",
            gpu_cache_tokens=512,
            cpu_cache_tokens=16384,
            block_size=32,
            quantize_kv=False,
            group_size=32,
            cpu_delegation=True,
            swap_space_gb=8,
            kv_cache_dtype="auto",
        )
        self.assertEqual(custom.gpu_cache_tokens, 512)
        self.assertEqual(custom.cpu_cache_tokens, 16384)
        self.assertEqual(custom.block_size, 32)
        self.assertFalse(custom.quantize_kv)
        self.assertEqual(custom.group_size, 32)
        self.assertTrue(custom.cpu_delegation)
        self.assertEqual(custom.swap_space_gb, 8)
        self.assertEqual(custom.kv_cache_dtype, "auto")

        kwargs = custom.engine_kwargs(context_length=2048, max_tokens=64)
        self.assertEqual(kwargs["swap_space"], 8)
        self.assertEqual(kwargs["kv_cache_dtype"], "auto")
        self.assertEqual(kwargs["block_size"], 32)

        mock_out = SimpleNamespace(prompt_tokens=4096, generated_tokens=64)
        metrics = custom.extra_metrics(None, mock_out)
        self.assertEqual(metrics["gpu_cache_tokens"], 512)
        self.assertEqual(metrics["cpu_cache_tokens"], 16384)
        self.assertEqual(metrics["active_gpu_tokens"], 512)
        self.assertEqual(metrics["offloaded_cpu_tokens"], 4160 - 512)
        self.assertEqual(metrics["compression_ratio"], 1.0)
        self.assertIn("crossover_t_pcie_ms", metrics)
        self.assertIn("crossover_t_cpu_ms", metrics)

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            get_technique("flexgen_kv", gpu_cache_tokens=-1)
        with self.assertRaises(ValueError):
            get_technique("flexgen_kv", cpu_cache_tokens=0)
        with self.assertRaises(ValueError):
            get_technique("flexgen_kv", block_size=0)
        with self.assertRaises(ValueError):
            get_technique("flexgen_kv", group_size=30)  # Not a power of 2
        with self.assertRaises(ValueError):
            get_technique("flexgen_kv", swap_space_gb=-1)

    def test_boundary_calculations(self):
        # 1. Sequence fits within GPU capacity (512 <= 1024)
        b1 = compute_flexgen_boundaries(seq_len=512, gpu_cache_tokens=1024, cpu_cache_tokens=4096)
        self.assertEqual(b1.active_gpu_tokens, 512)
        self.assertEqual(b1.offloaded_cpu_tokens, 0)
        self.assertEqual(b1.evicted_tokens, 0)
        self.assertEqual(b1.total_retained, 512)
        self.assertFalse(b1.is_offloaded)
        self.assertEqual(b1.gpu_range, (0, 512))

        # 2. Sequence exceeds GPU capacity, fits within combined GPU+CPU (2048 > 1024)
        b2 = compute_flexgen_boundaries(seq_len=2048, gpu_cache_tokens=1024, cpu_cache_tokens=4096)
        self.assertEqual(b2.active_gpu_tokens, 1024)
        self.assertEqual(b2.offloaded_cpu_tokens, 1024)
        self.assertEqual(b2.evicted_tokens, 0)
        self.assertEqual(b2.total_retained, 2048)
        self.assertTrue(b2.is_offloaded)
        self.assertEqual(b2.gpu_range, (1024, 2048))
        self.assertEqual(b2.cpu_range, (0, 1024))

        # 3. Sequence exceeds total capacity (6000 > 1024 + 4096 = 5120)
        b3 = compute_flexgen_boundaries(seq_len=6000, gpu_cache_tokens=1024, cpu_cache_tokens=4096)
        self.assertEqual(b3.active_gpu_tokens, 1024)
        self.assertEqual(b3.offloaded_cpu_tokens, 4096)
        self.assertEqual(b3.evicted_tokens, 880)
        self.assertEqual(b3.total_retained, 5120)
        self.assertTrue(b3.is_offloaded)
        self.assertEqual(b3.gpu_range, (4976, 6000))
        self.assertEqual(b3.cpu_range, (880, 4976))

    def test_paged_block_boundaries(self):
        # 160 tokens with block_size=16 -> 10 total blocks
        # 4 GPU blocks, 4 CPU blocks -> 2 evicted blocks
        pb = compute_flexgen_paged_block_boundaries(
            num_computed_tokens=160,
            gpu_blocks=4,
            cpu_blocks=4,
            block_size=16,
        )
        self.assertEqual(pb.total_blocks, 10)
        self.assertEqual(pb.gpu_blocks, [6, 7, 8, 9])
        self.assertEqual(pb.cpu_blocks, [2, 3, 4, 5])
        self.assertEqual(pb.evicted_blocks, [0, 1])

    def test_crossover_latency_model(self):
        # Normal sequence with DDR5 (50 GB/s) vs PCIe Gen4 (25 GB/s)
        res = estimate_crossover_latency(
            seq_len=4096,
            num_kv_heads=4,
            head_dim=128,
            pcie_bandwidth_gb_s=25.0,
            cpu_dram_bandwidth_gb_s=50.0,
        )
        self.assertGreater(res["t_pcie_ms"], res["t_cpu_ms"])
        self.assertAlmostEqual(res["cpu_speedup_factor"], 2.0, places=1)
        self.assertEqual(res["delegation_preferred"], 1.0)

    def test_quantization_and_dequantization_invariants(self):
        torch.manual_seed(42)
        x = torch.randn(2, 4, 32, 128, dtype=torch.bfloat16)
        packed, scales, mins = quantize_4bit_group(x, group_size=64)

        # Packed uint8 has shape [2, 4, 32, 2, 32]
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.shape, (2, 4, 32, 2, 32))
        self.assertEqual(scales.shape, (2, 4, 32, 2, 1))
        self.assertEqual(mins.shape, (2, 4, 32, 2, 1))

        recon = dequantize_4bit_group(packed, scales, mins, orig_dtype=torch.bfloat16, group_size=64)
        self.assertEqual(recon.shape, x.shape)
        self.assertEqual(recon.dtype, torch.bfloat16)

        # Numerical SNR check (> 18 dB signal-to-noise ratio)
        diff = x.float() - recon.float()
        snr = 10 * torch.log10((x.float() ** 2).mean() / (diff ** 2).mean()).item()
        self.assertGreater(snr, 18.0)
        self.assertLess(diff.abs().max().item(), 0.5)

    def test_flexgen_layer_within_capacity(self):
        layer = FlexGenKVLayer(gpu_cache_tokens=16, quantize_kv=True, group_size=64)
        k = torch.randn(1, 2, 10, 128, dtype=torch.bfloat16)
        v = torch.randn(1, 2, 10, 128, dtype=torch.bfloat16)

        out_k, out_v = layer.update(k, v)
        self.assertEqual(out_k.shape, (1, 2, 10, 128))
        self.assertEqual(out_v.shape, (1, 2, 10, 128))
        self.assertEqual(layer.get_seq_length(), 10)
        self.assertEqual(layer.gpu_keys.shape[-2], 10)
        self.assertIsNone(layer.cpu_keys)
        self.assertEqual(layer.total_bytes_transferred_pcie, 0)

    def test_flexgen_layer_offloading_and_reconstruction(self):
        layer = FlexGenKVLayer(gpu_cache_tokens=8, quantize_kv=True, group_size=64)

        # Step 1: 6 tokens
        k1 = torch.randn(1, 2, 6, 128, dtype=torch.bfloat16)
        v1 = torch.randn(1, 2, 6, 128, dtype=torch.bfloat16)
        layer.update(k1, v1)
        self.assertEqual(layer.get_seq_length(), 6)
        self.assertEqual(layer.gpu_keys.shape[-2], 6)

        # Step 2: 6 more tokens -> total 12 tokens (> 8 capacity)
        # 4 tokens must be offloaded to CPU tier
        k2 = torch.randn(1, 2, 6, 128, dtype=torch.bfloat16)
        v2 = torch.randn(1, 2, 6, 128, dtype=torch.bfloat16)
        out_k, out_v = layer.update(k2, v2)

        self.assertEqual(layer.get_seq_length(), 12)
        self.assertEqual(layer.gpu_keys.shape[-2], 8)
        self.assertIsNotNone(layer.cpu_keys)
        self.assertEqual(layer.cpu_keys[0].shape[-3], 4)  # 4 offloaded tokens
        self.assertEqual(out_k.shape, (1, 2, 12, 128))
        self.assertEqual(out_v.shape, (1, 2, 12, 128))
        self.assertGreater(layer.total_bytes_transferred_pcie, 0)

    def test_flexgen_cache_lifecycle(self):
        cache = FlexGenKVCache(gpu_cache_tokens=8, quantize_kv=True, group_size=64)
        self.assertEqual(get_active_flexgen_cache(), cache)

        k1 = torch.randn(1, 2, 6, 128, dtype=torch.bfloat16)
        v1 = torch.randn(1, 2, 6, 128, dtype=torch.bfloat16)
        cache.update(k1, v1, layer_idx=0)

        k2 = torch.randn(1, 2, 6, 128, dtype=torch.bfloat16)
        v2 = torch.randn(1, 2, 6, 128, dtype=torch.bfloat16)
        cache.update(k2, v2, layer_idx=0)

        self.assertEqual(cache.active_gpu_tokens, 8)
        self.assertEqual(cache.offloaded_cpu_tokens, 4)
        self.assertGreater(cache.total_bytes_transferred_pcie, 0)
        self.assertEqual(cache.get_seq_length(0), 12)

    def test_pytorch_runner_dry_run_with_flexgen_cache(self):
        technique = get_technique("flexgen_kv", gpu_cache_tokens=64)
        cache_factory = technique.pytorch_cache_factory()

        runner = PyTorchRunner(
            model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
            revision="cdbee75f17c01a7cc42f958dc650907174af0554",
            cache_factory=cache_factory,
            dry_run=True,
        )
        prompts = ["Test prompt for FlexGen KV offload."]
        outputs = runner.generate(prompts, max_tokens=16)

        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(outputs[0].outputs[0].token_ids), 16)


if __name__ == "__main__":
    unittest.main()

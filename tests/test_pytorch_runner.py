import unittest

from src.core.pytorch_runner import (
    PyTorchCompletionOutput,
    PyTorchRequestOutput,
    PyTorchRunner,
)
from src.core.quality import extract_step_logprobs
from src.core.workload import SyntheticFiller, extract_request_timings


class TestPyTorchRunner(unittest.TestCase):
    def setUp(self):
        self.runner = PyTorchRunner(
            model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
            revision="cdbee75f17c01a7cc42f958dc650907174af0554",
            dry_run=True,
        )

    def test_dry_run_generation_structure(self):
        prompts = ["Hello world, this is a test prompt."]
        outputs = self.runner.generate(prompts, max_tokens=16)

        self.assertEqual(len(outputs), 1)
        req_out = outputs[0]
        self.assertIsInstance(req_out, PyTorchRequestOutput)
        self.assertEqual(len(req_out.outputs), 1)

        comp = req_out.outputs[0]
        self.assertIsInstance(comp, PyTorchCompletionOutput)
        self.assertEqual(len(comp.token_ids), 16)
        self.assertIsNotNone(comp.logprobs)
        self.assertEqual(len(comp.logprobs), 16)

    def test_timing_extraction_compatibility(self):
        prompts = ["Timing test prompt."]
        outputs = self.runner.generate(prompts, max_tokens=32)
        req_out = outputs[0]

        ttft, decode, total = extract_request_timings(req_out, fallback_wall_sec=2.0, gen_tokens=32)
        self.assertGreater(ttft, 0.0)
        self.assertGreater(decode, 0.0)
        self.assertGreaterEqual(total, ttft + decode - 0.01)

    def test_logprob_extraction_compatibility(self):
        prompts = ["Logprob extraction test prompt."]
        outputs = self.runner.generate(prompts, max_tokens=10)
        req_out = outputs[0]

        step_lps = extract_step_logprobs(req_out)
        self.assertEqual(len(step_lps), 10)
        for step in step_lps:
            self.assertIsInstance(step, dict)
            self.assertTrue(len(step) > 0)

    def test_synthetic_filler_workload_compatibility(self):
        workload = SyntheticFiller()
        output = workload.run(self.runner, context_length=128, max_tokens=16)

        self.assertEqual(output.generated_tokens, 16)
        self.assertGreater(output.ttft_sec, 0.0)
        self.assertGreater(output.tpot_mean_ms, 0.0)
        self.assertEqual(len(output.generated_token_ids), 16)
        self.assertEqual(len(output.step_logprobs), 16)

    def test_multi_prompt_generation(self):
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        outputs = self.runner.generate(prompts, max_tokens=8)

        self.assertEqual(len(outputs), 3)
        for idx, out in enumerate(outputs):
            self.assertEqual(out.prompt, prompts[idx])
            self.assertEqual(len(out.outputs[0].token_ids), 8)


if __name__ == "__main__":
    unittest.main()

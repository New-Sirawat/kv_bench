"""Unit tests for diagnostic KV-cache retention and Schema v2 profiler (WP1-4-1)."""

import unittest
from unittest.mock import MagicMock

from src.core.diagnostic import (
    DiagnosticResult,
    classify_token_is_content,
    extract_active_positions_per_layer,
    inspect_sample_diagnostic,
    parse_session_token_ranges,
    summarize_diagnostic_results,
)
from src.workloads.niah import NIAHWorkload


class DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Deterministic token mapping based on whitespace and words
        words = text.split()
        return [abs(hash(w)) % 10000 + 1 for w in words]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"tok_{t}" for t in token_ids)


class TestDiagnosticModule(unittest.TestCase):
    def test_classify_token_is_content(self):
        # Delimiters and syntax
        self.assertFalse(classify_token_is_content(","))
        self.assertFalse(classify_token_is_content("."))
        self.assertFalse(classify_token_is_content("\n"))
        self.assertFalse(classify_token_is_content("  "))
        self.assertFalse(classify_token_is_content("User:"))
        self.assertFalse(classify_token_is_content("Assistant:"))
        self.assertFalse(classify_token_is_content("[Session Break]"))

        # Semantic content words
        self.assertTrue(classify_token_is_content("Marcus"))
        self.assertTrue(classify_token_is_content("Vance"))
        self.assertTrue(classify_token_is_content("12"))
        self.assertTrue(classify_token_is_content("GB"))
        self.assertTrue(classify_token_is_content("Alpha"))

    def test_parse_session_token_ranges_exact_delimiter(self):
        tok = DummyTokenizer()
        delim_str = "[Session Break]"
        delim_ids = tok.encode(delim_str, add_special_tokens=False)

        # Build 3 sessions separated by exact delimiter token sequences
        s0_ids = [101, 102, 103, 104]
        s1_ids = [201, 202, 203]
        s2_ids = [301, 302, 303, 304, 305]

        prompt_token_ids = s0_ids + delim_ids + s1_ids + delim_ids + s2_ids
        prompt_text = "Turn 0 [Session Break] Turn 1 [Session Break] Turn 2"

        ranges = parse_session_token_ranges(prompt_text, prompt_token_ids, tok)

        self.assertEqual(len(ranges), 3)
        # Session 0: length 4 (0..4)
        self.assertEqual(ranges[0], (0, len(s0_ids)))
        # Session 1: starts after first delimiter
        s1_start = len(s0_ids) + len(delim_ids)
        s1_end = s1_start + len(s1_ids)
        self.assertEqual(ranges[1], (s1_start, s1_end))
        # Session 2: ends at total_tokens
        s2_start = s1_end + len(delim_ids)
        self.assertEqual(ranges[2], (s2_start, len(prompt_token_ids)))

    def test_parse_session_token_ranges_overlapping_delimiters(self):
        class OverlappingTokenizer:
            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                if text == "[Session Break]":
                    return [10, 20]
                elif text == "\n\n[Session Break]\n\n":
                    return [9, 10, 20, 21]
                return []

            def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
                return "text"

        tok = OverlappingTokenizer()
        # [1, 2, 3] + [9, 10, 20, 21] + [4, 5, 6]
        # Contains both inner [10, 20] and outer [9, 10, 20, 21]
        prompt_tokens = [1, 2, 3, 9, 10, 20, 21, 4, 5, 6]
        ranges = parse_session_token_ranges("prompt", prompt_tokens, tok)
        self.assertEqual(len(ranges), 2)
        self.assertEqual(ranges[0], (0, 3))
        self.assertEqual(ranges[1], (7, 10))

    def test_parse_session_token_ranges_single_session_fallback(self):
        prompt_token_ids = list(range(50))
        tok = DummyTokenizer()
        ranges = parse_session_token_ranges("Single continuous prompt without breaks", prompt_token_ids, tok)
        self.assertEqual(len(ranges), 1)
        self.assertEqual(ranges[0], (0, 50))

    def test_extract_active_positions_per_layer(self):
        mock_cache = MagicMock()
        mock_layer1 = MagicMock()
        mock_layer1.original_positions = [0, 1, 2, 5, 6]
        mock_layer2 = MagicMock()
        mock_layer2.original_positions = [0, 1, 3, 7, 8]
        mock_cache.layers = [mock_layer1, mock_layer2]

        positions = extract_active_positions_per_layer(mock_cache, total_tokens=10)
        self.assertEqual(len(positions), 2)
        self.assertEqual(positions[0], {0, 1, 2, 5, 6})
        self.assertEqual(positions[1], {0, 1, 3, 7, 8})

    def test_baseline_controlled_causal_verdicts(self):
        tok = DummyTokenizer()
        mock_cache = MagicMock()
        mock_layer = MagicMock()
        mock_layer.original_positions = list(range(20))
        mock_cache.layers = [mock_layer]

        # 1. Correct output -> PASS_FIDELITY_MAINTAINED
        res_pass = inspect_sample_diagnostic(
            cache=mock_cache,
            prompt_text="User: Hello\nAssistant: Hi",
            prompt_token_ids=list(range(20)),
            sample_meta={"id": "s1", "task_type": "fact_recall", "expected_answer": "Hi"},
            is_correct=True,
            baseline_correct=True,
            tokenizer=tok,
        )
        self.assertEqual(res_pass.causal_verdict, "PASS_FIDELITY_MAINTAINED")
        self.assertIsNone(res_pass.most_degraded_session_idx)  # 100% retained

        # 2. Incorrect output, but baseline was correct -> QUALITY_REGRESSION
        mock_cache_evicted = MagicMock()
        mock_layer_evicted = MagicMock()
        # Keeps only 5 of 20 tokens (25% retention -> 4.0 bits/scalar)
        mock_layer_evicted.original_positions = list(range(15, 20))
        mock_cache_evicted.layers = [mock_layer_evicted]

        res_regress = inspect_sample_diagnostic(
            cache=mock_cache_evicted,
            prompt_text="User: Question\nAssistant: Wrong",
            prompt_token_ids=list(range(20)),
            sample_meta={"id": "s2", "task_type": "cross_session_reasoning", "expected_answer": "Right"},
            is_correct=False,
            baseline_correct=True,
            tokenizer=tok,
            bits_per_scalar=16.0,
        )
        self.assertEqual(res_regress.causal_verdict, "QUALITY_REGRESSION")
        self.assertEqual(res_regress.overall_retention_ratio, 0.25)
        self.assertEqual(res_regress.overall_effective_bitwidth, 4.0)

        # 3. Incorrect output, and baseline was also incorrect -> BASELINE_LIMITATION
        res_limit = inspect_sample_diagnostic(
            cache=mock_cache,
            prompt_text="User: Hard puzzle\nAssistant: Failed",
            prompt_token_ids=list(range(20)),
            sample_meta={"id": "s3", "task_type": "cross_session_reasoning", "expected_answer": "Correct"},
            is_correct=False,
            baseline_correct=False,
            tokenizer=tok,
        )
        self.assertEqual(res_limit.causal_verdict, "BASELINE_LIMITATION")

        # 4. Standalone run without baseline reference -> INCORRECT_STANDALONE
        res_standalone = inspect_sample_diagnostic(
            cache=mock_cache,
            prompt_text="User: Query\nAssistant: Ans",
            prompt_token_ids=list(range(20)),
            sample_meta={"id": "s4", "task_type": "fact_recall", "expected_answer": "Target"},
            is_correct=False,
            baseline_correct=None,
            tokenizer=tok,
        )
        self.assertEqual(res_standalone.causal_verdict, "INCORRECT_STANDALONE")

    def test_effective_bitwidth_and_multi_session_degradation(self):
        tok = DummyTokenizer()
        delim_str = "[Session Break]"
        delim_ids = tok.encode(delim_str, add_special_tokens=False)

        d_len = len(delim_ids)
        s0_len = 10
        s1_len = 10
        s2_len = 10

        s0_range = list(range(0, s0_len))
        s1_start = s0_len + d_len
        s1_range = list(range(s1_start, s1_start + s1_len))
        s2_start = s1_start + s1_len + d_len
        s2_range = list(range(s2_start, s2_start + s2_len))

        full_tokens = [1] * (s2_start + s2_len)
        # Put delimiters in place
        full_tokens[s0_len : s0_len + d_len] = delim_ids
        full_tokens[s1_start + s1_len : s2_start] = delim_ids

        # Simulate cache that wiped S0 (0 tokens kept), kept half of S1 (5 kept), and kept all of S2 (10 kept)
        mock_cache = MagicMock()
        mock_layer = MagicMock()
        # Active positions: none in S0, 5 in S1, 10 in S2
        active_positions = s1_range[:5] + s2_range
        mock_layer.original_positions = active_positions
        mock_cache.layers = [mock_layer]

        res = inspect_sample_diagnostic(
            cache=mock_cache,
            prompt_text="S0 [Session Break] S1 [Session Break] S2",
            prompt_token_ids=full_tokens,
            sample_meta={"id": "m1", "task_type": "cross_session_reasoning"},
            is_correct=False,
            baseline_correct=True,
            tokenizer=tok,
            bits_per_scalar=16.0,
        )

        self.assertEqual(len(res.session_retention_ratios), 3)
        self.assertEqual(res.session_retention_ratios[0], 0.0)
        self.assertEqual(res.session_retention_ratios[1], 0.5)
        self.assertEqual(res.session_retention_ratios[2], 1.0)

        # Effective bitwidths: S0=0.0, S1=8.0, S2=16.0
        self.assertEqual(res.session_effective_bitwidths, [0.0, 8.0, 16.0])
        self.assertEqual(res.min_session_effective_bitwidth, 0.0)
        self.assertEqual(res.most_degraded_session_idx, 0)
        self.assertEqual(res.causal_verdict, "QUALITY_REGRESSION")

    def test_summarize_diagnostic_results_schema_v2(self):
        diag1 = DiagnosticResult(
            sample_id="s1",
            task_type="fact_recall",
            is_correct=True,
            baseline_correct=True,
            causal_verdict="PASS_FIDELITY_MAINTAINED",
            session_token_counts=[50],
            session_retention_ratios=[1.0],
            session_effective_bitwidths=[16.0],
            overall_retention_ratio=1.0,
            overall_effective_bitwidth=16.0,
            min_session_effective_bitwidth=16.0,
            most_degraded_session_idx=None,
            session_content_retention=[1.0],
            session_syntax_retention=[1.0],
        )
        diag2 = DiagnosticResult(
            sample_id="s2",
            task_type="cross_session_reasoning",
            is_correct=False,
            baseline_correct=True,
            causal_verdict="QUALITY_REGRESSION",
            session_token_counts=[25, 25],
            session_retention_ratios=[0.1, 0.9],
            session_effective_bitwidths=[1.6, 14.4],
            overall_retention_ratio=0.5,
            overall_effective_bitwidth=8.0,
            min_session_effective_bitwidth=1.6,
            most_degraded_session_idx=0,
            session_content_retention=[0.1, 0.9],
            session_syntax_retention=[0.1, 0.9],
        )
        diag3 = DiagnosticResult(
            sample_id="s3",
            task_type="fact_recall",
            is_correct=False,
            baseline_correct=False,
            causal_verdict="BASELINE_LIMITATION",
            session_token_counts=[50],
            session_retention_ratios=[1.0],
            session_effective_bitwidths=[16.0],
            overall_retention_ratio=1.0,
            overall_effective_bitwidth=16.0,
            min_session_effective_bitwidth=16.0,
            most_degraded_session_idx=None,
            session_content_retention=[1.0],
            session_syntax_retention=[1.0],
        )

        summary = summarize_diagnostic_results([diag1, diag2, diag3])

        self.assertEqual(summary["diag_schema_version"], 2)
        self.assertEqual(summary["diag_total_inspected"], 3)
        self.assertEqual(summary["diag_pass_count"], 1)
        self.assertEqual(summary["diag_quality_regression_count"], 1)
        self.assertEqual(summary["diag_baseline_limitation_count"], 1)
        self.assertEqual(summary["diag_ancient_session_starvation_count"], 1)  # Legacy alias
        self.assertAlmostEqual(summary["diag_mean_overall_retention_ratio"], 0.8333, places=3)
        self.assertAlmostEqual(summary["diag_mean_overall_effective_bitwidth"], 13.33, places=2)
        self.assertAlmostEqual(summary["diag_mean_min_session_effective_bitwidth"], 11.20, places=2)


class TestNIAHWorkload(unittest.TestCase):
    def test_niah_init_and_attributes(self):
        workload = NIAHWorkload(passcode="998877", depths=[0.1, 0.5, 0.9])
        self.assertEqual(workload.passcode, "998877")
        self.assertEqual(workload.depths, [0.1, 0.5, 0.9])

    def test_niah_run_mock(self):
        workload = NIAHWorkload(passcode="849204", depths=[0.1, 0.9])

        mock_llm = MagicMock()
        mock_out1 = MagicMock()
        mock_out1.prompt_token_ids = [1] * 100
        mock_choice1 = MagicMock()
        mock_choice1.text = "The secret verification code is 849204."
        mock_choice1.token_ids = [10, 20, 30]
        mock_out1.outputs = [mock_choice1]

        mock_out2 = MagicMock()
        mock_out2.prompt_token_ids = [1] * 100
        mock_choice2 = MagicMock()
        mock_choice2.text = "I do not know the code."
        mock_choice2.token_ids = [10, 20]
        mock_out2.outputs = [mock_choice2]

        mock_llm.generate.return_value = [mock_out1, mock_out2]
        mock_llm.get_tokenizer.return_value = None

        result = workload.run(mock_llm, context_length=128, max_tokens=16)
        self.assertEqual(result.extra_summary["niah_retrieved_count"], 1)
        self.assertEqual(result.extra_summary["niah_total_depths"], 2)
        self.assertEqual(result.extra_summary["niah_accuracy"], 0.5)
        self.assertEqual(len(result.extra_summary["niah_grid"]), 2)

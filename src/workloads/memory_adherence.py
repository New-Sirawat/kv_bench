"""LongMemEval Workload (Section 3.2 and 3.3.2 in proposal.md).

Evaluates Long-Term Memory Adherence across 5 core conversational capabilities:
1. Fact Recall
2. Cross-Session Reasoning
3. Information Update
4. Relative Timeframes
5. Unanswerable Question Refusal
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from src.core.metrics import calc_percentile
from src.core.quality import (
    DEFAULT_REFUSAL_TRIGGERS,
    extract_step_logprobs,
    score_longmemeval_dataset,
    score_longmemeval_sample,
)
from src.core.workload import Workload, WorkloadOutput, extract_request_timings

BENCHMARK_DATASET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "longmemeval", "longmemeval_benchmark.json")
)
FALLBACK_DATASET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "longmemeval_subset.json")
)
DEFAULT_DATASET_PATH = (
    BENCHMARK_DATASET_PATH if os.path.exists(BENCHMARK_DATASET_PATH) else FALLBACK_DATASET_PATH
)

REFUSAL_TRIGGERS = DEFAULT_REFUSAL_TRIGGERS


def classify_length_tier(words: int) -> tuple[int, str]:
    """Classify context word count into approximate token count and length tier."""
    approx_tokens = int(words * 1.35)
    if approx_tokens <= 2500:
        return approx_tokens, "short"
    elif approx_tokens <= 8000:
        return approx_tokens, "medium"
    else:
        return approx_tokens, "long"


class LongMemEvalWorkload(Workload):
    """Workload that executes multi-turn conversations from the LongMemEval subset
    and evaluates category-level and overall memory adherence accuracy.
    """

    def __init__(
        self,
        dataset_path: str = DEFAULT_DATASET_PATH,
        samples_per_task: int | None = None,
        max_samples: int | None = None,
        seed: int | None = None,
        length_tier: str | None = None,
        select_by_length: bool = False,
    ):
        import random

        self.dataset_path = dataset_path
        self.samples_per_task = samples_per_task
        self.max_samples = max_samples
        self.seed = seed
        self.length_tier = length_tier if length_tier != "all" else None
        self.select_by_length = select_by_length or (length_tier == "auto")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            all_samples: list[dict[str, Any]] = json.load(f)

        # Ensure all samples have length metadata
        for s in all_samples:
            if "word_count" not in s or "approx_tokens" not in s or "length_tier" not in s:
                words = len(s.get("context", "").split())
                approx_tok, tier = classify_length_tier(words)
                s["word_count"] = words
                s["approx_tokens"] = approx_tok
                s["length_tier"] = tier

        self.all_samples = all_samples

        # Filter by static length_tier if explicitly specified and not auto
        if self.length_tier and self.length_tier not in ("auto", "all"):
            filtered_samples = [s for s in all_samples if s.get("length_tier") == self.length_tier]
        else:
            filtered_samples = all_samples

        effective_per_task = samples_per_task
        if effective_per_task is None and max_samples is not None and max_samples > 0:
            effective_per_task = max(1, max_samples // 5)

        if effective_per_task is not None and effective_per_task > 0:
            tasks: dict[str, list[dict[str, Any]]] = {}
            for s in filtered_samples:
                tasks.setdefault(s.get("task_type", "unknown"), []).append(s)

            rng = random.Random(seed) if seed is not None else random
            selected: list[dict[str, Any]] = []
            for task_type in sorted(tasks.keys()):
                category_samples = tasks[task_type]
                n = min(effective_per_task, len(category_samples))
                selected.extend(rng.sample(category_samples, n))
            self.samples = selected
        else:
            self.samples = filtered_samples

    def get_active_samples(self, context_length: int) -> list[dict[str, Any]]:
        """Resolve active samples for the current context length."""
        if not self.select_by_length and self.length_tier != "auto":
            return self.samples

        # Select samples that naturally fit the target context window without truncation
        import random
        rng = random.Random(self.seed) if self.seed is not None else random

        if context_length <= 1024:
            candidates = [s for s in self.all_samples if s.get("approx_tokens", 0) <= 1200]
        elif context_length <= 2048:
            candidates = [s for s in self.all_samples if s.get("approx_tokens", 0) <= 2500]
        elif context_length <= 4096:
            candidates = [s for s in self.all_samples if 2000 <= s.get("approx_tokens", 0) <= 6500]
        elif context_length <= 8192:
            candidates = [s for s in self.all_samples if 4000 <= s.get("approx_tokens", 0) <= 10000]
        else:  # 16384+
            candidates = [s for s in self.all_samples if s.get("approx_tokens", 0) >= 7000]

        if not candidates:
            candidates = self.all_samples

        effective_per_task = self.samples_per_task
        if effective_per_task is None and self.max_samples is not None and self.max_samples > 0:
            effective_per_task = max(1, self.max_samples // 5)

        if effective_per_task is not None and effective_per_task > 0:
            tasks: dict[str, list[dict[str, Any]]] = {}
            for s in candidates:
                tasks.setdefault(s.get("task_type", "unknown"), []).append(s)
            selected: list[dict[str, Any]] = []
            for task_type in sorted(tasks.keys()):
                cat_samples = tasks[task_type]
                n = min(effective_per_task, len(cat_samples))
                selected.extend(rng.sample(cat_samples, n))
            return selected
        return candidates

    def run(self, llm, context_length: int, max_tokens: int) -> WorkloadOutput:
        try:
            from vllm import SamplingParams
            sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=5)
        except ImportError:
            sampling_params = None

        from src.core.workload import _get_tokenizer, load_continuous_prefix

        active_samples = self.get_active_samples(context_length)
        target_words = max(30, int(context_length * 0.6))
        tok = _get_tokenizer(llm)

        prompts = []
        for s in active_samples:
            words = s.get("context", "").split()
            if len(words) > target_words:
                ctx_text = " ".join(words[-target_words:])
                prefix = ""
            else:
                ctx_text = s.get("context", "")
                remaining = target_words - len(words)
                # Pad with natural PG-19 background passage instead of synthetic repeated sentence
                needed_tokens = max(10, int(remaining * 1.45))
                prefix = (
                    load_continuous_prefix(needed_tokens, tokenizer=tok)
                    + "\n\n[Chat History Continued]\n"
                ) if remaining > 20 else ""

            prompts.append(f"{prefix}{ctx_text}\nUser: {s['question']}\nAssistant:")

        diagnostic_records = []
        def _post_hoc_diag_hook(cache: Any, req_idx: int, req_out: Any) -> dict[str, Any] | None:
            if req_idx < len(active_samples):
                sample = active_samples[req_idx]
                pred_text = req_out.outputs[0].text.strip()
                from src.core.quality import score_longmemeval_sample
                is_correct = score_longmemeval_sample(
                    pred_text,
                    sample.get("expected_answer", ""),
                    is_unanswerable=sample.get("unanswerable", False),
                    refusal_triggers=REFUSAL_TRIGGERS,
                )
                from src.core.diagnostic import inspect_sample_diagnostic
                diag = inspect_sample_diagnostic(
                    cache=cache,
                    prompt_text=req_out.prompt,
                    prompt_token_ids=req_out.prompt_token_ids,
                    sample_meta=sample,
                    is_correct=is_correct,
                    baseline_correct=sample.get("baseline_correct", None),
                    tokenizer=tok,
                )
                diagnostic_records.append(diag)
                return diag.to_dict()
            return None

        generate_kwargs: dict[str, Any] = {}
        import inspect
        if "post_hoc_hook" in inspect.signature(llm.generate).parameters:
            generate_kwargs["post_hoc_hook"] = _post_hoc_diag_hook

        start_wall = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, **generate_kwargs)
        end_wall = time.perf_counter()

        predictions: list[str] = []
        total_prompt_tok = 0
        total_gen_tok = 0
        ttfts: list[float] = []
        decodes: list[float] = []
        tpots: list[float] = []
        all_token_ids: list[int] = []
        all_step_logprobs: list[dict[int, float]] = []

        total_wall = end_wall - start_wall

        for out in outputs:
            gen_text = out.outputs[0].text.strip()
            predictions.append(gen_text)
            total_prompt_tok += len(out.prompt_token_ids)
            gen_len = len(out.outputs[0].token_ids)
            total_gen_tok += gen_len

            all_token_ids.extend(out.outputs[0].token_ids)
            all_step_logprobs.extend(extract_step_logprobs(out))

            t, d, _ = extract_request_timings(out, total_wall, gen_len)
            ttfts.append(round(t, 4))
            decodes.append(round(d, 4))
            decode_tok = gen_len - 1 if gen_len > 1 else gen_len
            if d > 0 and decode_tok > 0:
                tpots.append(round(d / decode_tok * 1000.0, 2))

        # Evaluate quality metrics via unified quality scorer
        eval_summary = score_longmemeval_dataset(active_samples, predictions, refusal_triggers=REFUSAL_TRIGGERS)
        if self.samples_per_task is not None:
            eval_summary["longmem_samples_per_task"] = self.samples_per_task
        if self.seed is not None:
            eval_summary["longmem_seed"] = self.seed
        eval_summary["longmem_length_tier"] = self.length_tier or "all"
        eval_summary["longmem_select_by_length"] = self.select_by_length
        eval_summary["longmem_active_samples_count"] = len(active_samples)

        # Calculate per-length-tier breakdown if multiple tiers are present
        tier_scores: dict[str, list[bool]] = {}
        for sample, pred in zip(active_samples, predictions):
            tier = sample.get("length_tier", "unknown")
            is_correct = score_longmemeval_sample(
                pred,
                sample.get("expected_answer", ""),
                is_unanswerable=sample.get("unanswerable", False),
                refusal_triggers=REFUSAL_TRIGGERS,
            )
            tier_scores.setdefault(tier, []).append(is_correct)
        eval_summary["tier_accuracy"] = {
            t: round(sum(scores) / len(scores), 4) for t, scores in sorted(tier_scores.items()) if scores
        }

        if diagnostic_records:
            from src.core.diagnostic import summarize_diagnostic_results
            diag_summary = summarize_diagnostic_results(diagnostic_records)
            eval_summary.update(diag_summary)

        total_samples = len(active_samples)
        mean_ttft = sum(ttfts) / len(ttfts) if ttfts else (total_wall / total_samples if total_samples > 0 else 0.0)
        p95_ttft = calc_percentile(ttfts, 95.0) if ttfts else mean_ttft
        total_decode = max(decodes) if decodes else max(0.0, total_wall - mean_ttft)

        decode_tokens = total_gen_tok - total_samples if total_gen_tok > total_samples else total_gen_tok
        fallback_tpot = (total_decode / decode_tokens * 1000.0) if (decode_tokens > 0 and total_decode > 0) else 0.0
        mean_tpot = sum(tpots) / len(tpots) if tpots else fallback_tpot
        p95_tpot = calc_percentile(tpots, 95.0) if tpots else round(mean_tpot, 2)

        out = WorkloadOutput(
            prompt_tokens=total_prompt_tok,
            generated_tokens=total_gen_tok,
            ttft_sec=round(mean_ttft, 4),
            decode_time_sec=round(total_decode, 4),
            total_time_sec=round(total_wall, 4),
            tpot_mean_ms=round(mean_tpot, 2),
            tpot_p95_ms=round(p95_tpot, 2),
            ttft_p95_sec=round(p95_ttft, 4),
            inter_token_latencies_ms=tpots,
            request_ttfts_sec=ttfts,
            generated_token_ids=all_token_ids,
            step_logprobs=all_step_logprobs,
            extra_summary=eval_summary,
        )
        return out


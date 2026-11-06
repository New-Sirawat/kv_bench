"""Mechanistic Needle-In-A-Haystack (NIAH) Workload (WP1-4-1).

Surgically evaluates token retrieval accuracy across context lengths and needle depths
by burying a deterministic synthetic passcode in PG-19 natural language text.
"""

from __future__ import annotations

import time
from typing import Any

from src.core.metrics import calc_percentile
from src.core.workload import Workload, WorkloadOutput, _get_tokenizer, extract_request_timings, load_continuous_prefix

DEFAULT_DEPTHS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_PASSCODE = "849204"


class NIAHWorkload(Workload):
    """Workload executing Needle-In-A-Haystack 2D evaluation (Length x Depth)."""

    def __init__(
        self,
        passcode: str = DEFAULT_PASSCODE,
        depths: list[float] | None = None,
    ):
        self.passcode = str(passcode)
        self.depths = depths if depths is not None else list(DEFAULT_DEPTHS)

    def run(self, llm: Any, context_length: int, max_tokens: int = 16) -> WorkloadOutput:
        tok = _get_tokenizer(llm)

        needle_text = f"\n[Special Notice: The secret verification code for project Alpha is {self.passcode}.]\n"
        query_text = "\nWhat is the secret verification code for project Alpha? The secret verification code is"

        needle_toks = tok.encode(needle_text, add_special_tokens=False) if tok else [1] * 20
        query_toks = tok.encode(query_text, add_special_tokens=False) if tok else [1] * 15

        overhead = len(needle_toks) + len(query_toks)
        haystack_tokens_needed = max(10, context_length - overhead)

        # Load natural background text from PG-19
        haystack_text = load_continuous_prefix(haystack_tokens_needed, tokenizer=tok)
        haystack_toks = tok.encode(haystack_text, add_special_tokens=False) if tok else [1] * haystack_tokens_needed

        if len(haystack_toks) > haystack_tokens_needed:
            haystack_toks = haystack_toks[:haystack_tokens_needed]

        prompts: list[str] = []
        depth_metadata: list[dict[str, Any]] = []

        for d in self.depths:
            insert_idx = int(round(d * len(haystack_toks)))
            insert_idx = max(0, min(insert_idx, len(haystack_toks)))

            combined_toks = haystack_toks[:insert_idx] + needle_toks + haystack_toks[insert_idx:] + query_toks
            if tok:
                prompt_str = tok.decode(combined_toks, skip_special_tokens=True)
            else:
                prompt_str = haystack_text + needle_text + query_text

            prompts.append(prompt_str)
            depth_metadata.append({
                "depth": d,
                "insert_token_index": insert_idx,
                "needle_distance": len(combined_toks) - insert_idx,
            })

        try:
            from vllm import SamplingParams
            sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        except ImportError:
            sampling_params = None

        start_wall = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, max_tokens=max_tokens)
        end_wall = time.perf_counter()
        total_wall = end_wall - start_wall

        grid_results: list[dict[str, Any]] = []
        retrieved_count = 0
        total_prompt_tok = 0
        total_gen_tok = 0
        ttfts: list[float] = []
        decodes: list[float] = []
        tpots: list[float] = []

        for idx, out in enumerate(outputs):
            gen_text = out.outputs[0].text.strip()
            gen_len = len(out.outputs[0].token_ids)
            total_prompt_tok += len(out.prompt_token_ids)
            total_gen_tok += gen_len

            is_retrieved = 1.0 if self.passcode in gen_text else 0.0
            if is_retrieved > 0.0:
                retrieved_count += 1

            meta = depth_metadata[idx]
            grid_results.append({
                "depth": meta["depth"],
                "needle_distance": meta["needle_distance"],
                "retrieved": is_retrieved,
                "generated_text": gen_text,
            })

            t, d_time, _ = extract_request_timings(out, total_wall, gen_len)
            ttfts.append(round(t, 4))
            decodes.append(round(d_time, 4))
            decode_tok = gen_len - 1 if gen_len > 1 else gen_len
            if d_time > 0 and decode_tok > 0:
                tpots.append(round(d_time / decode_tok * 1000.0, 2))

        accuracy = round(retrieved_count / len(self.depths), 4) if self.depths else 0.0
        mean_ttft = sum(ttfts) / len(ttfts) if ttfts else (total_wall / len(prompts) if prompts else 0.0)
        p95_ttft = calc_percentile(ttfts, 95.0) if ttfts else mean_ttft
        total_decode = max(decodes) if decodes else max(0.0, total_wall - mean_ttft)
        mean_tpot = sum(tpots) / len(tpots) if tpots else 0.0
        p95_tpot = calc_percentile(tpots, 95.0) if tpots else mean_tpot

        extra = {
            "niah_accuracy": accuracy,
            "niah_passcode": self.passcode,
            "niah_depths": self.depths,
            "niah_grid": grid_results,
            "niah_retrieved_count": retrieved_count,
            "niah_total_depths": len(self.depths),
        }

        return WorkloadOutput(
            prompt_tokens=total_prompt_tok,
            generated_tokens=total_gen_tok,
            ttft_sec=round(mean_ttft, 4),
            decode_time_sec=round(total_decode, 4),
            total_time_sec=round(total_wall, 4),
            tpot_mean_ms=round(mean_tpot, 2),
            tpot_p95_ms=round(p95_tpot, 2),
            ttft_p95_sec=round(p95_ttft, 4),
            extra_summary=extra,
        )

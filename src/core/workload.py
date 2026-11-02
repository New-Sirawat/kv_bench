"""A Workload defines *what gets sent to the engine and how it's timed* --
kept separate from Technique so several techniques can share the same
request pattern (e.g. every prefix-reuse method needs a cold/warm pair)
without copy-pasting the request logic into every technique file.

Add new workloads here as your evaluation needs grow -- e.g. a future
`NeedleInHaystack(Workload)` that buries a fact in the context and scores
retrieval accuracy would slot in exactly like SyntheticFiller/ColdWarm do,
and its score would flow into ResultEntry.extra via Technique.extra_metrics().
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from .metrics import calc_percentile


FILLER_SENTENCE = "The quick brown fox jumps over the lazy dog. "


@dataclass
class WorkloadOutput:
    prompt_tokens: int
    generated_tokens: int
    ttft_sec: float
    decode_time_sec: float
    total_time_sec: float
    # Cold-run metrics -- populated only by ColdWarm; zero for SyntheticFiller.
    # Having them on the same type means ResultEntry.cold_ttft_sec is always
    # present (just 0.0 for single-shot workloads), so CSV columns are stable.
    cold_ttft_sec: float = 0.0
    cold_total_sec: float = 0.0

    # Latency distributions & percentiles
    tpot_mean_ms: float = 0.0
    tpot_p95_ms: float = 0.0
    ttft_p95_sec: float = 0.0
    inter_token_latencies_ms: list[float] = field(default_factory=list)
    request_ttfts_sec: list[float] = field(default_factory=list)

    # Output tokens, logprobs, and workload-specific summaries
    generated_token_ids: list[int] = field(default_factory=list)
    step_logprobs: list[dict[int, float]] = field(default_factory=list)
    extra_summary: dict[str, Any] = field(default_factory=dict)



class Workload(ABC):
    @abstractmethod
    def run(self, llm, context_length: int, max_tokens: int) -> WorkloadOutput:
        raise NotImplementedError


def _get_tokenizer(llm: Any) -> Any:
    """Safely obtain tokenizer from LLM instance if present and valid."""
    if hasattr(llm, "get_tokenizer") and callable(getattr(llm, "get_tokenizer")):
        try:
            tok = llm.get_tokenizer()
            if hasattr(tok, "encode") and hasattr(tok, "decode"):
                return tok
        except Exception:
            pass
    return None


def _make_prompt(context_length: int) -> str:
    return FILLER_SENTENCE * max(1, context_length // 10)


def load_continuous_prefix(
    context_length: int,
    pg19_dir: str | None = None,
    tokenizer: Any | None = None,
) -> str:
    """Load a continuous natural text passage from staged PG-19 books,
    falling back to synthetic filler if unavailable.

    When tokenizer is available, encodes candidate text and slices to exact
    target tokens. When tokenizer is None, slices to target words calibrated
    for BPE tokenizers (~1.45 tokens/word on 19th-century text).
    """
    if pg19_dir is None:
        pg19_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "pg19"))
    if os.path.isdir(pg19_dir):
        files = sorted([os.path.join(pg19_dir, f) for f in os.listdir(pg19_dir) if f.endswith(".txt")])
        if files:
            # Need enough words to cover context_length across multiple files
            target_words = max(10, int(context_length / 1.45) + 32)
            accumulated_words: list[str] = []
            for filepath in files:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    accumulated_words.extend(f.read().split())
                if len(accumulated_words) >= target_words:
                    break

            if accumulated_words:
                raw_text = " ".join(accumulated_words)
                if tokenizer is not None and hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"):
                    try:
                        tokens = tokenizer.encode(raw_text, add_special_tokens=False)
                        if len(tokens) >= context_length:
                            return tokenizer.decode(tokens[:context_length])
                        return tokenizer.decode(tokens)
                    except Exception:
                        pass
                # Fallback word-slice if tokenizer is None or failed
                calibrated_words = max(10, int(context_length / 1.45))
                return " ".join(accumulated_words[:calibrated_words])
    return _make_prompt(context_length)


def extract_request_timings(
    out: Any, fallback_wall_sec: float, gen_tokens: int
) -> tuple[float, float, float]:
    """Extract (ttft_sec, decode_sec, total_sec) from vLLM RequestOutput across V0 and V1 APIs."""
    metrics = getattr(out, "metrics", None)
    if metrics is not None:
        def _is_num(val: Any) -> bool:
            return isinstance(val, (int, float)) and not isinstance(val, bool)

        # 1. vLLM V1 API: RequestStateStats
        # Attribute first_token_latency is the wall-clock interval from arrival to first token.
        # first_token_ts and last_token_ts are monotonic timestamps of first and last generation steps.
        first_token_latency = getattr(metrics, "first_token_latency", None)
        first_token_ts = getattr(metrics, "first_token_ts", None)
        last_token_ts = getattr(metrics, "last_token_ts", None)
        scheduled_ts = getattr(metrics, "scheduled_ts", None)

        if _is_num(first_token_ts) and _is_num(last_token_ts) and first_token_ts > 0.0:
            decode = max(0.0, float(last_token_ts - first_token_ts))
            if _is_num(first_token_latency) and first_token_latency > 0.0:
                ttft = float(first_token_latency)
            elif _is_num(scheduled_ts) and scheduled_ts > 0.0:
                ttft = max(0.0, float(first_token_ts - scheduled_ts))
            else:
                ttft = max(0.0, fallback_wall_sec - decode)
            total = ttft + decode
            return round(ttft, 4), round(decode, 4), round(total, 4)

        # 2. vLLM V0 API: RequestMetrics (first_token_time, arrival_time, finished_time)
        ft = getattr(metrics, "first_token_time", None)
        arr = getattr(metrics, "arrival_time", None)
        fin = getattr(metrics, "finished_time", None)
        if _is_num(ft) and _is_num(arr) and _is_num(fin):
            ttft = max(0.0, float(ft - arr))
            decode = max(0.0, float(fin - ft))
            total = max(0.0, float(fin - arr))
            return round(ttft, 4), round(decode, 4), round(total, 4)

    # 3. Fallback when engine metrics are absent:
    total = fallback_wall_sec
    ttft = total
    decode = 0.0
    return round(ttft, 4), round(decode, 4), round(total, 4)


def _timed_generate(llm, prompt: str, max_tokens: int) -> WorkloadOutput:
    try:
        from vllm import SamplingParams
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=5)
    except ImportError:
        sampling_params = None

    start_wall = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    end_wall = time.perf_counter()

    out = outputs[0]
    prompt_tokens = len(out.prompt_token_ids)
    gen_tokens = len(out.outputs[0].token_ids)

    ttft, decode, total = extract_request_timings(out, end_wall - start_wall, gen_tokens)

    decode_tokens = gen_tokens - 1 if gen_tokens > 1 else gen_tokens
    tpot_mean = (decode / decode_tokens * 1000.0) if (decode > 0 and decode_tokens > 0) else 0.0

    from .quality import extract_step_logprobs

    gen_token_ids = list(out.outputs[0].token_ids) if len(out.outputs) > 0 else []
    step_lps = extract_step_logprobs(out)

    return WorkloadOutput(
        prompt_tokens=prompt_tokens,
        generated_tokens=gen_tokens,
        ttft_sec=round(ttft, 4),
        decode_time_sec=round(decode, 4),
        total_time_sec=round(total, 4),
        tpot_mean_ms=round(tpot_mean, 2),
        tpot_p95_ms=round(tpot_mean, 2),
        ttft_p95_sec=round(ttft, 4),
        request_ttfts_sec=[round(ttft, 4)],
        generated_token_ids=gen_token_ids,
        step_logprobs=step_lps,
    )



class SyntheticFiller(Workload):
    """One cold prompt of the target length, one short generation. This is
    what baseline/flexgen-style techniques need -- pure memory/latency
    measurement, no reuse involved.

    cold_ttft_sec and cold_total_sec in the output will be 0.0 for this
    workload since there is no separate cold vs. warm distinction.
    """

    def run(self, llm, context_length: int, max_tokens: int) -> WorkloadOutput:
        tok = _get_tokenizer(llm)
        prompt = load_continuous_prefix(context_length, tokenizer=tok)
        return _timed_generate(llm, prompt, max_tokens)


class ColdWarm(Workload):
    """Two-phase prefix-reuse workload for chat-style scenarios: a 'cold'
    request that populates the KV cache (simulating the first message in a
    conversation or a shared system prompt), then a 'warm' request reusing
    that prefix with a new instruction (simulating subsequent turns).

    This models the SME/team chat use-case: 10-15 people sharing context
    (e.g. a long document or thread) where repeated queries all reuse the
    same prefix. A warm TTFT that is meaningfully lower than cold TTFT is
    direct evidence that prefix reuse is working.

    Both cold and warm timings are returned so the harness can surface the
    delta as a first-class reported field rather than a derived annotation.
    """

    def run(self, llm, context_length: int, max_tokens: int) -> WorkloadOutput:
        tok = _get_tokenizer(llm)
        prefix = load_continuous_prefix(context_length, tokenizer=tok)
        cold_prompt = prefix + "\n\nInstruction: Summarize."
        warm_prompt = prefix + "\n\nInstruction: What is the theme?"

        cold_out = _timed_generate(llm, cold_prompt, max_tokens)
        warm_out = _timed_generate(llm, warm_prompt, max_tokens)

        # Surface cold timing on the warm result so the harness gets both in
        # one return value and can compute the delta without a second output type.
        warm_out.cold_ttft_sec = cold_out.ttft_sec
        warm_out.cold_total_sec = cold_out.total_time_sec
        return warm_out


class ConcurrentChatWorkload(Workload):
    """Multi-user concurrent chat workload simulating 10-15 SME staff members
    sharing a single GPU (Section 1.3.4 & 3.3.1 in proposal.md).

    Issues N concurrent requests against a shared prefix/document, exercising
    continuous batching in vLLM. Measures individual TTFT distributions,
    P95 TTFT tail latency, and aggregated decode throughput.
    """

    def __init__(self, num_users: int = 10):
        self.num_users = num_users

    def run(self, llm, context_length: int, max_tokens: int) -> WorkloadOutput:
        try:
            from vllm import SamplingParams
            sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        except ImportError:
            sampling_params = None

        tok = _get_tokenizer(llm)
        prefix = load_continuous_prefix(context_length, tokenizer=tok)
        queries = [
            f"User {i+1} inquiry: Answer query {i+1} concerning the document."
            for i in range(self.num_users)
        ]
        prompts = [f"{prefix}\n\n{q}" for q in queries]

        start_wall = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        end_wall = time.perf_counter()

        ttfts = []
        decodes = []
        tpots = []
        tot_prompt_tok = 0
        tot_gen_tok = 0

        for out in outputs:
            tot_prompt_tok += len(out.prompt_token_ids)
            gen_len = len(out.outputs[0].token_ids)
            tot_gen_tok += gen_len

            t, d, _ = extract_request_timings(out, end_wall - start_wall, gen_len)
            decode_tok = gen_len - 1 if gen_len > 1 else gen_len
            if d > 0 and decode_tok > 0:
                tpots.append(round(d / decode_tok * 1000.0, 2))
            ttfts.append(round(t, 4))
            decodes.append(round(d, 4))

        mean_ttft = sum(ttfts) / len(ttfts) if ttfts else (end_wall - start_wall)
        p95_ttft = calc_percentile(ttfts, 95.0)

        total_wall = end_wall - start_wall
        total_decode = max(decodes) if decodes else max(0.0, total_wall - mean_ttft)

        decode_toks = tot_gen_tok - len(outputs) if tot_gen_tok > len(outputs) else tot_gen_tok
        tpot_mean = (sum(tpots) / len(tpots)) if tpots else ((total_decode / decode_toks * 1000.0) if (total_decode > 0 and decode_toks > 0) else 0.0)
        p95_tpot = calc_percentile(tpots, 95.0) if tpots else round(tpot_mean, 2)

        return WorkloadOutput(
            prompt_tokens=tot_prompt_tok,
            generated_tokens=tot_gen_tok,
            ttft_sec=round(mean_ttft, 4),
            decode_time_sec=round(total_decode, 4),
            total_time_sec=round(total_wall, 4),
            tpot_mean_ms=round(tpot_mean, 2),
            tpot_p95_ms=round(p95_tpot, 2),
            ttft_p95_sec=round(p95_ttft, 4),
            inter_token_latencies_ms=tpots,
            request_ttfts_sec=ttfts,
            extra_summary={"num_users": self.num_users},
        )

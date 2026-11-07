"""Canonical Diagnostic KV-Cache Profiler & Multi-Layer Bitwidth Telemetry.

Provides zero-bias, post-hoc physical inspection of active KV-cache tensors across
transformer layers. Evaluates physical presence (retention ratio) and continuous
representation budget (effective bitwidth per scalar element) across conversation sessions
with baseline-controlled causal attribution.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DiagnosticResult:
    sample_id: str
    task_type: str
    is_correct: bool
    baseline_correct: bool | None
    causal_verdict: str
    session_token_counts: list[int]
    session_retention_ratios: list[float]
    session_effective_bitwidths: list[float]
    overall_retention_ratio: float
    overall_effective_bitwidth: float
    min_session_effective_bitwidth: float
    most_degraded_session_idx: int | None
    session_content_retention: list[float]
    session_syntax_retention: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_token_is_content(token_str: str) -> bool:
    """Classify whether a token string is a semantic content token vs syntax/delimiter/whitespace."""
    cleaned = token_str.strip().replace(" ", "").replace("Ġ", "")
    if not cleaned:
        return False
    # Check if purely punctuation or control syntax
    if re.fullmatch(r"[\.,!?;:\(\)\[\]\"'\\/\-_=`~@#\$%\^&\*\+\{\}<>|]+", cleaned):
        return False
    if cleaned in {"User:", "Assistant:", "[Session", "Break]", "[SessionBreak]", "[Session Break]"}:
        return False
    if "SessionBreak" in cleaned or "ChatHistoryContinued" in cleaned:
        return False
    return True


def parse_session_token_ranges(
    prompt_text: str,
    prompt_token_ids: list[int],
    tokenizer: Any = None,
) -> list[tuple[int, int]]:
    """Segment prompt tokens into exact session epochs based on delimiter token sequences.

    Uses direct delimiter token-sequence scanning over prompt_token_ids to guarantee
    exact boundary alignment and prevent BPE chunk re-tokenization length drift.
    """
    total_tokens = len(prompt_token_ids)
    if total_tokens == 0:
        return [(0, 0)]

    session_delimiter_patterns = [
        "[Session Break]",
        "\n\n[Session Break]\n\n",
        "\n\n[Chat History Continued]\n",
    ]

    delimiter_spans: list[tuple[int, int]] = []

    if tokenizer is not None:
        for delim_str in session_delimiter_patterns:
            delim_ids = tokenizer.encode(delim_str, add_special_tokens=False)
            if not delim_ids:
                continue
            d_len = len(delim_ids)
            i = 0
            while i <= total_tokens - d_len:
                if prompt_token_ids[i : i + d_len] == delim_ids:
                    delimiter_spans.append((i, i + d_len))
                    i += d_len
                else:
                    i += 1

    # Sort and merge overlapping or contiguous delimiter spans
    merged_delimiters: list[tuple[int, int]] = []
    for s_start, s_end in sorted(delimiter_spans, key=lambda s: s[0]):
        if not merged_delimiters or s_start > merged_delimiters[-1][1]:
            merged_delimiters.append((s_start, s_end))
        else:
            merged_delimiters[-1] = (merged_delimiters[-1][0], max(merged_delimiters[-1][1], s_end))

    if not merged_delimiters:
        return [(0, total_tokens)]

    ranges: list[tuple[int, int]] = []
    current_pos = 0

    for d_start, d_end in merged_delimiters:
        if d_start > current_pos:
            ranges.append((current_pos, d_start))
        current_pos = max(current_pos, d_end)

    if current_pos < total_tokens:
        ranges.append((current_pos, total_tokens))

    return ranges if ranges else [(0, total_tokens)]


def extract_active_positions_per_layer(cache: Any, total_tokens: int) -> list[set[int]]:
    """Extract set of active original prompt token positions across all transformer layers."""
    if cache is None:
        return [set(range(total_tokens))]

    layers = getattr(cache, "layers", None)
    if layers is None or len(layers) == 0:
        return [set(range(total_tokens))]

    layer_positions: list[set[int]] = []
    for layer in layers:
        orig = getattr(layer, "original_positions", None)
        if orig is not None and isinstance(orig, (list, set)):
            layer_positions.append(set(orig))
        elif hasattr(layer, "get_active_positions"):
            layer_positions.append(set(layer.get_active_positions(total_tokens)))
        else:
            # Fallback for untracked caches (e.g. uncompressed DynamicCache): all tokens retained
            layer_positions.append(set(range(total_tokens)))

    return layer_positions


def inspect_sample_diagnostic(
    cache: Any,
    prompt_text: str,
    prompt_token_ids: list[int],
    sample_meta: dict[str, Any],
    is_correct: bool,
    baseline_correct: bool | None = None,
    tokenizer: Any = None,
    bits_per_scalar: float = 16.0,
) -> DiagnosticResult:
    """Execute post-hoc physical inspection of active KV cache with baseline-controlled causal attribution."""
    sample_id = sample_meta.get("id", "unknown")
    task_type = sample_meta.get("task_type", "unknown")
    total_tokens = len(prompt_token_ids)

    # 1. Extract layer-wise active token positions
    layer_positions = extract_active_positions_per_layer(cache, total_tokens)
    layers_total = max(1, len(layer_positions))

    # 2. Segment prompt into session epochs
    session_ranges = parse_session_token_ranges(prompt_text, prompt_token_ids, tokenizer)

    # 3. Classify content vs syntax tokens
    token_strings = [
        tokenizer.decode([tok_id]) if tokenizer is not None else str(tok_id)
        for tok_id in prompt_token_ids
    ]
    is_content_mask = [classify_token_is_content(s) for s in token_strings]

    session_token_counts: list[int] = []
    session_retention_ratios: list[float] = []
    session_effective_bitwidths: list[float] = []
    session_content_retention: list[float] = []
    session_syntax_retention: list[float] = []

    for s_start, s_end in session_ranges:
        s_count = max(0, s_end - s_start)
        session_token_counts.append(s_count)
        s_len = max(1, s_count)
        s_indices = set(range(s_start, s_end))
        content_indices = {i for i in s_indices if is_content_mask[i]}
        syntax_indices = s_indices - content_indices

        layer_overall: list[float] = []
        layer_content: list[float] = []
        layer_syntax: list[float] = []

        for l_pos in layer_positions:
            retained_in_layer = len(l_pos & s_indices)
            retained_s = retained_in_layer / s_len
            layer_overall.append(retained_s)

            if content_indices:
                retained_c = len(l_pos & content_indices) / len(content_indices)
                layer_content.append(retained_c)
            else:
                layer_content.append(1.0)

            if syntax_indices:
                retained_sy = len(l_pos & syntax_indices) / len(syntax_indices)
                layer_syntax.append(retained_sy)
            else:
                layer_syntax.append(1.0)

        mean_retention = round(sum(layer_overall) / len(layer_overall), 4)
        session_retention_ratios.append(mean_retention)
        session_effective_bitwidths.append(round(mean_retention * bits_per_scalar, 2))
        session_content_retention.append(round(sum(layer_content) / len(layer_content), 4))
        session_syntax_retention.append(round(sum(layer_syntax) / len(layer_syntax), 4))

    # Overall physical prompt token retention across all layers (exact prompt token set)
    prompt_token_set = set(range(total_tokens))
    total_active_instances = sum(len(l_pos & prompt_token_set) for l_pos in layer_positions)
    max_possible_instances = max(1, total_tokens * layers_total)
    overall_retention_ratio = round(total_active_instances / max_possible_instances, 4)
    overall_effective_bitwidth = round(overall_retention_ratio * bits_per_scalar, 2)
    min_session_effective_bitwidth = (
        min(session_effective_bitwidths) if session_effective_bitwidths else overall_effective_bitwidth
    )

    # Determine most degraded session (None if uncompressed / fully preserved)
    if not session_effective_bitwidths or all(r >= 0.999 for r in session_retention_ratios):
        most_degraded_session_idx = None
    else:
        most_degraded_session_idx = int(
            session_effective_bitwidths.index(min(session_effective_bitwidths))
        )

    # 4. Controlled Causal Attribution
    if is_correct:
        causal_verdict = "PASS_FIDELITY_MAINTAINED"
    else:
        if baseline_correct is True:
            causal_verdict = "QUALITY_REGRESSION"
        elif baseline_correct is False:
            causal_verdict = "BASELINE_LIMITATION"
        else:
            causal_verdict = "INCORRECT_STANDALONE"

    return DiagnosticResult(
        sample_id=sample_id,
        task_type=task_type,
        is_correct=is_correct,
        baseline_correct=baseline_correct,
        causal_verdict=causal_verdict,
        session_token_counts=session_token_counts,
        session_retention_ratios=session_retention_ratios,
        session_effective_bitwidths=session_effective_bitwidths,
        overall_retention_ratio=overall_retention_ratio,
        overall_effective_bitwidth=overall_effective_bitwidth,
        min_session_effective_bitwidth=min_session_effective_bitwidth,
        most_degraded_session_idx=most_degraded_session_idx,
        session_content_retention=session_content_retention,
        session_syntax_retention=session_syntax_retention,
    )


def summarize_diagnostic_results(results: list[DiagnosticResult]) -> dict[str, Any]:
    """Aggregate DiagnosticResults into canonical benchmark telemetry."""
    if not results:
        return {}

    total = len(results)
    verdict_counts: dict[str, int] = {}
    for r in results:
        verdict_counts[r.causal_verdict] = verdict_counts.get(r.causal_verdict, 0) + 1

    overall_retentions = [r.overall_retention_ratio for r in results]
    overall_bitwidths = [r.overall_effective_bitwidth for r in results]
    min_bitwidths = [r.min_session_effective_bitwidth for r in results]

    mean_bitwidth = (
        round(sum(overall_bitwidths) / len(overall_bitwidths), 2) if overall_bitwidths else 16.0
    )

    summary: dict[str, Any] = {
        "diag_total_inspected": total,
        "diag_pass_count": verdict_counts.get("PASS_FIDELITY_MAINTAINED", 0),
        "diag_quality_regression_count": verdict_counts.get("QUALITY_REGRESSION", 0),
        "diag_baseline_limitation_count": verdict_counts.get("BASELINE_LIMITATION", 0),
        "diag_incorrect_standalone_count": verdict_counts.get("INCORRECT_STANDALONE", 0),
        "diag_mean_overall_retention_ratio": (
            round(sum(overall_retentions) / len(overall_retentions), 4) if overall_retentions else 1.0
        ),
        "diag_mean_overall_effective_bitwidth": mean_bitwidth,
        "diag_mean_min_session_effective_bitwidth": (
            round(sum(min_bitwidths) / len(min_bitwidths), 2) if min_bitwidths else 16.0
        ),
        "diag_mean_compression_ratio": (
            round(16.0 / max(0.01, mean_bitwidth), 2) if mean_bitwidth > 0 else 0.0
        ),
    }
    return summary

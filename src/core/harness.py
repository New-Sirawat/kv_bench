"""The unified runner. This is the part that used to be copy-pasted into
every benchmark_*.py file: reset memory, build the engine, run the workload,
measure, handle failures, clean up. Written once here; every technique just
plugs into it.

Optional --profile mode wraps the workload in torch.profiler, saves the
Chrome trace, and extracts a memcpy summary (bytes transferred CPU↔GPU and
peak achieved PCIe bandwidth) into ResultEntry.extra. This gives you the
same transfer accounting that the old profiler_trace_phase3 runs produced,
now integrated directly into the benchmark result JSON.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from .axis import axis_names
from .config import resolve_model_target
from .metrics import (
    ResultEntry,
    format_metrics_summary,
    host_rss_mb,
    peak_gpu_memory_mb,
    reset_gpu_memory,
    throughput,
    validate_metrics,
)
from .quality import (
    evaluate_fidelity_against_reference,
    load_quality_reference,
    save_quality_reference,
)
from .technique import Technique
from .workload import Workload



@dataclass
class RunResult:
    model: str
    technique: str
    axes: list[str]
    results: list[ResultEntry]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Profiler trace parsing
# ---------------------------------------------------------------------------

def _parse_memcpy_summary(trace_path: str) -> dict:
    """Extract GPU↔CPU memory transfer stats from a Chrome trace file.

    Returns bytes_htod (CPU→GPU), bytes_dtoh (GPU→CPU), and peak bandwidth
    seen across all memcpy events. These come from the 'gpu_memcpy' category
    events that PyTorch profiler emits when profile_memory=True.
    """
    try:
        with open(trace_path) as f:
            trace = json.load(f)
    except Exception:
        return {}

    events = trace.get("traceEvents", [])
    htod_bytes = 0
    dtoh_bytes = 0
    peak_bw_gbs = 0.0

    for ev in events:
        if ev.get("cat") != "gpu_memcpy":
            continue
        args = ev.get("args", {})
        name = ev.get("name", "")
        b = args.get("bytes", 0)
        bw = args.get("memory bandwidth (GB/s)", 0.0)
        if "HtoD" in name:
            htod_bytes += b
        elif "DtoH" in name:
            dtoh_bytes += b
        if bw > peak_bw_gbs:
            peak_bw_gbs = bw

    return {
        "memcpy_htod_bytes": htod_bytes,
        "memcpy_dtoh_bytes": dtoh_bytes,
        "memcpy_peak_bw_gbs": round(peak_bw_gbs, 3),
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def _parse_extra_vllm_args(extra_args: tuple[str, ...]) -> dict:
    """Convert CLI-style extra_vllm_args from ModelSpec into LLM() keyword arguments."""
    kwargs: dict = {}
    i = 0
    while i < len(extra_args):
        arg = extra_args[i]
        if arg.startswith("--"):
            key = arg[2:].replace("-", "_")
            if i + 1 < len(extra_args) and not extra_args[i + 1].startswith("--"):
                kwargs[key] = extra_args[i + 1]
                i += 2
            else:
                kwargs[key] = True
                i += 1
        else:
            i += 1
    return kwargs


def run_benchmark(
    technique: Technique,
    model: str,
    contexts: list[int],
    max_tokens: int = 64,
    gpu_utilization: float = 0.85,
    profile: bool = False,
    profile_dir: str = "results/traces",
    dry_run: bool = False,
    workload_override: "Workload | None" = None,
    max_vram_gb: float | None = 16.0,
    save_reference: str | None = None,
    eval_reference: str | None = None,
    engine: str = "vllm",
) -> RunResult:

    """Run a technique across all requested context lengths.

    Args:
        technique: The Technique instance to benchmark.
        model: HuggingFace model ID or local path.
        contexts: List of prompt token lengths to sweep.
        max_tokens: Tokens to generate per request.
        gpu_utilization: Fraction of VRAM vLLM may use for KV blocks.
        profile: If True, wrap the workload in torch.profiler and save a
            Chrome trace per context length. Memcpy stats are extracted and
            added to ResultEntry.extra automatically.
        profile_dir: Directory for trace files (created if absent).
        dry_run: If True, skip the actual LLM() call -- useful for verifying
            technique registration and routing without a GPU.
        workload_override: Optional custom Workload instance to run instead of
            technique.workload().
        max_vram_gb: Cap VRAM to simulate consumer GPU limits (e.g. 16.0 GB)
            when running on higher-capacity cluster cards like RTX 4090 24GB.
    """
    # Keep vLLM V1 EngineCore in-process so torch.cuda.set_per_process_memory_fraction,
    # torch.cuda.max_memory_allocated, and torch.profiler apply to the GPU worker.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # Avoid FlashInfer JIT sampling compilation against older system nvcc (e.g. CUDA 12.2)
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    # Enable PyTorch expandable segments to prevent allocator fragmentation across long multi-request sweeps
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if not dry_run and engine == "vllm":
        from vllm import LLM  # noqa: F401 -- deferred so dry-run works without vllm

    technique.setup()
    workload = workload_override if workload_override is not None else technique.workload()
    axes = axis_names(technique.axes)
    entries: list[ResultEntry] = []
    reference_records: dict[str, Any] = {}

    gpu_name = torch.cuda.get_device_name(0) if (torch is not None and torch.cuda.is_available()) else "CPU (dry-run)"
    effective_utilization = gpu_utilization
    total_mem_gb = 0.0
    if max_vram_gb is not None and torch is not None and torch.cuda.is_available():
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_mem_gb > max_vram_gb:
            cap_fraction = min(1.0, max_vram_gb / total_mem_gb)
            torch.cuda.set_per_process_memory_fraction(cap_fraction, 0)
            effective_utilization = round(gpu_utilization * cap_fraction, 4)
            print(f"Hardware VRAM: {total_mem_gb:.1f} GB -> Capped to {max_vram_gb:.1f} GB (allocator cap: {cap_fraction:.3f}, vLLM utilization: {effective_utilization})")

    target_model, model_revision, model_extra_args = resolve_model_target(model)
    extra_model_kwargs = _parse_extra_vllm_args(model_extra_args)

    print(f"=== {technique.name} [engine: {engine}] :: axes={axes} ===")
    rev_str = f" (revision: {model_revision[:8]})" if model_revision else ""
    print(f"Model: {target_model}{rev_str} | GPU: {gpu_name}" + (" [DRY RUN]" if dry_run else ""))
    print("-" * 65)

    for ctx in contexts:
        print(f"\n[context length: {ctx}]")
        reset_gpu_memory()
        rss_before = host_rss_mb()
        llm = None

        try:
            engine_kwargs = dict(technique.engine_kwargs(ctx, max_tokens))
            for k, v in extra_model_kwargs.items():
                engine_kwargs.setdefault(k, v)
            if model_revision is not None:
                engine_kwargs.setdefault("revision", model_revision)

            if dry_run:
                # Emit a synthetic SUCCESS entry so callers get a parseable result.
                entry = ResultEntry(
                    requested_ctx=ctx,
                    status="DRY_RUN",
                    axes=axes,
                    extra={
                        "max_vram_gb": max_vram_gb,
                        "engine": engine,
                    } if max_vram_gb is not None else {"engine": engine},
                )
                print(f" -> DRY RUN OK (engine: {engine}, kwargs: {engine_kwargs})")
                entries.append(entry)
                continue

            if engine == "pytorch":
                from .pytorch_runner import PyTorchRunner
                cache_factory = technique.pytorch_cache_factory()
                llm = PyTorchRunner(
                    model_name_or_path=target_model,
                    revision=model_revision,
                    cache_factory=cache_factory,
                    trust_remote_code=True,
                    dry_run=dry_run,
                )
            else:
                from vllm import LLM
                dtype_choice = "bfloat16" if (torch is not None and torch.cuda.is_bf16_supported()) else "half"
                llm = LLM(
                    model=target_model,
                    dtype=dtype_choice,
                    gpu_memory_utilization=effective_utilization,
                    max_model_len=engine_kwargs.pop("max_model_len", max(2048, ctx + max_tokens + 2048)),
                    trust_remote_code=True,
                    enforce_eager=True,
                    disable_log_stats=False,
                    **engine_kwargs,
                )

            if profile:
                trace_path = _run_with_profiler(llm, workload, ctx, max_tokens, profile_dir, technique.name)
            else:
                trace_path = None

            output = workload.run(llm, ctx, max_tokens)
            rss_after = host_rss_mb()

            peak_alloc, peak_reserved = peak_gpu_memory_mb()
            tp = throughput(output.generated_tokens, output.decode_time_sec, output.total_time_sec)

            extra = technique.extra_metrics(llm, output)
            extra["engine"] = engine
            if engine == "pytorch":
                extra["attn_implementation"] = "sdpa"
            if max_vram_gb is not None:
                extra["max_vram_gb"] = max_vram_gb
                extra["effective_gpu_utilization"] = effective_utilization
                if total_mem_gb > 0:
                    extra["hardware_vram_gb"] = round(total_mem_gb, 2)
            if hasattr(output, "extra_summary") and isinstance(output.extra_summary, dict):
                extra.update(output.extra_summary)

            # Evaluate fidelity against reference if requested
            if eval_reference and os.path.exists(eval_reference):
                ref_data = load_quality_reference(eval_reference)
                fidelity = evaluate_fidelity_against_reference(
                    ref_data,
                    ctx,
                    target_token_ids=getattr(output, "generated_token_ids", []),
                    target_step_logprobs=getattr(output, "step_logprobs", None),
                )
                extra.update(fidelity)

            # Record reference generation if requested
            if save_reference:
                reference_records[str(ctx)] = {
                    "token_ids": getattr(output, "generated_token_ids", []),
                    "step_logprobs": getattr(output, "step_logprobs", []),
                    "prompt_tokens": output.prompt_tokens,
                    "generated_tokens": output.generated_tokens,
                }

            memcpy_bytes_per_token = 0.0
            if trace_path:
                memcpy_stats = _parse_memcpy_summary(trace_path)
                extra.update(memcpy_stats)
                total_memcpy = memcpy_stats.get("memcpy_htod_bytes", 0) + memcpy_stats.get("memcpy_dtoh_bytes", 0)
                if output.generated_tokens > 0:
                    memcpy_bytes_per_token = round(total_memcpy / output.generated_tokens, 2)

            entry = ResultEntry(
                requested_ctx=ctx,
                status="SUCCESS",
                axes=axes,
                prompt_tokens=output.prompt_tokens,
                generated_tokens=output.generated_tokens,
                ttft_sec=output.ttft_sec,
                decode_time_sec=output.decode_time_sec,
                total_time_sec=output.total_time_sec,
                throughput_tok_per_sec=tp,
                tpot_mean_ms=output.tpot_mean_ms,
                tpot_p95_ms=output.tpot_p95_ms,
                cold_ttft_sec=output.cold_ttft_sec,
                cold_total_sec=output.cold_total_sec,
                ttft_p95_sec=output.ttft_p95_sec,
                peak_alloc_mb=peak_alloc,
                peak_reserved_mb=peak_reserved,
                host_rss_delta_mb=round(rss_after - rss_before, 2),
                memcpy_bytes_per_token=memcpy_bytes_per_token,
                extra=extra,
            )

            max_vram_mb = (max_vram_gb * 1024.0) if max_vram_gb is not None else None
            validation_errs = validate_metrics(entry, max_vram_mb=max_vram_mb)
            print(f" -> {format_metrics_summary(entry)}")
            for v_err in validation_errs:
                print(f"    [METRIC WARNING] {v_err}")

        except Exception as e:  # noqa: BLE001 -- a failed context length shouldn't kill the sweep
            entry = ResultEntry(requested_ctx=ctx, status="FAILED", axes=axes, error=str(e))
            print(f" -> FAILED: {e}")

        finally:
            del llm
            reset_gpu_memory()

        entries.append(entry)

    if save_reference and reference_records:
        save_quality_reference(
            save_reference,
            {
                "model": model,
                "technique": technique.name,
                "contexts": reference_records,
            },
        )
        print(f"Saved quality reference -> {save_reference}")

    return RunResult(
        model=model,
        technique=technique.name,
        axes=axes,
        results=entries,
    )



def _run_with_profiler(llm, workload, ctx: int, max_tokens: int, profile_dir: str, technique_name: str) -> str:
    """Wrap workload.run() in torch.profiler and return the saved trace path."""
    import torch.profiler as profiler

    Path(profile_dir).mkdir(parents=True, exist_ok=True)
    trace_path = os.path.join(profile_dir, f"{technique_name}_ctx{ctx}.pt.trace.json")

    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        workload.run(llm, ctx, max_tokens)

    prof.export_chrome_trace(trace_path)
    return trace_path

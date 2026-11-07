import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from src.core.axis import axis_names
from src.core.config import BenchmarkConfig
from src.core.harness import RunResult, run_benchmark
from src.core.metrics import ResultEntry
from src.core.quality import load_quality_reference, save_quality_reference
from src.core.registry import get_technique, list_techniques


def parse_contexts(raw) -> list[int]:
    """Accepts space-separated (argparse nargs='+') or comma-separated input."""
    if isinstance(raw, list):
        joined = ",".join(str(c) for c in raw)
    else:
        joined = str(raw)
    return [int(c) for c in joined.replace(",", " ").split() if c.strip()]


def parse_technique_args(raw: list[str]) -> dict:
    """Parse --technique-args key=val key=val into a dict.

    Values are cast to int or float if possible, otherwise kept as str.
    This lets you tune technique parameters from the CLI without touching
    the technique file -- e.g. --technique-args swap_space_gb=8.

    All Technique.__init__ parameters MUST have defaults (zero-arg convention)
    so the registry can instantiate any technique without required arguments.
    These overrides are then applied on top of the defaults.
    """
    result = {}
    for item in raw or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"--technique-args items must be key=value, got: {item!r}"
            )
        k, v = item.split("=", 1)
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        result[k] = v
    return result


def run_isolated_sweep(args, contexts: list[int], max_tokens: int, output_path: str) -> RunResult:
    """Run each context length in an isolated subprocess to prevent CUDA memory retention."""
    technique_kwargs = parse_technique_args(args.technique_args)
    technique_cls_instance = get_technique(args.technique, **technique_kwargs)
    axes = axis_names(technique_cls_instance.axes)

    tmp_dir = tempfile.mkdtemp(prefix="kv_bench_iso_")
    all_results: list[ResultEntry] = []
    combined_ref_records: dict[str, Any] = {}

    try:
        for idx, ctx in enumerate(contexts, 1):
            print(f"\n=======================================================")
            print(f"=== [Context Sweep {idx}/{len(contexts)}: {ctx} tokens (Isolated Subprocess)] ===")
            print(f"=======================================================")
            sys.stdout.flush()

            tmp_out = os.path.join(tmp_dir, f"out_{ctx}.json")
            tmp_ref = os.path.join(tmp_dir, f"ref_{ctx}.json") if args.save_reference else None

            cmd = [
                sys.executable,
                os.path.abspath(sys.argv[0]),
                "--technique", args.technique,
                "--model", args.model,
                "--contexts", str(ctx),
                "--max-tokens", str(max_tokens),
                "--gpu-utilization", str(args.gpu_utilization),
                "--workload", args.workload,
                "--num-users", str(args.num_users),
                "--output", tmp_out,
                "--no-isolation",
                "--engine", args.engine,
            ]
            if args.dry_run:
                cmd.append("--dry-run")
            if args.max_vram_gb is not None:
                cmd.extend(["--max-vram-gb", str(args.max_vram_gb)])
            if args.technique_args:
                cmd.append("--technique-args")
                cmd.extend(args.technique_args)
            if args.profile:
                cmd.append("--profile")
                cmd.extend(["--profile-dir", args.profile_dir])
            if args.samples_per_task is not None:
                cmd.extend(["--samples-per-task", str(args.samples_per_task)])
            if args.seed is not None:
                cmd.extend(["--seed", str(args.seed)])
            if args.max_samples is not None:
                cmd.extend(["--max-samples", str(args.max_samples)])
            if args.dataset_path:
                cmd.extend(["--dataset-path", args.dataset_path])
            if getattr(args, "length_tier", "all") != "all":
                cmd.extend(["--length-tier", args.length_tier])
            if getattr(args, "select_by_length", False):
                cmd.append("--select-by-length")
            if tmp_ref:
                cmd.extend(["--save-reference", tmp_ref])
            if args.eval_reference:
                cmd.extend(["--eval-reference", args.eval_reference])

            proc = subprocess.run(cmd)

            child_entry: ResultEntry | None = None
            if os.path.exists(tmp_out):
                try:
                    with open(tmp_out) as f:
                        child_data = json.load(f)
                    res_list = child_data.get("results", [])
                    if res_list:
                        raw_dict = res_list[0]
                        child_entry = ResultEntry(**raw_dict)
                except Exception as ex:
                    print(f"Failed to parse child result for ctx={ctx}: {ex}", file=sys.stderr)

            if child_entry is None:
                child_entry = ResultEntry(
                    requested_ctx=ctx,
                    status="FAILED",
                    axes=axes,
                    error=f"Subprocess terminated with exit code {proc.returncode}",
                )

            all_results.append(child_entry)

            if tmp_ref and os.path.exists(tmp_ref):
                try:
                    ref_data = load_quality_reference(tmp_ref)
                    if "contexts" in ref_data and isinstance(ref_data["contexts"], dict):
                        combined_ref_records.update(ref_data["contexts"])
                except Exception as ex:
                    print(f"Failed to parse child reference for ctx={ctx}: {ex}", file=sys.stderr)

        if args.save_reference and combined_ref_records:
            save_quality_reference(
                args.save_reference,
                {
                    "model": args.model,
                    "technique": args.technique,
                    "contexts": combined_ref_records,
                },
            )
            print(f"Saved consolidated quality reference -> {args.save_reference}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return RunResult(
        model=args.model,
        technique=args.technique,
        axes=axes,
        results=all_results,
    )


def main() -> None:
    defaults = BenchmarkConfig()
    available = list_techniques()

    parser = argparse.ArgumentParser(
        description="Unified KV-cache technique benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run main.py --technique baseline --model qwen3-4b --contexts 512 1024 2048
  uv run main.py --technique baseline --model phi4-mini --dry-run
  uv run main.py --technique flexgen_style --contexts 4096 --technique-args swap_space_gb=8
  uv run main.py --technique lmcache --contexts 2048 --profile
""",
    )
    parser.add_argument("--technique", default=None, help=f"one of: {', '.join(available)}")
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--contexts", nargs="+", default=defaults.contexts)
    parser.add_argument("--max-tokens", type=int, default=defaults.max_tokens)
    parser.add_argument("--gpu-utilization", type=float, default=defaults.gpu_utilization)
    parser.add_argument(
        "--technique-args",
        nargs="*",
        metavar="KEY=VAL",
        help="Override technique __init__ kwargs, e.g. swap_space_gb=8 cpu_offload_gb=4",
    )
    parser.add_argument(
        "--workload",
        choices=["default", "filler", "coldwarm", "concurrent", "longmem", "niah"],
        default="default",
        help="Workload pattern: default (use technique default), filler (single prompt), coldwarm (prefix reuse), concurrent (10-15 simulated users), longmem (LongMemEval adherence), niah (Needle-In-A-Haystack)",
    )
    parser.add_argument(
        "--num-users",
        type=int,
        default=10,
        help="Number of simulated concurrent users for --workload concurrent (default: 10)",
    )
    parser.add_argument(
        "--engine",
        choices=["vllm", "pytorch"],
        default="vllm",
        help="Inference engine backend: vllm (industry standard) or pytorch (scientific control)",
    )
    parser.add_argument(
        "--max-vram-gb",
        type=float,
        default=16.0,
        help="Simulate a maximum VRAM ceiling in GB (default: 16.0 for consumer GPU ceiling on 24GB RTX 4090)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Wrap workload in torch.profiler; saves Chrome trace and extracts memcpy stats into results",
    )
    parser.add_argument(
        "--profile-dir",
        default="results/traces",
        help="Directory for profiler trace files (default: results/traces)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM() construction -- verifies technique registration and routing without a GPU",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Fast verification mode: run a single short prompt (128 ctx, 16 tokens) to verify CUDA memory capping and clean engine initialization",
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=None,
        help="Number of samples to draw per task category in LongMemEval (e.g. 1, 2, up to 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic sampling across evaluation runs (default: None for non-deterministic)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of dataset samples (e.g. for workloads like longmem)",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Custom dataset path for workloads like longmem",
    )
    parser.add_argument(
        "--length-tier",
        choices=["all", "short", "medium", "long", "auto"],
        default="all",
        help="Filter LongMemEval samples by length tier: short (<=2.5K tok), medium (2.5K-8K tok), long (>8K tok), auto (match each context length), or all (default)",
    )
    parser.add_argument(
        "--select-by-length",
        action="store_true",
        help="Dynamically select LongMemEval samples matching each context length in the sweep",
    )
    parser.add_argument(
        "--save-reference",
        default=None,
        help="Save baseline token IDs and logprobs to JSON as a reference for quality evaluation",
    )
    parser.add_argument(
        "--eval-reference",
        default=None,
        help="Path to reference JSON to evaluate quality (Top-1 agreement, KL divergence) against",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: results/<technique>.json or results/verify_<technique>.json)",
    )
    parser.add_argument(
        "--no-isolation",
        action="store_true",
        help="Disable subprocess isolation across context lengths (run all contexts in a single process)",
    )
    parser.add_argument(
        "--force-isolation",
        action="store_true",
        help="Force subprocess isolation even during dry-run (useful for verifying subprocess mechanics)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate Phase 2 Baseline Profiling Report and visualization figures from results directory",
    )
    args = parser.parse_args()

    if args.report:
        from src.analysis.report_generator import (
            load_baseline_results,
            reconcile_hardware_gaps,
            generate_publication_figures,
            generate_baseline_markdown_report,
        )
        print("[Report] Loading baseline results...")
        data = load_baseline_results()
        reconciled_hw = reconcile_hardware_gaps(
            data["hardware"]["results"],
            data["concurrent"].get("concurrency_grid"),
        )
        generate_publication_figures(data)
        generate_baseline_markdown_report(data, reconciled_hw)
        print("[Report] Phase 2 Baseline Profiling Report and figures successfully generated.")
        sys.exit(0)

    if not args.technique:
        parser.error("the following arguments are required: --technique (unless --report is specified)")

    technique_kwargs = parse_technique_args(args.technique_args)
    technique_cls_instance = get_technique(args.technique, **technique_kwargs)

    workload_override = None
    if args.workload == "filler" or (args.verify and args.workload == "default"):
        from src.core.workload import SyntheticFiller
        workload_override = SyntheticFiller()
    elif args.workload == "coldwarm":
        from src.core.workload import ColdWarm
        workload_override = ColdWarm()
    elif args.workload == "concurrent":
        from src.core.workload import ConcurrentChatWorkload
        workload_override = ConcurrentChatWorkload(num_users=args.num_users)
    elif args.workload == "longmem":
        from src.workloads.memory_adherence import LongMemEvalWorkload
        kw = {
            "samples_per_task": args.samples_per_task,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "length_tier": args.length_tier,
            "select_by_length": args.select_by_length,
            "eval_reference": args.eval_reference,
        }
        if args.dataset_path:
            kw["dataset_path"] = args.dataset_path
        workload_override = LongMemEvalWorkload(**kw)
    elif args.workload == "niah":
        from src.workloads.niah import NIAHWorkload
        workload_override = NIAHWorkload()



    engine_suffix = f"_{args.engine}" if args.engine != "vllm" else ""
    if args.verify:
        contexts = [128]
        max_tokens = 16
        default_output = f"results/verify_{args.technique}{engine_suffix}.json"
    else:
        contexts = parse_contexts(args.contexts)
        max_tokens = args.max_tokens
        default_output = f"results/{args.technique}{engine_suffix}.json"

    output_path = args.output or default_output
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    should_isolate = (
        (len(contexts) > 1 and not args.no_isolation and not args.verify)
        and (not args.dry_run or args.force_isolation)
    )

    if should_isolate:
        run_result = run_isolated_sweep(args, contexts, max_tokens, output_path)
    else:
        run_result = run_benchmark(
            technique_cls_instance,
            args.model,
            contexts,
            max_tokens=max_tokens,
            gpu_utilization=args.gpu_utilization,
            profile=args.profile,
            profile_dir=args.profile_dir,
            dry_run=args.dry_run,
            workload_override=workload_override,
            max_vram_gb=args.max_vram_gb,
            save_reference=args.save_reference,
            eval_reference=args.eval_reference,
            engine=args.engine,
        )


    with open(output_path, "w") as f:
        json.dump(run_result.to_dict(), f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())

    print(f"\nSaved {args.technique} results -> {output_path}")

    exit_code = 0
    if args.verify:
        failed_entries = [r for r in run_result.results if r.status == "FAILED"]
        if failed_entries:
            print(f"[VERIFY FAILED] Engine failed on {len(failed_entries)} context(s): {failed_entries[0].error}", file=sys.stderr)
            exit_code = 1
        elif not args.dry_run:
            entry = run_result.results[0]
            vram_cap_mb = (args.max_vram_gb * 1024.0) if args.max_vram_gb else float("inf")
            if entry.generated_tokens <= 0:
                print("[VERIFY FAILED] Engine generated 0 tokens.", file=sys.stderr)
                exit_code = 1
            elif entry.peak_alloc_mb > vram_cap_mb:
                print(
                    f"[VERIFY FAILED] Peak VRAM {entry.peak_alloc_mb:.1f} MB exceeded cap {vram_cap_mb:.1f} MB.",
                    file=sys.stderr,
                )
                exit_code = 1
            else:
                print(
                    f"[VERIFY OK] Clean engine init | generated {entry.generated_tokens} tokens | "
                    f"peak VRAM {entry.peak_alloc_mb:.1f} MB <= {vram_cap_mb:.1f} MB cap"
                )
        else:
            print(f"[VERIFY OK] Dry-run verification passed (contexts={contexts}, max_tokens={max_tokens}).")
    elif run_result.results and all(r.status == "FAILED" for r in run_result.results):
        print("[BENCHMARK FAILED] All context lengths failed.", file=sys.stderr)
        exit_code = 1

    sys.stdout.flush()
    sys.stderr.flush()

    if args.dry_run:
        # Use normal sys.exit in dry-run so CI gets a clean exit and we don't
        # swallow real failures. os._exit() bypasses Python cleanup.
        sys.exit(exit_code)

    # vLLM spawns background threads that can hang a clean process exit.
    os._exit(exit_code)


if __name__ == "__main__":
    main()

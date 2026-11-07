#!/usr/bin/env python3
"""Dataset Downloader and Preprocessor for kv_bench.

Downloads, preprocesses, and stages evaluation corpora specified in Proposal Section 3.2:
1. LongMemEval (High-level conversational memory adherence across 5 capabilities)
2. PG-19 (Low-level continuous long-context text scaling to 32K tokens)

Generates dataset provenance manifest with SHA-256 checksums at results/manifests/dataset_manifest.json.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import urllib.request
from typing import Any

# Default PG-19 test book IDs (from deepmind/pg19 test set)
DEFAULT_PG19_TEST_BOOKS = [
    "10146",  # Reminiscences of Pioneer Days in St. Paul
    "10321",  # A Critical History of Free Thought in reference to the Christian Religion
    "10356",  # The Life of the Fields
    "10762",  # The Expositor's Bible: The Book of the Twelve Prophets, Vol. 1
    "12204",  # A Little Tour in France
]

PG19_BASE_URL = "https://storage.googleapis.com/deepmind-gutenberg/test"
LONGMEMEVAL_ORACLE_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json"
)

# Mapping upstream LongMemEval question_type to our 5 benchmark capabilities
# Exclude single-session-preference as it contains open-ended subjective rubrics
TASK_TYPE_MAP = {
    "single-session-user": "fact_recall",
    "single-session-assistant": "fact_recall",
    "multi-session": "cross_session_reasoning",
    "knowledge-update": "information_update",
    "temporal-reasoning": "relative_timeframe",
}


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def fetch_url_bytes(url: str, timeout: int = 60) -> bytes:
    """Fetch content over HTTP with user-agent."""
    req = urllib.request.Request(url, headers={"User-Agent": "kv-bench-dataset-loader/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def classify_length_tier(words: int) -> tuple[int, str]:
    """Classify context word count into approximate token count and length tier."""
    approx_tokens = int(words * 1.35)
    if approx_tokens <= 2500:
        return approx_tokens, "short"
    elif approx_tokens <= 8000:
        return approx_tokens, "medium"
    else:
        return approx_tokens, "long"


def format_haystack_context(haystack_sessions: list[list[dict[str, Any]]]) -> str:
    """Format nested multi-session turns into unified dialogue string."""
    session_blocks = []
    for s_idx, session in enumerate(haystack_sessions):
        turns = []
        for turn in session:
            role = "User" if turn.get("role") == "user" else "Assistant"
            content = turn.get("content", "").strip()
            turns.append(f"{role}: {content}")
        if turns:
            session_blocks.append("\n".join(turns))
    return "\n\n[Session Break]\n\n".join(session_blocks)


def generate_mock_longmemeval(samples_per_task: int = 2) -> list[dict[str, Any]]:
    """Generate deterministic mock LongMemEval samples for dry-run / offline testing."""
    records = []
    task_types = [
        "fact_recall",
        "cross_session_reasoning",
        "information_update",
        "relative_timeframe",
        "unanswerable_refusal",
    ]

    for t_idx, task in enumerate(task_types):
        for s_idx in range(samples_per_task):
            sample_id = f"mock_{task}_{s_idx+1}"
            if task == "fact_recall":
                context = (
                    "User: I'm leading the systems research team in Austin.\n"
                    "Assistant: Understood, systems research in Austin.\n"
                    "User: My project codename is Project Falcon.\n"
                    "Assistant: Recorded Project Falcon as your project codename."
                )
                question = "What is the project codename for my systems research project?"
                answer = "Project Falcon"
                unanswerable = False
            elif task == "cross_session_reasoning":
                context = (
                    "User: Session 1: Team A completed Phase 1 in 4 days.\n"
                    "Assistant: Noted 4 days for Team A.\n\n"
                    "[Session Break]\n\n"
                    "User: Session 2: Team B took twice as long as Team A to finish.\n"
                    "Assistant: Noted that Team B took twice as long as Team A."
                )
                question = "How many days did Team B take to finish Phase 1?"
                answer = "8 days"
                unanswerable = False
            elif task == "information_update":
                context = (
                    "User: The cache policy was initially set to Least Recently Used (LRU).\n"
                    "Assistant: LRU is recorded as the active cache policy.\n"
                    "User: Please update: our team decided to switch to Attention Sinks with Sliding Window.\n"
                    "Assistant: Active cache policy updated to Attention Sinks with Sliding Window."
                )
                question = "What is the current active cache policy?"
                answer = "Attention Sinks"
                unanswerable = False
            elif task == "relative_timeframe":
                context = (
                    "User: Today is Thursday, October 10. We deployed the first test run three days ago.\n"
                    "Assistant: So the deployment took place on Monday, October 7.\n"
                    "User: Correct, and the retro is scheduled two days from today.\n"
                    "Assistant: That will be on Saturday, October 12."
                )
                question = "On what date was the first test run deployed?"
                answer = "October 7"
                unanswerable = False
            else:  # unanswerable_refusal
                context = (
                    "User: We benchmarked vLLM throughput on RTX 4090.\n"
                    "Assistant: Recorded benchmarking on RTX 4090.\n"
                    "User: Peak allocated VRAM reached 14.8 GB during the 32K context test.\n"
                    "Assistant: 14.8 GB peak allocation recorded."
                )
                question = "What was the brand of CPU used in the workstation testbed?"
                answer = "not mentioned"
                unanswerable = True

            words = len(context.split())
            approx_tok, tier = classify_length_tier(words)
            records.append({
                "id": sample_id,
                "task_type": task,
                "context": context,
                "question": question,
                "expected_answer": answer,
                "unanswerable": unanswerable,
                "word_count": words,
                "approx_tokens": approx_tok,
                "length_tier": tier,
            })
    return records


def generate_mock_pg19(num_books: int = 2) -> dict[str, str]:
    """Generate synthetic long continuous text books exceeding 32K tokens."""
    books = {}
    base_paragraph = (
        "In the tranquil valley of Oakhaven, the morning mist rose slowly above the canopy of ancient trees. "
        "The observers documented the gradual progression of the experiment with meticulous detail, "
        "noting each subtle fluctuation in temperature, pressure, and electromagnetic resonance across the valley. "
        "Generations of scholars had gathered within the stone archive to preserve the historical records of the region, "
        "ensuring that every discovery in navigation, philosophy, and metallurgy was faithfully recorded for posterity. "
    )
    # 50 words per paragraph. 700 paragraphs = 35,000 words (~45,000 tokens), exceeding 32K.
    full_text = "\n\n".join([f"Chapter {i+1}\n" + (base_paragraph * 7) for i in range(100)])

    for i in range(num_books):
        book_id = DEFAULT_PG19_TEST_BOOKS[i]
        books[book_id] = f"PROJECT GUTENBERG PREVIEW BOOK {book_id}\n\n{full_text}"
    return books


def stage_longmemeval(
    output_path: str,
    target_samples_per_task: int = 10,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Download, preprocess, and stage LongMemEval dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if dry_run:
        print("[Dry-Run] Generating mock LongMemEval dataset...")
        records = generate_mock_longmemeval(samples_per_task=2)
    else:
        print(f"Downloading LongMemEval oracle dataset from {LONGMEMEVAL_ORACLE_URL}...")
        raw_bytes = fetch_url_bytes(LONGMEMEVAL_ORACLE_URL)
        raw_items = json.loads(raw_bytes.decode("utf-8"))
        print(f"Loaded {len(raw_items)} raw questions from LongMemEval oracle.")

        # Group items by canonical task_type
        grouped: dict[str, list[dict[str, Any]]] = {
            "fact_recall": [],
            "cross_session_reasoning": [],
            "information_update": [],
            "relative_timeframe": [],
            "unanswerable_refusal": [],
        }

        for item in raw_items:
            raw_type = item.get("question_type", "")
            mapped_type = TASK_TYPE_MAP.get(raw_type)
            if not mapped_type:
                continue

            q_id = str(item.get("question_id", ""))
            question = str(item.get("question", "")).strip()
            raw_ans = item.get("answer", "")
            if isinstance(raw_ans, list):
                answer = ", ".join(str(x) for x in raw_ans)
            else:
                answer = str(raw_ans).strip()
            haystack_sessions = item.get("haystack_sessions", [])
            context = format_haystack_context(haystack_sessions)

            if not question or not answer or not context:
                continue

            words = len(context.split())
            approx_tok, tier = classify_length_tier(words)
            entry = {
                "id": q_id,
                "task_type": mapped_type,
                "context": context,
                "question": question,
                "expected_answer": answer,
                "unanswerable": False,
                "word_count": words,
                "approx_tokens": approx_tok,
                "length_tier": tier,
            }
            grouped[mapped_type].append(entry)

        # Create paired unanswerable questions to balance the unanswerable_refusal capability
        # By taking a sample's context and pairing with a question from a different unrelated session
        unanswerable_pool = []
        for i in range(min(target_samples_per_task, len(grouped["fact_recall"]))):
            source_context = grouped["fact_recall"][i]["context"]
            u_words = len(source_context.split())
            u_approx, u_tier = classify_length_tier(u_words)
            # Query about completely unrelated knowledge not in the context
            unanswerable_pool.append({
                "id": f"ur_abs_{i+1:02d}",
                "task_type": "unanswerable_refusal",
                "context": source_context,
                "question": "What is the secret Wi-Fi password and network configuration of the lab?",
                "expected_answer": "not mentioned",
                "unanswerable": True,
                "word_count": u_words,
                "approx_tokens": u_approx,
                "length_tier": u_tier,
            })
        grouped["unanswerable_refusal"] = unanswerable_pool

        # Select target samples per task
        records = []
        for task, entries in grouped.items():
            selected = entries[:target_samples_per_task]
            print(f"  Staging {len(selected)} samples for capability '{task}'")
            records.extend(selected)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    task_counts: dict[str, int] = {}
    for r in records:
        task_counts[r["task_type"]] = task_counts.get(r["task_type"], 0) + 1

    sha256 = compute_sha256(output_path)
    file_size = os.path.getsize(output_path)

    return {
        "path": os.path.relpath(output_path),
        "sha256": sha256,
        "size_bytes": file_size,
        "sample_count": len(records),
        "task_distribution": task_counts,
    }


def stage_pg19(
    output_dir: str,
    book_ids: list[str],
    dry_run: bool = False,
    force_download: bool = False,
) -> list[dict[str, Any]]:
    """Download and stage curated continuous long-text books from PG-19."""
    os.makedirs(output_dir, exist_ok=True)
    manifest_entries = []

    if dry_run:
        print("[Dry-Run] Generating mock PG-19 continuous text books...")
        mock_books = generate_mock_pg19(num_books=len(book_ids))
        for b_id, text in mock_books.items():
            book_path = os.path.join(output_dir, f"book_{b_id}.txt")
            with open(book_path, "w", encoding="utf-8") as f:
                f.write(text)

            words = len(text.split())
            manifest_entries.append({
                "path": os.path.relpath(book_path),
                "book_id": b_id,
                "sha256": compute_sha256(book_path),
                "size_bytes": os.path.getsize(book_path),
                "word_count": words,
                "approx_tokens": int(words * 1.3),
            })
    else:
        for b_id in book_ids:
            book_path = os.path.join(output_dir, f"book_{b_id}.txt")
            if os.path.exists(book_path) and not force_download:
                print(f"Reusing existing staged PG-19 book {b_id} at {book_path}...")
                with open(book_path, "rb") as f:
                    text_bytes = f.read()
            else:
                url = f"{PG19_BASE_URL}/{b_id}.txt"
                print(f"Downloading PG-19 book {b_id} from {url}...")
                text_bytes = fetch_url_bytes(url)
                with open(book_path, "wb") as f:
                    f.write(text_bytes)

            text = text_bytes.decode("utf-8", errors="ignore")
            words = len(text.split())
            approx_tokens = int(words * 1.3)
            print(f"  Staged book {b_id}: {words:,} words (~{approx_tokens:,} tokens, {len(text_bytes):,} bytes)")

            manifest_entries.append({
                "path": os.path.relpath(book_path),
                "book_id": b_id,
                "sha256": compute_sha256(book_path),
                "size_bytes": len(text_bytes),
                "word_count": words,
                "approx_tokens": approx_tokens,
            })

    return manifest_entries


def write_manifest(
    manifest_path: str,
    longmemeval_entry: dict[str, Any],
    pg19_entries: list[dict[str, Any]],
    is_mock: bool = False,
) -> None:
    """Write the dataset provenance manifest, preserving existing entries on partial updates."""
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    existing_longmemeval = {}
    existing_pg19 = []

    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            existing_longmemeval = (
                existing_data.get("datasets", {}).get("longmemeval", {}).get("file", {})
            )
            existing_pg19 = (
                existing_data.get("datasets", {}).get("pg19", {}).get("files", [])
            )
        except Exception:
            pass

    final_longmemeval = longmemeval_entry if longmemeval_entry else existing_longmemeval
    final_pg19 = pg19_entries if pg19_entries else existing_pg19

    manifest = {
        "manifest_version": "1.0",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "is_dry_run": is_mock,
        "datasets": {
            "longmemeval": {
                "name": "LongMemEval",
                "role": "High-Level Long-Term Memory Adherence (5 capabilities)",
                "upstream": LONGMEMEVAL_ORACLE_URL,
                "file": final_longmemeval,
            },
            "pg19": {
                "name": "PG-19",
                "role": "Low-Level Continuous Long-Context Stream (up to 32K tokens)",
                "upstream": PG19_BASE_URL,
                "files": final_pg19,
            },
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote dataset provenance manifest to {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and preprocess datasets for kv_bench.")
    parser.add_argument(
        "--dataset",
        choices=["all", "longmemeval", "pg19"],
        default="all",
        help="Dataset to stage (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Root output directory for staged datasets (default: data)",
    )
    parser.add_argument(
        "--manifest-path",
        default="results/manifests/dataset_manifest.json",
        help="Destination path for dataset_manifest.json",
    )
    parser.add_argument(
        "--longmemeval-samples-per-task",
        type=int,
        default=10,
        help="Target samples per capability for LongMemEval (default: 10, total 50)",
    )
    parser.add_argument(
        "--pg19-books",
        type=int,
        default=5,
        help="Number of PG-19 continuous test books to stage (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate synthetic mock data without making network requests",
    )

    args = parser.parse_args()

    longmemeval_entry = {}
    pg19_entries = []

    if args.dataset in ("all", "longmemeval"):
        longmemeval_path = os.path.join(args.output_dir, "longmemeval", "longmemeval_benchmark.json")
        longmemeval_entry = stage_longmemeval(
            longmemeval_path,
            target_samples_per_task=args.longmemeval_samples_per_task,
            dry_run=args.dry_run,
        )

    if args.dataset in ("all", "pg19"):
        pg19_dir = os.path.join(args.output_dir, "pg19")
        selected_books = DEFAULT_PG19_TEST_BOOKS[: args.pg19_books]
        pg19_entries = stage_pg19(
            pg19_dir,
            book_ids=selected_books,
            dry_run=args.dry_run,
        )

    write_manifest(
        args.manifest_path,
        longmemeval_entry=longmemeval_entry,
        pg19_entries=pg19_entries,
        is_mock=args.dry_run,
    )
    print("Dataset staging and manifest generation completed successfully.")


if __name__ == "__main__":
    main()

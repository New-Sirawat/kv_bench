"""Unit tests for dataset staging, preprocessing, and provenance manifest validation."""

import hashlib
import json
import os
import tempfile
import unittest

from scripts.download_datasets import (
    compute_sha256,
    format_haystack_context,
    generate_mock_longmemeval,
    generate_mock_pg19,
    stage_longmemeval,
    stage_pg19,
    write_manifest,
)
from src.core.workload import load_continuous_prefix
from src.workloads.memory_adherence import LongMemEvalWorkload


class TestDatasetStagingAndManifest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(self.temp_dir.name, "data")
        self.manifest_path = os.path.join(
            self.temp_dir.name, "results", "manifests", "dataset_manifest.json"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compute_sha256(self):
        test_file = os.path.join(self.temp_dir.name, "sample.txt")
        content = b"Benchmark KV Cache Optimization"
        with open(test_file, "wb") as f:
            f.write(content)

        expected_hash = hashlib.sha256(content).hexdigest()
        self.assertEqual(compute_sha256(test_file), expected_hash)

    def test_format_haystack_context(self):
        sessions = [
            [
                {"role": "user", "content": "Hello assistant"},
                {"role": "assistant", "content": "Hello user"},
            ],
            [
                {"role": "user", "content": "What is the update?"},
                {"role": "assistant", "content": "All systems nominal."},
            ],
        ]
        context = format_haystack_context(sessions)
        self.assertIn("User: Hello assistant", context)
        self.assertIn("Assistant: Hello user", context)
        self.assertIn("[Session Break]", context)
        self.assertIn("User: What is the update?", context)

    def test_mock_longmemeval_schema_and_capabilities(self):
        records = generate_mock_longmemeval(samples_per_task=2)
        self.assertEqual(len(records), 10)

        required_capabilities = {
            "fact_recall",
            "cross_session_reasoning",
            "information_update",
            "relative_timeframe",
            "unanswerable_refusal",
        }
        present_capabilities = {r["task_type"] for r in records}
        self.assertEqual(required_capabilities, present_capabilities)

        for record in records:
            self.assertIn("id", record)
            self.assertIn("task_type", record)
            self.assertIn("context", record)
            self.assertIn("question", record)
            self.assertIn("expected_answer", record)
            self.assertIn("word_count", record)
            self.assertIn("approx_tokens", record)
            self.assertIn("length_tier", record)
            self.assertIn(record["length_tier"], {"short", "medium", "long"})
            self.assertIsInstance(record["unanswerable"], bool)
            self.assertTrue(len(record["context"]) > 0)
            self.assertTrue(len(record["question"]) > 0)
            self.assertTrue(len(record["expected_answer"]) > 0)

    def test_mock_pg19_token_capacity(self):
        books = generate_mock_pg19(num_books=2)
        self.assertEqual(len(books), 2)
        for book_id, text in books.items():
            words = len(text.split())
            approx_tokens = int(words * 1.3)
            # Must exceed our 32K context scaling ceiling
            self.assertGreater(approx_tokens, 32768, f"Book {book_id} should exceed 32K tokens")

    def test_dry_run_pipeline_and_manifest_integrity(self):
        longmemeval_path = os.path.join(self.data_dir, "longmemeval", "longmemeval_benchmark.json")
        longmem_entry = stage_longmemeval(
            longmemeval_path, target_samples_per_task=2, dry_run=True
        )

        pg19_dir = os.path.join(self.data_dir, "pg19")
        pg19_entries = stage_pg19(pg19_dir, book_ids=["10146", "10321"], dry_run=True)

        write_manifest(
            self.manifest_path,
            longmemeval_entry=longmem_entry,
            pg19_entries=pg19_entries,
            is_mock=True,
        )

        self.assertTrue(os.path.exists(self.manifest_path))
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["manifest_version"], "1.0")
        self.assertTrue(manifest["is_dry_run"])
        self.assertIn("longmemeval", manifest["datasets"])
        self.assertIn("pg19", manifest["datasets"])

        # Check recorded longmemeval sha256 matches actual file
        self.assertEqual(
            compute_sha256(longmemeval_path),
            manifest["datasets"]["longmemeval"]["file"]["sha256"],
        )

        # Check recorded pg19 sha256 matches actual files
        for entry in manifest["datasets"]["pg19"]["files"]:
            book_file = os.path.join(self.temp_dir.name, entry["path"])
            self.assertTrue(os.path.exists(book_file))
            self.assertEqual(compute_sha256(book_file), entry["sha256"])

    def test_workload_loading_from_staged_benchmark(self):
        # Test that LongMemEvalWorkload loads properly
        workload = LongMemEvalWorkload()
        self.assertGreater(len(workload.samples), 0)
        first_sample = workload.samples[0]
        self.assertIn("task_type", first_sample)
        self.assertIn("question", first_sample)
        self.assertIn("expected_answer", first_sample)
        self.assertIn("length_tier", first_sample)

    def test_workload_length_tier_filtering_and_selection(self):
        # Test explicit length_tier filtering
        short_wl = LongMemEvalWorkload(length_tier="short")
        self.assertTrue(all(s.get("length_tier") == "short" for s in short_wl.samples))

        medium_wl = LongMemEvalWorkload(length_tier="medium")
        self.assertTrue(all(s.get("length_tier") == "medium" for s in medium_wl.samples))

        long_wl = LongMemEvalWorkload(length_tier="long")
        self.assertTrue(all(s.get("length_tier") == "long" for s in long_wl.samples))

        # Test dynamic select_by_length
        auto_wl = LongMemEvalWorkload(select_by_length=True)
        samples_512 = auto_wl.get_active_samples(context_length=512)
        self.assertTrue(all(s.get("approx_tokens", 0) <= 1200 for s in samples_512))

        samples_2048 = auto_wl.get_active_samples(context_length=2048)
        self.assertTrue(all(s.get("approx_tokens", 0) <= 2500 for s in samples_2048))

        samples_16k = auto_wl.get_active_samples(context_length=16384)
        self.assertTrue(all(s.get("approx_tokens", 0) >= 7000 for s in samples_16k))

    def test_load_continuous_prefix_from_pg19(self):
        pg19_dir = os.path.join(self.data_dir, "pg19")
        os.makedirs(pg19_dir, exist_ok=True)
        sample_book = os.path.join(pg19_dir, "book_test.txt")
        with open(sample_book, "w", encoding="utf-8") as f:
            f.write("Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi.")

        prefix = load_continuous_prefix(context_length=8, pg19_dir=pg19_dir)
        self.assertTrue(len(prefix) > 0)
        self.assertIn("Alpha beta gamma", prefix)

        # Fallback when dir is missing
        fallback_prefix = load_continuous_prefix(context_length=50, pg19_dir="/non/existent/path")
        self.assertTrue("The quick brown fox" in fallback_prefix)

    def test_load_continuous_prefix_multibook_concatenation(self):
        pg19_dir = os.path.join(self.data_dir, "pg19_multi")
        os.makedirs(pg19_dir, exist_ok=True)
        # Create 2 books with 20 words each
        b1_words = [f"wordA{i}" for i in range(20)]
        b2_words = [f"wordB{i}" for i in range(20)]
        with open(os.path.join(pg19_dir, "book_1.txt"), "w") as f:
            f.write(" ".join(b1_words))
        with open(os.path.join(pg19_dir, "book_2.txt"), "w") as f:
            f.write(" ".join(b2_words))

        # Request 40 tokens -> requires concatenating both books
        prefix = load_continuous_prefix(context_length=40, pg19_dir=pg19_dir)
        self.assertIn("wordA0", prefix)
        self.assertIn("wordB0", prefix)

    def test_load_continuous_prefix_with_tokenizer(self):
        pg19_dir = os.path.join(self.data_dir, "pg19_tok")
        os.makedirs(pg19_dir, exist_ok=True)
        sample_words = [f"token{i}" for i in range(100)]
        with open(os.path.join(pg19_dir, "book_1.txt"), "w") as f:
            f.write(" ".join(sample_words))

        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [hash(w) % 1000 for w in text.split()]

            def decode(self, token_ids):
                return " ".join([f"id_{tid}" for tid in token_ids])

        mock_tok = MockTokenizer()
        prefix = load_continuous_prefix(context_length=15, pg19_dir=pg19_dir, tokenizer=mock_tok)
        self.assertEqual(len(prefix.split()), 15)
        self.assertTrue(prefix.startswith("id_"))

    def test_write_manifest_preserves_sibling(self):
        manifest_file = os.path.join(self.temp_dir.name, "manifest_preserve.json")
        # 1. Write longmemeval only
        lm_entry = {"path": "data/sample.json", "sha256": "1234"}
        write_manifest(manifest_file, longmemeval_entry=lm_entry, pg19_entries=[])

        with open(manifest_file) as f:
            data1 = json.load(f)
        self.assertEqual(data1["datasets"]["longmemeval"]["file"]["sha256"], "1234")
        self.assertEqual(data1["datasets"]["pg19"]["files"], [])

        # 2. Update with pg19 only -> should preserve longmemeval
        pg_entries = [{"path": "data/pg19/book_1.txt", "sha256": "abcd"}]
        write_manifest(manifest_file, longmemeval_entry={}, pg19_entries=pg_entries)

        with open(manifest_file) as f:
            data2 = json.load(f)
        self.assertEqual(data2["datasets"]["longmemeval"]["file"]["sha256"], "1234")
        self.assertEqual(len(data2["datasets"]["pg19"]["files"]), 1)
        self.assertEqual(data2["datasets"]["pg19"]["files"][0]["sha256"], "abcd")


if __name__ == "__main__":
    unittest.main()

"""Small synthetic tests for the optional BM provider-file routing boundary."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))
from migrate import sha
from study_inputs import Sources, prepare


class ProviderBridgeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.original, self.native, self.snapshot = [self.root / p for p in ("original", "native", "snapshot")]
        for p in (self.original, self.native, self.snapshot):
            p.mkdir()
        self.relative = "benchmark_results/bone_marrow/maxfuse_v2/processed/bm_codex/active_input_48.csv.gz"
        self.rebuilt = self.native / "bm_codex/active_input_48.csv.gz"
        self.rebuilt.parent.mkdir()
        self.rebuilt.write_bytes(b"pinned native values")
        (self.original / "other.json").write_text("{}")
        self.manifest = self.native / "manifest.json"
        self.manifest.write_text("{}")
        self.complete = self.native / "COMPLETE.json"
        self.write_completion()
        (self.snapshot / "catalog.json").write_text(json.dumps({"files": [
            {"source": self.relative, "sha256": sha(self.rebuilt)},
            {"source": "other.json", "sha256": sha(self.original / "other.json")}
        ]}))

    def write_completion(self, **changes):
        record = {"status": "complete", "cells": 88031, "manifest_sha256": sha(self.manifest)}
        record.update(changes)
        self.complete.write_text(json.dumps(record))

    def sources(self, native=True):
        # Catalog-copy verification is independently tested against the real
        # 173-file snapshot. These fixtures isolate the new routing logic.
        with patch("study_inputs.verify_local"):
            return Sources(self.original, self.snapshot, self.native if native else None)

    def test_only_six_explicit_files_redirected(self):
        sources = self.sources()
        self.assertEqual(len(sources.native_paths), 6)
        self.assertEqual(sources.read(self.relative), self.rebuilt)
        self.assertEqual(sources.used[self.relative]["read_from_provider_rebuild"], str(self.rebuilt))
        self.assertEqual(sources.read("other.json"), self.original / "other.json")
        self.assertNotIn("read_from_provider_rebuild", sources.used["other.json"])
        sources.recheck()

    def test_changed_values_rejected_despite_self_reported_completion(self):
        self.rebuilt.write_bytes(b"different values")
        self.write_completion()
        with self.assertRaisesRegex(ValueError, "Frozen source changed"):
            self.sources().read(self.relative)

    def test_incomplete_or_wrong_manifest_rejected(self):
        for changes in ({"status": "partial"}, {"cells": 1}, {"manifest_sha256": "0" * 64}):
            self.write_completion(**changes)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                self.sources()

    def test_missing_rebuild_file_never_falls_back(self):
        old = self.original / self.relative
        old.parent.mkdir(parents=True)
        old.write_bytes(self.rebuilt.read_bytes())
        self.rebuilt.unlink()
        with self.assertRaises(FileNotFoundError):
            self.sources().read(self.relative)

    def test_outside_symlink_rejected(self):
        outside = self.root / "outside"
        self.rebuilt.rename(outside)
        self.rebuilt.symlink_to(outside)
        with self.assertRaises(ValueError):
            self.sources().read(self.relative)

    def test_recheck_detects_changed_file(self):
        sources = self.sources()
        sources.read(self.relative)
        self.rebuilt.write_bytes(b"changed after reading")
        with self.assertRaises(ValueError):
            sources.recheck()

    def test_recheck_detects_changed_completion(self):
        sources = self.sources()
        self.write_completion(status="partial")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            sources.recheck()

    def test_ordinary_sources_have_no_override(self):
        old = self.original / self.relative
        old.parent.mkdir(parents=True)
        old.write_bytes(self.rebuilt.read_bytes())
        sources = self.sources(native=False)
        self.assertEqual(sources.native_paths, {})
        self.assertIsNone(sources.native_provenance)
        self.assertEqual(sources.location(self.relative), self.original / self.relative)

    def test_other_case_rejected_before_output_creation(self):
        output = self.root / "output"
        with self.assertRaisesRegex(ValueError, "only"):
            prepare("unregistered_case", self.original, output, native_root=self.native)
        self.assertFalse(output.exists())

    def test_output_within_native_tree_rejected(self):
        sources = self.sources()
        sources.catalog["cases"] = [{"case": "bone_marrow", "config": "other.json"}]
        output = self.native / "new-output"
        with patch("study_inputs.Sources", return_value=sources):
            with self.assertRaises(ValueError):
                prepare("bone_marrow", self.original, output, native_root=self.native)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

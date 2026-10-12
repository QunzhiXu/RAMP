"""TNBC-specific routing checks, no real data or MaxFuse fitting."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from migrate import sha
from study_inputs import Sources, prepare


class TNBCBridgeTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.native, self.original, self.snapshot = [self.root / n for n in ("native", "original", "snapshot")]
        for p in (self.native, self.original, self.snapshot): p.mkdir()
        self.relative = "benchmark_results/tnbc/maxfuse_archive/input/shared_processed_float64.npy"
        self.array = self.native / "shared_processed_float64.npy"
        self.array.write_bytes(b"synthetic pinned array bytes")
        self.manifest = self.native / "input_manifest.json"
        self.manifest.write_text("{}")
        self.complete = self.native / "COMPLETE.json"
        self.completion()
        (self.snapshot / "catalog.json").write_text(json.dumps({"cases": [], "files": [
            {"source": self.relative, "sha256": sha(self.array)}]}))

    def completion(self, **changes):
        m = {"status":"complete", "case":"tnbc", "cells":62701, "manifest_sha256":sha(self.manifest)}
        m.update(changes)
        self.complete.write_text(json.dumps(m))

    def sources(self):
        with patch("study_inputs.verify_local"):
            return Sources(self.original, self.snapshot, self.native, native_case="tnbc")

    def test_seven_files_include_combined_npy_no_bm_paths(self):
        s = self.sources()
        self.assertEqual(len(s.native_paths), 7)
        self.assertTrue(all(k.startswith("benchmark_results/tnbc/maxfuse_archive/input/") for k in s.native_paths))
        self.assertEqual(s.read(self.relative), self.array)
        self.assertIn("read_from_provider_rebuild", s.used[self.relative])
        s.recheck()

    def test_hash_mismatch_rejected_despite_new_completion(self):
        self.array.write_bytes(b"wrong")
        self.completion()
        with self.assertRaisesRegex(ValueError, "Frozen source changed"):
            self.sources().read(self.relative)

    def test_missing_new_npy_never_falls_back(self):
        old = self.original / self.relative
        old.parent.mkdir(parents=True)
        old.write_bytes(self.array.read_bytes())
        self.array.unlink()
        with self.assertRaises(FileNotFoundError): self.sources().read(self.relative)

    def test_case_count_and_status_checked(self):
        for change in ({"case":"bone_marrow"}, {"cells":88031}, {"status":"partial"}):
            self.completion(**change)
            with self.assertRaises(ValueError): self.sources()

    def test_recheck_uses_tnbc_manifest_name(self):
        s = self.sources()
        self.manifest.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "manifest changed"): s.recheck()

    def test_wrong_manifest_hash_refused(self):
        self.completion(manifest_sha256="0"*64)
        with self.assertRaises(ValueError): self.sources()

    def test_tnbc_does_not_accept_bm_replay_flag(self):
        s = self.sources()
        s.catalog["cases"] = [{"case":"tnbc"}]
        with patch("study_inputs.Sources", return_value=s):
            with self.assertRaisesRegex(ValueError, "replay cases"):
                prepare("tnbc", self.original, self.root / "out", with_cache=True, native_root=self.native)
        self.assertFalse((self.root / "out").exists())

    def test_tnbc_refuses_output_inside_native_root(self):
        s = self.sources()
        config = self.original / "config.json"
        config.write_text("{}")
        s.expected["config.json"] = sha(config)
        s.catalog["cases"] = [{"case":"tnbc", "config":"config.json"}]
        with patch("study_inputs.Sources", return_value=s):
            with self.assertRaises(ValueError):
                prepare("tnbc", self.original, self.native / "out", native_root=self.native)
        self.assertFalse((self.native / "out").exists())


if __name__ == "__main__":
    unittest.main()

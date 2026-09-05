"""Review regressions: provenance gates, material identity and shape metrics."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from audit_schneider_response_scope import schneider_identity
from compare_electron_response_geometries import joint_shape_distance
from run_electron_response_diagnostic import extract_config_metadata
from test_electron_response_campaign import make_shard_report

MERGE = ROOT / "tools/merge_electron_response_diagnostics.py"
RUN = ROOT / "tools/run_electron_response_diagnostic.py"


class ReviewTests(unittest.TestCase):
    def test_material_not_density_segment(self):
        path = ROOT / "data/HUtoMaterialSchneider.txt"
        for hu, section in [(-1000, 0), (-951, 0), (-950, 1), (-550, 1), (-100, 2)]:
            record = schneider_identity(path, hu)
            self.assertEqual(record["material_section"], section)
            self.assertEqual(record["density_formula_segment"], 0)
        self.assertAlmostEqual(schneider_identity(path, -1000)["density_g_cm3"], .00121*9.35212)
        result = subprocess.run([sys.executable, str(ROOT/"tools/audit_schneider_response_scope.py"),
                                 "--hu", "-1000", "-550", "-100"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("scope mismatch", result.stderr)

    def test_shape_not_amplitude(self):
        a = np.array([[1., 2.], [3., 4.]])
        self.assertAlmostEqual(joint_shape_distance(a, 20*a), 0)
        self.assertAlmostEqual(joint_shape_distance([[1, 0]], [[0, 1]]), 2)
        for b in ([[1, -1], [3, 4]], [[1, float("nan")], [3, 4]]):
            with self.assertRaises(ValueError):
                joint_shape_distance(a, b)

    def test_vector_config_modules(self):
        record = extract_config_metadata('sv:Ph/Default/Modules = 3 "a" "b" "c"\n')
        self.assertEqual(record["Ph/Default/Modules"], '3 "a" "b" "c"')

    def test_raw_verification_is_real(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            report = make_shard_report(base, "a")
            cmd = [sys.executable, str(MERGE), "--reports", str(report),
                   "--output", str(base/"merged.json"), "--verify-raw"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            raw = Path(json.loads(report.read_text())["raw_steps"])
            with raw.open("a") as stream:
                stream.write("\n")
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SHA mismatch", result.stderr)

    def test_missing_config_sidecar_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            report = make_shard_report(base, "a")
            result = subprocess.run([sys.executable, str(MERGE), "--reports", str(report),
                                     "--output", str(base/"out.json"),
                                     "--expected-config-sha", "abc"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing metadata", result.stderr)

    def test_each_shard_closes_not_just_sum(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            reports = [make_shard_report(base, "a"), make_shard_report(base, "b", seed_shift=1)]
            for report, delta in zip(reports, [.2, -.2]):
                r = json.loads(report.read_text())
                r["joint_deposited_MeV"][0][0] += delta
                report.write_text(json.dumps(r))
            result = subprocess.run([sys.executable, str(MERGE), "--reports", *map(str, reports),
                                     "--output", str(base/"out.json")], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)

    def test_unsafe_submission_and_invalid_resources_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg = base/"base.txt"
            cfg.write_text("")
            cmd = [sys.executable, str(RUN), "--base-config", str(cfg),
                   "--output-dir", str(base/"out"), "--shards", "1", "--seed-start", "1"]
            for extra in (["--submit"], ["--cpus-per-shard", "-2"],
                          ["--mem-gb-per-shard", "nan"], ["--case-prefix", "../escape"]):
                result = subprocess.run(cmd+extra, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((base/"out").exists())


if __name__ == "__main__":
    unittest.main()

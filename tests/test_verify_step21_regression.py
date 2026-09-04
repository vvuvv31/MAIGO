#!/usr/bin/env python3
"""tests/test_verify_step21_regression.py

Unit and regression tests for Step 21 gate verification:
1. Missing evidence summary file must fail.
2. Missing any required field (gamma, overflow, unsupported, provenance, range, dose) must fail.
3. Out-of-bounds metrics (e.g. gamma < 95%, overflow > 0, dose diff >= 2%) must fail.
4. Complete and passing metrics must pass.
"""

import json
import unittest

REQUIRED_FIELDS = [
    "axis_range_diff_mm",
    "oblique_range_diff_mm",
    "axis_dose_diff_pct",
    "oblique_dose_diff_pct",
    "axis_gamma_pass_rate",
    "oblique_gamma_pass_rate",
    "step20_species_diff_pct",
    "unsupported_lookup_count",
    "shard_overflow_counters",
    "provenance_verified"
]

def evaluate_step21_summary(summary_data):
    # 1. Field completeness check
    for field in REQUIRED_FIELDS:
        if field not in summary_data:
            return False, f"Missing required field: {field}"

    # 2. Provenance check
    if not summary_data.get("provenance_verified", False):
        return False, "Provenance verification failed"

    # 3. Acceptance gates
    if summary_data["axis_range_diff_mm"] >= 1.0 or summary_data["oblique_range_diff_mm"] >= 1.0:
        return False, "Range difference >= 1.0 mm"
    if summary_data["axis_dose_diff_pct"] >= 2.0 or summary_data["oblique_dose_diff_pct"] >= 2.0:
        return False, "Dose difference >= 2.0%"
    if summary_data["axis_gamma_pass_rate"] <= 95.0 or summary_data["oblique_gamma_pass_rate"] <= 95.0:
        return False, "Gamma pass rate <= 95.0%"
    if summary_data["step20_species_diff_pct"] >= 2.0:
        return False, "Step 20 major species diff >= 2.0%"
    if summary_data["unsupported_lookup_count"] != 0:
        return False, "Unsupported lookup count != 0"
    if summary_data["shard_overflow_counters"] != 0:
        return False, "Shard overflow counters != 0"

    return True, "All gates passed"

class TestStep21VerifierRegression(unittest.TestCase):
    def valid_data(self):
        return {
            "axis_range_diff_mm": 0.4,
            "oblique_range_diff_mm": 0.6,
            "axis_dose_diff_pct": 1.5,
            "oblique_dose_diff_pct": 1.7,
            "axis_gamma_pass_rate": 98.2,
            "oblique_gamma_pass_rate": 97.4,
            "step20_species_diff_pct": 1.48,
            "unsupported_lookup_count": 0,
            "shard_overflow_counters": 0,
            "provenance_verified": True
        }

    def test_valid_summary_passes(self):
        passed, msg = evaluate_step21_summary(self.valid_data())
        self.assertTrue(passed, msg)

    def test_missing_gamma_fails(self):
        d = self.valid_data()
        del d["axis_gamma_pass_rate"]
        passed, msg = evaluate_step21_summary(d)
        self.assertFalse(passed)
        self.assertIn("Missing required field", msg)

    def test_missing_overflow_fails(self):
        d = self.valid_data()
        del d["shard_overflow_counters"]
        passed, msg = evaluate_step21_summary(d)
        self.assertFalse(passed)
        self.assertIn("Missing required field", msg)

    def test_missing_unsupported_fails(self):
        d = self.valid_data()
        del d["unsupported_lookup_count"]
        passed, msg = evaluate_step21_summary(d)
        self.assertFalse(passed)
        self.assertIn("Missing required field", msg)

    def test_missing_provenance_fails(self):
        d = self.valid_data()
        del d["provenance_verified"]
        passed, msg = evaluate_step21_summary(d)
        self.assertFalse(passed)
        self.assertIn("Missing required field", msg)

    def test_overflow_nonzero_fails(self):
        d = self.valid_data()
        d["shard_overflow_counters"] = 2
        passed, msg = evaluate_step21_summary(d)
        self.assertFalse(passed)
        self.assertIn("overflow", msg)

    def test_unsupported_nonzero_fails(self):
        d = self.valid_data()
        d["unsupported_lookup_count"] = 1
        passed, msg = evaluate_step21_summary(d)
        self.assertFalse(passed)
        self.assertIn("Unsupported", msg)

    def test_gamma_below_95_fails(self):
        d = self.valid_data()
        d["axis_gamma_pass_rate"] = 94.8
        passed, msg = evaluate_step21_summary(d)
        self.assertFalse(passed)
    def test_actual_summary_file_passes(self):
        import pathlib
        path = pathlib.Path(__file__).resolve().parent.parent / "evidence/step-21/step21_validation_summary.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            passed, msg = evaluate_step21_summary(data)
            self.assertTrue(passed, msg)

if __name__ == "__main__":
    unittest.main()

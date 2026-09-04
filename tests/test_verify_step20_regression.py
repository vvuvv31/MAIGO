#!/usr/bin/env python3
"""tests/test_verify_step20_regression.py

Unit and regression tests for Step 20 verifier logic:
1. 12 pass + 1 fail + suite mean < 2% must evaluate to overall FAIL.
2. All pass + suite mean < 2% must evaluate to overall PASS.
3. All pass + suite mean >= 2% must evaluate to overall FAIL.
4. Empty results must evaluate to overall FAIL.
"""

import unittest

def compute_overall_pass(results, suite_mean_diff):
    return (
        bool(results)
        and all(r.get("pass", False) for r in results)
        and suite_mean_diff < 0.02
    )

class TestStep20VerifierLogic(unittest.TestCase):
    def test_single_failure_fails_overall(self):
        # 12 pass, 1 fail, mean diff 1.48% (< 2%)
        results = [{"pass": True, "major_species_relative_diff": 0.01} for _ in range(12)]
        results.append({"pass": False, "major_species_relative_diff": 0.0481})
        suite_mean = sum(r["major_species_relative_diff"] for r in results) / len(results)
        self.assertLess(suite_mean, 0.02)
        overall = compute_overall_pass(results, suite_mean)
        self.assertFalse(overall, "Single case failure must result in overall FAIL even if suite mean < 2%")

    def test_all_pass_within_mean_passes(self):
        results = [{"pass": True, "major_species_relative_diff": 0.015} for _ in range(13)]
        suite_mean = 0.015
        overall = compute_overall_pass(results, suite_mean)
        self.assertTrue(overall)

    def test_all_pass_exceeding_mean_fails(self):
        results = [{"pass": True, "major_species_relative_diff": 0.025} for _ in range(13)]
        suite_mean = 0.025
        overall = compute_overall_pass(results, suite_mean)
        self.assertFalse(overall, "Suite mean >= 2% must result in overall FAIL")

    def test_empty_results_fails(self):
        overall = compute_overall_pass([], 0.0)
        self.assertFalse(overall)

if __name__ == "__main__":
    unittest.main()

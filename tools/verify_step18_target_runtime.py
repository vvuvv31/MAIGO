#!/usr/bin/env python3
"""Step 18 Verification Script: Material Target Selection & GPU Replay.

Validates:
Gate 1: Categorical target distribution and Pearson Chi-square goodness-of-fit.
Gate 2: Density independence of target fractions across density variations.
Gate 3: Cross-section element library reuse (Oxygen Z=8 across sections 5, 8, 20).
Gate 4: Fail-closed missing/unsupported target behavior (zero aliasing to O/H).
Gate 5: CPU/GPU bitwise equivalence and atomic diagnostics tracking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent

def run_tests() -> tuple[bool, str]:
    res = subprocess.run([str(REPO_DIR / "build" / "carbon_tests"), "test_step18"],
                         cwd=REPO_DIR, capture_output=True, text=True)
    return (res.returncode == 0 and "All carbon_tests passed" in res.stdout), res.stdout

def run_regression_suite() -> bool:
    res = subprocess.run([str(REPO_DIR / "build" / "carbon_tests")],
                         cwd=REPO_DIR, capture_output=True, text=True)
    return (res.returncode == 0 and "All carbon_tests passed" in res.stdout)

def main():
    parser = argparse.ArgumentParser(description="Step 18 Verification Script")
    parser.add_argument("--out-summary", type=Path, default=REPO_DIR / "evidence/step-18/step18_target_runtime_summary.json")
    args = parser.parse_args()

    print("=" * 80)
    print("Step 18 Acceptance Verification: Material Target Runtime Selection & Replay")
    print("=" * 80)

    ok, test_stdout = run_tests()
    if not ok:
        print("  [FAIL] Step 18 test suite failed!")
        print(test_stdout)
        sys.exit(1)

    print("Step 18 Unit Tests Passed:")
    for line in test_stdout.splitlines():
        if "[step18-test]" in line:
            print("  " + line)

    print("\n--- Gate 1: Categorical Target Distribution & Chi-Square Goodness-of-Fit ---")
    g1_ok = "Statistical goodness-of-fit PASSED" in test_stdout
    print(f"  [Gate 1 {'PASS' if g1_ok else 'FAIL'}]")

    print("\n--- Gate 2: Density Independence of Target Fractions ---")
    g2_ok = "Density independence and library reuse PASSED" in test_stdout
    print(f"  [Gate 2 {'PASS' if g2_ok else 'FAIL'}]")

    print("\n--- Gate 3: Universal Element Library Reuse Across Media ---")
    g3_ok = g2_ok
    print(f"  [Gate 3 {'PASS' if g3_ok else 'FAIL'}]")

    print("\n--- Gate 4: Fail-Closed Missing/Unsupported Target Handling ---")
    g4_ok = "Fail-closed missing target checks PASSED" in test_stdout
    print(f"  [Gate 4 {'PASS' if g4_ok else 'FAIL'}]")

    print("\n--- Gate 5: CPU/GPU Bitwise Equivalence & Diagnostics ---")
    g5_ok = "CPU/GPU equivalence and diagnostics PASSED" in test_stdout
    print(f"  [Gate 5 {'PASS' if g5_ok else 'FAIL'}]")

    print("\nRunning full regression test suite across all previous steps...")
    reg_ok = run_regression_suite()
    print(f"Regression Suite Status: {'PASS' if reg_ok else 'FAIL'}")

    all_pass = (g1_ok and g2_ok and g3_ok and g4_ok and g5_ok and reg_ok)

    summary = {
        "step": 18,
        "name": "material_target_runtime_and_cinel03_replay",
        "overall_passed": all_pass,
        "gates": {
            "gate1_goodness_of_fit": {"passed": g1_ok},
            "gate2_density_independence": {"passed": g2_ok},
            "gate3_library_reuse": {"passed": g3_ok},
            "gate4_fail_closed": {"passed": g4_ok},
            "gate5_cpu_gpu_equivalence": {"passed": g5_ok},
            "full_regression_suite": {"passed": reg_ok}
        }
    }

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nVerification summary written to {args.out_summary}")

    print("=" * 80)
    if all_pass:
        print(">>> ALL 5 GATES PASSED (100%) <<<")
        sys.exit(0)
    else:
        print(">>> VERIFICATION FAILED <<<")
        sys.exit(1)

if __name__ == "__main__":
    main()

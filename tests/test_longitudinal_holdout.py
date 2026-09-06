import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import analyze_longitudinal_holdout as audit

class HoldoutTests(unittest.TestCase):
    def run_case(self, error=0., noise=0., bad_quality=False):
        contract = dict(energy_MeV_u=175, scale=1, histories_per_run=20000,
            density_g_cm3=0.01131606474518776, windows_mm=[[2,20],[20,50],[50,100],[100,120]],
            diagnostic_gates=dict(window_replica_relative_difference_max=.01,
                rebin_abs_relative_error_max=.02, window_abs_relative_error_max=.01),
            limitations=[])
        def read(path, *args, **kwargs):
            if path.name == "campaign.json":
                return json.dumps(contract)
            candidate = "gpu_long" in str(path)
            codes = ["unvalidated_longitudinal_candidate"] if candidate else []
            if bad_quality:
                codes.append("physical_energy_residual_exceeded")
            return json.dumps(dict(accepted=not codes, failures=[dict(code=c) for c in codes]))
        def topas(path):
            return np.full(440, 20000.*(1+noise if "s1" in str(path) else 1-noise))
        def gpu(path):
            return np.full(440, 20000.*(1+error if path.name == "gpu_long" else 1.06))
        with patch.object(Path, "read_text", read), patch.object(audit, "lateral_integral", topas), patch.object(audit, "gpu_curve", gpu), patch.object(audit, "sha", return_value="mock"):
            return audit.analyze(Path("/mock"))

    def test_absolute_normalization(self):
        r = self.run_case()
        self.assertEqual(r["status"], "PASS_PILOT_ONLY")
        self.assertAlmostEqual(r["windows"][0]["gpu_base_relative_error"], .06)
        expected = 440*.01131606474518776*2e-6/1.602176634e-13
        self.assertAlmostEqual(r["integrals_MeV_per_primary"]["topas"]/expected, 1.)

    def test_systematic_error_not_fitted_away(self):
        self.assertEqual(self.run_case(error=.03)["status"], "FAIL_PILOT")

    def test_insufficient_statistics(self):
        self.assertEqual(self.run_case(noise=.02)["status"], "INCONCLUSIVE")

    def test_quality_failure_prevents_pass(self):
        self.assertEqual(self.run_case(bad_quality=True)["status"], "INCONCLUSIVE")

if __name__ == "__main__":
    unittest.main()

"""End-to-end schema-v3 generating-step binding regressions."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ("run event track parent pdg step birth_x_mm birth_y_mm birth_z_mm birth_ke_MeV "
           "pre_x_mm pre_y_mm pre_z_mm post_x_mm post_y_mm post_z_mm edep_MeV pre_ke_MeV "
           "post_ke_MeV weight density_g_cm3 schema_version parent_valid parent_ke_MeV "
           "parent_dir_x parent_dir_y parent_dir_z creator_process_id birth_density_g_cm3 "
           "parent_step_id parent_post_ke_MeV parent_post_dir_x parent_post_dir_y parent_post_dir_z").split()
INTS = set(range(6)) | {21, 22, 27, 29}


class V3Test(unittest.TestCase):
    def run_case(self, mutation=None):
        d = np.zeros((3, 34))
        d[:, :6] = [[0,0,1,0,1000060120,2], [0,0,1,0,1000060120,3], [0,0,2,1,11,1]]
        d[:2, 8:10] = [-1, 10]
        d[2, 8:10] = [1,1]
        d[:, 12] = [0,4,1]
        d[:, 15] = [4,8,3]
        d[:, 16] = [2,2,1]
        d[:, 17:19] = [[10,7], [7,5], [1,0]]
        d[:, 19:21] = 1
        d[:, 21] = 3
        d[:, 23] = -1
        d[:, 28] = 1
        d[:, 29:31] = -1
        d[2, 22:24] = [1,10]
        d[2, 26:28] = [1,1]
        d[2, 29:31] = [2,7]
        d[2, 33] = 1
        if mutation:
            mutation(d)
        with tempfile.TemporaryDirectory() as tmp:
            b = Path(tmp)
            dtype = np.dtype([(name, "<i4" if i in INTS else "<f8") for i,name in enumerate(COLUMNS)])
            raw = np.zeros(3, dtype=dtype)
            for i,name in enumerate(COLUMNS):
                raw[name] = d[:, i]
            raw.tofile(b/"steps.phsp")
            (b/"steps.header").write_text(
                "Number of Original Histories: 1\nNumber of Scored Entries: 3\n"
                f"Number of Bytes per Particle: {dtype.itemsize}\n" +
                "\n".join(f"{'i4' if i in INTS else 'f8'}: {name}" for i,name in enumerate(COLUMNS)))
            np.savetxt(b/"dose.csv", [[0,0,0,5/(2e-6*6.241509074e12)]], delimiter=",")
            result = subprocess.run([sys.executable, str(ROOT/"tools/analyze_electron_deposit_steps.py"),
                "--steps", str(b/"steps.phsp"), "--dose", str(b/"dose.csv"),
                "--histories", "1", "--voxel-volume-mm3", "2", "--format", "topas-binary-le",
                "--slab-bounds-mm", "-1", "1", "-1", "1", "0", "8",
                "--output", str(b/"report.json")], capture_output=True, text=True)
            report = json.loads((b/"report.json").read_text()) if result.returncode == 0 else None
            return result, report

    def test_birth_step_not_exit_step(self):
        result, rep = self.run_case()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(rep["record_schema_version"], 3)
        self.assertTrue(rep["parent_step_binding_verified"])
        self.assertFalse(rep["parent_conditioning_runtime_eligible"])
        self.assertEqual(rep["finite_slab_energy_audit"]["per_family_residuals"][0]["matched_parent_ke_MeV"], 10)
        self.assertEqual(rep["parent_conditioning"]["missing_parent_roots"], 0)

    def test_wrong_parent_step(self):
        result, _ = self.run_case(lambda d: d.__setitem__((2,29), 99))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("parent step missing", result.stderr)

    def test_wrong_parent_energy(self):
        result, _ = self.run_case(lambda d: d.__setitem__((2,30), 8))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("KE mismatch", result.stderr)

    def test_downgraded_version(self):
        result, _ = self.run_case(lambda d: d.__setitem__((slice(None),21), 2))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("schema_version", result.stderr)


if __name__ == "__main__":
    unittest.main()

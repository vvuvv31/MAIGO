import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data/schneider/schneider_section0_c12_delta_longitudinal_v1.csv"
META = REPO / "data/schneider/schneider_section0_c12_delta_longitudinal_v1.metadata.json"


def lookup(energies, fracs, lambdas, energy):
    # Supplemental Python check only; CTest exercises the real C++/SYCL code.
    if not energies[0] <= energy <= energies[-1]:
        return 0.0, 0.0
    if energy <= energies[0]:
        return fracs[0], lambdas[0]
    if energy >= energies[-1]:
        return fracs[-1], lambdas[-1]
    e0 = 0
    ef = 0.0
    if len(energies) > 1:
        if energy == energies[-1]:
            e0, ef = len(energies) - 2, 1.0
        elif energy > energies[0]:
            while e0 + 1 < len(energies) - 1 and energy > energies[e0 + 1]:
                e0 += 1
            de = energies[e0 + 1] - energies[e0]
            ef = (energy - energies[e0]) / de if de > 0 else 0.0
    e1 = e0 + 1 if len(energies) > 1 else e0
    return fracs[e0] + ef * (fracs[e1] - fracs[e0]), lambdas[e0] + ef * (lambdas[e1] - lambdas[e0])


class LongitudinalTableTests(unittest.TestCase):
    def test_compiler_refuses_existing_output(self):
        cmd = [sys.executable, str(REPO / "tools/compile_schneider_delta_longitudinal.py"),
               "--input", "200:unused.csv", "--output", str(DATA), "--metadata", str(META)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Refusing to overwrite", r.stderr)

    def test_compiler_rejects_incomplete_and_flat_scorers(self):
        sys.path.insert(0, str(REPO / "tools"))
        from compile_schneider_delta_longitudinal import lateral_integral, fit_forward
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("# not a 3D scorer\n" * 8 + "0,0,0,1\n")
            with self.assertRaises(ValueError):
                lateral_integral(path)
        with self.assertRaises(ValueError):
            fit_forward((np.arange(440) + .5) * .5, np.ones(440))

    def test_candidate_verifier(self):
        sys.path.insert(0, str(REPO / "tools"))
        from verify_schneider_longitudinal_candidate import verify
        self.assertEqual(verify(DATA)["status"], "unvalidated_diagnostic")
    def test_csv_and_metadata_consistent(self):
        with DATA.open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(
            list(rows[0].keys()),
            ["energy_MeV_per_u", "forward_fraction", "lambda_mm"])
        energies = [float(r["energy_MeV_per_u"]) for r in rows]
        self.assertEqual(energies, sorted(energies))
        for r in rows:
            self.assertGreater(float(r["forward_fraction"]), 0.0)
            self.assertLess(float(r["forward_fraction"]), 0.5)
            self.assertGreater(float(r["lambda_mm"]), 0.0)
        meta = json.loads(META.read_text())
        self.assertEqual(meta["data_filename"], DATA.name)
        h = hashlib.sha256()
        with DATA.open("rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        self.assertEqual(h.hexdigest(), meta["data_sha256"])
        for entry, row in zip(meta["inputs"], rows):
            self.assertAlmostEqual(
                entry["fitted_forward_fraction"], float(row["forward_fraction"]))
            self.assertAlmostEqual(
                entry["fitted_lambda_mm"], float(row["lambda_mm"]))

    def test_lookup_mirrors_device_interpolation(self):
        with DATA.open() as f:
            rows = list(csv.DictReader(f))
        energies = [float(r["energy_MeV_per_u"]) for r in rows]
        fracs = [float(r["forward_fraction"]) for r in rows]
        lambdas = [float(r["lambda_mm"]) for r in rows]
        f, lam = lookup(energies, fracs, lambdas, energies[1])
        self.assertAlmostEqual(f, fracs[1])
        self.assertAlmostEqual(lam, lambdas[1])
        mid = 0.5 * (energies[0] + energies[1])
        f, lam = lookup(energies, fracs, lambdas, mid)
        self.assertAlmostEqual(f, 0.5 * (fracs[0] + fracs[1]))
        self.assertAlmostEqual(lam, 0.5 * (lambdas[0] + lambdas[1]))
        # Out-of-domain requests retain local energy rather than extrapolate.
        self.assertEqual(lookup(energies, fracs, lambdas, energies[0] - 0.1),
                         (0.0, 0.0))
        self.assertEqual(lookup(energies, fracs, lambdas, energies[-1] + 0.1),
                         (0.0, 0.0))

    def test_exponential_range_covers_entrance_scale(self):
        # Mean range must resolve the observed ~15-20 mm entrance decay and
        # stay well inside the scorer so most moved energy is not escaped.
        with DATA.open() as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            lam = float(r["lambda_mm"])
            self.assertGreater(lam, 5.0)
            self.assertLess(lam, 60.0)

    def test_compile_script_reproduces_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "long.csv"
            meta = Path(tmp) / "long.meta.json"
            cmd = [sys.executable, str(REPO / "tools/compile_schneider_delta_longitudinal.py"),
                   "--input", "150:/mnt/sda/wuwei/delta_section0/energy_scan/topas_hu_n1000_e150.csv",
                   "--input", "200:/mnt/sda/wuwei/delta_section0/topas_dose3d_hu_n1000_em.csv",
                   "--input", "225:/mnt/sda/wuwei/delta_section0/energy_scan/topas_hu_n1000_e225.csv",
                   "--kernel-a-scale", "1.2", "--kernel-lambda-scale", "1.3",
                   "--output", str(out), "--metadata", str(meta)]
            subprocess.run(cmd, check=True, cwd=REPO)
            self.assertEqual(out.read_text(), DATA.read_text())

    def test_candidate_manifest_is_diagnostic(self):
        manifest = json.loads(DATA.with_suffix(".candidate.json").read_text())
        self.assertEqual(manifest["status"], "unvalidated_diagnostic")
        self.assertEqual(manifest["scale"], 1)
        self.assertEqual(manifest["data_sha256"], hashlib.sha256(DATA.read_bytes()).hexdigest())
        self.assertEqual(manifest["metadata_sha256"], hashlib.sha256(META.read_bytes()).hexdigest())

if __name__ == "__main__":
    unittest.main()

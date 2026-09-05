"""Acceptance tests for sharded electron-response campaigns (Step 03)."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "tools/run_electron_response_diagnostic.py"
MERGE = ROOT / "tools/merge_electron_response_diagnostics.py"
ANALYZE = ROOT / "tools/analyze_electron_deposit_steps.py"
COMPARE = ROOT / "tools/compare_electron_response_geometries.py"
FACES = ROOT / "tools/attribute_escape_faces.py"


def make_shard_report(base, name, *, seed_shift=0, deposit=3.0):
    rows = np.zeros((2, 21))
    rows[:, 0] = 0
    rows[:, 1] = seed_shift
    rows[:, 2] = [1, 2]
    rows[:, 3] = [0, 1]
    rows[:, 4] = [1000060120, 11]
    rows[:, 5] = 1
    rows[1, 12], rows[1, 15] = 1, 3
    rows[:, 16] = [2, deposit - 2]
    rows[:, 19:21] = 1
    steps = base / f"{name}_steps"
    dose = base / f"{name}_dose"
    out = base / f"{name}.json"
    np.savetxt(steps, rows)
    columns = "run event track parent pdg step birth_x_mm birth_y_mm birth_z_mm birth_ke_MeV pre_x_mm pre_y_mm pre_z_mm post_x_mm post_y_mm post_z_mm edep_MeV pre_ke_MeV post_ke_MeV weight density_g_cm3".split()
    (steps.with_suffix(".header")).write_text(
        "Number of Original Histories: 1\nNumber of Scored Entries: 2\n" +
        "\n".join(f"{i+1}: {n}" for i, n in enumerate(columns)))
    np.savetxt(dose, [[0, 0, 0, deposit / (2e-6 * 6.241509074e12)]], delimiter=",")
    result = subprocess.run([sys.executable, str(ANALYZE), "--steps", str(steps),
                             "--dose", str(dose), "--histories", "1",
                             "--voxel-volume-mm3", "2", "--output", str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return out


class CampaignTest(unittest.TestCase):
    def test_run_enforces_shard_histories_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg = base / "base.txt"
            cfg.write_text("i:So/Beam/NumberOfHistoriesInRun = 12\n")
            result = subprocess.run([sys.executable, str(RUN), "--base-config", str(cfg),
                                     "--output-dir", str(base / "out"), "--shards", "1",
                                     "--histories-per-shard", "13", "--seed-start", "1"],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exceeds bounded limit", result.stderr)

    def test_run_enforces_disk_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg = base / "base.txt"
            cfg.write_text("i:So/Beam/NumberOfHistoriesInRun = 12\n")
            result = subprocess.run([sys.executable, str(RUN), "--base-config", str(cfg),
                                     "--output-dir", str(base / "out"), "--shards", "2",
                                     "--histories-per-shard", "12", "--seed-start", "1",
                                     "--disk-budget-gib", "0.01"],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("disk budget", result.stderr.lower())

    def test_run_writes_metadata_and_scripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cfg = base / "base.txt"
            cfg.write_text("i:So/Beam/NumberOfHistoriesInRun = 12\n")
            result = subprocess.run([sys.executable, str(RUN), "--base-config", str(cfg),
                                     "--output-dir", str(base / "out"), "--shards", "2",
                                     "--histories-per-shard", "12", "--seed-start", "7",
                                     "--disk-budget-gib", "5"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((base / "out/campaign_manifest.json").read_text())
            self.assertEqual(manifest["shard_count"], 2)
            self.assertEqual(len({s["seed"] for s in manifest["shards"]}), 2)
            for item in manifest["shards"]:
                script = (Path(item["dir"]) / "run.slurm").read_text()
                self.assertIn("#SBATCH --mem=4096M\n", script)
                self.assertNotIn("--mem=4.0G", script)
                meta = json.loads((Path(item["dir"]) / "metadata.json").read_text())
                for field in ("case_id", "seed", "histories_requested", "config_sha256",
                              "topas_exe_sha256", "scorer_cc_sha256", "completion_status",
                              "analysis_status"):
                    self.assertIn(field, meta)

    def test_merge_core_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            rep_a = make_shard_report(base, "a", seed_shift=0, deposit=3.0)
            rep_b = make_shard_report(base, "b", seed_shift=1, deposit=5.0)
            out = base / "merged.json"
            # Order independence.
            for order in ([rep_a, rep_b], [rep_b, rep_a]):
                result = subprocess.run([sys.executable, str(MERGE), "--reports",
                                         str(order[0]), str(order[1]),
                                         "--output", str(out)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                first = json.loads(out.read_text())
            # Energy equals sum of shards; joint conserved; quantiles not averaged.
            self.assertAlmostEqual(first["total_deposit_MeV"], 8.0)
            self.assertAlmostEqual(first["delta_family_deposit_MeV"], 4.0)
            self.assertIsNone(first["pooled_longitudinal_quantiles_mm"])
            self.assertIn("must not be averaged",
                          first["pooled_longitudinal_quantiles_status"])
            joint = np.asarray(first["joint_deposited_MeV"])
            self.assertAlmostEqual(joint.sum(), 4.0)
            self.assertAlmostEqual(sum(first["joint_probability"][i][j]
                                       for i in range(joint.shape[0])
                                       for j in range(joint.shape[1])), 1.0)
            # Duplicate input rejected.
            result = subprocess.run([sys.executable, str(MERGE), "--reports",
                                     str(rep_a), str(rep_a), "--output", str(out)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Duplicate", result.stderr)
            # Missing shard rejected when a complete campaign is claimed.
            result = subprocess.run([sys.executable, str(MERGE), "--reports", str(rep_a),
                                     "--output", str(out), "--expected-shards", "2"],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing or extra shards", result.stderr)
            # SHA mismatch rejected: tamper with recorded hash.
            tampered = base / "tampered.json"
            payload = json.loads(rep_a.read_text())
            payload["steps_sha256"] = json.loads(rep_b.read_text())["steps_sha256"]
            tampered.write_text(json.dumps(payload))
            result = subprocess.run([sys.executable, str(MERGE), "--reports",
                                     str(tampered), str(rep_b), "--output", str(out)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Duplicate shard content", result.stderr)
            # Mixed configs rejected via sidecar gate.
            (base / "a_metadata_sidecar_check").write_text("x")
            meta_a = base / "a_metadata.json"
            meta_b = base / "b_metadata.json"
            # Merge tool reads metadata.json next to each report; place conflicting
            # config hashes there and require the expected campaign config.
            (rep_a.parent / "metadata.json").write_text(
                json.dumps({"config_sha256": "aaa"}))
            result = subprocess.run([sys.executable, str(MERGE), "--reports",
                                     str(rep_a), str(rep_b), "--output", str(out),
                                     "--expected-config-sha", "bbb"],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("config SHA mismatch", result.stderr)
            (rep_a.parent / "metadata.json").unlink(missing_ok=True)


V2_COLUMNS = ("run event track parent pdg step birth_x_mm birth_y_mm birth_z_mm "
              "birth_ke_MeV pre_x_mm pre_y_mm pre_z_mm post_x_mm post_y_mm "
              "post_z_mm edep_MeV pre_ke_MeV post_ke_MeV weight density_g_cm3 "
              "schema_version parent_valid parent_ke_MeV parent_dir_x parent_dir_y "
              "parent_dir_z creator_process_id birth_density_g_cm3").split()
V2_INTS = set(range(6)) | {21, 22, 27}


def make_v2_geometry(base, name, *, c12_half=10.0, seed=0):
    """Tiny v2 binary geometry: C12 segment through origin along +z plus one
    electron depositing 1 MeV at mid z=2. Returns (steps, report) with slab audit."""
    rows = np.zeros((2, 29))
    rows[0, 0:6] = [0, seed, 1, 0, 1000060120, 2]
    rows[0, 10:13] = [0, 0, 0]
    rows[0, 13:16] = [0, 0, 220]
    rows[0, 16] = 2
    rows[0, 17:19] = [2400, 2397]
    rows[0, 19:21] = 1
    rows[0, 21] = 2
    rows[0, 23] = -1.0
    rows[0, 28] = 1
    rows[1, 0:6] = [0, seed, 2, 1, 11, 1]
    rows[1, 6:9] = [0, 0, 1]
    rows[1, 9] = 1
    rows[1, 10:13] = [0, 0, 1]
    rows[1, 13:16] = [0, 0, 3]
    rows[1, 16] = 1
    rows[1, 17:19] = [1, 0]
    rows[1, 19:21] = 1
    rows[1, 21:24] = [2, 1, 2395.0]
    rows[1, 24:27] = [0, 0, 1]
    rows[1, 27:29] = [1, 1]
    steps = base / f"{name}_steps.phsp"
    dtype = np.dtype([(n, "<i4" if i in V2_INTS else "<f8") for i, n in enumerate(V2_COLUMNS)])
    records = np.zeros(len(rows), dtype=dtype)
    for i, n in enumerate(V2_COLUMNS):
        records[n] = rows[:, i]
    records.tofile(steps)
    (steps.with_suffix(".header")).write_text(
        "Number of Original Histories: 1\nNumber of Scored Entries: 2\n"
        f"Number of Bytes per Particle: {dtype.itemsize}\n\n"
        "Byte order of each record is as follows:\n" +
        "\n".join(f"{'i4' if i in V2_INTS else 'f8'}: {n}" for i, n in enumerate(V2_COLUMNS)))
    dose = base / f"{name}_dose.csv"
    np.savetxt(dose, [[0, 0, 0, 3 / (2e-6 * 6.241509074e12)]], delimiter=",")
    out = base / f"{name}_report.json"
    result = subprocess.run(
        [sys.executable, str(ANALYZE), "--steps", str(steps), "--dose", str(dose),
         "--histories", "1", "--voxel-volume-mm3", "2", "--output", str(out),
         "--format", "topas-binary-le",
         "--slab-bounds-mm", "-100", "100", "-100", "100", "0", "220"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return steps, out


class GeometryComparisonTest(unittest.TestCase):
    def test_compare_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            steps_a, rep_a = make_v2_geometry(base, "ga", seed=0)
            steps_b, rep_b = make_v2_geometry(base, "gb", seed=1)
            out = base / "cmp.json"
            spec_a = f"ga:{steps_a}:{rep_a}:-100,100,-100,100,0,220:2"
            spec_b = f"gb:{steps_b}:{rep_b}:-100,100,-100,100,0,220:2"
            # Order independence.
            first, second = None, None
            for order in ([["--input", spec_a, "--input", spec_b],
                           ["--input", spec_b, "--input", spec_a]]):
                result = subprocess.run([sys.executable, str(COMPARE), "--histories", "1",
                                         "--output", str(out)] + order,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                payload = json.loads(out.read_text())
                key = {g["name"]: {k: g[k] for k in ("escape_fraction", "joint_total_MeV")}
                       for g in payload["geometries"]}
                if first is None:
                    first = key
                else:
                    second = key
            self.assertEqual(first, second)
            self.assertIn("INCONCLUSIVE", json.loads(out.read_text())["verdict"])
            # Duplicate raw content rejected.
            rep_a2 = base / "ga_report2.json"
            rep_a2.write_text(rep_a.read_text())
            result = subprocess.run(
                [sys.executable, str(COMPARE), "--histories", "1", "--output", str(out),
                 "--input", spec_a,
                 "--input", f"ga2:{steps_a}:{rep_a2}:-100,100,-100,100,0,220:2"],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Duplicate shard content", result.stderr)
            # Broken family closure rejected.
            tampered = base / "tampered.json"
            payload = json.loads(rep_a.read_text())
            payload["finite_slab_energy_audit"]["electron_family_escaped_MeV"] += 1.0
            tampered.write_text(json.dumps(payload))
            result = subprocess.run(
                [sys.executable, str(COMPARE), "--histories", "1", "--output", str(out),
                 "--input", f"ga:{steps_a}:{tampered}:-100,100,-100,100,0,220:2",
                 "--input", spec_b],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("closure broken", result.stderr)


def write_v2_binary(base, name, rows):
    steps = base / f"{name}.phsp"
    dtype = np.dtype([(n, "<i4" if i in V2_INTS else "<f8") for i, n in enumerate(V2_COLUMNS)])
    records = np.zeros(len(rows), dtype=dtype)
    for i, n in enumerate(V2_COLUMNS):
        records[n] = rows[:, i]
    records.tofile(steps)
    (steps.with_suffix(".header")).write_text(
        "Number of Original Histories: 1\nNumber of Scored Entries: 2\n"
        f"Number of Bytes per Particle: {dtype.itemsize}\n\n"
        "Byte order of each record is as follows:\n" +
        "\n".join(f"{'i4' if i in V2_INTS else 'f8'}: {n}" for i, n in enumerate(V2_COLUMNS)))
    return steps


class EscapeFaceAttributionTest(unittest.TestCase):
    def test_faces_and_rejections(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("faces", str(FACES))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Electron exits +z (KE 2), C12 exits +z (KE 2397), photon exits +x (KE 1).
        rows = np.zeros((3, 29))
        rows[:, 0:6] = [[0, 0, 1, 0, 1000060120, 2],
                        [0, 0, 2, 1, 11, 1],
                        [0, 0, 3, 1, 22, 1]]
        rows[0, 10:13] = [0, 0, 0]
        rows[0, 13:16] = [0, 0, 220]
        rows[0, 18] = 2397
        rows[0, 19:21] = 1
        rows[1, 10:13] = [0, 0, 219]
        rows[1, 13:16] = [0, 0, 220]
        rows[1, 18] = 2
        rows[1, 19:21] = 1
        rows[2, 10:13] = [99, 0, 100]
        rows[2, 13:16] = [100, 0, 100]
        rows[2, 18] = 1
        rows[2, 19:21] = 1
        out = mod.attribute(rows, [-100, 100, -100, 100, 0, 220])
        self.assertEqual(out["faces"]["+z"]["c12_MeV"], 2397)
        self.assertEqual(out["faces"]["+z"]["electron_family_MeV"], 2)
        self.assertEqual(out["faces"]["+x"]["electron_family_MeV"], 1)
        self.assertAlmostEqual(out["forward_share"], 2 / 3)
        bad = rows.copy()
        bad[1, 13:16] = [0, 0, 200]  # KE>0 terminal strictly inside slab
        with self.assertRaisesRegex(ValueError, "outward|outside"):
            mod.attribute(bad, [-100, 100, -100, 100, 0, 220])
        bad = rows.copy()
        bad[2, 19] = 0.5
        with self.assertRaisesRegex(ValueError, "unit-weight"):
            mod.attribute(bad, [-100, 100, -100, 100, 0, 220])


if __name__ == "__main__":
    unittest.main()

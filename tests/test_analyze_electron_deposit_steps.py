"""Small synthetic acceptance/rejection tests for the diagnostic, not physics."""
import json
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "tools/analyze_electron_deposit_steps.py"
SPEC = importlib.util.spec_from_file_location("electron_audit", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class ElectronDepositAuditTest(unittest.TestCase):
    def run_case(self, *, parent=1, histories=1, dose_scale=1.0, entries=2, bad_schema=False, binary=False):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # Primary deposits 2 MeV; its electron deposits 1 MeV between
            # z=1 and z=3, born at z=0. All units match the 21-column scorer.
            rows = np.zeros((2, 21))
            rows[:, 2] = [1, 2]
            rows[:, 3] = [0, parent]
            rows[:, 4] = [1000060120, 11]
            rows[:, 5] = 1
            rows[1, 12], rows[1, 15] = 1, 3
            rows[:, 16] = [2, 1]
            rows[:, 19:21] = 1
            np.savetxt(base / "steps", rows)
            columns = "run event track parent pdg step birth_x_mm birth_y_mm birth_z_mm birth_ke_MeV pre_x_mm pre_y_mm pre_z_mm post_x_mm post_y_mm post_z_mm edep_MeV pre_ke_MeV post_ke_MeV weight density_g_cm3".split()
            if bad_schema:
                columns[16] = "wrong_edep"
            if binary:
                dtype = np.dtype([(name, "<i4" if i < 6 else "<f8") for i, name in enumerate(columns)])
                records = np.zeros(2, dtype=dtype)
                for i, name in enumerate(columns):
                    records[name] = rows[:, i]
                records.tofile(base / "steps")
            description = ("Number of Bytes per Particle: 144\n" if binary else "")
            description += "\n".join(
                f"{('i4' if i < 6 else 'f8') if binary else i+1}: {name}"
                for i, name in enumerate(columns))
            (base / "steps.header").write_text(
                f"Number of Original Histories: {histories}\nNumber of Scored Entries: {entries}\n" +
                description)
            np.savetxt(base / "dose", [[0, 0, 0, dose_scale*3/(2e-6*6.241509074e12)]], delimiter=",")
            result = subprocess.run([sys.executable, str(SCRIPT),
                "--steps", str(base / "steps"), "--dose", str(base / "dose"),
                "--histories", str(histories), "--voxel-volume-mm3", "2",
                "--output", str(base / "report.json"),
                "--format", "topas-binary-le" if binary else "ascii"], capture_output=True, text=True)
            report = json.loads((base / "report.json").read_text()) if result.returncode == 0 else None
            return result, report

    def test_closed_ancestry_offsets(self):
        result, report = self.run_case()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["primary_local_MeV"], 2)
        self.assertEqual(report["delta_family_deposit_MeV"], 1)
        self.assertEqual(report["longitudinal_mid_mm"], [2]*4)
        self.assertAlmostEqual(report["dose3d_over_steps"], 1)
        self.assertAlmostEqual(np.asarray(report["joint_deposited_MeV"]).sum(), 1)

    def test_missing_ancestor(self):
        result, _ = self.run_case(parent=9)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing ancestor", result.stderr)

    def test_incomplete_events(self):
        result, _ = self.run_case(histories=2)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Incomplete event coverage", result.stderr)

    def test_energy_mismatch(self):
        result, _ = self.run_case(dose_scale=1.02)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("closure failed", result.stderr)

    def test_truncation(self):
        result, _ = self.run_case(entries=3)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Truncated ntuple", result.stderr)

    def test_column_schema(self):
        result, _ = self.run_case(bad_schema=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("schema", result.stderr)

    def test_binary_matches_ascii(self):
        result, report = self.run_case(binary=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        _, ascii_report = self.run_case()
        for field in ["joint_deposited_MeV", "longitudinal_mid_mm", "total_deposit_MeV"]:
            self.assertEqual(report[field], ascii_report[field])

    def test_binary_truncation(self):
        result, _ = self.run_case(binary=True, entries=3)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("binary byte count mismatch", result.stderr)

    def test_escape_accounting_and_rejections(self):
        rows = np.zeros((2, 21))
        rows[:, 2] = [1, 2]
        rows[:, 3] = [0, 1]
        rows[:, 4] = [1000060120, 11]
        rows[:, 5] = 1
        rows[:, 9] = [10, 3]
        rows[:, 12] = [0, 1]
        rows[:, 15] = 4
        rows[:, 16] = [2, 1]
        rows[:, 17] = [10, 3]
        rows[:, 18] = [5, 2]
        rows[:, 19:21] = 1
        root_key = (0, 0, 2)
        tracks = {root_key: (1, 11, np.array([0, 0, 1]), 3)}
        def root(key):
            return root_key if key[2] == 2 else None
        report = AUDIT.audit_slab_energy(rows, root, tracks, [-1, 1, -1, 1, 0, 4])
        self.assertEqual(report["electron_family_escaped_MeV"], 2)
        self.assertEqual(report["electron_relative_residual"], 0)
        self.assertEqual(report["global_relative_residual"], 0)
        # New gates: per-event and per-family detail with worst-family identity.
        self.assertEqual(len(report["per_event_residuals"]), 1)
        self.assertEqual(report["per_event_residuals"][0]["global_relative_residual"], 0)
        self.assertEqual(report["per_event_residuals"][0]["electron_birth_MeV"], 3)
        self.assertEqual(report["per_event_residuals"][0]["electron_relative_residual"], 0)
        self.assertEqual(len(report["per_family_residuals"]), 1)
        self.assertEqual(report["worst_family"]["track"], 2)
        self.assertEqual(report["finite_slab_scope"] if "finite_slab_scope" in report else 0, 0)
        # Descendants must not duplicate their root's birth energy.
        descendant = rows[1].copy()
        descendant[2:4] = [3, 2]
        descendant[9], descendant[16:19] = 1, [.5, 1, .5]
        extended = np.vstack([rows, descendant])
        extended[1, 16], extended[1, 18] = .5, 1.5
        descendant_tracks = dict(tracks)
        descendant_tracks[(0, 0, 3)] = (2, 11, np.array([0, 0, 1]), 1)
        r = AUDIT.audit_slab_energy(extended, lambda key: root_key if key[2] != 1 else None,
                                   descendant_tracks, [-1, 1, -1, 1, 0, 4])
        self.assertEqual(r["per_event_residuals"][0]["electron_birth_MeV"], 3)
        self.assertEqual(r["per_event_residuals"][0]["electron_relative_residual"], 0)
        # Global event energies close while electron errors cancel across events.
        paired = np.vstack([rows, rows])
        paired[2:, 1] = 1
        paired[:, 16] += [.25, -.25, -.25, .25]
        event_tracks = dict(tracks)
        event_tracks[(0, 1, 2)] = tracks[root_key]
        with self.assertRaisesRegex(ValueError, "in event.*electron"):
            AUDIT.audit_slab_energy(paired, lambda k: k if k[2] == 2 else None,
                                   event_tracks, [-1, 1, -1, 1, 0, 4])
        # Two families in one event have opposite errors; event sum still closes.
        paired = np.vstack([rows, rows[1]])
        paired[2, 2] = 3
        paired[0, 18] = 2
        family_tracks = dict(tracks)
        family_tracks[root_key] = (1, 11, np.array([0, 0, 1]), 4)
        family_tracks[(0, 0, 3)] = (1, 11, np.array([0, 0, 1]), 2)
        with self.assertRaisesRegex(ValueError, "in family"):
            AUDIT.audit_slab_energy(paired, lambda k: k if k[2] != 1 else None,
                                   family_tracks, [-1, 1, -1, 1, 0, 4])
        bad = rows.copy()
        bad[1, 15] = 2
        with self.assertRaisesRegex(ValueError, "inside slab"):
            AUDIT.audit_slab_energy(bad, root, tracks, [-1, 1, -1, 1, 0, 4])
        bad = rows.copy()
        bad[1, 18] = 1
        with self.assertRaisesRegex(ValueError, "Escape closure failed"):
            AUDIT.audit_slab_energy(bad, root, tracks, [-1, 1, -1, 1, 0, 4])
        bad = rows.copy()
        bad[1, 5] = 2
        with self.assertRaisesRegex(ValueError, "first-step gap"):
            AUDIT.audit_slab_energy(bad, root, tracks, [-1, 1, -1, 1, 0, 4])

    def _write_case(self, base, rows, *, histories, entries, dose_scale=1.0, binary=False):
        columns = "run event track parent pdg step birth_x_mm birth_y_mm birth_z_mm birth_ke_MeV pre_x_mm pre_y_mm pre_z_mm post_x_mm post_y_mm post_z_mm edep_MeV pre_ke_MeV post_ke_MeV weight density_g_cm3".split()
        if binary:
            dtype = np.dtype([(name, "<i4" if i < 6 else "<f8") for i, name in enumerate(columns)])
            records = np.zeros(len(rows), dtype=dtype)
            for i, name in enumerate(columns):
                records[name] = rows[:, i]
            records.tofile(base / "steps")
        else:
            np.savetxt(base / "steps", rows)
        description = ("Number of Bytes per Particle: 144\n" if binary else "")
        description += "\n".join(
            f"{('i4' if i < 6 else 'f8') if binary else i+1}: {name}"
            for i, name in enumerate(columns))
        (base / "steps.header").write_text(
            f"Number of Original Histories: {histories}\nNumber of Scored Entries: {entries}\n" +
            description)
        total = float(np.sum(rows[:, 16] * rows[:, 19]))
        np.savetxt(base / "dose", [[0, 0, 0, dose_scale*total/(2e-6*6.241509074e12)]], delimiter=",")
        result = subprocess.run([sys.executable, str(SCRIPT),
            "--steps", str(base / "steps"), "--dose", str(base / "dose"),
            "--histories", str(histories), "--voxel-volume-mm3", "2",
            "--output", str(base / "report.json"),
            "--format", "topas-binary-le" if binary else "ascii"], capture_output=True, text=True)
        report = None
        if result.returncode == 0:
            report = json.loads((base / "report.json").read_text())
            self.assertNotIn("NaN", (base / "report.json").read_text())
            self.assertNotIn("Infinity", (base / "report.json").read_text())
        return result, report

    @staticmethod
    def _base_rows():
        rows = np.zeros((2, 21))
        rows[:, 2] = [1, 2]
        rows[:, 3] = [0, 1]
        rows[:, 4] = [1000060120, 11]
        rows[:, 5] = 1
        rows[1, 12], rows[1, 15] = 1, 3
        rows[:, 16] = [2, 1]
        rows[:, 19:21] = 1
        return rows

    def test_per_track_birth_consistency_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            rows = self._base_rows()
            # Two rows for the same electron track disagree on birth position.
            extra = rows[1:2].copy()
            extra[0, 6] = 5.0
            stacked = np.vstack([rows, extra])
            result, _ = self._write_case(base, stacked, histories=1, entries=3)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Inconsistent per-track", result.stderr)

    def test_duplicate_step_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            rows = self._base_rows()
            extra = rows[1:2].copy()
            stacked = np.vstack([rows, extra])
            result, _ = self._write_case(base, stacked, histories=1, entries=3)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue("Duplicate step" in result.stderr
                            or "Inconsistent per-track" in result.stderr
                            or "Missing/duplicate" in result.stderr)

    def test_primary_count_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            rows = self._base_rows()
            # Second primary in the same event: two parent==0 tracks.
            rows[1, 3] = 0
            rows[1, 4] = 1000060120
            result, _ = self._write_case(base, rows, histories=1, entries=2)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("primary tracks", result.stderr)

    def test_negative_ke_rejected(self):
        import tempfile
        for col, label in ((9, "birth KE"), (17, "pre-step KE"), (18, "post-step KE")):
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                rows = self._base_rows()
                rows[1, col] = -1.0
                result, _ = self._write_case(base, rows, histories=1, entries=2)
                self.assertNotEqual(result.returncode, 0, label)
                self.assertIn("KE", result.stderr)

    def test_nonfinite_ke_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            rows = self._base_rows()
            rows[1, 9] = np.inf
            result, _ = self._write_case(base, rows, histories=1, entries=2)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue("finite" in result.stderr or "KE" in result.stderr)

    def test_uncomputed_slab_field_is_null(self):
        result, report = self.run_case()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNone(report["finite_slab_energy_audit"])
        self.assertIn("not computed", report["finite_slab_energy_audit_status"])

    def test_slab_rejects_nonunit_weight_and_bad_pdg(self):
        rows = np.zeros((2, 21))
        rows[:, 2] = [1, 2]
        rows[:, 3] = [0, 1]
        rows[:, 4] = [1000060120, 11]
        rows[:, 5] = 1
        rows[:, 9] = [10, 3]
        rows[:, 12] = [0, 1]
        rows[:, 15] = 4
        rows[:, 16] = [2, 1]
        rows[:, 17] = [10, 3]
        rows[:, 18] = [5, 2]
        rows[:, 19:21] = 1
        root_key = (0, 0, 2)
        tracks = {root_key: (1, 11, np.array([0, 0, 1]), 3)}
        def root(key):
            return root_key if key[2] == 2 else None
        bad = rows.copy()
        bad[:, 19] = 0.5
        with self.assertRaisesRegex(ValueError, "unit-weight"):
            AUDIT.audit_slab_energy(bad, root, tracks, [-1, 1, -1, 1, 0, 4])
        bad = rows.copy()
        bad[1, 4] = 211
        with self.assertRaisesRegex(ValueError, "EM C12"):
            AUDIT.audit_slab_energy(bad, root, tracks, [-1, 1, -1, 1, 0, 4])
        bad = rows.copy()
        bad[:, 18] = [-1, 2]
        with self.assertRaisesRegex(ValueError, "non-negative|Negative terminal"):
            AUDIT.audit_slab_energy(bad, root, tracks, [-1, 1, -1, 1, 0, 4])

    def test_event_cancellation_does_not_pass(self):
        rows = np.zeros((4, 21))
        rows[:, 3] = [0, 1, 0, 1]
        rows[:, 2] = [1, 2, 1, 2]
        rows[:, 1] = [0, 0, 1, 1]
        rows[:, 4] = [1000060120, 11, 1000060120, 11]
        rows[:, 5] = 1
        rows[:, 9] = [10, 3, 10, 3]
        rows[:, 12] = [0, 1, 0, 1]
        rows[:, 15] = 4
        rows[:, 16] = [2, 1, 2, 1]
        rows[:, 17] = [10, 3, 10, 3]
        # Event 0 escapes 0.5 MeV too little, event 1 escapes 0.5 too much
        # relative to birth=3 (deposit 1): residuals +-0.167 cancel globally.
        rows[:, 18] = [5, 1.5, 5, 2.5]
        rows[:, 19:21] = 1
        tracks = {(0, 0, 2): (1, 11, np.array([0, 0, 1]), 3),
                  (0, 1, 2): (1, 11, np.array([0, 0, 1]), 3)}
        def root(key):
            if key[2] == 2:
                return (key[0], key[1], 2)
            return None
        with self.assertRaisesRegex(ValueError, "Escape closure failed in event"):
            AUDIT.audit_slab_energy(rows, root, tracks, [-1, 1, -1, 1, 0, 4])


V2_COLUMNS = ("run event track parent pdg step birth_x_mm birth_y_mm birth_z_mm "
              "birth_ke_MeV pre_x_mm pre_y_mm pre_z_mm post_x_mm post_y_mm "
              "post_z_mm edep_MeV pre_ke_MeV post_ke_MeV weight density_g_cm3 "
              "schema_version parent_valid parent_ke_MeV parent_dir_x parent_dir_y "
              "parent_dir_z creator_process_id birth_density_g_cm3").split()
V2_INT_POSITIONS = set(range(6)) | {21, 22, 27}


def _v2_rows(*, parent_dir=(0, 0, 1), parent_ke=2400.0, parent_valid=1,
             creator=1, deposit_z=(1, 3), include_c12=True,
             c12_ke=(2400.0, 2390.0), c12_shift=(0.0, 0.0, 0.0),
             scorer_parent_dir=None, scorer_parent_ke=None):
    """Primary C12 + electron v2 rows. The C12 segment runs through the
    electron birth (origin) along parent_dir so the offline birth match
    recovers that direction; scorer-cached parent_* still carry parent_ke."""
    rows = []
    if include_c12:
        c12 = np.zeros(29)
        c12[0:6] = [0, 0, 1, 0, 1000060120, 2]
        direction = np.asarray(parent_dir, dtype=float)
        direction = direction / np.linalg.norm(direction)
        shift = np.asarray(c12_shift, dtype=float)
        c12[10:13] = -10.0 * direction + shift
        c12[13:16] = 10.0 * direction + shift
        c12[16] = 2
        c12[17], c12[18] = c12_ke
        c12[19:21] = 1
        c12[21] = 2
        c12[23] = -1.0
        c12[28] = 1
        rows.append(c12)
    ele = np.zeros(29)
    ele[0:6] = [0, 0, 2, 1, 11, 1]
    ele[12], ele[15] = deposit_z
    ele[16] = 1
    ele[19:21] = 1
    ele[21] = 2
    ele[22] = parent_valid
    cached_ke = scorer_parent_ke if scorer_parent_ke is not None else parent_ke
    cached_dir = scorer_parent_dir if scorer_parent_dir is not None else parent_dir
    ele[23] = cached_ke if parent_valid else -1.0
    ele[24:27] = cached_dir if parent_valid else (0, 0, 0)
    ele[27] = creator
    ele[28] = 1
    rows.append(ele)
    out = np.array(rows)
    # Primary local row first (track 1), electron second (track 2).
    return out


def _write_v2(base, rows, *, histories=1, binary=False, name="steps"):
    import tempfile  # noqa: F401 (kept local for symmetry with v1 helpers)
    steps = base / name
    dose = base / f"{name}_dose"
    out = base / f"{name}_report.json"
    if binary:
        dtype = np.dtype([(n, "<i4" if i in V2_INT_POSITIONS else "<f8")
                          for i, n in enumerate(V2_COLUMNS)])
        records = np.zeros(len(rows), dtype=dtype)
        for i, n in enumerate(V2_COLUMNS):
            records[n] = rows[:, i]
        records.tofile(steps)
        header_body = "\n".join(
            f"{'i4' if i in V2_INT_POSITIONS else 'f8'}: {n}"
            for i, n in enumerate(V2_COLUMNS))
        (steps.with_suffix(".header")).write_text(
            f"Number of Original Histories: {histories}\n"
            f"Number of Scored Entries: {len(rows)}\n"
            f"Number of Bytes per Particle: {dtype.itemsize}\n" + header_body)
    else:
        np.savetxt(steps, rows)
        (steps.with_suffix(".header")).write_text(
            f"Number of Original Histories: {histories}\n"
            f"Number of Scored Entries: {len(rows)}\n" +
            "\n".join(f"{i+1}: {n}" for i, n in enumerate(V2_COLUMNS)))
    total = float(np.sum(rows[:, 16] * rows[:, 19]))
    np.savetxt(dose, [[0, 0, 0, total / (2e-6 * 6.241509074e12)]], delimiter=",")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--steps", str(steps), "--dose", str(dose),
         "--histories", str(histories), "--voxel-volume-mm3", "2",
         "--output", str(out),
         "--format", "topas-binary-le" if binary else "ascii"],
        capture_output=True, text=True)
    report = json.loads(out.read_text()) if result.returncode == 0 else None
    return result, report


class ElectronDepositV2Test(unittest.TestCase):
    def test_v1_backward_compat(self):
        result, report = ElectronDepositAuditTest().run_case()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["record_schema_version"], 1)
        self.assertIsNone(report["parent_conditioning"])

    def test_v2_happy_path_ascii(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            result, report = _write_v2(Path(tmp), _v2_rows())
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(report["record_schema_version"], 2)
            pc = report["parent_conditioning"]
            self.assertEqual(pc["missing_parent_roots"], 0)
            self.assertEqual(pc["missing_parent_root_rate"], 0)
            self.assertEqual(pc["projected_status"], "computed")
            # Deposit midpoint z=2 with parent +z: projected longitude is 2.
            self.assertEqual(pc["projected"]["projected_longitudinal_mm"], [2.0] * 4)
            self.assertEqual(pc["creator_process_id_counts"], {"1": 1})
            # Birth-time match recovers the C12 segment midpoint KE (2395),
            # not the scorer-cached exit proxy (2400).
            self.assertAlmostEqual(pc["matched_parent_ke_quantiles_MeV"][1], 2395.0)
            self.assertLess(pc["match_distance_mm_quantiles"][1], 1e-6)

    def test_v2_birth_match_ignores_exit_proxy(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            # Scorer cache claims parent +x at 100 MeV, but the true C12
            # trajectory through the birth runs along +z: projection must
            # follow the birth match (+z), not the exit proxy (+x).
            rows = _v2_rows(parent_dir=(0, 0, 1), scorer_parent_dir=(1, 0, 0),
                            scorer_parent_ke=100.0)
            result, report = _write_v2(Path(tmp), rows, name="proxy")
            self.assertEqual(result.returncode, 0, result.stderr)
            proj = report["parent_conditioning"]["projected"]
            self.assertEqual(proj["projected_longitudinal_mm"], [2.0] * 4)

    def test_v2_binary_matches_ascii(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            result, binary_report = _write_v2(base, _v2_rows(), binary=True, name="bin")
            self.assertEqual(result.returncode, 0, result.stderr)
            _, ascii_report = _write_v2(base, _v2_rows(), name="asc")
            self.assertEqual(binary_report["parent_conditioning"],
                             ascii_report["parent_conditioning"])
            self.assertEqual(binary_report["total_deposit_MeV"],
                             ascii_report["total_deposit_MeV"])

    def test_v2_rotation_projections(self):
        import tempfile
        # Electron born at origin, deposit midpoint at (0, 0, 2).
        cases = {
            "plus_z": ((0, 0, 1), 2.0, 0.0),
            "plus_x": ((1, 0, 0), 0.0, 2.0),
            "plus_y": ((0, 1, 0), 0.0, 2.0),
            "minus_z": ((0, 0, -1), -2.0, 0.0),
            "oblique": ((1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)),
                        2.0 / np.sqrt(3), np.sqrt(8.0 / 3.0)),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for label, (direction, expected_long, expected_rad) in cases.items():
                result, report = _write_v2(
                    Path(tmp), _v2_rows(parent_dir=direction), name=f"rot_{label}")
                self.assertEqual(result.returncode, 0, f"{label}: {result.stderr}")
                proj = report["parent_conditioning"]["projected"]
                self.assertAlmostEqual(proj["projected_longitudinal_mm"][1],
                                       expected_long, places=9, msg=label)
                self.assertAlmostEqual(proj["projected_radial_median_mm"],
                                       expected_rad, places=9, msg=label)

    def test_v2_unknown_schema_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            rows = np.zeros((2, 25))  # neither v1 (21) nor v2 (29)
            rows[:, 16] = [2, 1]
            rows[:, 19:21] = 1
            np.savetxt(base / "steps", rows)
            (base / "steps.header").write_text(
                "Number of Original Histories: 1\nNumber of Scored Entries: 2\n" +
                "\n".join(f"{i+1}: col{i}" for i in range(25)))
            np.savetxt(base / "dose", [[0, 0, 0, 1.0]], delimiter=",")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--steps", str(base / "steps"),
                 "--dose", str(base / "dose"), "--histories", "1",
                 "--voxel-volume-mm3", "2", "--output", str(base / "r.json")],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unknown ntuple schema", result.stderr)

    def test_v2_missing_parent_counted_not_zero_filled(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            # C12 trajectory far from the electron birth: no birth match.
            result, report = _write_v2(
                Path(tmp), _v2_rows(c12_shift=(50.0, 0.0, 0.0)), name="noparent")
            self.assertEqual(result.returncode, 0, result.stderr)
            pc = report["parent_conditioning"]
            self.assertEqual(pc["missing_parent_roots"], 1)
            self.assertEqual(pc["missing_parent_root_rate"], 1.0)
            self.assertIsNone(pc["projected"])
            self.assertIn("no valid-parent", pc["projected_status"])

    def test_v2_bad_parent_direction_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            rows = _v2_rows(parent_dir=(0, 0, 0.5))  # non-unit
            result, _ = _write_v2(Path(tmp), rows, name="baddir")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unit", result.stderr)

    def test_v2_wrong_schema_version_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            rows = _v2_rows()
            rows[:, 21] = 3
            result, _ = _write_v2(Path(tmp), rows, name="badver")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("schema_version", result.stderr)


if __name__ == "__main__":
    unittest.main()

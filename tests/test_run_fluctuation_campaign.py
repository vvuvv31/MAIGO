"""End-to-end tests for running and collecting a synthetic TOPAS campaign."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "startup/proton_water/generate_fluctuation_campaign.py"
RUNNER = ROOT / "startup/proton_water/run_fluctuation_campaign.py"
FINALIZER = ROOT / "startup/proton_water/finalize_fluctuation_campaign.py"
COMPILER = ROOT / "startup/package_tools/compile_energy_loss_fluctuation_grid.py"


FAKE_TOPAS = r'''#!/usr/bin/env python3
from pathlib import Path
import os
import re
import sys
import time

time.sleep(float(os.environ.get("FAKE_TOPAS_DELAY", "0")))

config = Path(sys.argv[1]).read_text(encoding="utf-8")
def value(pattern):
    match = re.search(pattern, config)
    if match is None:
        raise SystemExit(f"missing config pattern: {pattern}")
    return match.group(1)

histories = int(value(r"NumberOfHistoriesInRun = (\d+)"))
energy = float(value(r"BeamEnergy = ([0-9.eE+-]+) MeV"))
z = int(value(r"ProjectileZ = (\d+)"))
a = int(value(r"ProjectileA = (\d+)"))
path_mm = 2.0 * float(value(r"Ge/Slab/HLZ = ([0-9.eE+-]+) mm"))
stem = value(r'OutputFile = "([^"]+)"')
columns = [
    "Run ID", "Event ID", "Thread ID", "Primary Track ID", "Atomic Number Z",
    "Mass Number A", "Material Name", "Entry Kinetic Energy (MeV)",
    "Exit Kinetic Energy (MeV)", "Primary Kinetic Energy Loss (MeV)",
    "Primary Local Deposit (MeV)", "Primary Path Length (mm)",
    "Primary Step Count", "Material Consistent", "Completed", "Completion Status",
]
header = [f"Number of Original Histories: {histories}",
          f"Number of Scored Entries: {histories}", "",
          "Columns of data are as follows:"]
header.extend(f"{i}: {name}" for i, name in enumerate(columns, 1))
Path(f"{stem}.header").write_text("\n".join(header) + "\n", encoding="utf-8")
rows = []
for event in range(histories):
    loss = 0.05 + 0.01 * event
    rows.append(f"0 {event} 0 1 {z} {a} G4_WATER {energy:g} {energy-loss:g} "
                f"{loss:g} {loss:g} {path_mm:g} 1 1 1 exited")
Path(f"{stem}.phsp").write_text("\n".join(rows) + "\n", encoding="utf-8")
print("Welcome to TOPAS, Tool for Particle Simulation (Version 4.2.p3)")
print(" Geant4 version Name: geant4-11-03-patch-02 [MT]")
print("Fluctuations of dE/dx are enabled                  1")
print("TOPAS run sequence complete.")
'''


class RunFluctuationCampaignTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.g4_data = self.root / "G4DATA"
        self.g4_data.mkdir()
        self.topas = self.root / "topas"
        self.topas.write_text(FAKE_TOPAS, encoding="utf-8")
        self.topas.chmod(self.topas.stat().st_mode | stat.S_IXUSR)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def generate(self, *, topas_version: str = "4.2.p3") -> Path:
        campaign = self.root / f"campaign_{topas_version.replace('.', '_')}"
        result = subprocess.run(
            [
                sys.executable, str(GENERATOR), "--output-dir", str(campaign),
                "--projectile-z", "1", "--projectile-a", "1",
                "--energies-mevu", "100,200",
                "--areal-densities-g-per-cm2", "0.01,0.02",
                "--histories", "4", "--threads", "1",
                "--seed-base", "2026082200", "--topas-version", topas_version,
                "--geant4-version", "11.3.2",
            ],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return campaign

    def run_campaign(
        self, campaign: Path, mode: str = "--run"
    ) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable, str(RUNNER), mode, "--campaign-dir", str(campaign),
            "--topas-bin", str(self.topas),
        ]
        if mode == "--run":
            command.extend(["--g4-data-dir", str(self.g4_data)])
        return subprocess.run(command, text=True, capture_output=True, check=False)

    def test_runs_collects_and_records_versions_for_all_points(self) -> None:
        campaign = self.generate()
        result = self.run_campaign(campaign)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("campaign points prepared: 4", result.stdout)
        point_files = sorted(campaign.glob("runs/*/fluctuation_point.json"))
        result_files = sorted(campaign.glob("runs/*/run_result.json"))
        self.assertEqual((len(point_files), len(result_files)), (4, 4))
        for path in result_files:
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                record["actual_versions"],
                {"TOPAS": "4.2.p3", "Geant4": "11.3.2"},
            )
            self.assertEqual(len(record["topas_binary"]["sha256"]), 64)

    def test_collect_only_recreates_prepared_records(self) -> None:
        campaign = self.generate()
        first = json.loads(
            (campaign / "campaign_manifest.json").read_text(encoding="utf-8")
        )["points"][0]["label"]
        initial = subprocess.run(
            [
                sys.executable, str(RUNNER), "--run", "--campaign-dir", str(campaign),
                "--topas-bin", str(self.topas), "--g4-data-dir", str(self.g4_data),
                "--point", first,
            ],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(initial.returncode, 0, initial.stderr)
        run_dir = campaign / "runs" / first
        (run_dir / "fluctuation_point.json").unlink()
        (run_dir / "run_result.json").unlink()
        collected = subprocess.run(
            [
                sys.executable, str(RUNNER), "--collect-only",
                "--campaign-dir", str(campaign), "--topas-bin", str(self.topas),
                "--point", first,
            ],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(collected.returncode, 0, collected.stderr)
        self.assertTrue((run_dir / "fluctuation_point.json").is_file())

    def test_rejects_runtime_version_mismatch(self) -> None:
        campaign = self.generate(topas_version="9.9")
        result = self.run_campaign(campaign)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("runtime versions", result.stderr)
        self.assertFalse(any(campaign.glob("runs/*/fluctuation_point.json")))

    def test_rejects_config_hash_mismatch_before_running(self) -> None:
        campaign = self.generate()
        config = next(campaign.glob("runs/*/run.txt"))
        config.write_text(config.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
        result = self.run_campaign(campaign)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SHA-256 mismatch", result.stderr)
        self.assertFalse(any(campaign.glob("runs/*/topas.log")))

    def test_disjoint_points_can_run_concurrently(self) -> None:
        campaign = self.generate()
        points = json.loads(
            (campaign / "campaign_manifest.json").read_text(encoding="utf-8")
        )["points"]
        environment = os.environ.copy()
        environment["FAKE_TOPAS_DELAY"] = "0.4"

        def command(label: str) -> list[str]:
            return [
                sys.executable, str(RUNNER), "--run",
                "--campaign-dir", str(campaign),
                "--topas-bin", str(self.topas),
                "--g4-data-dir", str(self.g4_data),
                "--point", label,
            ]

        first = subprocess.Popen(
            command(points[0]["label"]), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
        )
        time.sleep(0.1)
        second = subprocess.run(
            command(points[1]["label"]), text=True,
            capture_output=True, check=False, env=environment,
        )
        first_stdout, first_stderr = first.communicate(timeout=5)
        self.assertEqual(first.returncode, 0, first_stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("campaign points prepared: 1", first_stdout)
        self.assertIn("campaign points prepared: 1", second.stdout)

    def test_same_point_concurrent_run_is_rejected(self) -> None:
        campaign = self.generate()
        label = json.loads(
            (campaign / "campaign_manifest.json").read_text(encoding="utf-8")
        )["points"][0]["label"]
        command = [
            sys.executable, str(RUNNER), "--run",
            "--campaign-dir", str(campaign),
            "--topas-bin", str(self.topas),
            "--g4-data-dir", str(self.g4_data),
            "--point", label,
        ]
        environment = os.environ.copy()
        environment["FAKE_TOPAS_DELAY"] = "0.5"
        first = subprocess.Popen(
            command, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=environment,
        )
        time.sleep(0.1)
        duplicate = subprocess.run(
            command, text=True, capture_output=True,
            check=False, env=environment,
        )
        first_stdout, first_stderr = first.communicate(timeout=5)
        self.assertEqual(first.returncode, 0, first_stderr)
        self.assertIn("campaign points prepared: 1", first_stdout)
        self.assertNotEqual(duplicate.returncode, 0)
        self.assertIn("another process owns point lock", duplicate.stderr)

    def test_finalizes_complete_campaign_into_runtime_package(self) -> None:
        campaign = self.generate()
        run = self.run_campaign(campaign)
        self.assertEqual(run.returncode, 0, run.stderr)
        output = self.root / "runtime.csv"
        metadata = self.root / "runtime.metadata.json"
        audit = self.root / "runtime.audit.json"
        result = subprocess.run(
            [
                sys.executable, str(FINALIZER),
                "--campaign-dir", str(campaign),
                "--compiler", str(COMPILER),
                "--output", str(output),
                "--output-metadata", str(metadata),
                "--audit-output", str(audit),
                "--expected-point-count", "4",
                "--expected-probability-count", "335",
            ],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 5)
        record = json.loads(audit.read_text(encoding="utf-8"))
        self.assertEqual(record["points"], 4)
        self.assertEqual(record["total_histories"], 16)
        self.assertEqual(record["probability_count"], 335)
        self.assertLess(
            record["maximum_inverse_cdf_compression_mean_relative_error"],
            0.001,
        )
        self.assertTrue(record["raw_source_hashes_verified"])
        self.assertEqual(len(record["runtime_package"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()

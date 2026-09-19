#!/usr/bin/env python3
"""Run controlled old/new species-FE and secondary-EM minibeam screens."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import yaml


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("/mnt/sda/wuwei/minibeam_species_fe_ab_20260919")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--energies", type=int, nargs="+", default=(250, 300))
    parser.add_argument("--variants", nargs="+",
                        help="optional subset of A/B/C variant labels")
    parser.add_argument("--energy-band-roi-scoring", action="store_true")
    args = parser.parse_args()
    binary = REPO / "build/oneapi-nvidia-minibeam/carbon_mc"
    env = dict(os.environ)
    env["ONEAPI_DEVICE_SELECTOR"] = "cuda:*"
    env["LD_LIBRARY_PATH"] = (
        "/home/wuwei/sycl_workspace/llvm/build/install/lib:" +
        env.get("LD_LIBRARY_PATH", ""))
    variants = (
        ("A_shared_fe_legacy_em", "shared_c12", False),
        ("B_species_fe_legacy_em", "species_water", False),
        ("C_species_fe_unified_em", "species_water", True),
    )
    records = []
    for energy in args.energies:
        source = REPO / f"config/beam_minibeam_field3cm_copper_e{energy}_256k.yaml"
        base = yaml.safe_load(source.read_text())
        for label, parameter_set, unified in variants:
            if args.variants and label not in args.variants:
                continue
            directory = args.output_root / f"e{energy}" / label
            directory.mkdir(parents=True, exist_ok=True)
            data_link = directory / "data"
            if not data_link.exists():
                data_link.symlink_to(REPO / "data")
            config = dict(base)
            config.update({
                "fermi_eyges_parameter_set": parameter_set,
                "enable_secondary_unified_em": unified,
                "enable_secondary_energy_straggling": unified,
                "enable_charged_origin_voxel_scoring": True,
                "enable_minibeam_energy_band_roi_scoring":
                    args.energy_band_roi_scoring,
                "charged_origin_voxel_mhd_output_prefix": str(
                    directory / "component"),
                "dose_to_medium_name": "dose",
            })
            config_path = directory / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            command = [
                str(binary), "--config", str(config_path), "--device", "cuda",
                "--output", str(directory / "depth.csv"),
                "--dose-output", str(directory / "dose.raw"),
                "--write-canonical-config", str(directory / "config.canonical.yaml"),
            ]
            with (directory / "run.log").open("w") as log:
                result = subprocess.run(command, cwd=directory, env=env,
                                        stdout=log, stderr=subprocess.STDOUT)
            mhd_link = directory / "dose.mhd"
            if not mhd_link.exists():
                mhd_link.symlink_to(directory / "out/config/dose.mhd")
            quality_path = directory / "out/config/quality_report.json"
            ledger_path = directory / "out/config/energy_ledger.json"
            quality = json.loads(quality_path.read_text())
            ledger = json.loads(ledger_path.read_text())
            if (result.returncode != 0 or not quality.get("accepted") or
                    quality.get("failures") or
                    ledger.get("secondary_queue_overflow") or
                    ledger.get("cascade_queue_overflow") or
                    ledger.get("neutral_queue_overflow")):
                raise RuntimeError(f"failed {energy} {label}")
            records.append({
                "energy_MeVu": energy,
                "variant": label,
                "parameter_set": parameter_set,
                "secondary_unified_em": unified,
                "energy_band_roi_scoring": args.energy_band_roi_scoring,
                "histories": ledger["histories"],
                "energy_balance_error": ledger["energy_balance_error"],
                "directory": str(directory),
            })
            print(energy, label, "complete", flush=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()

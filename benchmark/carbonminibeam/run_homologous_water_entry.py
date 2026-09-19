#!/usr/bin/env python3
"""Drive the same GPU water transport from GPU vs TOPAS C12 entrance files."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

import yaml


REPO = Path(__file__).resolve().parents[2]
BRAGG = {250: 124.375, 300: 168.625}
TOPAS_ENTRY = {
    250: Path("/mnt/sda/wuwei/minibeam_water_replay_e250_256k/input/water_entrance_primary_c12_gpu.csv"),
    300: Path("/mnt/sda/wuwei/minibeam_water_multienergy_replay_20260919/e300/input/water_entrance_primary_c12_gpu.csv"),
}


def count_spots(path: Path) -> int:
    with path.open() as stream:
        return max(0, sum(1 for _ in stream) - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--energies", type=int, nargs="+", default=(250, 300))
    parser.add_argument("--incident-histories", type=int, default=256000)
    args = parser.parse_args()
    binary = REPO / "build/oneapi-nvidia-minibeam/carbon_mc"
    env = dict(os.environ)
    env["ONEAPI_DEVICE_SELECTOR"] = "cuda:*"
    env["LD_LIBRARY_PATH"] = (
        "/home/wuwei/sycl_workspace/llvm/build/install/lib:" +
        env.get("LD_LIBRARY_PATH", ""))
    converter = REPO / "benchmark/carbonminibeam/convert_gpu_phase_space_to_spots.py"
    records = []
    for energy in args.energies:
        diag = args.diag_root / f"e{energy}" / "C_species_fe_unified_em"
        canonical = yaml.safe_load((diag / "config.canonical.yaml").read_text())
        gpu_spots = args.output_root / f"e{energy}" / "gpu_entry_spots.csv"
        subprocess.check_call([
            "python3", str(converter),
            "--input", str(diag / "water_entrance.csv"),
            "--output", str(gpu_spots),
            "--incident-histories", str(args.incident_histories),
        ])
        sources = {
            "gpu_entry": gpu_spots,
            "topas_entry": TOPAS_ENTRY[energy],
        }
        bragg = BRAGG[energy]
        for label, spots in sources.items():
            directory = args.output_root / f"e{energy}" / label
            directory.mkdir(parents=True, exist_ok=True)
            data_link = directory / "data"
            if not data_link.exists():
                data_link.symlink_to(REPO / "data")
            n = count_spots(spots)
            config = dict(canonical)
            config.update({
                "number_of_histories": n,
                "tps_spots_file": str(spots),
                "tps_histories_scale": 1.0,
                "tps_spot_weight_mode": "histories",
                "beam_energy_spread": 0.0,
                "enable_emittance_source": False,
                "minibeam_transport_mode": "absorbing_geometry",
                "charged_origin_voxel_mhd_output_prefix": str(directory / "component"),
                "minibeam_water_primary_plane_output_file": str(
                    directory / "primary_planes.csv"),
                "minibeam_water_primary_plane_depths_mm":
                    f"{40.0}, {bragg - 2.0}, {bragg}, {bragg + 2.0}",
                "primary_voxel_fluence_mhd_output_file": str(
                    directory / "primary_fluence.mhd"),
                "enable_minibeam_energy_band_roi_scoring": True,
                "enable_minibeam_primary_c12_roi_scoring": True,
            })
            config_path = directory / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            command = [
                str(binary), "--config", str(config_path), "--device", "cuda",
                "--output", str(directory / "depth.csv"),
                "--dose-output", str(directory / "dose.raw"),
            ]
            with (directory / "run.log").open("w") as log:
                result = subprocess.run(command, cwd=directory, env=env,
                                        stdout=log, stderr=subprocess.STDOUT)
            quality = json.loads((directory / "out/config/quality_report.json").read_text())
            ledger = json.loads((directory / "out/config/energy_ledger.json").read_text())
            if result.returncode != 0 or not quality.get("accepted"):
                raise RuntimeError(f"failed {energy} {label}")
            records.append({
                "energy_MeVu": energy,
                "entry": label,
                "replay_particles": n,
                "incident_histories": args.incident_histories,
                "directory": str(directory),
                "energy_balance_error": ledger["energy_balance_error"],
            })
            print(energy, label, "particles", n, "complete", flush=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()

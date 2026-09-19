#!/usr/bin/env python3
"""Prepare secondary-queue GPU replays matching the TOPAS water slabs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_RUN_ROOT = Path("/mnt/sda/wuwei/fe_species_water_em10gev_20260919")
BASE_CONFIG = REPO / "config/beam_minibeam_water_fragment_replay_e250_256k.yaml"


def write_replay(path: Path, case: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "origin", "run_id", "event_id", "track_id", "parent_id", "pdg",
        "atomic_number", "mass_number", "kinetic_energy_MeV", "weight",
        "x_mm", "y_mm", "z_mm", "direction_x", "direction_y", "direction_z",
    ]
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        energy = case["energy_MeVu"] * case["mass_number"]
        for event in range(case["histories"]):
            # Translational invariance lets the replay distribute otherwise
            # identical pencil tracks over a 500x500 grid.  This prevents an
            # irrelevant FP32 atomic-closure failure from 250k histories
            # depositing into one voxel; analysis subtracts this known origin.
            x_mm = ((event % 500) - 249.5) * 0.2
            y_mm = ((event // 500) - 249.5) * 0.2
            writer.writerow((
                "fragment", 0, event, 1, 1, case["pdg"],
                case["atomic_number"], case["mass_number"], energy, 1,
                x_mm, y_mm, 0, 0, 0, 1,
            ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--gpu-subdir", default="gpu")
    parser.add_argument("--secondary-em", choices=("legacy", "unified"),
                        default="legacy")
    parser.add_argument("--straggling", choices=("off", "on"), default="on")
    parser.add_argument("--case", action="append", default=[],
                        help="case name to prepare; repeat to select several")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    manifest = json.loads((run_root / "manifest.json").read_text())
    base = yaml.safe_load(BASE_CONFIG.read_text())
    gpu_root = run_root / args.gpu_subdir
    inputs: dict[tuple[str, int], Path] = {}
    gpu_cases = []
    for case in manifest["cases"]:
        if args.case and case["name"] not in args.case:
            continue
        species = case["name"].split("_", 1)[0]
        input_key = (species, case["energy_MeVu"])
        if input_key not in inputs:
            input_path = gpu_root / "input" / f"{species}_e{case['energy_MeVu']}.csv"
            write_replay(input_path, case)
            inputs[input_key] = input_path
        output = gpu_root / case["name"]
        output.mkdir(parents=True, exist_ok=True)
        data_link = output / "data"
        if not data_link.exists():
            data_link.symlink_to(REPO / "data")
        config = dict(base)
        config.update({
            "number_of_histories": case["histories"],
            "initial_energy_MeVu": case["energy_MeVu"],
            "phantom_length_mm": case["thickness_mm"] + 0.1,
            "depth_bin_width_mm": 0.1,
            "maximum_step_mm": 0.1,
            "scorer_area_mm2": 40000.0,
            "enable_voxel_scoring": True,
            "dose_to_medium": True,
            "voxel_bins_x": 500,
            "voxel_bins_y": 500,
            "voxel_size_x_mm": 0.2,
            "voxel_size_y_mm": 0.2,
            "minibeam_water_entry_secondary_replay_file": str(inputs[input_key]),
            "minibeam_water_entry_secondary_replay_allow_primary_c12": False,
            # Score the actual water exit. Endpoint handling belongs to the
            # transport/scorer contract and must not be hidden by an inward
            # diagnostic offset.
            "minibeam_water_primary_plane_depths_mm": case["thickness_mm"],
            "minibeam_water_primary_plane_output_file": str(output / "plane.csv"),
            "multiple_scattering_model": "fermi_eyges",
            "fermi_eyges_species": "all_charged",
            "fermi_eyges_max_segment_mm": 0.1,
            "enable_inelastic": False,
            "enable_nuclear_elastic": False,
            "enable_secondary_unified_em": args.secondary_em == "unified",
            "enable_secondary_energy_straggling": args.straggling == "on",
            "secondary_queue_capacity": 300000,
            "random_seed": case["seed"],
            "dose_to_medium_name": "dose",
        })
        # The common selector is authoritative; remove the old C12-only
        # development selector to keep this validation single-path.
        config.pop("minibeam_water_secondary_c12_mcs_model", None)
        config.pop("minibeam_water_secondary_c12_mcs_max_segment_mm", None)
        config_path = output / "gpu.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        gpu_cases.append({**case, "config": str(config_path),
                          "plane": str(output / "plane.csv"),
                          "secondary_em": args.secondary_em,
                          "straggling": args.straggling})
    (gpu_root / "manifest.json").write_text(json.dumps({
        "binary": str(REPO / "build/oneapi-nvidia-minibeam/carbon_mc"),
        "secondary_em": args.secondary_em,
        "straggling": args.straggling,
        "cases": gpu_cases,
    }, indent=2) + "\n")
    print(f"prepared {len(gpu_cases)} GPU cases in {gpu_root}")


if __name__ == "__main__":
    main()

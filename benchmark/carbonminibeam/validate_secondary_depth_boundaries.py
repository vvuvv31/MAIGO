#!/usr/bin/env python3
"""Prepare/analyze deterministic forward/backward secondary z-boundary cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml


CASES = (
    ("forward_before", 1.0 - 5.0e-7, 1.0, 3),
    ("forward_on", 1.0, 1.0, 4),
    ("forward_after", 1.0 + 5.0e-7, 1.0, 4),
    ("backward_before", 1.0 - 5.0e-7, -1.0, 3),
    ("backward_on", 1.0, -1.0, 3),
    ("backward_after", 1.0 + 5.0e-7, -1.0, 4),
)


def prepare(root: Path) -> None:
    base_path = Path(
        "config/beam_minibeam_water_secondary_c12_replay_e250_1622795_legacy.yaml"
    )
    with base_path.open(encoding="utf-8") as stream:
        base = yaml.safe_load(stream)
    source = root / "boundary_particles.csv"
    root.mkdir(parents=True, exist_ok=True)
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "origin", "run_id", "event_id", "track_id", "parent_id",
            "pdg", "atomic_number", "mass_number", "kinetic_energy_MeV",
            "weight", "x_mm", "y_mm", "z_mm", "direction_x",
            "direction_y", "direction_z", "rng_stream", "generation",
            "birth_region",
        ])
        for index, (_, z_mm, direction_z, _) in enumerate(CASES):
            writer.writerow([
                "fragment", 0, index, index + 1, index + 1, 1000060120,
                6, 12, 120.0, 1.0, -2.5 + index, 0.0, z_mm,
                0.0, 0.0, direction_z, 1000 + index, 1, 2,
            ])

    common = {
        "number_of_histories": len(CASES),
        "secondary_queue_capacity": 64,
        "minibeam_water_entry_secondary_replay_file": str(source),
        "minibeam_water_entry_secondary_replay_allow_internal_births": True,
        "enable_secondary_unified_em": False,
        "enable_inelastic": False,
        "enable_multiple_scattering": False,
        "enable_energy_straggling": False,
        "enable_secondary_energy_straggling": False,
        "minibeam_water_secondary_c12_post_sample_loss_scale": 1.0,
        "minibeam_water_secondary_c12_mcs_model": "legacy_highland",
        "phantom_length_mm": 2.0,
        "depth_bin_width_mm": 0.25,
        "maximum_step_mm": 0.1,
        "voxel_bins_x": 6,
        "voxel_bins_y": 1,
        "voxel_size_x_mm": 1.0,
        "voxel_size_y_mm": 10.0,
        "scorer_area_mm2": 6.0,
    }
    for name, unified in (("legacy", False), ("unified", True)):
        config = dict(base)
        config.update(common)
        config.pop("minibeam_water_primary_plane_output_file", None)
        config.pop("minibeam_water_primary_plane_depths_mm", None)
        config["minibeam_water_secondary_c12_enable_unified_em"] = unified
        case_dir = root / name
        case_dir.mkdir(exist_ok=True)
        with (case_dir / "config.yaml").open("w", encoding="utf-8") as stream:
            stream.write("# Deterministic secondary depth-boundary validation.\n")
            yaml.safe_dump(config, stream, sort_keys=True)


def analyze(root: Path) -> None:
    report = {"cases": {}}
    for path_name in ("legacy", "unified"):
        raw_path = root / path_name / "dose.raw"
        dose = np.fromfile(raw_path, dtype="<f4").reshape(8, 1, 6)[:, 0, :]
        path_report = []
        for index, (name, _, direction_z, expected_start_bin) in enumerate(CASES):
            column = dose[:, index]
            nonzero = np.flatnonzero(column > 0.0)
            if nonzero.size == 0:
                raise AssertionError(f"{path_name}/{name}: no scored dose")
            observed_start = int(nonzero.min() if direction_z > 0 else nonzero.max())
            if observed_start != expected_start_bin:
                raise AssertionError(
                    f"{path_name}/{name}: expected directed start bin "
                    f"{expected_start_bin}, observed {observed_start}"
                )
            path_report.append({
                "name": name,
                "expected_directed_start_bin": expected_start_bin,
                "observed_directed_start_bin": observed_start,
                "dose_sum_Gy": float(column.sum()),
                "nonzero_bins": nonzero.tolist(),
            })
        report["cases"][path_name] = path_report
        report[f"{path_name}_dose_sum_Gy"] = float(dose.sum())
    denominator = report["legacy_dose_sum_Gy"]
    report["unified_over_legacy_total_dose"] = (
        report["unified_dose_sum_Gy"] / denominator
    )
    (root / "boundary_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.analyze:
        analyze(args.root)
    else:
        prepare(args.root)


if __name__ == "__main__":
    main()

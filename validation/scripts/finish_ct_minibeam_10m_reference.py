#!/usr/bin/env python3
"""Wait for the second TOPAS batch and finish high-statistics CT validation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from match_gpu_to_physical_dose import read_mhd


ROOT = Path(__file__).resolve().parents[2]


def run(arguments: list[str]) -> None:
    print("+", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=ROOT, check=True)


def wait_for_stable_file(path: Path, poll_seconds: float, timeout_hours: float) -> None:
    deadline = time.monotonic() + timeout_hours * 3600.0
    previous_size = -1
    stable_checks = 0
    while time.monotonic() < deadline:
        if path.exists():
            size = path.stat().st_size
            if size > 0 and size == previous_size:
                stable_checks += 1
                if stable_checks >= 2:
                    print(f"Stable input detected: {path} ({size} bytes)", flush=True)
                    return
            else:
                stable_checks = 0
            previous_size = size
        print(f"Waiting for {path}...", flush=True)
        time.sleep(poll_seconds)
    raise TimeoutError(f"Timed out waiting for {path}")


def beam_y_fit_scale(
    gpu_path: Path,
    reference_path: Path,
    threshold_fraction: float = 0.1,
) -> float:
    _, reference_values = read_mhd(reference_path)
    gpu_metadata, gpu_values = read_mhd(gpu_path)
    gx, gy, gz = (int(value) for value in gpu_metadata["DimSize"].split())
    reference = np.asarray(reference_values, dtype=np.float64)
    gpu = (
        np.asarray(gpu_values, dtype=np.float64)
        .reshape(gz, gy, gx)
        .transpose(1, 0, 2)
        .reshape(-1)
    )
    selected = reference >= threshold_fraction * float(reference.max())
    return float(
        np.dot(gpu[selected], reference[selected])
        / np.dot(gpu[selected], gpu[selected])
    )


def identity_fit_scale(
    evaluation_path: Path,
    reference_path: Path,
    threshold_fraction: float = 0.1,
) -> float:
    _, reference_values = read_mhd(reference_path)
    _, evaluation_values = read_mhd(evaluation_path)
    reference = np.asarray(reference_values, dtype=np.float64)
    evaluation = np.asarray(evaluation_values, dtype=np.float64)
    selected = reference >= threshold_fraction * float(reference.max())
    return float(
        np.dot(evaluation[selected], reference[selected])
        / np.dot(evaluation[selected], evaluation[selected])
    )


def match_command(
    evaluation: Path,
    reference: Path,
    output: Path,
    mapping: str,
    histories: int,
    multiplier: float,
) -> list[str]:
    command = [
        sys.executable,
        "validation/scripts/match_gpu_to_physical_dose.py",
        str(evaluation),
        str(reference),
        "--output-dir",
        str(output),
        "--mapping",
        mapping,
        "--histories",
        str(histories),
        "--no-flip-x",
        "--no-flip-y",
        "--dose-scale-multiplier",
        f"{multiplier:.12g}",
        "--gamma-points",
        "300000",
        "--gamma-resolution-mm",
        "0.1",
        "--only-custom-gamma",
        "--gamma-criterion",
        "3",
        "0.3",
    ]
    if mapping == "beam_y":
        command.extend(["--patient-shape", "960", "607", "42"])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-hours", type=float, default=6.0)
    args = parser.parse_args()

    topas_directory = ROOT / "ct/fullplan_local_ct_compare/20022516"
    first_dicom = topas_directory / "minibeam_plane_e200_5M.dcm"
    second_dicom = topas_directory / "minibeam_plane_e200_5M_seed20260803.dcm"
    wait_for_stable_file(second_dicom, args.poll_seconds, args.timeout_hours)

    first_topas = ROOT / "out/ct/20022516/minibeam_plane_e200_5M/topas/dose.mhd"
    second_topas = (
        ROOT
        / "out/ct/20022516/minibeam_plane_e200_5M_seed20260803/topas/dose.mhd"
    )
    combined_topas = (
        ROOT / "out/ct/20022516/minibeam_plane_e200_10M_combined/topas/dose.mhd"
    )
    first_gpu = (
        ROOT / "out/ct/20022516/minibeam_plane_e200_10M_high_accuracy_valid/gpu/dose.mhd"
    )
    second_gpu = (
        ROOT
        / "out/ct/20022516/minibeam_plane_e200_10M_high_accuracy_seed20260731_valid/gpu/dose.mhd"
    )
    combined_gpu = (
        ROOT / "out/ct/20022516/minibeam_plane_e200_20M_high_accuracy_combined/gpu/dose.mhd"
    )

    run(
        [
            sys.executable,
            "validation/scripts/convert_topas_rtdose_to_mhd.py",
            str(second_dicom),
            str(second_topas),
            "--metrics-json",
            str(second_topas.parent / "conversion_metrics.json"),
        ]
    )
    run(
        [
            sys.executable,
            "validation/scripts/combine_mhd_dose.py",
            str(first_topas),
            str(second_topas),
            "--output",
            str(combined_topas),
            "--mode",
            "sum",
        ]
    )
    if not combined_gpu.exists():
        run(
            [
                sys.executable,
                "validation/scripts/combine_mhd_dose.py",
                str(first_gpu),
                str(second_gpu),
                "--output",
                str(combined_gpu),
                "--mode",
                "sum",
            ]
        )

    topas_fit = identity_fit_scale(second_topas, first_topas)
    run(
        match_command(
            second_topas,
            first_topas,
            ROOT
            / "out/ct/20022516/minibeam_plane_e200_5M_topas_topas/"
            "gamma_3pct_0p3mm_absolute_all_voxels",
            "identity",
            5_000_000,
            1.0 / topas_fit,
        )
    )

    combined_fit = beam_y_fit_scale(combined_gpu, combined_topas)
    combined_output = (
        ROOT
        / "out/ct/20022516/minibeam_plane_e200_20M_vs_topas_10M/"
        "gamma_3pct_0p3mm_absolute_all_voxels"
    )
    run(
        match_command(
            combined_gpu,
            combined_topas,
            combined_output,
            "beam_y",
            20_000_000,
            0.5 / combined_fit,
        )
    )

    uncertainty_output = (
        ROOT
        / "out/ct/20022516/minibeam_plane_e200_20M_vs_topas_10M/"
        "uncertainty_aware_local_gamma.json"
    )
    run(
        [
            sys.executable,
            "validation/scripts/analyze_mc_uncertainty_gamma.py",
            "--reference-batch",
            str(first_topas),
            "--reference-batch",
            str(second_topas),
            "--evaluation-batch",
            str(first_gpu),
            "--evaluation-batch",
            str(second_gpu),
            "--reference-histories",
            "5000000",
            "--evaluation-histories",
            "10000000",
            "--output",
            str(uncertainty_output),
            "--gamma-points",
            "300000",
        ]
    )

    profile_output = (
        ROOT
        / "out/ct/20022516/minibeam_plane_e200_20M_vs_topas_10M/"
        "profiles_absolute_ct"
    )
    run(
        [
            sys.executable,
            "validation/scripts/analyze_ct_minibeam_plane.py",
            "--reference",
            str(combined_topas),
            "--gpu",
            str(combined_output / "gpu_scaled_to_physical.mhd"),
            "--output-dir",
            str(profile_output),
            "--depths-mm",
            "5",
            "50",
            "100",
            "150",
            "--slab-width-mm",
            "2",
        ]
    )

    summary = {
        "status": "complete",
        "second_topas_dicom": str(second_dicom),
        "combined_topas": str(combined_topas),
        "combined_gpu": str(combined_gpu),
        "topas_topas_metrics": str(
            ROOT
            / "out/ct/20022516/minibeam_plane_e200_5M_topas_topas/"
            "gamma_3pct_0p3mm_absolute_all_voxels/match_metrics.json"
        ),
        "gpu_topas_metrics": str(combined_output / "match_metrics.json"),
        "uncertainty_metrics": str(uncertainty_output),
        "profiles": str(profile_output),
    }
    status_path = (
        ROOT
        / "out/ct/20022516/minibeam_plane_e200_20M_vs_topas_10M/"
        "automation_status.json"
    )
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

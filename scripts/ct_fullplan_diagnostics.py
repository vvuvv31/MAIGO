#!/usr/bin/env python3
"""Audit CT full-plan geometry, normalization, beam-depth and material response."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from array import array
from pathlib import Path

import numpy as np


CASES = {
    "20022516": {
        "ct": "benchmark/ct/20022516/grid/patient_ct.bin",
        "budget": "benchmark/ct/20022516/fullplan_mc/particle_budget.json",
        "spots": "benchmark/ct/20022516/fullplan_mc/spots_full_plan.txt",
        "topas": "out/fullplan_profiles/20022516/topas/dose.mhd",
        "body": "out/fullplan_profiles/20022516/body_mask.mhd",
        "beam_axis": "y", "beam_sign": 1,
    },
    "RT06423": {
        "ct": "benchmark/ct/RT06423/grid/patient_ct.bin",
        "budget": "benchmark/ct/RT06423/fullplan_mc/particle_budget.json",
        "spots": "benchmark/ct/RT06423/fullplan_mc/spots_full_plan.txt",
        "topas": "out/fullplan_result/RT06423/topas/dose.mhd",
        "body": "out/fullplan_result/RT06423/body_mask.mhd",
        "beam_axis": "x", "beam_sign": -1,
    },
    "RT07575": {
        "ct": "benchmark/ct/grids/patient_ct_rt07575_edge_corrected.bin",
        "budget": "benchmark/ct/RT07575/conventional/fullplan_mc/particle_budget.json",
        "spots": "benchmark/ct/RT07575/conventional/fullplan_mc/spots_full_plan.txt",
        "topas": "out/fullplan_result/RT07575/topas/dose.mhd",
        "body": "out/fullplan_result/RT07575/body_mask.mhd",
        "beam_axis": "x", "beam_sign": -1,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_mhd(path: Path) -> tuple[dict[str, str], np.ndarray]:
    meta: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key.strip()] = value.strip()
    nx, ny, nz = map(int, meta["DimSize"].split())
    raw_path = path.parent / meta["ElementDataFile"]
    values = array("f")
    values.frombytes(raw_path.read_bytes())
    if len(values) != nx * ny * nz:
        raise ValueError(f"{path}: RAW size mismatch")
    return meta, np.asarray(values, dtype=np.float32).reshape(nz, ny, nx)


def mhd_geometry(meta: dict[str, str]) -> dict[str, object]:
    return {
        "dim_xyz": [int(v) for v in meta["DimSize"].split()],
        "spacing_xyz_mm": [float(v) for v in meta["ElementSpacing"].split()],
        "first_center_xyz_mm": [float(v) for v in meta["Offset"].split()],
        "transform": [float(v) for v in meta.get(
            "TransformMatrix", "1 0 0 0 1 0 0 0 1").split()],
    }


def read_cctg(path: Path) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        magic, version = struct.unpack("<II", stream.read(8))
        if magic != 0x47544343 or version not in (1, 2, 3):
            raise ValueError(f"{path}: unsupported CCTG magic/version")
        nx, ny, nz = struct.unpack("<III", stream.read(12))
        origin = struct.unpack("<fff", stream.read(12))
        spacing = struct.unpack("<fff", stream.read(12))
        count = nx * ny * nz
        if count <= 0 or count > 512 * 1024 * 1024:
            raise ValueError(f"{path}: invalid CCTG dimensions {nx}x{ny}x{nz}")
        density = np.frombuffer(stream.read(4 * count), dtype="<f4").copy()
        material = np.frombuffer(stream.read(count), dtype=np.uint8).copy()
    if density.size != count or material.size != count:
        raise ValueError(f"{path}: truncated CCTG")
    geometry = {
        "version": version, "dim_xyz": [nx, ny, nz],
        "spacing_xyz_mm": list(spacing), "low_edge_xyz_mm": list(origin),
        "first_center_xyz_mm": [origin[i] + 0.5 * spacing[i] for i in range(3)],
    }
    return geometry, density.reshape(nz, ny, nx), material.reshape(nz, ny, nx)


def parse_spot_histories(path: Path) -> tuple[int, int]:
    count = total = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "Tf/Scatterer1/L4/Values" not in line or "=" not in line:
            continue
        payload = line.split("=", 1)[1].split("#", 1)[0].strip().split()
        if payload:
            declared = int(payload[0])
            values = [int(float(v)) for v in payload[1:]]
            if len(values) != declared:
                raise ValueError(f"{path}: L4 declared {declared}, got {len(values)}")
            return len(values), sum(values)
    # Generated files can use one Step line per spot instead of a vector.
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "/L4/Value" in line and "=" in line:
            try:
                total += int(float(line.split("=", 1)[1].split()[0]))
                count += 1
            except ValueError:
                pass
    return count, total


def summarize(case: str, profile: str, repo: Path) -> dict[str, object]:
    spec = CASES[case]
    paths = {key: repo / str(spec[key]) for key in ("ct", "budget", "spots", "topas", "body")}
    gpu_path = repo / "benchmark/ct/result" / case / profile / "dose.mhd"
    budget = json.loads(paths["budget"].read_text())
    topas_meta, topas = read_mhd(paths["topas"])
    gpu_meta, gpu = read_mhd(gpu_path)
    _, body = read_mhd(paths["body"])
    ct_geometry, density, section = read_cctg(paths["ct"])
    if topas.shape != gpu.shape or topas.shape != density.shape or topas.shape != body.shape:
        raise ValueError(
            f"shape mismatch TOPAS={topas.shape} GPU={gpu.shape} CT={density.shape} BODY={body.shape}")
    spot_count, spot_histories = parse_spot_histories(paths["spots"])
    expected_histories = int(budget["total_particles_this_run"])
    topas_geometry = mhd_geometry(topas_meta)
    gpu_geometry = mhd_geometry(gpu_meta)
    ct_center = np.asarray(ct_geometry["first_center_xyz_mm"], dtype=float)
    topas_center = np.asarray(topas_geometry["first_center_xyz_mm"], dtype=float)
    gpu_center = np.asarray(gpu_geometry["first_center_xyz_mm"], dtype=float)

    inside = body > 0.5
    peak = float(topas[inside].max())
    selected = inside & (topas >= 0.1 * peak)
    axis_xyz = "xyz".index(str(spec["beam_axis"]))
    axis_zyx = 2 - axis_xyz
    transverse = tuple(i for i in range(3) if i != axis_zyx)
    topas_depth = (topas * inside).sum(axis=transverse, dtype=np.float64)
    gpu_depth = (gpu * inside).sum(axis=transverse, dtype=np.float64)
    depth_mask = topas_depth >= 0.1 * topas_depth.max()

    material_rows = []
    class_masks = {
        "air": section == 0,
        "lung": section == 1,
        "soft_tissue": (section >= 2) & (section <= 8),
        "bone": section >= 9,
    }
    for label, class_mask in class_masks.items():
        mask = selected & class_mask
        ref_sum = float(topas[mask].sum(dtype=np.float64))
        eval_sum = float(gpu[mask].sum(dtype=np.float64))
        material_rows.append({
            "class": label, "voxels": int(mask.sum()),
            "topas_dose_sum_Gy": ref_sum, "gpu_dose_sum_Gy": eval_sum,
            "gpu_to_topas": eval_sum / ref_sum if ref_sum > 0.0 else None,
            "mean_density_g_per_cm3": float(density[mask].mean()) if mask.any() else None,
        })

    return {
        "case": case, "profile": profile,
        "files": {name: {"path": str(path.relative_to(repo)), "sha256": sha256(path)}
                  for name, path in paths.items()},
        "gpu_file": {"path": str(gpu_path.relative_to(repo)), "sha256": sha256(gpu_path)},
        "normalization": {
            "scale_integer_K": int(budget["scale_integer_K"]),
            "declared_topas_dose_rule": budget["definition"]["absolute_dose"],
            "budget_run_histories": expected_histories,
            "spot_count_from_L4": spot_count, "spot_histories_from_L4": spot_histories,
            "history_check_pass": spot_histories == expected_histories,
        },
        "geometry": {
            "ct": ct_geometry, "topas": topas_geometry, "gpu": gpu_geometry,
            "gpu_minus_ct_first_center_mm": (gpu_center - ct_center).tolist(),
            "topas_minus_ct_first_center_mm": (topas_center - ct_center).tolist(),
            "shape_check_pass": list(gpu.shape) == list(topas.shape) == list(density.shape),
            "gpu_ct_geometry_pass": bool(np.allclose(gpu_center, ct_center, atol=1e-6)),
            "topas_ct_geometry_pass": bool(np.allclose(topas_center, ct_center, atol=1e-6)),
        },
        "selection": {"definition": "BODY and TOPAS dose >=10% BODY Dmax",
                      "voxels": int(selected.sum()), "topas_body_dmax_Gy": peak},
        "beam_depth": {
            "axis": spec["beam_axis"], "sign": spec["beam_sign"],
            "plane_integral_gpu_to_topas": float(gpu_depth[depth_mask].sum() /
                                                   topas_depth[depth_mask].sum()),
            "topas_peak_plane": int(np.argmax(topas_depth)),
            "gpu_peak_plane": int(np.argmax(gpu_depth)),
        },
        "high_dose_integral_gpu_to_topas": float(gpu[selected].sum(dtype=np.float64) /
                                                   topas[selected].sum(dtype=np.float64)),
        "material_response": material_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--case", choices=tuple(CASES), action="append")
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark/ct/result/diagnostics/comparison_integrity.json"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    cases = args.case or list(CASES)
    result = {"cases": [summarize(case, args.profile, repo) for case in cases]}
    output = args.output if args.output.is_absolute() else repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

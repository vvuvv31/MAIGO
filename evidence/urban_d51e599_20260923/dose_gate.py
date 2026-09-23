#!/usr/bin/env python3
"""Dose acceptance using scorer metadata and independent run uncertainty.

The frozen acceptance.yaml is read at runtime. GPU MetaImage headers and
TOPAS scorer headers are checked against run manifests, configs, hashes,
voxel geometry, density, normalization, and history counts. Missing independent
TOPAS batches or invalid measurements return INCONCLUSIVE (exit 2).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import yaml


class GateError(RuntimeError):
    pass


def require(ok: bool, msg: str) -> None:
    if not ok:
        raise GateError(msg)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_key_value_header(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def parse_gpu_mhd(path: Path) -> dict:
    h = parse_key_value_header(path)
    required = ("NDims", "DimSize", "ElementSpacing", "Offset", "ElementType",
                "ElementDataFile", "DoseUnits")
    missing = [k for k in required if k not in h]
    require(not missing, f"GPU MHD header lacks {missing}")
    require(h["NDims"] == "3" and h["DoseUnits"] == "Gy",
            "GPU scorer must be a 3D Gy MetaImage")
    dims = tuple(int(x) for x in h["DimSize"].split())
    spacing = tuple(float(x) for x in h["ElementSpacing"].split())
    offset = tuple(float(x) for x in h["Offset"].split())
    require(len(dims) == len(spacing) == len(offset) == 3,
            "GPU MHD geometry vectors must have three components")
    require(dims[1] == 1 and h["ElementType"] == "MET_FLOAT",
            "GPU dose scorer must expose a single FP32 lateral slab")
    raw = path.parent / h["ElementDataFile"]
    require(raw.is_file(), f"GPU dose raw file missing: {raw}")
    require(raw.stat().st_size == int(np.prod(dims)) * 4,
            "GPU dose raw size differs from MHD dimensions/FP32 type")
    return {"header": h, "dims_xyz": dims, "spacing_xyz_mm": spacing,
            "offset_center_xyz_mm": offset, "raw": raw}


def parse_topas_header(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    require("DoseToMedium ( Gy ) : Sum" in raw,
            "TOPAS scorer must be DoseToMedium (Gy), Sum")
    bins, pitch_cm = {}, {}
    axes = {"X": 0, "Y": 1, "Z": 2}
    for line in raw.splitlines():
        m = re.search(r"\b([XYZ]) in (\d+) bins?\s+of\s+([0-9.eE+-]+) cm", line)
        if m:
            axis = m.group(1)
            bins[axis] = int(m.group(2))
            pitch_cm[axis] = float(m.group(3))
    require(set(bins) == {"X", "Y", "Z"},
            "TOPAS scorer header lacks all three binned axes")
    records = None
    return {"raw": raw, "bins_xyz": (bins["X"], bins["Y"], bins["Z"]),
            "spacing_xyz_mm": tuple(10.0 * pitch_cm[a] for a in "XYZ"),
            "sha256": sha256(path)}


def load_manifest(path: Path, expected_key: str) -> dict:
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GateError(f"cannot read {expected_key} manifest: {exc}")
    require(m.get("schema") == "maigo.dose-scorer-manifest.v1",
            f"unsupported {expected_key} manifest schema")
    require(m.get("role") == expected_key, f"manifest role is not {expected_key}")
    require(bool(m.get("runs")), f"{expected_key} manifest has no runs")
    return m


def validate_run_common(run: dict, role: str) -> tuple[Path, Path, float, float]:
    required = ("effective_config", "config_sha256", "history_count", "random_seed",
                "normalization", "density_g_cm3", "dose_sha256",
                "geometry_axes", "origin_centers_mm")
    missing = [k for k in required if k not in run]
    require(not missing, f"{role} run manifest missing {missing}")
    cfg = Path(run["effective_config"])
    dose = Path(run.get("mhd" if role == "gpu" else "dose_bin", ""))
    require(cfg.is_file() and sha256(cfg) == run["config_sha256"],
            f"{role} effective config missing/hash mismatch: {cfg}")
    require(dose.is_file() and sha256(dose) == run["dose_sha256"],
            f"{role} dose file missing/hash mismatch: {dose}")
    require(int(run["history_count"]) > 0 and int(run["random_seed"]) >= 0,
            f"{role} run has invalid history count or seed")
    require(run["normalization"] in ("Gy_per_primary", "Gy_sum_over_histories"),
            f"{role} normalization must be explicit")
    require(float(run["density_g_cm3"]) > 0.0,
            f"{role} material density must be positive")
    require(run["geometry_axes"] == {
        "transverse": "x", "depth": "z", "slab_thickness": "y"},
        f"{role} scorer axis convention differs from x/transverse, z/depth")
    origin = run["origin_centers_mm"]
    require(len(origin) == 2 and all(math.isfinite(float(x)) for x in origin),
            f"{role} must provide first transverse/depth voxel-center coordinates")
    per_primary = 1.0 if run["normalization"] == "Gy_per_primary" else 1.0 / int(run["history_count"])
    return dose, cfg, float(run["density_g_cm3"]), per_primary


def load_gpu_run(run: dict, expected_count: int) -> tuple[np.ndarray, dict]:
    mhd, cfg, density, scale = validate_run_common(run, "gpu")
    g = parse_gpu_mhd(mhd)
    require(run.get("mhd_sha256") == sha256(mhd), "GPU MHD hash mismatch")
    require(run.get("raw_sha256") == sha256(g["raw"]), "GPU raw hash mismatch")
    nx, ny, nz = g["dims_xyz"]
    dx, dy, dz = g["spacing_xyz_mm"]
    ox, oy, oz = g["offset_center_xyz_mm"]
    require(ny == 1 and (nx, nz) == tuple(run.get("shape_xz", [])),
            "GPU scorer dimensions differ from run manifest")
    require(run["origin_centers_mm"] == [ox, oz],
            "GPU run manifest origin differs from MHD offset")
    require(run.get("spacing_xz_mm") == [dx, dz] and
            float(run.get("slab_thickness_mm", 0.0)) == dy,
            "GPU run manifest spacing/slab thickness differs from MHD")
    raw = np.fromfile(g["raw"], dtype="<f4")
    dose = raw.reshape((nz, ny, nx))[:, 0, :].astype(np.float64) * scale
    require(dose.shape == (expected_count, nx) or
            (expected_count < dose.shape[0] and dose.shape[0] > expected_count),
            "GPU dose data do not cover the declared scorer dimensions")
    require(np.isfinite(dose).all() and np.all(dose >= 0.0),
            "GPU dose contains NaN/Inf or negative values")
    mass_kg = dx * dz * dy * density / 1.0e6  # mm^3 -> cm^3 -> kg
    return dose, {"nx": nx, "nz": nz, "dx_mm": dx, "dz_mm": dz,
                  "dy_mm": dy, "x0_center_mm": ox, "z0_center_mm": oz,
                  "density_g_cm3": density, "voxel_mass_kg": mass_kg,
                  "config": str(cfg), "seed": int(run["random_seed"]),
                  "histories": int(run["history_count"]), "dose_file": str(g["raw"])}


def load_topas_run(run: dict, expected_shape: tuple[int, int],
                   reference_geometry: dict) -> tuple[np.ndarray, dict]:
    dose_path, cfg, density, scale = validate_run_common(run, "topas")
    header_path = Path(run.get("dose_header", ""))
    require(header_path.is_file() and sha256(header_path) == run.get("header_sha256"),
            "TOPAS scorer header missing/hash mismatch")
    h = parse_topas_header(header_path)
    nx, ny, nz = h["bins_xyz"]
    dx, dy, dt = h["spacing_xyz_mm"]
    require((nx, ny, nz) == (expected_shape[1], expected_shape[0], 1),
            "TOPAS scorer dimensions do not map to GPU x/depth grid")
    require(abs(dx-reference_geometry["dx_mm"]) < 1.0e-9 and
            abs(dy-reference_geometry["dz_mm"]) < 1.0e-9 and
            abs(dt-reference_geometry["dy_mm"]) < 1.0e-9,
            "TOPAS scorer spacing or lateral slab thickness differs from GPU")
    require(run.get("shape_xz") == [nx, ny], "TOPAS manifest shape differs from scorer header")
    require(run.get("spacing_xz_mm") == [dx, dy] and
            float(run.get("slab_thickness_mm", 0.0)) == dt,
            "TOPAS manifest spacing differs from scorer header")
    require(run["origin_centers_mm"] == [reference_geometry["x0_center_mm"],
                                          reference_geometry["z0_center_mm"]],
            "TOPAS scorer origin does not match GPU bin centers")
    elem = run.get("element_type")
    require(elem in ("float32", "float64"), "TOPAS binary element_type must be declared")
    dtype = "<f4" if elem == "float32" else "<f8"
    count = nx * ny * nz
    require(dose_path.stat().st_size == count * np.dtype(dtype).itemsize,
            "TOPAS dose file size disagrees with scorer dimensions/element type")
    values = np.fromfile(dose_path, dtype=dtype).astype(np.float64)
    # TOPAS axes are X,Y,Z; Y is depth and X is transverse. X is contiguous.
    dose = values.reshape((nz, ny, nx))[0, :, :] * scale
    require(np.isfinite(dose).all() and np.all(dose >= 0.0),
            "TOPAS dose contains NaN/Inf or negative values")
    mass_kg = dx * dy * dt * density / 1.0e6
    return dose, {"nx": nx, "nz": ny, "dx_mm": dx, "dz_mm": dy,
                  "dy_mm": dt, "x0_center_mm": run["origin_centers_mm"][0],
                  "z0_center_mm": run["origin_centers_mm"][1],
                  "density_g_cm3": density, "voxel_mass_kg": mass_kg,
                  "config": str(cfg), "seed": int(run["random_seed"]),
                  "histories": int(run["history_count"]), "dose_file": str(dose_path),
                  "header_sha256": h["sha256"]}


def _stats(v: list[float]) -> dict:
    a = np.asarray(v, dtype=np.float64)
    require(len(a) >= 3, "fewer than three independent runs/batches")
    require(np.isfinite(a).all(), "metric contains non-finite run values")
    return {"mean": float(a.mean()),
            "se": float(a.std(ddof=1) / math.sqrt(len(a))),
            "n_independent_runs": int(len(a)), "values": [float(x) for x in a]}


def _ratio_gate(gpu: list[float], topas: list[float], threshold: float) -> dict:
    gs, ts = _stats(gpu), _stats(topas)
    require(ts["mean"] > 0.0, "reference metric is not positive")
    ratio = gs["mean"] / ts["mean"]
    se = math.hypot(gs["se"] / ts["mean"],
                    gs["mean"] * ts["se"] / ts["mean"]**2)
    margin = abs(ratio - 1.0) + 2.0 * se
    return {"status": "PASS" if margin < threshold else "FAIL", "ratio": ratio,
            "ratio_se": se, "threshold": threshold,
            "equivalence_margin_2se": margin, "gpu": gs, "topas": ts}


def _abs_gate(gpu: list[float], topas: list[float], threshold: float) -> dict:
    gs, ts = _stats(gpu), _stats(topas)
    diff = gs["mean"] - ts["mean"]
    se = math.hypot(gs["se"], ts["se"])
    margin = abs(diff) + 2.0 * se
    return {"status": "PASS" if margin < threshold else "FAIL", "difference": diff,
            "difference_se": se, "threshold": threshold,
            "equivalence_margin_2se": margin, "gpu": gs, "topas": ts}


def _fwhm_1d(profile: np.ndarray, x: np.ndarray,
             peak_window: tuple[float, float] | None = (-0.25, 0.25)) -> float | None:
    candidates = np.arange(len(x)) if peak_window is None else np.flatnonzero(
        (x > peak_window[0]) & (x < peak_window[1]))
    if len(candidates) == 0:
        return None
    peak = int(candidates[np.argmax(profile[candidates])])
    half = float(profile[peak]) * 0.5
    if not math.isfinite(half) or half <= 0.0:
        return None
    left_below = peak
    while left_below > 0 and profile[left_below] >= half:
        left_below -= 1
    right_below = peak
    while right_below + 1 < len(profile) and profile[right_below] >= half:
        right_below += 1
    if profile[left_below] >= half or profile[right_below] >= half:
        return None  # both half-height crossings must be bracketed
    if left_below == peak or right_below == peak:
        return None
    left_above, right_above = left_below + 1, right_below - 1
    if (left_above < peak and profile[left_above] == half and
            profile[left_above+1] == half) or \
       (right_above > peak and profile[right_above] == half and
            profile[right_above-1] == half):
        return None
    xl = float(x[left_above]) if profile[left_above] == half else \
        x[left_below] + (half-profile[left_below]) * \
        (x[left_above]-x[left_below]) / (profile[left_above]-profile[left_below])
    xr = float(x[right_above]) if profile[right_above] == half else \
        x[right_above] + (half-profile[right_above]) * \
        (x[right_below]-x[right_above]) / (profile[right_below]-profile[right_above])
    return float(xr - xl) if xr > xl else None


def _r80(profile: np.ndarray, z: np.ndarray) -> float | None:
    if len(profile) < 2 or not np.isfinite(profile).all() or profile.max() <= 0.0:
        return None
    peak = int(np.argmax(profile))
    target = float(profile[peak]) * 0.8
    for i in range(len(profile)-1, peak, -1):
        y0, y1 = float(profile[i-1]), float(profile[i])
        if y0 >= target and y1 < target:
            if y0 == y1:
                return None
            return float(z[i-1] + (target-y0) * (z[i]-z[i-1]) / (y1-y0))
    return None


def run(args) -> tuple[dict, int]:
    accept_path = Path(args.accept)
    try:
        accept = yaml.safe_load(accept_path.read_text(encoding="utf-8"))["carbon_em_only_dose"]
    except Exception as exc:
        raise GateError(f"cannot parse carbon_em_only_dose acceptance: {exc}")
    gm, tm = load_manifest(Path(args.gpu_manifest), "gpu"), load_manifest(Path(args.topas_manifest), "topas")
    require(len(gm["runs"]) >= int(accept["seeds_required"]),
            "GPU independent seed count below frozen seeds_required")
    gpu_seeds = [int(r.get("random_seed", -1)) for r in gm["runs"]]
    require(len(set(gpu_seeds)) == len(gpu_seeds),
            "GPU runs must use independent unique seeds")
    topas_seeds = [int(r.get("random_seed", -1)) for r in tm["runs"]]
    require(len(tm["runs"]) >= 3 and len(set(topas_seeds)) == len(topas_seeds),
            "TOPAS requires at least three independent, uniquely seeded reference batches")
    gpu_runs = []
    geom = None
    shape = None
    for run in gm["runs"]:
        dose, g = load_gpu_run(run, expected_count=1)
        if shape is None:
            shape = dose.shape
            geom = g
        else:
            require(dose.shape == shape and g["dx_mm"] == geom["dx_mm"] and
                    g["dz_mm"] == geom["dz_mm"] and
                    g["x0_center_mm"] == geom["x0_center_mm"] and
                    g["z0_center_mm"] == geom["z0_center_mm"],
                    "GPU seed scorer geometry differs")
        gpu_runs.append((dose, g))
    topas_runs = []
    for run in tm["runs"]:
        dose, g = load_topas_run(run, shape, geom)
        require(dose.shape == shape, "TOPAS reference grid differs from GPU")
        topas_runs.append((dose, g))
    require(len(set((g["dx_mm"], g["dz_mm"], g["dy_mm"],
                     g["x0_center_mm"], g["z0_center_mm"]) for _, g in gpu_runs+topas_runs)) == 1,
            "GPU/TOPAS scoring grids are not registered identically")

    nx, nz = shape[1], shape[0]
    x = geom["x0_center_mm"] + np.arange(nx) * geom["dx_mm"]
    z = geom["z0_center_mm"] + np.arange(nz) * geom["dz_mm"]
    peak_mask = np.abs(x) < 0.25
    valley_mask = np.abs(np.abs(x) - 1.8) < 0.45
    require(np.any(peak_mask) and np.any(valley_mask), "frozen peak/valley ROI is outside scorer")
    nx_required = [float(d) for d in accept["depths_mm"]]
    rows = []
    any_fail, any_block = False, False
    for dep in nx_required:
        iz = int(np.argmin(np.abs(z-dep)))
        require(abs(z[iz]-dep) <= geom["dz_mm"]*0.5 + 1e-9,
                f"required depth {dep} mm is outside scorer bins")
        metric_sets = {"dose_sum": [], "peak": [], "valley": [], "pvdr": [],
                       "plane_energy_J_per_primary": []}
        topas_sets = {key: [] for key in metric_sets}
        for dose, g in gpu_runs:
            line = dose[iz]
            pm, vm = float(line[peak_mask].mean()), float(line[valley_mask].mean())
            metric_sets["dose_sum"].append(float(line.sum()))
            metric_sets["peak"].append(pm)
            metric_sets["valley"].append(vm)
            metric_sets["pvdr"].append(pm/vm if vm > 0 else math.nan)
            metric_sets["plane_energy_J_per_primary"].append(float(line.sum()*g["voxel_mass_kg"]))
        for dose, g in topas_runs:
            line = dose[iz]
            pm, vm = float(line[peak_mask].mean()), float(line[valley_mask].mean())
            topas_sets["dose_sum"].append(float(line.sum()))
            topas_sets["peak"].append(pm)
            topas_sets["valley"].append(vm)
            topas_sets["pvdr"].append(pm/vm if vm > 0 else math.nan)
            topas_sets["plane_energy_J_per_primary"].append(float(line.sum()*g["voxel_mass_kg"]))
        gates = {
            "transverse_row_sum": _ratio_gate(metric_sets["dose_sum"], topas_sets["dose_sum"], float(accept["total_deposit_rel"])),
            "peak_roi_mean": _ratio_gate(metric_sets["peak"], topas_sets["peak"], float(accept["peak_mean_rel"])),

        }
        topas_val = float(np.mean(topas_sets["valley"]))
        if all(math.isfinite(v) for v in metric_sets["pvdr"] + topas_sets["pvdr"]):
            gates["pvdr"] = _ratio_gate(metric_sets["pvdr"], topas_sets["pvdr"], float(accept["pvdr_rel"]))
        else:
            gates["pvdr"] = {"status": "INCONCLUSIVE", "reason": "zero valley: PVDR undefined"}
        if topas_val < float(accept["valley_abs_floor_Gy_per_history"]):
            gates["valley_roi_mean"] = _abs_gate(metric_sets["valley"], topas_sets["valley"],
                                                 float(accept["valley_abs_floor_Gy_per_history"]))
        else:
            gates["valley_roi_mean"] = _ratio_gate(metric_sets["valley"], topas_sets["valley"], float(accept["valley_mean_rel"]))
        widths_gpu = [_fwhm_1d(dose[iz], x) for dose, _ in gpu_runs]
        widths_topas = [_fwhm_1d(dose[iz], x) for dose, _ in topas_runs]
        if any(v is None for v in widths_gpu+widths_topas):
            gates["transverse_beamlet_fwhm"] = {"status": "INCONCLUSIVE", "reason": "undefined half-height crossings"}
        else:
            gates["transverse_beamlet_fwhm"] = _ratio_gate(
                [float(v) for v in widths_gpu], [float(v) for v in widths_topas], float(accept["fwhm_rel"]))
            gates["transverse_beamlet_fwhm"]["definition"] = "D(x, fixed depth), linearly interpolated half-height crossings"
        row_statuses = [v["status"] for v in gates.values()]
        status = "FAIL" if "FAIL" in row_statuses else "INCONCLUSIVE" if "INCONCLUSIVE" in row_statuses else "PASS"
        any_fail |= status == "FAIL"
        any_block |= status == "INCONCLUSIVE"
        rows.append({"depth_mm": dep, "scorer_z_bin_center_mm": float(z[iz]),
                     "plane_energy_J_per_primary_gpu": _stats(metric_sets["plane_energy_J_per_primary"]),
                     "plane_energy_J_per_primary_topas": _stats(topas_sets["plane_energy_J_per_primary"]),
                     "gates": gates, "status": status})

    # Whole scored volume: sum(D_i * voxel_mass_i), reported separately from
    # depth-wise transverse row integrals used by the frozen gate.
    gpu_volume = [float(dose.sum()*g["voxel_mass_kg"]) for dose, g in gpu_runs]
    topas_volume = [float(dose.sum()*g["voxel_mass_kg"]) for dose, g in topas_runs]
    full_volume = _ratio_gate(gpu_volume, topas_volume, float(accept["total_deposit_rel"]))
    any_fail |= full_volume["status"] == "FAIL"
    any_block |= full_volume["status"] == "INCONCLUSIVE"
    central = int(np.argmin(np.abs(x)))
    widths_long_g = [_fwhm_1d(dose[:, central], z, None) for dose, _ in gpu_runs]
    widths_long_t = [_fwhm_1d(dose[:, central], z, None) for dose, _ in topas_runs]
    # Here _fwhm_x is a generic 1D linear crossing finder; label the axis
    # explicitly so it cannot be confused with transverse beamlet FWHM.
    longitudinal = {"gpu_mm": widths_long_g, "topas_mm": widths_long_t,
                    "definition": "central-axis D(z) depth-profile width; diagnostic only"}
    if any(v is None for v in widths_long_g+widths_long_t):
        longitudinal["status"] = "INCONCLUSIVE"
    else:
        longitudinal["gpu_stats"] = _stats([float(v) for v in widths_long_g])
        longitudinal["topas_stats"] = _stats([float(v) for v in widths_long_t])

    # Distal 80% crossing of the historical central-axis depth profile,
    # now linearly interpolated between voxel centers.
    r80_g = [_r80(dose[:, central], z) for dose, _ in gpu_runs]
    r80_t = [_r80(dose[:, central], z) for dose, _ in topas_runs]
    if any(v is None for v in r80_g+r80_t):
        r80 = {"status": "INCONCLUSIVE", "reason": "distal 80% crossing is undefined"}
        any_block = True
    else:
        gs, ts = _stats([float(v) for v in r80_g]), _stats([float(v) for v in r80_t])
        bin_unc = math.sqrt(2.0) * geom["dz_mm"] / math.sqrt(12.0)
        diff = gs["mean"] - ts["mean"]
        se = math.hypot(gs["se"], ts["se"])
        margin = abs(diff) + 2.0*se + bin_unc
        r80 = {"status": "PASS" if margin < float(accept["r80_abs_mm"]) else "FAIL",
               "gpu_mm": gs, "topas_mm": ts, "difference_mm": diff,
               "combined_run_se_mm": se, "bin_resolution_uncertainty_mm": bin_unc,
               "equivalence_margin": margin, "threshold_mm": float(accept["r80_abs_mm"]),
               "definition": "distal 80% crossing of central-axis D(z), linearly interpolated"}
        any_fail |= r80["status"] == "FAIL"

    status = "FAIL" if any_fail else "INCONCLUSIVE" if any_block else "PASS"
    result = {"schema": "maigo.dose-gate.v2", "state": status,
              "acceptance": accept, "acceptance_sha256": sha256(accept_path),
              "gpu_manifest": str(Path(args.gpu_manifest)),
              "gpu_manifest_sha256": sha256(Path(args.gpu_manifest)),
              "topas_manifest": str(Path(args.topas_manifest)),
              "topas_manifest_sha256": sha256(Path(args.topas_manifest)),
              "gpu_runs": [g for _, g in gpu_runs],
              "topas_runs": [g for _, g in topas_runs],
              "depth_rows": rows,
              "whole_scored_volume_energy_J_per_primary": full_volume,
              "longitudinal_depth_profile_fwhm": longitudinal,
              "r80": r80,
              "roi": {"peak": "abs(x_mm) < 0.25", "valley": "abs(abs(x_mm)-1.8) < 0.45",
                      "registration": "no shift; exact scorer coordinate centers"}}
    return result, 1 if status == "FAIL" else 2 if status == "INCONCLUSIVE" else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-manifest", required=True)
    ap.add_argument("--topas-manifest", required=True)
    ap.add_argument("--accept", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        result, code = run(args)
    except Exception as exc:
        result = {"schema": "maigo.dose-gate.v2", "state": "INCONCLUSIVE",
                  "issues": [str(exc)]}
        code = 2
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                              encoding="utf-8")
    print(f"{result['state']}: wrote {args.out}")
    for issue in result.get("issues", []):
        print("INCONCLUSIVE:", issue)
    return code


if __name__ == "__main__":
    sys.exit(main())

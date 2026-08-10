#!/usr/bin/env python3
"""Localize RT07575 equal-history GPU/TOPAS dose residuals without scaling.

The GPU dose MHD is mapped into the TOPAS patient grid using the validated
TPS-90 x-negative convention in ``compare_rt07575_equal_history_groups``.
All comparisons use the fixed BODY intersect TOPAS-seed-1 >= 10% BODY-Dmax
mask.  The script is diagnostic-only: it never changes dose inputs.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import struct
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from compare_topas_seed_gamma import load  # noqa: E402
from match_gpu_to_physical_dose import dose_only_pass_rate, gamma_3d  # noqa: E402
from schneider_hu import load_schneider_table  # noqa: E402


HEADER = struct.Struct("<IIIIIffffff")
MAGIC = 0x47544343
MATERIAL_LABELS = {0: "air", 1: "lung", 2: "soft_tissue", 3: "bone"}
DENSITY_EDGES = (0.0, 0.1, 0.5, 0.9, 1.05, 1.2, 1.6, math.inf)
DOSE_LEVEL_EDGES = (10.0, 20.0, 40.0, 60.0, 80.0, 100.000001)
GRADIENT_EDGES_PERCENT_DMAX_PER_MM = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, math.inf)


def read_ct_grid(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read density and Schneider-section fields from the CCTG grid."""
    payload = path.read_bytes()
    if len(payload) < HEADER.size:
        raise ValueError(f"truncated CCTG header: {path}")
    magic, version, nx, ny, nz, *_ = HEADER.unpack_from(payload)
    if magic != MAGIC or version not in {1, 2, 3}:
        raise ValueError(f"unsupported CCTG magic/version in {path}")
    count = nx * ny * nz
    density_offset = HEADER.size
    material_offset = density_offset + 4 * count
    if len(payload) < material_offset + count:
        raise ValueError(f"truncated CCTG arrays: {path}")
    shape = (nz, ny, nx)
    density = np.frombuffer(payload, dtype="<f4", count=count, offset=density_offset)
    section = np.frombuffer(payload, dtype=np.uint8, count=count, offset=material_offset)
    return density.reshape(shape), section.reshape(shape)


def load_patient(path: Path) -> tuple[dict[str, str], np.ndarray]:
    meta, values = load(path)
    shape_xyz = tuple(int(v) for v in meta["DimSize"].split())
    return meta, np.asarray(values, dtype=np.float32).reshape(
        shape_xyz[2], shape_xyz[1], shape_xyz[0]
    )


def load_gpu_mapped(path: Path, patient_shape_zyx: tuple[int, int, int]) -> tuple[dict[str, str], np.ndarray]:
    """Map GPU [depth=patient-X, patient-Z, patient-Y] to patient [Z,Y,X]."""
    meta, values = load(path)
    shape_xyz = tuple(int(v) for v in meta["DimSize"].split())
    if shape_xyz != (505, 35, 417):
        raise ValueError(f"{path}: expected GPU DimSize 505 35 417, got {shape_xyz}")
    gpu = np.asarray(values, dtype=np.float32).reshape(shape_xyz[2], shape_xyz[1], shape_xyz[0])
    mapped = np.transpose(gpu, (1, 2, 0))[:, :, ::-1]
    if mapped.shape != patient_shape_zyx:
        raise ValueError(f"{path}: mapped shape {mapped.shape} != patient {patient_shape_zyx}")
    return meta, mapped


def pair_metrics(reference: np.ndarray, evaluation: np.ndarray, selection: np.ndarray, shape_xyz: tuple[int, int, int], spacing_xyz: tuple[float, float, float], *, gamma: bool = True) -> dict[str, Any]:
    """Dose-only metrics; signed bias is evaluation minus reference."""
    ref = reference[selection]
    ev = evaluation[selection]
    delta = ev - ref
    dmax = float(np.max(ref))
    result: dict[str, Any] = {
        "selected_voxels": int(ref.size),
        "reference_dmax": dmax,
        "nrmse_over_reference_dmax_percent": float(100.0 * np.sqrt(np.mean(delta * delta)) / max(dmax, 1.0e-30)),
        "mean_signed_bias": float(np.mean(delta)),
        "mean_signed_bias_over_reference_dmax_percent": float(100.0 * np.mean(delta) / max(dmax, 1.0e-30)),
        "mae_over_reference_dmax_percent": float(100.0 * np.mean(np.abs(delta)) / max(dmax, 1.0e-30)),
        "pearson_r": float(np.corrcoef(ref, ev)[0, 1]),
        "gamma_3pct_0mm": {
            "global": dose_only_pass_rate(reference.ravel(), evaluation.ravel(), 3.0, 10.0, selection_mask=selection.ravel()),
            "local": dose_only_pass_rate(reference.ravel(), evaluation.ravel(), 3.0, 10.0, local_dose=True, selection_mask=selection.ravel()),
        },
    }
    if gamma:
        result["gamma_1pct_1mm"] = {
            "global": gamma_3d(reference.ravel(), evaluation.ravel(), shape_xyz, spacing_xyz, 1.0, 1.0, 10.0, 50_000, 0, interpolation_step_mm=0.5, selection_mask=selection.ravel()),
            "local": gamma_3d(reference.ravel(), evaluation.ravel(), shape_xyz, spacing_xyz, 1.0, 1.0, 10.0, 50_000, 0, local_dose=True, interpolation_step_mm=0.5, selection_mask=selection.ravel()),
        }
    return result


def bin_metrics(reference: np.ndarray, evaluation: np.ndarray, mask: np.ndarray, dmax: float) -> dict[str, Any]:
    if not np.any(mask):
        return {"voxels": 0}
    delta = evaluation[mask] - reference[mask]
    return {
        "voxels": int(delta.size),
        "fraction_of_selection": float(delta.size),  # normalized by caller
        "mean_signed_bias": float(np.mean(delta)),
        "mean_signed_bias_over_topas_dmax_percent": float(100.0 * np.mean(delta) / dmax),
        "mae_over_topas_dmax_percent": float(100.0 * np.mean(np.abs(delta)) / dmax),
        "rmse_over_topas_dmax_percent": float(100.0 * np.sqrt(np.mean(delta * delta)) / dmax),
        "squared_error_sum": float(np.sum(delta * delta, dtype=np.float64)),
    }


def add_binned_metrics(reference: np.ndarray, evaluation: np.ndarray, selection: np.ndarray, density: np.ndarray, material_class: np.ndarray, spacing_xyz: tuple[float, float, float]) -> dict[str, Any]:
    """Residual strata for engine means; all percentage values use TOPAS Dmax."""
    dmax = float(np.max(reference[selection]))
    total = int(np.count_nonzero(selection))
    result: dict[str, Any] = {"dose_level_percent_of_topas_dmax": {}, "material_class": {}, "density_g_cm3": {}, "reference_gradient_percent_topas_dmax_per_mm": {}}

    def add(container: dict[str, Any], label: str, mask: np.ndarray) -> None:
        entry = bin_metrics(reference, evaluation, mask, dmax)
        entry["fraction_of_selection"] = float(entry["voxels"] / max(total, 1))
        container[label] = entry

    dose_percent = 100.0 * reference / dmax
    for low, high in zip(DOSE_LEVEL_EDGES[:-1], DOSE_LEVEL_EDGES[1:]):
        close = "]" if high > 100.0 else ")"
        add(result["dose_level_percent_of_topas_dmax"], f"[{low:g},{min(high, 100.0):g}{close}", selection & (dose_percent >= low) & (dose_percent < high))
    for class_id, label in MATERIAL_LABELS.items():
        add(result["material_class"], label, selection & (material_class == class_id))
    for low, high in zip(DENSITY_EDGES[:-1], DENSITY_EDGES[1:]):
        close = ")" if math.isfinite(high) else "]"
        high_label = f"{high:g}" if math.isfinite(high) else "inf"
        add(result["density_g_cm3"], f"[{low:g},{high_label}{close}", selection & (density >= low) & (density < high))
    # np.gradient receives (z,y,x) spacing, while MHD records (x,y,z).
    gz, gy, gx = np.gradient(reference, spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    grad_percent = 100.0 * np.sqrt(gx * gx + gy * gy + gz * gz) / dmax
    for low, high in zip(GRADIENT_EDGES_PERCENT_DMAX_PER_MM[:-1], GRADIENT_EDGES_PERCENT_DMAX_PER_MM[1:]):
        close = ")" if math.isfinite(high) else "]"
        high_label = f"{high:g}" if math.isfinite(high) else "inf"
        add(result["reference_gradient_percent_topas_dmax_per_mm"], f"[{low:g},{high_label}{close}", selection & (grad_percent >= low) & (grad_percent < high))
    del gz, gy, gx, grad_percent
    return result


def sample_trilinear(volume: np.ndarray, z: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Trilinearly sample valid index coordinates; callers guarantee bounds."""
    z0 = np.floor(z).astype(np.intp)
    y0 = np.floor(y).astype(np.intp)
    x0 = np.floor(x).astype(np.intp)
    z1 = np.minimum(z0 + 1, volume.shape[0] - 1)
    y1 = np.minimum(y0 + 1, volume.shape[1] - 1)
    x1 = np.minimum(x0 + 1, volume.shape[2] - 1)
    fz, fy, fx = z - z0, y - y0, x - x0
    c00 = volume[z0, y0, x0] * (1.0 - fx) + volume[z0, y0, x1] * fx
    c01 = volume[z0, y1, x0] * (1.0 - fx) + volume[z0, y1, x1] * fx
    c10 = volume[z1, y0, x0] * (1.0 - fx) + volume[z1, y0, x1] * fx
    c11 = volume[z1, y1, x0] * (1.0 - fx) + volume[z1, y1, x1] * fx
    return (c00 * (1.0 - fy) + c01 * fy) * (1.0 - fz) + (c10 * (1.0 - fy) + c11 * fy) * fz


def sampled_shifted_values(volume: np.ndarray, linear: np.ndarray, shift_xyz_mm: tuple[float, float, float], shape_xyz: tuple[int, int, int], spacing_xyz: tuple[float, float, float]) -> np.ndarray:
    nx, ny, _nz = shape_xyz
    z = linear // (nx * ny)
    rem = linear % (nx * ny)
    y, x = rem // nx, rem % nx
    # A positive shift moves the evaluation field toward positive patient axis:
    # E_shift(r) = E(r - shift), so it is sampled at index - shift/spacing.
    sx, sy, sz = shift_xyz_mm
    return sample_trilinear(volume, z - sz / spacing_xyz[2], y - sy / spacing_xyz[1], x - sx / spacing_xyz[0])


def linear_shift_axis(volume: np.ndarray, shift_index: float, axis: int) -> np.ndarray:
    """Zero-padded linear resampling where output(i)=input(i-shift_index)."""
    n = volume.shape[axis]
    coordinates = np.arange(n, dtype=np.float64) - shift_index
    low = np.floor(coordinates).astype(np.intp)
    frac = (coordinates - low).astype(volume.dtype, copy=False)
    valid = (low >= 0) & (low < n)
    low_clamped = np.clip(low, 0, n - 1)
    high_clamped = np.clip(low + 1, 0, n - 1)
    lo = np.take(volume, low_clamped, axis=axis)
    hi = np.take(volume, high_clamped, axis=axis)
    shape = [1, 1, 1]
    shape[axis] = n
    out = lo * (1.0 - frac.reshape(shape)) + hi * frac.reshape(shape)
    return np.where(valid.reshape(shape), out, 0.0)


def shifted_volume(volume: np.ndarray, shift_xyz_mm: tuple[float, float, float], spacing_xyz: tuple[float, float, float]) -> np.ndarray:
    sx, sy, sz = shift_xyz_mm
    result = linear_shift_axis(volume, sx / spacing_xyz[0], 2)
    result = linear_shift_axis(result, sy / spacing_xyz[1], 1)
    return linear_shift_axis(result, sz / spacing_xyz[2], 0)


def rigid_search(reference: np.ndarray, evaluation: np.ndarray, selection: np.ndarray, shape_xyz: tuple[int, int, int], spacing_xyz: tuple[float, float, float]) -> dict[str, Any]:
    """Coarse 3-D and focused y-axis shift search using a fixed 50k subset."""
    _nx, _ny, _nz = shape_xyz
    # Avoid materializing three full index grids: shifts are bounded by 1 mm,
    # so this fixed edge exclusion keeps every sampled coordinate in-bounds.
    eligible = selection.copy()
    eligible[:, :, :2] = False
    eligible[:, :, -2:] = False
    eligible[:, :2, :] = False
    eligible[:, -2:, :] = False
    eligible[:1, :, :] = False
    eligible[-1:, :, :] = False
    indices = np.flatnonzero(eligible.ravel())
    rng = np.random.default_rng(20260804)
    picked = rng.choice(indices, size=min(50_000, indices.size), replace=False)
    ref = reference.ravel()[picked]
    dmax = float(np.max(reference[selection]))
    shifts = tuple(float(v) for v in np.arange(-1.0, 1.0 + 1e-9, 0.25))
    candidates: list[tuple[float, tuple[float, float, float]]] = []
    for shift in product(shifts, repeat=3):
        values = sampled_shifted_values(evaluation, picked, shift, shape_xyz, spacing_xyz)
        candidates.append((float(np.sqrt(np.mean((values - ref) ** 2)) / dmax), shift))
    baseline = next(score for score, shift in candidates if shift == (0.0, 0.0, 0.0))
    coarse_score, coarse_shift = min(candidates, key=lambda item: item[0])
    # The coarse optimum is at y=+0.25 mm.  Refine this physically meaningful
    # single-axis registration sensitivity at 0.05 mm without broadening the
    # search into a fitted affine transform or dose calibration.
    focused_y = tuple(float(v) for v in np.arange(coarse_shift[1] - 0.25, coarse_shift[1] + 0.250001, 0.05))
    focused_candidates: list[tuple[float, tuple[float, float, float]]] = []
    for y_shift in focused_y:
        shift = (coarse_shift[0], y_shift, coarse_shift[2])
        values = sampled_shifted_values(evaluation, picked, shift, shape_xyz, spacing_xyz)
        focused_candidates.append((float(np.sqrt(np.mean((values - ref) ** 2)) / dmax), shift))
    best_score, best_shift = min(focused_candidates, key=lambda item: item[0])
    best_volume = shifted_volume(evaluation, best_shift, spacing_xyz)
    unshifted_full = float(np.sqrt(np.mean((evaluation[selection] - reference[selection]) ** 2)) / dmax)
    shifted_full = float(np.sqrt(np.mean((best_volume[selection] - reference[selection]) ** 2)) / dmax)
    strict = pair_metrics(reference, best_volume, selection, shape_xyz, spacing_xyz, gamma=True)
    return {
        "definition": "positive component moves GPU evaluation dose toward positive patient x/y/z; E_shift(r)=E(r-shift)",
        "search_range_mm": [-1.0, 1.0],
        "increment_mm": 0.25,
        "candidate_count": len(candidates),
        "optimization": "exhaustive deterministic 50,000-voxel subset of an edge-safe fixed selection; no dose scale fit",
        "eligible_selected_voxels": int(indices.size),
        "subset_voxels": int(picked.size),
        "subset_unshifted_nrmse_over_dmax_percent": 100.0 * baseline,
        "coarse_best_shift_xyz_mm": list(coarse_shift),
        "coarse_subset_best_nrmse_over_dmax_percent": 100.0 * coarse_score,
        "focused_refinement": {"axis": "patient_y", "range_mm": [focused_y[0], focused_y[-1]], "increment_mm": 0.05, "candidate_count": len(focused_candidates), "subset_best_nrmse_over_dmax_percent": 100.0 * best_score},
        "best_shift_xyz_mm": list(best_shift),
        "full_selection_unshifted_nrmse_over_dmax_percent": 100.0 * unshifted_full,
        "full_selection_best_nrmse_over_dmax_percent": 100.0 * shifted_full,
        "full_selection_nrmse_relative_improvement_percent": float(100.0 * (unshifted_full - shifted_full) / unshifted_full),
        "strict_metrics_at_best_shift": strict,
    }


def dose_weighted_com(volume: np.ndarray, selection: np.ndarray, spacing_xyz: tuple[float, float, float], offset_xyz: tuple[float, float, float]) -> list[float]:
    """Physical patient xyz COM on the fixed selection, weighted by dose."""
    weights = np.where(selection, volume, 0.0)
    total = float(np.sum(weights, dtype=np.float64))
    if total <= 0.0:
        return [float("nan")] * 3
    x = offset_xyz[0] + spacing_xyz[0] * np.arange(volume.shape[2])
    y = offset_xyz[1] + spacing_xyz[1] * np.arange(volume.shape[1])
    z = offset_xyz[2] + spacing_xyz[2] * np.arange(volume.shape[0])
    return [
        float(np.dot(np.sum(weights, axis=(0, 1), dtype=np.float64), x) / total),
        float(np.dot(np.sum(weights, axis=(0, 2), dtype=np.float64), y) / total),
        float(np.dot(np.sum(weights, axis=(1, 2), dtype=np.float64), z) / total),
    ]


def gaussian_kernel(sigma_index: float) -> np.ndarray:
    if sigma_index <= 1.0e-12:
        return np.array([1.0])
    radius = int(math.ceil(3.0 * sigma_index))
    coordinate = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (coordinate / sigma_index) ** 2)
    return kernel / np.sum(kernel)


def gaussian_axis(volume: np.ndarray, sigma_index: float, axis: int) -> np.ndarray:
    kernel = gaussian_kernel(sigma_index)
    radius = kernel.size // 2
    if radius == 0:
        return volume.copy()
    padded = np.pad(volume, [(radius, radius) if i == axis else (0, 0) for i in range(3)], mode="edge")
    out = np.zeros_like(volume)
    for k, weight in enumerate(kernel):
        source = [slice(None)] * 3
        source[axis] = slice(k, k + volume.shape[axis])
        out += np.asarray(weight, dtype=volume.dtype) * padded[tuple(source)]
    return out


def gaussian_smooth(volume: np.ndarray, sigma_mm: float, spacing_xyz: tuple[float, float, float]) -> np.ndarray:
    result = volume
    for axis, spacing in ((2, spacing_xyz[0]), (1, spacing_xyz[1]), (0, spacing_xyz[2])):
        result = gaussian_axis(result, sigma_mm / spacing, axis)
    return result


def conserved_anisotropic_gaussian(
    volume: np.ndarray,
    sigma_lateral_yz_mm: float,
    sigma_beam_x_mm: float,
    spacing_xyz: tuple[float, float, float],
) -> tuple[np.ndarray, dict[str, float]]:
    """Blur GPU dose only, then restore its own total dose exactly by design.

    Patient X is the beam-depth axis for the TPS-90 x-negative field; patient
    Y/Z are lateral.  The final factor is an internal finite-grid conservation
    correction, not a fit to TOPAS or a dose normalization calibration.
    """
    result = volume
    for axis, sigma_mm, spacing in (
        (2, sigma_beam_x_mm, spacing_xyz[0]),
        (1, sigma_lateral_yz_mm, spacing_xyz[1]),
        (0, sigma_lateral_yz_mm, spacing_xyz[2]),
    ):
        if sigma_mm > 0.0:
            result = gaussian_axis(result, sigma_mm / spacing, axis)
    total_before = float(np.sum(volume, dtype=np.float64))
    total_blurred = float(np.sum(result, dtype=np.float64))
    conservation_factor = total_before / max(total_blurred, 1.0e-30)
    # This normalizes only to the input GPU total and makes edge treatment
    # exactly energy-conserving to float precision; it never uses TOPAS.
    result = result * conservation_factor
    total_after = float(np.sum(result, dtype=np.float64))
    return result, {
        "input_total_dose": total_before,
        "blurred_total_dose_before_conservation": total_blurred,
        "input_over_blurred_conservation_factor": conservation_factor,
        "output_total_dose": total_after,
        "output_over_input_total": total_after / max(total_before, 1.0e-30),
    }


def gpu_only_blur_scan(
    reference: np.ndarray,
    gpu_evaluation: np.ndarray,
    selection: np.ndarray,
    shape_xyz: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
) -> dict[str, Any]:
    """Grid-aware 0--1 mm electronic-nonlocality proxy scan, no scale/shift."""
    widths_mm = (0.0, 0.25, 0.5, 0.75, 1.0)
    candidates: list[dict[str, Any]] = []
    best_volume: np.ndarray | None = None
    best_candidate: dict[str, Any] | None = None
    for lateral_yz_mm, beam_x_mm in product(widths_mm, repeat=2):
        blurred, conservation = conserved_anisotropic_gaussian(
            gpu_evaluation, lateral_yz_mm, beam_x_mm, spacing_xyz
        )
        metric = pair_metrics(
            reference, blurred, selection, shape_xyz, spacing_xyz, gamma=False
        )
        candidate = {
            "sigma_lateral_patient_yz_mm": lateral_yz_mm,
            "sigma_beam_depth_patient_x_mm": beam_x_mm,
            "nrmse_over_reference_dmax_percent": metric[
                "nrmse_over_reference_dmax_percent"
            ],
            "gamma_3pct_0mm": metric["gamma_3pct_0mm"],
            "conservation": conservation,
        }
        candidates.append(candidate)
        if best_candidate is None or candidate[
            "nrmse_over_reference_dmax_percent"
        ] < best_candidate["nrmse_over_reference_dmax_percent"]:
            best_candidate, best_volume = candidate, blurred
        else:
            del blurred
    assert best_candidate is not None and best_volume is not None
    strict = pair_metrics(
        reference, best_volume, selection, shape_xyz, spacing_xyz, gamma=True
    )
    baseline = next(
        item
        for item in candidates
        if item["sigma_lateral_patient_yz_mm"] == 0.0
        and item["sigma_beam_depth_patient_x_mm"] == 0.0
    )
    improvement = 100.0 * (
        baseline["nrmse_over_reference_dmax_percent"]
        - best_candidate["nrmse_over_reference_dmax_percent"]
    ) / baseline["nrmse_over_reference_dmax_percent"]
    if (
        best_candidate["sigma_lateral_patient_yz_mm"] == 0.0
        and best_candidate["sigma_beam_depth_patient_x_mm"] == 0.0
    ):
        direction = (
            "No positive Gaussian broadening helps; this blur-only proxy does "
            "not support missing delta-electron nonlocality. If width were the "
            "only mechanism, the indicated direction would be sharpening, which "
            "this physical proxy does not test."
        )
    else:
        direction = (
            "Positive GPU broadening improves NRMSE, consistent with (but not "
            "proof of) missing nonlocal electronic dose deposition."
        )
    return {
        "purpose": "GPU-only energy-conserving anisotropic Gaussian mechanism diagnostic; not accepted calibration or physics substitution",
        "reference": "TOPAS engine mean, unchanged",
        "evaluation": "GPU engine mean only; no fitted dose scale and no translation applied or jointly fitted",
        "orientation": "patient X is beam depth; patient Y/Z are lateral for TPS-90 x-negative mapping",
        "widths_mm": list(widths_mm),
        "candidate_count": len(candidates),
        "grid_aware_kernel_note": "Gaussian sigma is converted independently by x/y/z voxel spacing (0.5/0.5/2.0 mm); sub-voxel z kernels are retained rather than rounded to a voxel",
        "candidates": candidates,
        "best_nrmse_candidate": best_candidate,
        "strict_metrics_at_best_nrmse_candidate": strict,
        "nrmse_relative_improvement_percent_vs_unblurred_gpu": improvement,
        "broadening_or_sharpening_interpretation": direction,
    }


def dominance(binned: dict[str, Any]) -> dict[str, Any]:
    """Select the populated stratum contributing most squared residual."""
    answer: dict[str, Any] = {}
    for name, entries in binned.items():
        valid = [(label, entry) for label, entry in entries.items() if entry["voxels"]]
        if valid:
            family_sse = sum(entry["squared_error_sum"] for _, entry in valid)
            label, entry = max(valid, key=lambda item: item[1]["squared_error_sum"])
            answer[name] = {"bin": label, "mean_signed_bias_over_topas_dmax_percent": entry["mean_signed_bias_over_topas_dmax_percent"], "rmse_over_topas_dmax_percent": entry["rmse_over_topas_dmax_percent"], "voxels": entry["voxels"], "fraction_of_family_squared_error": entry["squared_error_sum"] / family_sse}
    return answer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-seed1", type=Path, required=True)
    parser.add_argument("--gpu-seed2", type=Path, required=True)
    parser.add_argument("--topas-seed1", type=Path, required=True)
    parser.add_argument("--topas-seed2", type=Path, required=True)
    parser.add_argument("--body-mask", type=Path, required=True)
    parser.add_argument("--ct-grid", type=Path, required=True)
    parser.add_argument("--schneider-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    topas_meta, topas1 = load_patient(args.topas_seed1 / "dose.mhd")
    _topas2_meta, topas2 = load_patient(args.topas_seed2 / "dose.mhd")
    shape_xyz = tuple(int(v) for v in topas_meta["DimSize"].split())
    spacing_xyz = tuple(float(v) for v in topas_meta["ElementSpacing"].split())
    offset_xyz = tuple(float(v) for v in topas_meta.get("Offset", "0 0 0").split())
    _body_meta, body_values = load_patient(args.body_mask)
    body = body_values > 0.5
    if body.shape != topas1.shape:
        raise ValueError(f"BODY shape {body.shape} != TOPAS shape {topas1.shape}")
    _gpu1_meta, gpu1 = load_gpu_mapped(args.gpu_seed1 / "dose.mhd", topas1.shape)
    _gpu2_meta, gpu2 = load_gpu_mapped(args.gpu_seed2 / "dose.mhd", topas1.shape)
    topas_dmax = float(np.max(topas1[body]))
    selection = body & (topas1 >= 0.10 * topas_dmax)
    density_gpu, section_gpu = read_ct_grid(args.ct_grid)
    table = load_schneider_table(args.schneider_file)
    if int(section_gpu.max()) >= table.material_class.size:
        raise ValueError("CT-grid Schneider section exceeds Schneider table")
    material_gpu = table.material_class[section_gpu]
    # CT is stored in GPU [depth=patient-X, patient-Z, patient-Y]; map it to patient.
    density = np.transpose(density_gpu, (1, 2, 0))[:, :, ::-1]
    material = np.transpose(material_gpu, (1, 2, 0))[:, :, ::-1]
    if density.shape != topas1.shape:
        raise ValueError(f"mapped CT grid shape {density.shape} != patient {topas1.shape}")

    gpu_mean = 0.5 * (gpu1 + gpu2)
    topas_mean = 0.5 * (topas1 + topas2)
    topas_com = dose_weighted_com(topas_mean, selection, spacing_xyz, offset_xyz)
    gpu_com = dose_weighted_com(gpu_mean, selection, spacing_xyz, offset_xyz)
    print("Computing four fixed-mask comparisons (strict gamma is deterministic 50k/0.5mm)...", flush=True)
    comparisons = {
        "gpu_seed1_vs_topas_seed1": pair_metrics(topas1, gpu1, selection, shape_xyz, spacing_xyz),
        "gpu_mean_vs_topas_mean": pair_metrics(topas_mean, gpu_mean, selection, shape_xyz, spacing_xyz),
        "gpu_seed2_minus_seed1": pair_metrics(gpu1, gpu2, selection, shape_xyz, spacing_xyz),
        "topas_seed2_minus_seed1": pair_metrics(topas1, topas2, selection, shape_xyz, spacing_xyz),
    }
    print("Computing engine-mean strata, rigid translation search, and smoothing tests...", flush=True)
    binned = add_binned_metrics(topas_mean, gpu_mean, selection, density, material, spacing_xyz)
    rigid = rigid_search(topas_mean, gpu_mean, selection, shape_xyz, spacing_xyz)
    gc.collect()
    smoothing: dict[str, Any] = {}
    for sigma_mm in (0.5, 1.0):
        smoothing[f"sigma_{sigma_mm:.1f}_mm"] = pair_metrics(
            gaussian_smooth(topas_mean, sigma_mm, spacing_xyz),
            gaussian_smooth(gpu_mean, sigma_mm, spacing_xyz),
            selection, shape_xyz, spacing_xyz,
        )
        gc.collect()
    print("Scanning GPU-only energy-conserving anisotropic blur proxy...", flush=True)
    blur_scan = gpu_only_blur_scan(
        topas_mean, gpu_mean, selection, shape_xyz, spacing_xyz
    )

    report: dict[str, Any] = {
        "case": "RT07575",
        "histories_per_seed": 12_963_817,
        "normalization": "absolute equal-history dose; no fitted dose scale or case-specific calibration",
        "input_paths": {"gpu_seed1": str(args.gpu_seed1), "gpu_seed2": str(args.gpu_seed2), "topas_seed1": str(args.topas_seed1), "topas_seed2": str(args.topas_seed2), "body_mask": str(args.body_mask), "ct_grid": str(args.ct_grid), "schneider_file": str(args.schneider_file)},
        "grid": {"shape_xyz": list(shape_xyz), "shape_zyx": list(topas1.shape), "spacing_xyz_mm": list(spacing_xyz), "topas_offset_xyz_mm": list(offset_xyz)},
        "mapping": "GPU MHD [depth=patient-X, patient-Z, patient-Y] -> patient [Z,Y,X] by transpose(1,2,0) then reverse patient X; same convention as compare_rt07575_equal_history_groups.py",
        "selection": {"definition": "BODY intersect TOPAS seed1 dose >= 10% of BODY Dmax", "body_voxels": int(np.count_nonzero(body)), "selected_voxels": int(np.count_nonzero(selection)), "topas_seed1_body_dmax": topas_dmax},
        "comparisons": comparisons,
        "engine_mean_dose_weighted_com_patient_xyz_mm": {"selection": "fixed BODY intersect TOPAS seed1 >=10% Dmax", "topas_mean": topas_com, "gpu_mean": gpu_com, "gpu_minus_topas": [gpu_com[i] - topas_com[i] for i in range(3)]},
        "engine_mean_residual_bins": binned,
        "dominant_bins_by_total_squared_error": dominance(binned),
        "rigid_translation_gpu_engine_mean": rigid,
        "matched_gaussian_smoothing_engine_means": smoothing,
        "gpu_only_energy_conserving_anisotropic_blur_scan": blur_scan,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    mean = comparisons["gpu_mean_vs_topas_mean"]
    lines = [
        "# RT07575 equal-history dose residual diagnostic",
        "",
        "Absolute equal-history dose comparison; no fitted dose scale or calibration.",
        f"Fixed selection: BODY ∩ TOPAS seed 1 ≥10% BODY Dmax = {report['selection']['selected_voxels']:,} voxels (BODY {report['selection']['body_voxels']:,}); grid {shape_xyz} xyz at {spacing_xyz} mm.",
        "",
        "| Comparison (evaluation − reference) | NRMSE/Dmax | signed bias/Dmax | MAE/Dmax | Pearson r | 3%/0 mm G/L | 1%/1 mm G/L |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metric in comparisons.items():
        lines.append(f"| {name} | {metric['nrmse_over_reference_dmax_percent']:.3f}% | {metric['mean_signed_bias_over_reference_dmax_percent']:.3f}% | {metric['mae_over_reference_dmax_percent']:.3f}% | {metric['pearson_r']:.6f} | {metric['gamma_3pct_0mm']['global']['pass_percent']:.3f}% / {metric['gamma_3pct_0mm']['local']['pass_percent']:.3f}% | {metric['gamma_1pct_1mm']['global']['pass_percent']:.3f}% / {metric['gamma_1pct_1mm']['local']['pass_percent']:.3f}% |")
    lines.extend([
        "",
        f"Rigid GPU shift: best {rigid['best_shift_xyz_mm']} mm (positive moves GPU toward +patient axis), full-mask NRMSE {rigid['full_selection_unshifted_nrmse_over_dmax_percent']:.3f}% → {rigid['full_selection_best_nrmse_over_dmax_percent']:.3f}% ({rigid['full_selection_nrmse_relative_improvement_percent']:.2f}% relative improvement).",
        f"Dose-weighted COM (GPU mean − TOPAS mean) on the fixed mask: {[round(gpu_com[i] - topas_com[i], 5) for i in range(3)]} mm in patient xyz.",
        "",
        "## Matched smoothing of engine means",
        "",
        "| Gaussian sigma | NRMSE/Dmax | 3%/0 mm G/L | 1%/1 mm G/L |",
        "|---|---:|---:|---:|",
    ])
    for label, metric in smoothing.items():
        lines.append(f"| {label} | {metric['nrmse_over_reference_dmax_percent']:.3f}% | {metric['gamma_3pct_0mm']['global']['pass_percent']:.3f}% / {metric['gamma_3pct_0mm']['local']['pass_percent']:.3f}% | {metric['gamma_1pct_1mm']['global']['pass_percent']:.3f}% / {metric['gamma_1pct_1mm']['local']['pass_percent']:.3f}% |")
    blur_best = blur_scan["best_nrmse_candidate"]
    blur_strict = blur_scan["strict_metrics_at_best_nrmse_candidate"]
    lines.extend([
        "",
        "## GPU-only energy-conserving anisotropic blur scan",
        "",
        "Mechanism diagnostic only—not accepted calibration or a source-physics substitute. TOPAS is unchanged; GPU is blurred without fitted dose scale or translation. Patient X is beam depth and Y/Z are lateral.",
        "",
        "| Lateral Y/Z sigma \\ beam-X sigma (mm) | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    candidate_map = {
        (item["sigma_lateral_patient_yz_mm"], item["sigma_beam_depth_patient_x_mm"]): item
        for item in blur_scan["candidates"]
    }
    for lateral_yz_mm in blur_scan["widths_mm"]:
        cells = [
            f"{candidate_map[(lateral_yz_mm, beam_x_mm)]['nrmse_over_reference_dmax_percent']:.3f}%"
            for beam_x_mm in blur_scan["widths_mm"]
        ]
        lines.append(f"| {lateral_yz_mm:.2f} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        f"Best kernel: lateral Y/Z sigma={blur_best['sigma_lateral_patient_yz_mm']:.2f} mm, beam-X sigma={blur_best['sigma_beam_depth_patient_x_mm']:.2f} mm; NRMSE {blur_strict['nrmse_over_reference_dmax_percent']:.3f}% ({blur_scan['nrmse_relative_improvement_percent_vs_unblurred_gpu']:.2f}% relative vs unblurred).",
        f"Best-kernel strict 3%/0 mm: {blur_strict['gamma_3pct_0mm']['global']['pass_percent']:.3f}% / {blur_strict['gamma_3pct_0mm']['local']['pass_percent']:.3f}%; 1%/1 mm: {blur_strict['gamma_1pct_1mm']['global']['pass_percent']:.3f}% / {blur_strict['gamma_1pct_1mm']['local']['pass_percent']:.3f}%.",
        blur_scan["broadening_or_sharpening_interpretation"],
    ])
    lines.extend(["", "## Dominant engine-mean residual strata (total squared-error contribution)", ""])
    for family, entry in report["dominant_bins_by_total_squared_error"].items():
        lines.append(f"- {family}: {entry['bin']} ({entry['voxels']:,} voxels; {100.0 * entry['fraction_of_family_squared_error']:.1f}% of residual SSE), signed bias {entry['mean_signed_bias_over_topas_dmax_percent']:.3f}% Dmax; RMSE {entry['rmse_over_topas_dmax_percent']:.3f}% Dmax.")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

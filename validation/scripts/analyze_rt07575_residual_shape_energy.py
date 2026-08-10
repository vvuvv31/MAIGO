#!/usr/bin/env python3
"""RT07575 air_loss residual shape + energy decomposition (diagnostic only).

Quantifies how much residual SSE can be removed by non-physical transforms
(global scale, rigid shift, anisotropic blur) and decomposes residual against
primary/secondary partitions and energy-budget logs.

No dose calibration is applied to the reported baseline metrics; scale/shift/
blur are upper-bound mechanism diagnostics only.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_rt07575_equal_history_dose_residual import (  # noqa: E402
    conserved_anisotropic_gaussian,
    dose_weighted_com,
    gaussian_axis,
    load_gpu_mapped,
    load_patient,
    pair_metrics,
    rigid_search,
    shifted_volume,
)
from compare_topas_seed_gamma import load  # noqa: E402


def number(text: str, pattern: str, default: float | None = None) -> float:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        if default is not None:
            return default
        raise ValueError(f"pattern not found: {pattern}")
    return float(match.group(1))


def parse_energy_budget(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    keys = {
        "histories": r"^Histories:\s*([0-9.eE+-]+)$",
        "initial_energy_MeVu": r"^Initial energy:\s*([0-9.eE+-]+)\s+MeV/u",
        "initial_energy_MeV_per_C12": r"=\s*([0-9.eE+-]+)\s+MeV per C-12",
        "energy_balance_error": r"^Energy balance error:\s*([0-9.eE+-]+)$",
        "nuclear_interactions": r"^Nuclear interactions:\s*([0-9.eE+-]+)$",
        "sampled_reaction_packages": r"^Sampled reaction packages:\s*([0-9.eE+-]+)$",
        "generated_direct_secondaries": r"^Generated direct secondaries:\s*([0-9.eE+-]+)$",
        "generated_direct_secondary_energy_MeV": r"^Generated direct-secondary energy:\s*([0-9.eE+-]+)\s+MeV$",
        "queued_charged_secondaries": r"^Queued charged secondaries:\s*([0-9.eE+-]+)$",
        "queued_secondary_energy_MeV": r"^Queued secondary energy:\s*([0-9.eE+-]+)\s+MeV$",
        "untransported_neutral_energy_MeV": r"^Untransported neutral energy:\s*([0-9.eE+-]+)\s+MeV$",
        "nuclear_energy_not_in_secondaries_MeV": r"^Nuclear energy not in sampled direct secondaries:\s*([0-9.eE+-]+)\s+MeV$",
        "transported_charged_secondaries": r"^Transported charged secondaries:\s*([0-9.eE+-]+)$",
        "secondary_deposited_energy_MeV": r"^Secondary deposited energy:\s*([0-9.eE+-]+)\s+MeV$",
        "secondary_escaped_energy_MeV": r"^Secondary escaped energy:\s*([0-9.eE+-]+)\s+MeV$",
        "cascade_interactions": r"^Cascade interactions:\s*([0-9.eE+-]+)$",
        "generated_cascade_products": r"^Generated cascade products:\s*([0-9.eE+-]+)$",
        "queued_cascade_secondaries": r"^Queued cascade secondaries:\s*([0-9.eE+-]+)$",
        "untracked_nuclear_energy_MeV": r"^Untracked nuclear energy:\s*([0-9.eE+-]+)\s+MeV$",
        "dose_output_scale": r"^Dose output scale \(independent calibration\):\s*([0-9.eE+-]+)$",
    }
    out: dict[str, Any] = {}
    for key, pat in keys.items():
        try:
            out[key] = number(text, pat, 0.0)
        except ValueError:
            out[key] = 0.0
    gen = float(out["generated_direct_secondary_energy_MeV"])
    queued = float(out["queued_secondary_energy_MeV"])
    dep = float(out["secondary_deposited_energy_MeV"])
    esc = float(out["secondary_escaped_energy_MeV"])
    neut = float(out["untransported_neutral_energy_MeV"])
    residual_heat = float(out["nuclear_energy_not_in_secondaries_MeV"])
    untracked = float(out["untracked_nuclear_energy_MeV"])
    out["derived"] = {
        "queued_over_generated": queued / max(gen, 1.0),
        "deposited_over_queued": dep / max(queued, 1.0),
        "escaped_over_queued": esc / max(queued, 1.0),
        "neutral_over_generated": neut / max(gen, 1.0),
        "residual_heat_over_generated": residual_heat / max(gen, 1.0),
        "untracked_over_generated": untracked / max(gen, 1.0),
        "dep_plus_esc_over_queued": (dep + esc) / max(queued, 1.0),
        "mean_direct_secondary_energy_MeV": gen
        / max(float(out["generated_direct_secondaries"]), 1.0),
        "mean_queued_energy_MeV": queued
        / max(float(out["queued_charged_secondaries"]), 1.0),
        "nuclear_per_history": float(out["nuclear_interactions"])
        / max(float(out["histories"]), 1.0),
    }
    return out


def selection_metrics(
    reference: np.ndarray,
    evaluation: np.ndarray,
    selection: np.ndarray,
    dmax: float,
) -> dict[str, float]:
    delta = evaluation[selection] - reference[selection]
    sse = float(np.sum(delta * delta, dtype=np.float64))
    rmse = float(np.sqrt(np.mean(delta * delta, dtype=np.float64)))
    return {
        "nrmse_pct_dmax": 100.0 * rmse / dmax,
        "bias_pct_dmax": 100.0 * float(np.mean(delta)) / dmax,
        "mae_pct_dmax": 100.0 * float(np.mean(np.abs(delta))) / dmax,
        "sse": sse,
        "pos_sse_fraction": float(np.sum(delta[delta > 0] ** 2) / max(sse, 1e-30)),
        "pearson_r": float(np.corrcoef(reference[selection], evaluation[selection])[0, 1]),
    }


def optimal_global_scale(
    reference: np.ndarray, evaluation: np.ndarray, selection: np.ndarray, dmax: float
) -> dict[str, Any]:
    """Least-squares scale a*E ≈ R on selection; diagnostic only."""
    r = reference[selection].astype(np.float64)
    e = evaluation[selection].astype(np.float64)
    scale = float(np.dot(r, e) / max(np.dot(e, e), 1e-30))
    scaled = evaluation * scale
    base = selection_metrics(reference, evaluation, selection, dmax)
    after = selection_metrics(reference, scaled, selection, dmax)
    return {
        "scale": scale,
        "baseline": base,
        "after_scale": after,
        "sse_relative_reduction_pct": 100.0
        * (base["sse"] - after["sse"])
        / max(base["sse"], 1e-30),
    }


def residual_gradient_correlations(
    reference: np.ndarray,
    evaluation: np.ndarray,
    selection: np.ndarray,
    spacing_xyz: tuple[float, float, float],
) -> dict[str, float]:
    """corr(R, D), corr(R, ∂D/∂x), corr(R, |∇D|) — shift vs scale signatures."""
    residual = evaluation - reference
    r = residual[selection]
    d = reference[selection]
    gz, gy, gx = np.gradient(reference, spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz)

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 2 or np.std(a) < 1e-30 or np.std(b) < 1e-30:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "corr_R_vs_D": corr(r, d),
        "corr_R_vs_dD_dx": corr(r, gx[selection]),
        "corr_R_vs_dD_dy": corr(r, gy[selection]),
        "corr_R_vs_dD_dz": corr(r, gz[selection]),
        "corr_R_vs_grad_mag": corr(r, grad_mag[selection]),
        "corr_R_vs_D_times_sign_dD_dx": corr(
            r, d * np.sign(gx[selection] + 1e-30)
        ),
    }


def depth_residual_profile(
    reference: np.ndarray,
    evaluation: np.ndarray,
    selection: np.ndarray,
    spacing_x: float,
    offset_x: float,
) -> list[dict[str, float]]:
    """Mean residual vs patient-X depth on selection."""
    residual = evaluation - reference
    rows: list[dict[str, float]] = []
    for ix in range(reference.shape[2]):
        mask = selection[:, :, ix]
        n = int(np.count_nonzero(mask))
        if n == 0:
            continue
        r = residual[:, :, ix][mask]
        t = reference[:, :, ix][mask]
        g = evaluation[:, :, ix][mask]
        rows.append(
            {
                "ix": ix,
                "x_mm": offset_x + ix * spacing_x,
                "n": n,
                "mean_topas": float(np.mean(t)),
                "mean_gpu": float(np.mean(g)),
                "mean_R": float(np.mean(r)),
                "rmse_R": float(np.sqrt(np.mean(r * r))),
                "ratio": float(np.mean(g) / max(np.mean(t), 1e-30)),
            }
        )
    return rows


def lateral_sigma_by_depth(
    volume: np.ndarray,
    selection: np.ndarray,
    spacing_y: float,
    offset_y: float,
    axis_y: int = 1,
) -> list[dict[str, float]]:
    """Dose-weighted lateral σ_y vs depth index (patient X)."""
    rows: list[dict[str, float]] = []
    ny = volume.shape[axis_y]
    y = offset_y + spacing_y * np.arange(ny, dtype=np.float64)
    for ix in range(volume.shape[2]):
        # integrate over z for each y at fixed x
        weights = np.where(selection[:, :, ix], volume[:, :, ix], 0.0)
        w_y = np.sum(weights, axis=0, dtype=np.float64)  # shape (ny,)
        total = float(np.sum(w_y))
        if total <= 0.0:
            continue
        mean_y = float(np.dot(w_y, y) / total)
        var = float(np.dot(w_y, (y - mean_y) ** 2) / total)
        rows.append(
            {
                "ix": ix,
                "n_mass": total,
                "mean_y_mm": mean_y,
                "sigma_y_mm": math.sqrt(max(var, 0.0)),
            }
        )
    return rows


def component_ols(
    residual: np.ndarray,
    components: dict[str, np.ndarray],
    selection: np.ndarray,
) -> dict[str, Any]:
    """R ≈ sum a_i C_i + b on selection; report R² and coeffs."""
    y = residual[selection].astype(np.float64)
    names = list(components.keys())
    X = np.column_stack(
        [components[n][selection].astype(np.float64) for n in names]
        + [np.ones(y.size, dtype=np.float64)]
    )
    coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "names": names + ["intercept"],
        "coefficients": [float(c) for c in coef],
        "rank": int(rank),
        "r_squared": float(1.0 - sse / max(sst, 1e-30)),
        "sse_explained_fraction": float(1.0 - sse / max(float(np.sum(y * y)), 1e-30)),
    }


def blur_scan_nrmse(
    reference: np.ndarray,
    evaluation: np.ndarray,
    selection: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    dmax: float,
    widths_mm: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5),
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_vol: np.ndarray | None = None
    for lat, beam in product(widths_mm, repeat=2):
        blurred, cons = conserved_anisotropic_gaussian(
            evaluation, lat, beam, spacing_xyz
        )
        m = selection_metrics(reference, blurred, selection, dmax)
        cand = {
            "sigma_lateral_yz_mm": lat,
            "sigma_beam_x_mm": beam,
            "nrmse_pct_dmax": m["nrmse_pct_dmax"],
            "sse": m["sse"],
            "bias_pct_dmax": m["bias_pct_dmax"],
            "conservation_factor": cons["input_over_blurred_conservation_factor"],
        }
        candidates.append(cand)
        if best is None or cand["nrmse_pct_dmax"] < best["nrmse_pct_dmax"]:
            best, best_vol = cand, blurred
        else:
            del blurred
    assert best is not None and best_vol is not None
    baseline = next(
        c
        for c in candidates
        if c["sigma_lateral_yz_mm"] == 0.0 and c["sigma_beam_x_mm"] == 0.0
    )
    return {
        "candidates": candidates,
        "best": best,
        "baseline_nrmse_pct_dmax": baseline["nrmse_pct_dmax"],
        "relative_nrmse_improvement_pct": 100.0
        * (baseline["nrmse_pct_dmax"] - best["nrmse_pct_dmax"])
        / max(baseline["nrmse_pct_dmax"], 1e-30),
        "best_volume": best_vol,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu",
        type=Path,
        default=ROOT / "out/ct/RT07575/upstream_air_loss_ablation/air_loss/dose.mhd",
    )
    parser.add_argument(
        "--primary",
        type=Path,
        default=ROOT
        / "out/ct/RT07575/cascade_secondary_ablation/phys_primary_only/dose.mhd",
    )
    parser.add_argument(
        "--sec-nc",
        type=Path,
        default=ROOT
        / "out/ct/RT07575/cascade_secondary_ablation/phys_secondary_no_cascade/dose.mhd",
    )
    parser.add_argument(
        "--topas",
        type=Path,
        default=ROOT / "out/fullplan_result/RT07575/topas/dose.mhd",
    )
    parser.add_argument(
        "--body-mask",
        type=Path,
        default=ROOT / "out/fullplan_result/RT07575/body_mask.mhd",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "out/ct/RT07575/cascade_secondary_ablation/residual_shape_energy",
    )
    parser.add_argument("--skip-gamma", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, topas = load_patient(args.topas)
    _, body_vals = load_patient(args.body_mask)
    body = body_vals > 0.5
    patient_shape = topas.shape  # z,y,x
    _, gpu = load_gpu_mapped(args.gpu, patient_shape)
    _, prim = load_gpu_mapped(args.primary, patient_shape)
    _, sec_nc = load_gpu_mapped(args.sec_nc, patient_shape)

    meta, _ = load(args.topas)
    spacing_xyz = tuple(float(v) for v in meta["ElementSpacing"].split())
    offset_xyz = tuple(float(v) for v in meta["Offset"].split())
    shape_xyz = tuple(int(v) for v in meta["DimSize"].split())

    dmax = float(np.max(topas[body]))
    selection = body & (topas >= 0.10 * dmax)
    dmax_sel = float(np.max(topas[selection]))

    full = gpu
    S = sec_nc - prim  # accounting partition (not physical negative secondary)
    C = full - sec_nc
    residual = full - topas

    base = selection_metrics(topas, full, selection, dmax_sel)
    prim_m = selection_metrics(topas, prim, selection, dmax_sel)
    sec_m = selection_metrics(topas, sec_nc, selection, dmax_sel)

    # SSE attribution
    sse_full = float(np.sum(residual[selection] ** 2))
    sse_sec = float(np.sum((sec_nc - topas)[selection] ** 2))
    sse_cas_inc = float(np.sum(C[selection] ** 2))
    sse_cross = float(
        np.sum(2.0 * (sec_nc - topas)[selection] * C[selection], dtype=np.float64)
    )

    scale_diag = optimal_global_scale(topas, full, selection, dmax_sel)
    corr_grad = residual_gradient_correlations(topas, full, selection, spacing_xyz)

    print("Running rigid shift search...", flush=True)
    shift_diag = rigid_search(
        topas, full, selection, shape_xyz, spacing_xyz
    )
    # drop heavy nested gamma if skip? keep as-is from pair_metrics

    print("Running blur scan...", flush=True)
    blur = blur_scan_nrmse(topas, full, selection, spacing_xyz, dmax_sel)
    best_blur_vol = blur.pop("best_volume")
    blur_after = selection_metrics(topas, best_blur_vol, selection, dmax_sel)

    # shift then blur at best shift (mechanism upper bound)
    best_shift = tuple(shift_diag["best_shift_xyz_mm"])
    shifted = shifted_volume(full, best_shift, spacing_xyz)
    shift_then_blur = blur_scan_nrmse(
        topas, shifted, selection, spacing_xyz, dmax_sel,
        widths_mm=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    stb_vol = shift_then_blur.pop("best_volume")
    stb_after = selection_metrics(topas, stb_vol, selection, dmax_sel)

    # also blur on sec_nc
    blur_sec = blur_scan_nrmse(
        topas, sec_nc, selection, spacing_xyz, dmax_sel,
        widths_mm=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    blur_sec.pop("best_volume", None)

    ols = component_ols(
        residual,
        {"primary": prim, "secondary_partition": S, "cascade_inc": C},
        selection,
    )

    # correlations with residual
    def corr_sel(a: np.ndarray, b: np.ndarray) -> float:
        aa, bb = a[selection], b[selection]
        if np.std(aa) < 1e-30 or np.std(bb) < 1e-30:
            return float("nan")
        return float(np.corrcoef(aa, bb)[0, 1])

    corrs = {
        "R_vs_prim_minus_topas": corr_sel(residual, prim - topas),
        "R_vs_sec_nc_minus_topas": corr_sel(residual, sec_nc - topas),
        "R_vs_S": corr_sel(residual, S),
        "R_vs_C": corr_sel(residual, C),
        "R_vs_full": corr_sel(residual, full),
        "R_vs_topas": corr_sel(residual, topas),
    }

    depth_rows = depth_residual_profile(
        topas, full, selection, spacing_xyz[0], offset_xyz[0]
    )
    # write depth csv
    depth_csv = args.output_dir / "depth_residual_profile.csv"
    with depth_csv.open("w", encoding="utf-8") as f:
        f.write("ix,x_mm,n,mean_topas,mean_gpu,mean_R,rmse_R,ratio\n")
        for row in depth_rows:
            f.write(
                f"{row['ix']},{row['x_mm']},{row['n']},{row['mean_topas']},"
                f"{row['mean_gpu']},{row['mean_R']},{row['rmse_R']},{row['ratio']}\n"
            )

    # lateral sigma comparison
    sig_t = lateral_sigma_by_depth(topas, selection, spacing_xyz[1], offset_xyz[1])
    sig_g = lateral_sigma_by_depth(full, selection, spacing_xyz[1], offset_xyz[1])
    sig_by_ix = {r["ix"]: r for r in sig_t}
    sigma_deltas: list[float] = []
    sigma_pairs: list[dict[str, float]] = []
    for g in sig_g:
        t = sig_by_ix.get(g["ix"])
        if t is None or t["n_mass"] < 1e-12 or g["n_mass"] < 1e-12:
            continue
        delta = g["sigma_y_mm"] - t["sigma_y_mm"]
        sigma_deltas.append(delta)
        sigma_pairs.append(
            {
                "ix": g["ix"],
                "sigma_gpu": g["sigma_y_mm"],
                "sigma_topas": t["sigma_y_mm"],
                "delta": delta,
                "mass_topas": t["n_mass"],
            }
        )
    # mass-weighted mean delta
    if sigma_pairs:
        masses = np.array([p["mass_topas"] for p in sigma_pairs], dtype=np.float64)
        deltas = np.array([p["delta"] for p in sigma_pairs], dtype=np.float64)
        mean_delta_sigma = float(np.sum(masses * deltas) / np.sum(masses))
    else:
        mean_delta_sigma = float("nan")

    # COM
    com_t = dose_weighted_com(topas, selection, spacing_xyz, offset_xyz)
    com_g = dose_weighted_com(full, selection, spacing_xyz, offset_xyz)
    com_delta = [g - t for g, t in zip(com_g, com_t)]

    # energy budgets
    log_full = args.gpu.parent / "run.log"
    log_prim = args.primary.parent / "run.log"
    log_sec = args.sec_nc.parent / "run.log"
    budgets = {
        "full": parse_energy_budget(log_full) if log_full.exists() else {},
        "primary": parse_energy_budget(log_prim) if log_prim.exists() else {},
        "sec_nc": parse_energy_budget(log_sec) if log_sec.exists() else {},
    }

    # peak vs entrance residual ratio trend
    if depth_rows:
        # entrance: first 20% of depth span with data; peak: highest mean_topas region
        xs = [r["x_mm"] for r in depth_rows]
        x_min, x_max = min(xs), max(xs)
        entrance = [r for r in depth_rows if r["x_mm"] <= x_min + 0.2 * (x_max - x_min)]
        peak_idx = max(range(len(depth_rows)), key=lambda i: depth_rows[i]["mean_topas"])
        peak_x = depth_rows[peak_idx]["x_mm"]
        peak = [
            r
            for r in depth_rows
            if abs(r["x_mm"] - peak_x) <= 5.0  # ±5 mm around peak mean dose
        ]
        depth_summary = {
            "entrance_mean_ratio": float(np.mean([r["ratio"] for r in entrance]))
            if entrance
            else float("nan"),
            "peak_mean_ratio": float(np.mean([r["ratio"] for r in peak]))
            if peak
            else float("nan"),
            "entrance_mean_R": float(np.mean([r["mean_R"] for r in entrance]))
            if entrance
            else float("nan"),
            "peak_mean_R": float(np.mean([r["mean_R"] for r in peak]))
            if peak
            else float("nan"),
            "global_mean_ratio": float(
                np.mean(full[selection]) / max(np.mean(topas[selection]), 1e-30)
            ),
        }
    else:
        depth_summary = {}

    # optional gamma at best transforms
    gamma_block: dict[str, Any] = {}
    if not args.skip_gamma:
        print("Computing baseline + best-transform gammas...", flush=True)
        gamma_block["baseline"] = pair_metrics(
            topas, full, selection, shape_xyz, spacing_xyz, gamma=True
        )
        gamma_block["best_shift"] = shift_diag.get("strict_metrics_at_best_shift", {})
        gamma_block["best_blur"] = pair_metrics(
            topas, best_blur_vol, selection, shape_xyz, spacing_xyz, gamma=True
        )
        gamma_block["shift_then_blur"] = pair_metrics(
            topas, stb_vol, selection, shape_xyz, spacing_xyz, gamma=True
        )

    report = {
        "baseline": {
            "selection_voxels": int(np.count_nonzero(selection)),
            "dmax_selection": dmax_sel,
            "full_vs_topas": base,
            "primary_vs_topas": prim_m,
            "sec_nc_vs_topas": sec_m,
            "mean_E_over_R": float(
                np.mean(full[selection]) / max(np.mean(topas[selection]), 1e-30)
            ),
        },
        "sse_partition": {
            "sse_full_minus_topas": sse_full,
            "sse_sec_nc_minus_topas": sse_sec,
            "sse_sec_nc_over_full": sse_sec / max(sse_full, 1e-30),
            "sse_cascade_increment": sse_cas_inc,
            "sse_cross_2_sec_C": sse_cross,
            "cascade_fraction_of_full_sse": (sse_full - sse_sec) / max(sse_full, 1e-30),
        },
        "correlations": corrs,
        "ols_components": ols,
        "optimal_global_scale": scale_diag,
        "residual_gradient_signatures": corr_grad,
        "rigid_shift": {
            k: v
            for k, v in shift_diag.items()
            if k != "strict_metrics_at_best_shift" or not args.skip_gamma
        },
        "blur_scan_full": blur,
        "blur_after_metrics": blur_after,
        "blur_scan_sec_nc": blur_sec,
        "shift_then_blur": {
            "shift_xyz_mm": list(best_shift),
            "blur": shift_then_blur,
            "after_metrics": stb_after,
        },
        "com_patient_xyz_mm": {
            "topas": com_t,
            "gpu": com_g,
            "gpu_minus_topas": com_delta,
        },
        "lateral_sigma": {
            "mass_weighted_mean_delta_sigma_y_mm": mean_delta_sigma,
            "n_depth_bins": len(sigma_pairs),
            "interpretation": (
                "positive => GPU wider than TOPAS on dose-weighted σ_y"
            ),
        },
        "depth_summary": depth_summary,
        "energy_budgets": budgets,
        "gamma": gamma_block,
        "paths": {
            "gpu": str(args.gpu),
            "primary": str(args.primary),
            "sec_nc": str(args.sec_nc),
            "topas": str(args.topas),
            "body_mask": str(args.body_mask),
            "depth_csv": str(depth_csv),
        },
    }

    # strip non-serializable
    def scrub(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: scrub(v) for k, v in obj.items() if k != "best_volume"}
        if isinstance(obj, list):
            return [scrub(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    report = scrub(report)
    json_path = args.output_dir / "summary.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # markdown
    md = []
    md.append("# RT07575 air_loss 残差形状 + 能量分解")
    md.append("")
    md.append(
        f"掩膜 BODY∩≥10% Dmax = **{report['baseline']['selection_voxels']}** voxels；"
        f"无拟合标度基线。"
    )
    md.append("")
    md.append("## 1. 基线与物理分区")
    md.append("")
    md.append("| 配置 | NRMSE%Dmax | bias%Dmax | pos SSE 占比 | Pearson r |")
    md.append("|---|---:|---:|---:|---:|")
    for name, key in (
        ("full (air_loss)", "full_vs_topas"),
        ("secondary_no_cascade", "sec_nc_vs_topas"),
        ("primary_only", "primary_vs_topas"),
    ):
        m = report["baseline"][key]
        md.append(
            f"| {name} | {m['nrmse_pct_dmax']:.3f} | {m['bias_pct_dmax']:+.3f} | "
            f"{m['pos_sse_fraction']:.3f} | {m['pearson_r']:.6f} |"
        )
    md.append("")
    sp = report["sse_partition"]
    md.append(
        f"- `SSE(sec_nc−TOPAS)/SSE(full−TOPAS)` = **{sp['sse_sec_nc_over_full']:.4f}**"
    )
    md.append(
        f"- cascade 对 full SSE 贡献分数 ≈ **{sp['cascade_fraction_of_full_sse']*100:.2f}%**"
    )
    md.append(f"- mean E/R = **{report['baseline']['mean_E_over_R']:.6f}**")
    md.append("")
    md.append("## 2. 非物理变换可解释上限（机制诊断，非校准）")
    md.append("")
    sc = report["optimal_global_scale"]
    md.append(
        f"- **最优全局标度** a={sc['scale']:.6f}：NRMSE "
        f"{sc['baseline']['nrmse_pct_dmax']:.3f}% → {sc['after_scale']['nrmse_pct_dmax']:.3f}% "
        f"（SSE 相对降 {sc['sse_relative_reduction_pct']:.2f}%）"
    )
    rs = report["rigid_shift"]
    md.append(
        f"- **刚体位移** best={rs['best_shift_xyz_mm']} mm：NRMSE "
        f"{rs['full_selection_unshifted_nrmse_over_dmax_percent']:.3f}% → "
        f"{rs['full_selection_best_nrmse_over_dmax_percent']:.3f}% "
        f"（相对降 {rs['full_selection_nrmse_relative_improvement_percent']:.2f}%）"
    )
    bl = report["blur_scan_full"]
    md.append(
        f"- **各向异性模糊** best lat={bl['best']['sigma_lateral_yz_mm']} / "
        f"beam={bl['best']['sigma_beam_x_mm']} mm：NRMSE "
        f"{bl['baseline_nrmse_pct_dmax']:.3f}% → {bl['best']['nrmse_pct_dmax']:.3f}% "
        f"（相对降 {bl['relative_nrmse_improvement_pct']:.2f}%）"
    )
    stb = report["shift_then_blur"]
    md.append(
        f"- **位移+模糊串联** shift={stb['shift_xyz_mm']} + blur "
        f"lat={stb['blur']['best']['sigma_lateral_yz_mm']}/"
        f"beam={stb['blur']['best']['sigma_beam_x_mm']} mm：NRMSE → "
        f"**{stb['after_metrics']['nrmse_pct_dmax']:.3f}%**"
    )
    md.append("")
    md.append(
        "解读：若最优标度≈1 且 SSE 几乎不降 → 不是全局剂量标定问题；"
        "若位移/模糊只降几个点 NRMSE 而 local 3%/0mm 仍远低于 TOPAS–TOPAS，"
        "则残差是**不可被简单卷积/平移吸收的形状系统差**。"
    )
    md.append("")
    md.append("## 3. 残差签名（shift vs scale）")
    md.append("")
    cg = report["residual_gradient_signatures"]
    md.append("| 相关 | 值 |")
    md.append("|---|---:|")
    for k, v in cg.items():
        if isinstance(v, float):
            md.append(f"| {k} | {v:.4f} |")
        else:
            md.append(f"| {k} | {v} |")
    md.append("")
    md.append(
        f"- dose-weighted COM GPU−TOPAS = {report['com_patient_xyz_mm']['gpu_minus_topas']} mm"
    )
    md.append(
        f"- 质量加权横向 σ_y 差 (GPU−TOPAS) ≈ **{report['lateral_sigma']['mass_weighted_mean_delta_sigma_y_mm']:.4f} mm**"
    )
    md.append("")
    if report.get("depth_summary"):
        ds = report["depth_summary"]
        md.append("### 深度比")
        md.append(
            f"- 入口区 mean ratio GPU/TOPAS = {ds.get('entrance_mean_ratio')}"
        )
        md.append(f"- 峰区 mean ratio = {ds.get('peak_mean_ratio')}")
        md.append(f"- 全局 mean ratio = {ds.get('global_mean_ratio')}")
        md.append("")
    md.append("## 4. 分量 OLS / 相关")
    md.append("")
    md.append("| 相关 | 值 |")
    md.append("|---|---:|")
    for k, v in report["correlations"].items():
        md.append(f"| {k} | {v:.4f} |")
    ols_r = report["ols_components"]
    md.append("")
    md.append(
        f"OLS R≈aP+bS+cC+d：R²={ols_r['r_squared']:.4f}，"
        f"SSE explained={ols_r['sse_explained_fraction']:.4f}"
    )
    md.append(f"coeffs ({ols_r['names']}) = {ols_r['coefficients']}")
    md.append("")
    md.append("## 5. 能量预算（run.log）")
    md.append("")
    for label in ("full", "sec_nc", "primary"):
        b = report["energy_budgets"].get(label) or {}
        if not b:
            continue
        md.append(f"### {label}")
        md.append(
            f"- nuclear ints: {b.get('nuclear_interactions', 0):.0f}；"
            f"gen_sec_E: {b.get('generated_direct_secondary_energy_MeV', 0):.4e} MeV"
        )
        md.append(
            f"- queued_sec_E: {b.get('queued_secondary_energy_MeV', 0):.4e}；"
            f"sec_dep: {b.get('secondary_deposited_energy_MeV', 0):.4e}；"
            f"sec_esc: {b.get('secondary_escaped_energy_MeV', 0):.4e}"
        )
        md.append(
            f"- residual_heat(nuc not in sec): {b.get('nuclear_energy_not_in_secondaries_MeV', 0):.4e}；"
            f"neutral_untrans: {b.get('untransported_neutral_energy_MeV', 0):.4e}；"
            f"untracked: {b.get('untracked_nuclear_energy_MeV', 0):.4e}"
        )
        d = b.get("derived") or {}
        if d:
            md.append(
                f"- 比率：queued/gen={d.get('queued_over_generated', 0):.4f}，"
                f"dep/queued={d.get('deposited_over_queued', 0):.4f}，"
                f"esc/queued={d.get('escaped_over_queued', 0):.4f}，"
                f"residual_heat/gen={d.get('residual_heat_over_generated', 0):.4f}"
            )
        md.append("")

    def gamma_pass(entry: Any) -> float:
        if isinstance(entry, dict):
            if "pass_percent" in entry:
                return float(entry["pass_percent"])
            return float("nan")
        if isinstance(entry, (int, float)):
            return float(entry)
        return float("nan")

    if gamma_block:
        md.append("## 6. Gamma（可选）")
        md.append("")
        for name, block in gamma_block.items():
            if not block:
                continue
            g30 = block.get("gamma_3pct_0mm", {}) or {}
            g11 = block.get("gamma_1pct_1mm", {}) or {}
            nrmse = block.get("nrmse_over_reference_dmax_percent", float("nan"))
            nrmse_f = float(nrmse) if isinstance(nrmse, (int, float)) else float("nan")
            md.append(
                f"- **{name}**: NRMSE={nrmse_f:.3f}%；"
                f"3%/0mm G/L={gamma_pass(g30.get('global')):.3f}/"
                f"{gamma_pass(g30.get('local')):.3f}；"
                f"1%/1mm G/L={gamma_pass(g11.get('global')):.3f}/"
                f"{gamma_pass(g11.get('local')):.3f}"
            )
        md.append("")

    md.append("## 7. 结论与下一物理杠杆")
    md.append("")
    md.append(
        "1. cascade 增量对剂量残差 SSE 可忽略；问题在 primary+gen0 次级空间形状。"
    )
    md.append(
        "2. 全局标度接近 1 时，残差不是 normalization；正负 SSE 各半支持形状错位。"
    )
    md.append(
        "3. 刚体位移/小宽度模糊若只能回收小部分 NRMSE，则需改 **MCS/次级角谱/核反应末态角**，"
        "而不是再扫 residual-heat MFP 或 cascade 反应率。"
    )
    md.append(
        "4. 能量预算中 residual nuclear heat + untracked + neutral 占比用于约束："
        "核热就地沉积 vs 次级带走是否与 TOPAS 一致（需单点 ntuple 对照）。"
    )
    md.append(
        "5. **下一刀**： (a) gen0 MCS 关/开 A/B（accurate profile）；"
        " (b) primary reaction 产物 θ 谱 + residual heat 与 TOPAS CarbonCascadeNtuple 对照。"
    )
    md.append("")
    md_path = args.output_dir / "summary.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)
    print(
        f"Baseline NRMSE={base['nrmse_pct_dmax']:.3f}%  "
        f"scale={sc['scale']:.6f}  "
        f"shift_nrmse={rs['full_selection_best_nrmse_over_dmax_percent']:.3f}%  "
        f"blur_nrmse={bl['best']['nrmse_pct_dmax']:.3f}%  "
        f"shift+blur={stb['after_metrics']['nrmse_pct_dmax']:.3f}%",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

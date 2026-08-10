#!/usr/bin/env python3
"""Pre-peak high-gradient lateral profile diagnostic for RT07575 air_loss.

On the fixed BODY ∩ TOPAS≥10% Dmax mask:
  - dose-weighted lateral σ_y (and σ_z) vs patient-X depth
  - 1-D patient-Y profiles at entrance / pre-peak SSE hotspot / peak / distal
  - core vs wing residual SSE partition at pre-peak and peak slabs
  - wing–core bias signature (GPU too wide / too narrow)

No dose scale fit. Diagnostic only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_rt07575_equal_history_dose_residual import (  # noqa: E402
    load_gpu_mapped,
    load_patient,
)
from analyze_rt07575_residual_shape_energy import selection_metrics  # noqa: E402
from compare_topas_seed_gamma import load  # noqa: E402


def dose_weighted_moments_1d(
    dose: np.ndarray, coords: np.ndarray
) -> tuple[float, float, float]:
    """Return (total, mean, sigma) for non-negative weights."""
    w = np.clip(dose.astype(np.float64), 0.0, None)
    total = float(np.sum(w))
    if total <= 0.0:
        return 0.0, float("nan"), float("nan")
    mean = float(np.dot(w, coords) / total)
    var = float(np.dot(w, (coords - mean) ** 2) / total)
    return total, mean, math.sqrt(max(var, 0.0))


def lateral_sigma_depth_table(
    topas: np.ndarray,
    gpu: np.ndarray,
    selection: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    offset_xyz: tuple[float, float, float],
) -> list[dict[str, Any]]:
    """Per patient-X slice: σ_y and σ_z for TOPAS/GPU on selection."""
    nz, ny, nx = topas.shape
    y = offset_xyz[1] + spacing_xyz[1] * np.arange(ny, dtype=np.float64)
    z = offset_xyz[2] + spacing_xyz[2] * np.arange(nz, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for ix in range(nx):
        mask = selection[:, :, ix]
        n = int(np.count_nonzero(mask))
        if n == 0:
            continue
        # collapse to y: sum over z where selected
        t_yz = np.where(mask, topas[:, :, ix], 0.0)
        g_yz = np.where(mask, gpu[:, :, ix], 0.0)
        t_y = np.sum(t_yz, axis=0)
        g_y = np.sum(g_yz, axis=0)
        t_z = np.sum(t_yz, axis=1)
        g_z = np.sum(g_yz, axis=1)
        tt, tmy, tsy = dose_weighted_moments_1d(t_y, y)
        gt, gmy, gsy = dose_weighted_moments_1d(g_y, y)
        _, tmz, tsz = dose_weighted_moments_1d(t_z, z)
        _, gmz, gsz = dose_weighted_moments_1d(g_z, z)
        residual = gpu[:, :, ix][mask] - topas[:, :, ix][mask]
        rows.append(
            {
                "ix": ix,
                "x_mm": offset_xyz[0] + ix * spacing_xyz[0],
                "n": n,
                "mean_topas": float(np.mean(topas[:, :, ix][mask])),
                "mean_gpu": float(np.mean(gpu[:, :, ix][mask])),
                "sse": float(np.sum(residual * residual, dtype=np.float64)),
                "ratio": float(
                    np.mean(gpu[:, :, ix][mask])
                    / max(np.mean(topas[:, :, ix][mask]), 1e-30)
                ),
                "sigma_y_topas_mm": tsy,
                "sigma_y_gpu_mm": gsy,
                "delta_sigma_y_mm": gsy - tsy if (tsy == tsy and gsy == gsy) else float("nan"),
                "mean_y_topas_mm": tmy,
                "mean_y_gpu_mm": gmy,
                "delta_mean_y_mm": gmy - tmy if (tmy == tmy and gmy == gmy) else float("nan"),
                "sigma_z_topas_mm": tsz,
                "sigma_z_gpu_mm": gsz,
                "delta_sigma_z_mm": gsz - tsz if (tsz == tsz and gsz == gsz) else float("nan"),
                "topas_mass": tt,
                "gpu_mass": gt,
            }
        )
    return rows


def slab_lateral_profile(
    volume: np.ndarray,
    selection: np.ndarray,
    ix_list: list[int],
    axis: int,
    coords: np.ndarray,
) -> np.ndarray:
    """Sum dose over selected voxels in given depth indices, project onto axis."""
    acc = np.zeros(volume.shape[axis], dtype=np.float64)
    for ix in ix_list:
        mask = selection[:, :, ix]
        slab = np.where(mask, volume[:, :, ix], 0.0)
        if axis == 1:  # y
            acc += np.sum(slab, axis=0)
        elif axis == 0:  # z
            acc += np.sum(slab, axis=1)
        else:
            raise ValueError("axis must be 0 (z) or 1 (y)")
    return acc


def core_wing_partition(
    topas: np.ndarray,
    gpu: np.ndarray,
    selection: np.ndarray,
    ix_list: list[int],
    y_coords: np.ndarray,
    core_frac: float = 0.5,
) -> dict[str, Any]:
    """Partition residual SSE into lateral core vs wings using TOPAS mass on y.

    Core = voxels whose |y - COM_y| is within the central mass fraction
    (approx: cumulative TOPAS mass along |y-COM| radius).
    """
    # Build per-y TOPAS mass and residual SSE across the slab
    ny = topas.shape[1]
    t_y = np.zeros(ny, dtype=np.float64)
    g_y = np.zeros(ny, dtype=np.float64)
    sse_y = np.zeros(ny, dtype=np.float64)
    n_y = np.zeros(ny, dtype=np.int64)
    for ix in ix_list:
        mask = selection[:, :, ix]
        for iy in range(ny):
            col = mask[:, iy]
            if not np.any(col):
                continue
            t = topas[:, iy, ix][col]
            g = gpu[:, iy, ix][col]
            t_y[iy] += float(np.sum(t))
            g_y[iy] += float(np.sum(g))
            d = g - t
            sse_y[iy] += float(np.sum(d * d))
            n_y[iy] += int(np.count_nonzero(col))

    total_t = float(np.sum(t_y))
    if total_t <= 0:
        return {"voxels": 0}
    com = float(np.dot(t_y, y_coords) / total_t)
    order = np.argsort(np.abs(y_coords - com))
    cum = np.cumsum(t_y[order])
    thresh = core_frac * total_t
    # smallest radius whose enclosed TOPAS mass reaches core_frac
    hit = np.nonzero(cum >= thresh)[0]
    if hit.size == 0:
        radius = float(np.max(np.abs(y_coords - com)))
    else:
        radius = float(abs(y_coords[order[int(hit[0])]] - com))
    core_mask = (np.abs(y_coords - com) <= radius + 1e-9) & (n_y > 0)
    wing_mask = (~core_mask) & (n_y > 0)
    sse_tot = float(np.sum(sse_y))
    return {
        "com_y_mm": com,
        "core_frac_target": core_frac,
        "core_n_bins": int(np.count_nonzero(core_mask)),
        "wing_n_bins": int(np.count_nonzero(wing_mask)),
        "core_topas_mass_frac": float(np.sum(t_y[core_mask]) / total_t),
        "wing_topas_mass_frac": float(np.sum(t_y[wing_mask]) / total_t),
        "core_sse_frac": float(np.sum(sse_y[core_mask]) / max(sse_tot, 1e-30)),
        "wing_sse_frac": float(np.sum(sse_y[wing_mask]) / max(sse_tot, 1e-30)),
        "core_ratio": float(np.sum(g_y[core_mask]) / max(np.sum(t_y[core_mask]), 1e-30)),
        "wing_ratio": float(np.sum(g_y[wing_mask]) / max(np.sum(t_y[wing_mask]), 1e-30)),
        "core_minus_wing_ratio": float(
            np.sum(g_y[core_mask]) / max(np.sum(t_y[core_mask]), 1e-30)
            - np.sum(g_y[wing_mask]) / max(np.sum(t_y[wing_mask]), 1e-30)
        ),
        "interpretation": (
            "core_ratio > wing_ratio → GPU relatively peaked (too narrow); "
            "core_ratio < wing_ratio → GPU relatively flat/wide"
        ),
    }


def pick_ix_window(
    rows: list[dict[str, Any]], x_center: float, half_width_mm: float
) -> list[int]:
    return [
        r["ix"]
        for r in rows
        if abs(r["x_mm"] - x_center) <= half_width_mm
    ]


def mass_mean(rows: list[dict[str, Any]], key: str) -> float:
    m = np.array([r["topas_mass"] for r in rows], dtype=np.float64)
    v = np.array([r[key] for r in rows], dtype=np.float64)
    ok = np.isfinite(v) & (m > 0)
    if not np.any(ok):
        return float("nan")
    return float(np.sum(m[ok] * v[ok]) / np.sum(m[ok]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gpu",
        type=Path,
        default=ROOT / "out/ct/RT07575/upstream_air_loss_ablation/air_loss/dose.mhd",
    )
    ap.add_argument(
        "--topas",
        type=Path,
        default=ROOT / "out/fullplan_result/RT07575/topas/dose.mhd",
    )
    ap.add_argument(
        "--body-mask",
        type=Path,
        default=ROOT / "out/fullplan_result/RT07575/body_mask.mhd",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "out/ct/RT07575/cascade_secondary_ablation/prepeak_lateral_profiles",
    )
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, topas = load_patient(args.topas)
    _, body_v = load_patient(args.body_mask)
    body = body_v > 0.5
    shape = topas.shape
    _, gpu = load_gpu_mapped(args.gpu, shape)
    meta, _ = load(args.topas)
    spacing = tuple(float(v) for v in meta["ElementSpacing"].split())
    offset = tuple(float(v) for v in meta["Offset"].split())

    dmax_body = float(np.max(topas[body]))
    selection = body & (topas >= 0.10 * dmax_body)
    dmax = float(np.max(topas[selection]))
    base = selection_metrics(topas, gpu, selection, dmax)

    rows = lateral_sigma_depth_table(topas, gpu, selection, spacing, offset)
    total_sse = sum(r["sse"] for r in rows)
    peak_x = max(rows, key=lambda r: r["mean_topas"])["x_mm"]
    # SSE hotspot center (top 20 bins by sse)
    top_sse = sorted(rows, key=lambda r: -r["sse"])[:20]
    hotspot_x = float(np.average([r["x_mm"] for r in top_sse], weights=[r["sse"] for r in top_sse]))

    x_min = rows[0]["x_mm"]
    x_max = rows[-1]["x_mm"]
    span = x_max - x_min

    def region_rows(pred):
        return [r for r in rows if pred(r)]

    regions_def = {
        "entrance_0_20pct": region_rows(lambda r: r["x_mm"] <= x_min + 0.2 * span),
        "prepeak_hotspot_pm5mm": region_rows(lambda r: abs(r["x_mm"] - hotspot_x) <= 5.0),
        "peak_pm_5mm": region_rows(lambda r: abs(r["x_mm"] - peak_x) <= 5.0),
        "peak_pm_10mm": region_rows(lambda r: abs(r["x_mm"] - peak_x) <= 10.0),
        "distal_gt_peak_10mm": region_rows(lambda r: r["x_mm"] > peak_x + 10.0),
        "all": rows,
    }

    region_stats = {}
    for name, rr in regions_def.items():
        if not rr:
            region_stats[name] = {}
            continue
        sse = sum(r["sse"] for r in rr)
        region_stats[name] = {
            "depth_bins": len(rr),
            "voxels": sum(r["n"] for r in rr),
            "x_mm_range": [rr[0]["x_mm"], rr[-1]["x_mm"]],
            "sse_fraction": sse / max(total_sse, 1e-30),
            "mean_delta_sigma_y_mm": mass_mean(rr, "delta_sigma_y_mm"),
            "mean_delta_sigma_z_mm": mass_mean(rr, "delta_sigma_z_mm"),
            "mean_delta_mean_y_mm": mass_mean(rr, "delta_mean_y_mm"),
            "mean_ratio": float(
                np.average([r["ratio"] for r in rr], weights=[r["n"] for r in rr])
            ),
            "mean_sigma_y_topas_mm": mass_mean(rr, "sigma_y_topas_mm"),
            "mean_sigma_y_gpu_mm": mass_mean(rr, "sigma_y_gpu_mm"),
        }

    # 1D profiles at key slabs
    ny = topas.shape[1]
    nz = topas.shape[0]
    y = offset[1] + spacing[1] * np.arange(ny, dtype=np.float64)
    z = offset[2] + spacing[2] * np.arange(nz, dtype=np.float64)

    slabs = {
        "entrance": pick_ix_window(rows, x_min + 0.1 * span, 2.5),
        "prepeak_hotspot": pick_ix_window(rows, hotspot_x, 2.5),
        "peak": pick_ix_window(rows, peak_x, 2.5),
        "distal": pick_ix_window(rows, min(peak_x + 20.0, x_max - 2.0), 2.5),
    }

    profile_rows_out: list[dict[str, Any]] = []
    core_wing = {}
    for slab_name, ixs in slabs.items():
        if not ixs:
            continue
        t_y = slab_lateral_profile(topas, selection, ixs, 1, y)
        g_y = slab_lateral_profile(gpu, selection, ixs, 1, y)
        for iy in range(ny):
            if t_y[iy] <= 0 and g_y[iy] <= 0:
                continue
            profile_rows_out.append(
                {
                    "slab": slab_name,
                    "axis": "patient_y",
                    "coord_mm": y[iy],
                    "topas": t_y[iy],
                    "gpu": g_y[iy],
                    "delta": g_y[iy] - t_y[iy],
                    "ratio": g_y[iy] / max(t_y[iy], 1e-30),
                }
            )
        core_wing[slab_name] = core_wing_partition(
            topas, gpu, selection, ixs, y, core_frac=0.5
        )
        # also sigma from slab profiles
        _, _, tsy = dose_weighted_moments_1d(t_y, y)
        _, _, gsy = dose_weighted_moments_1d(g_y, y)
        core_wing[slab_name]["slab_sigma_y_topas_mm"] = tsy
        core_wing[slab_name]["slab_sigma_y_gpu_mm"] = gsy
        core_wing[slab_name]["slab_delta_sigma_y_mm"] = (
            gsy - tsy if (tsy == tsy and gsy == gsy) else float("nan")
        )
        core_wing[slab_name]["n_depth_bins"] = len(ixs)
        core_wing[slab_name]["x_mm_center"] = float(
            np.mean([offset[0] + ix * spacing[0] for ix in ixs])
        )

    # write CSVs
    sigma_csv = args.output_dir / "lateral_sigma_vs_depth.csv"
    with sigma_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    prof_csv = args.output_dir / "lateral_profiles_slabs.csv"
    with prof_csv.open("w", encoding="utf-8", newline="") as f:
        if profile_rows_out:
            w = csv.DictWriter(f, fieldnames=list(profile_rows_out[0].keys()))
            w.writeheader()
            w.writerows(profile_rows_out)

    # wing residual signature summary
    pre = core_wing.get("prepeak_hotspot", {})
    peak = core_wing.get("peak", {})
    signature = "ambiguous"
    if pre:
        dsig = pre.get("slab_delta_sigma_y_mm", 0.0)
        cw = pre.get("core_minus_wing_ratio", 0.0)
        if dsig is not None and dsig == dsig:
            if dsig > 0.05 and cw < -0.01:
                signature = "GPU_too_wide_prepeak"
            elif dsig < -0.05 and cw > 0.01:
                signature = "GPU_too_narrow_prepeak"
            elif abs(dsig) < 0.05 and abs(cw) < 0.02:
                signature = "lateral_width_matched_prepeak_residual_is_shape_or_depth"
            else:
                signature = "mixed_lateral_signature"

    report = {
        "baseline": base,
        "peak_x_mm": peak_x,
        "sse_hotspot_x_mm": hotspot_x,
        "depth_span_mm": [x_min, x_max],
        "regions": region_stats,
        "core_wing_slabs": core_wing,
        "lateral_signature": signature,
        "paths": {
            "sigma_csv": str(sigma_csv),
            "profiles_csv": str(prof_csv),
            "gpu": str(args.gpu),
            "topas": str(args.topas),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    md: list[str] = []
    md.append("# RT07575 峰前高梯度横向剖面诊断")
    md.append("")
    md.append(
        f"GPU air_loss；掩膜 NRMSE=**{base['nrmse_pct_dmax']:.3f}%**；"
        f"峰位 x=**{peak_x:.2f} mm**；SSE 热点中心 x=**{hotspot_x:.2f} mm**。"
    )
    md.append("")
    md.append("## 1. 分区横向宽度（质量加权 Δσ_y = GPU−TOPAS）")
    md.append("")
    md.append(
        "| 区域 | SSE% | Δσ_y mm | σ_y TOPAS | σ_y GPU | mean ratio |"
    )
    md.append("|---|---:|---:|---:|---:|---:|")
    for name in (
        "entrance_0_20pct",
        "prepeak_hotspot_pm5mm",
        "peak_pm_5mm",
        "peak_pm_10mm",
        "distal_gt_peak_10mm",
        "all",
    ):
        r = region_stats.get(name) or {}
        if not r:
            continue
        md.append(
            f"| {name} | {100*r['sse_fraction']:.1f} | "
            f"{r['mean_delta_sigma_y_mm']:+.4f} | "
            f"{r['mean_sigma_y_topas_mm']:.3f} | {r['mean_sigma_y_gpu_mm']:.3f} | "
            f"{r['mean_ratio']:.4f} |"
        )
    md.append("")
    md.append("## 2. 板层 core/wing（TOPAS 质量中心 50% core）")
    md.append("")
    md.append(
        "| slab | x_c mm | Δσ_y mm | core ratio | wing ratio | core−wing | wing SSE% |"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in ("entrance", "prepeak_hotspot", "peak", "distal"):
        r = core_wing.get(name) or {}
        if not r or r.get("core_n_bins", 0) == 0:
            continue
        md.append(
            f"| {name} | {r.get('x_mm_center', float('nan')):.2f} | "
            f"{r.get('slab_delta_sigma_y_mm', float('nan')):+.4f} | "
            f"{r.get('core_ratio', float('nan')):.4f} | "
            f"{r.get('wing_ratio', float('nan')):.4f} | "
            f"{r.get('core_minus_wing_ratio', float('nan')):+.4f} | "
            f"{100*r.get('wing_sse_frac', 0):.1f} |"
        )
    md.append("")
    md.append(
        f"**横向签名**：`{signature}`  \n"
        f"（core_ratio &lt; wing_ratio → GPU 相对更宽/翼更亮；反之更窄。）"
    )
    md.append("")
    md.append("## 3. 解读")
    md.append("")
    pre_s = region_stats.get("prepeak_hotspot_pm5mm") or {}
    if pre_s:
        dsy = pre_s.get("mean_delta_sigma_y_mm", 0.0)
        md.append(
            f"- 峰前热点区 Δσ_y ≈ **{dsy:+.4f} mm**（相对 σ_y~"
            f"{pre_s.get('mean_sigma_y_topas_mm', float('nan')):.2f} mm 为 "
            f"{100*dsy/max(pre_s.get('mean_sigma_y_topas_mm', 1), 1e-6):+.2f}%）"
        )
    if pre:
        md.append(
            f"- 峰前 core/wing：core ratio={pre.get('core_ratio'):.4f}，"
            f"wing ratio={pre.get('wing_ratio'):.4f}，"
            f"wing 占该板层 SSE **{100*pre.get('wing_sse_frac', 0):.1f}%**"
        )
    md.append(
        "- 若 |Δσ_y| ≪ 0.1 mm 且 wing SSE 不主导 → 残差不是简单“整体横向过宽/过窄”，"
        "而是 **深度方向峰形/梯度错位** 或 **局部斑驳形状**。"
    )
    md.append(
        "- 若 wing SSE 高且 GPU 翼偏亮 → 优先 fragment/H-He **横向展宽**（MCS scale 或角谱能区）。"
    )
    md.append("")
    md.append("## 4. 结论与下一杠杆")
    md.append("")
    if signature == "lateral_width_matched_prepeak_residual_is_shape_or_depth":
        md.append(
            "1. **峰前横向宽度已基本匹配** → 不要再靠全局横向 blur / emittance↑ / MCS 关开。"
        )
        md.append(
            "2. 残差更像 **沿束深度的形状错位**（峰前梯度）或 **斑驳系统差**；"
            "下一刀：**深度 1D 剖面（IDD）分层** + soft-tissue **200 MeV INCL++ package**。"
        )
    elif signature == "GPU_too_wide_prepeak":
        md.append("1. 峰前 GPU **偏宽** → 考虑减小 fragment/secondary MCS 或检查次级角谱高能翼。")
    elif signature == "GPU_too_narrow_prepeak":
        md.append("1. 峰前 GPU **偏窄** → 考虑增大 secondary MCS / 源角（注意 emittance×1.15 已有害）。")
    else:
        md.append(f"1. 横向签名为 `{signature}`，见上表定量。")
    md.append(
        "3. 已排除：核反应率深度、入口 emittance、heavy local、sec straggling、cascade。"
    )
    md.append("")
    md.append(f"CSV: `{sigma_csv.name}`, `{prof_csv.name}`")
    md.append("脚本: `validation/scripts/analyze_rt07575_prepeak_lateral_profiles.py`")
    (args.output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir / 'summary.md'}")
    print(f"hotspot_x={hotspot_x:.2f} peak_x={peak_x:.2f} signature={signature}")
    if pre_s:
        print(
            f"prepeak Δσ_y={pre_s.get('mean_delta_sigma_y_mm'):+.4f} "
            f"ratio={pre_s.get('mean_ratio'):.4f}"
        )
    if pre:
        print(
            f"prepeak core/wing ratios {pre.get('core_ratio'):.4f}/"
            f"{pre.get('wing_ratio'):.4f} wing_sse%={100*pre.get('wing_sse_frac',0):.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

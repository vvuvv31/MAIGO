#!/usr/bin/env python3
"""Full-plan entrance / source residual stratification for RT07575 air_loss.

Diagnostic only: no dose scale fit. Reports:
  - residual SSE by beam-depth (patient X) bins (entrance vs mid vs peak)
  - dose-weighted lateral sigma_y vs depth (GPU−TOPAS)
  - entrance shell metrics and ratio GPU/TOPAS
  - optional comparison of an alternate GPU dose (e.g. emittance-scaled arm)
"""

from __future__ import annotations

import argparse
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
    pair_metrics,
)
from analyze_rt07575_residual_shape_energy import (  # noqa: E402
    lateral_sigma_by_depth,
    selection_metrics,
)
from compare_topas_seed_gamma import load  # noqa: E402


def depth_bins(
    reference: np.ndarray,
    evaluation: np.ndarray,
    selection: np.ndarray,
    spacing_x: float,
    offset_x: float,
    dmax: float,
) -> list[dict[str, Any]]:
    residual = evaluation - reference
    rows: list[dict[str, Any]] = []
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
                "sse": float(np.sum(r * r, dtype=np.float64)),
                "ratio": float(np.mean(g) / max(np.mean(t), 1e-30)),
                "bias_pct_dmax": float(100.0 * np.mean(r) / dmax),
            }
        )
    return rows


def region_metrics(
    rows: list[dict[str, Any]],
    pred,
    total_sse: float,
) -> dict[str, Any]:
    sel = [r for r in rows if pred(r)]
    if not sel:
        return {"voxels": 0}
    n = sum(r["n"] for r in sel)
    sse = sum(r["sse"] for r in sel)
    # weight by voxel count
    mean_ratio = sum(r["ratio"] * r["n"] for r in sel) / max(n, 1)
    mean_bias = sum(r["bias_pct_dmax"] * r["n"] for r in sel) / max(n, 1)
    return {
        "voxels": n,
        "depth_bins": len(sel),
        "x_mm_range": [sel[0]["x_mm"], sel[-1]["x_mm"]],
        "sse_fraction": sse / max(total_sse, 1e-30),
        "mean_ratio": mean_ratio,
        "mean_bias_pct_dmax": mean_bias,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gpu",
        type=Path,
        default=ROOT / "out/ct/RT07575/upstream_air_loss_ablation/air_loss/dose.mhd",
    )
    ap.add_argument("--gpu-alt", type=Path, default=None, help="optional alternate GPU dose")
    ap.add_argument(
        "--alt-label", type=str, default="alternate", help="label for --gpu-alt arm"
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
        / "out/ct/RT07575/cascade_secondary_ablation/entrance_source_residual",
    )
    ap.add_argument("--skip-gamma", action="store_true")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, topas = load_patient(args.topas)
    _, body_v = load_patient(args.body_mask)
    body = body_v > 0.5
    patient_shape = topas.shape
    _, gpu = load_gpu_mapped(args.gpu, patient_shape)
    meta, _ = load(args.topas)
    spacing = tuple(float(v) for v in meta["ElementSpacing"].split())
    offset = tuple(float(v) for v in meta["Offset"].split())
    shape_xyz = tuple(int(v) for v in meta["DimSize"].split())

    dmax_body = float(np.max(topas[body]))
    selection = body & (topas >= 0.10 * dmax_body)
    dmax = float(np.max(topas[selection]))

    base = selection_metrics(topas, gpu, selection, dmax)
    rows = depth_bins(topas, gpu, selection, spacing[0], offset[0], dmax)
    total_sse = sum(r["sse"] for r in rows)

    xs = [r["x_mm"] for r in rows]
    x_min, x_max = min(xs), max(xs)
    span = x_max - x_min
    # peak = depth bin with max mean_topas
    peak_x = max(rows, key=lambda r: r["mean_topas"])["x_mm"]

    regions = {
        "entrance_0_20pct_span": region_metrics(
            rows, lambda r: r["x_mm"] <= x_min + 0.20 * span, total_sse
        ),
        "entrance_0_30mm_from_min": region_metrics(
            rows, lambda r: r["x_mm"] <= x_min + 30.0, total_sse
        ),
        "mid_20_60pct_span": region_metrics(
            rows,
            lambda r: (x_min + 0.20 * span) < r["x_mm"] <= (x_min + 0.60 * span),
            total_sse,
        ),
        "peak_pm_10mm": region_metrics(
            rows, lambda r: abs(r["x_mm"] - peak_x) <= 10.0, total_sse
        ),
        "distal_gt_peak_plus_10mm": region_metrics(
            rows, lambda r: r["x_mm"] > peak_x + 10.0, total_sse
        ),
    }

    # lateral sigma
    sig_t = {r["ix"]: r for r in lateral_sigma_by_depth(topas, selection, spacing[1], offset[1])}
    sig_g = lateral_sigma_by_depth(gpu, selection, spacing[1], offset[1])
    sigma_rows = []
    for g in sig_g:
        t = sig_t.get(g["ix"])
        if t is None or t["n_mass"] <= 0:
            continue
        sigma_rows.append(
            {
                "ix": g["ix"],
                "x_mm": offset[0] + g["ix"] * spacing[0],
                "sigma_topas": t["sigma_y_mm"],
                "sigma_gpu": g["sigma_y_mm"],
                "delta": g["sigma_y_mm"] - t["sigma_y_mm"],
                "mass": t["n_mass"],
            }
        )

    def mass_mean_delta(pred) -> float:
        selr = [r for r in sigma_rows if pred(r)]
        if not selr:
            return float("nan")
        m = np.array([r["mass"] for r in selr], dtype=np.float64)
        d = np.array([r["delta"] for r in selr], dtype=np.float64)
        return float(np.sum(m * d) / np.sum(m))

    sigma_regions = {
        "all": mass_mean_delta(lambda r: True),
        "entrance_0_30mm": mass_mean_delta(lambda r: r["x_mm"] <= x_min + 30.0),
        "peak_pm_10mm": mass_mean_delta(lambda r: abs(r["x_mm"] - peak_x) <= 10.0),
    }

    gamma = {}
    if not args.skip_gamma:
        g = pair_metrics(topas, gpu, selection, shape_xyz, spacing, gamma=True)

        def gp(block, key):
            e = block[key]
            return e["pass_percent"] if isinstance(e, dict) else e

        gamma = {
            "nrmse": g["nrmse_over_reference_dmax_percent"],
            "gamma_3pct_0mm": {
                "global": gp(g["gamma_3pct_0mm"], "global"),
                "local": gp(g["gamma_3pct_0mm"], "local"),
            },
            "gamma_1pct_1mm": {
                "global": gp(g["gamma_1pct_1mm"], "global"),
                "local": gp(g["gamma_1pct_1mm"], "local"),
            },
        }

    alt_report = None
    if args.gpu_alt is not None:
        _, alt = load_gpu_mapped(args.gpu_alt, patient_shape)
        alt_m = selection_metrics(topas, alt, selection, dmax)
        delta = alt - gpu
        alt_report = {
            "label": args.alt_label,
            "metrics": alt_m,
            "vs_baseline": {
                "mean_bias_pct_dmax": 100.0 * float(np.mean(delta[selection])) / dmax,
                "rmse_pct_dmax": 100.0
                * float(np.sqrt(np.mean(delta[selection] ** 2)))
                / dmax,
                "corr_with_baseline_residual": float(
                    np.corrcoef((gpu - topas)[selection], delta[selection])[0, 1]
                ),
            },
        }
        if not args.skip_gamma:
            ga = pair_metrics(topas, alt, selection, shape_xyz, spacing, gamma=True)

            def gp2(block, key):
                e = block[key]
                return e["pass_percent"] if isinstance(e, dict) else e

            alt_report["gamma_3pct_0mm"] = {
                "global": gp2(ga["gamma_3pct_0mm"], "global"),
                "local": gp2(ga["gamma_3pct_0mm"], "local"),
            }
            alt_report["gamma_1pct_1mm"] = {
                "global": gp2(ga["gamma_1pct_1mm"], "global"),
                "local": gp2(ga["gamma_1pct_1mm"], "local"),
            }
            alt_report["delta_local_3pct_0mm_pp"] = (
                alt_report["gamma_3pct_0mm"]["local"] - gamma["gamma_3pct_0mm"]["local"]
            )

    # write depth csv
    depth_csv = args.output_dir / "depth_residual_profile.csv"
    with depth_csv.open("w", encoding="utf-8") as f:
        f.write(
            "ix,x_mm,n,mean_topas,mean_gpu,mean_R,rmse_R,sse,ratio,bias_pct_dmax\n"
        )
        for r in rows:
            f.write(
                f"{r['ix']},{r['x_mm']},{r['n']},{r['mean_topas']},{r['mean_gpu']},"
                f"{r['mean_R']},{r['rmse_R']},{r['sse']},{r['ratio']},{r['bias_pct_dmax']}\n"
            )

    report = {
        "baseline": base,
        "gamma": gamma,
        "depth_span_mm": [x_min, x_max],
        "peak_x_mm": peak_x,
        "regions": regions,
        "lateral_sigma_y_gpu_minus_topas_mm": sigma_regions,
        "alternate": alt_report,
        "paths": {
            "gpu": str(args.gpu),
            "gpu_alt": str(args.gpu_alt) if args.gpu_alt else None,
            "topas": str(args.topas),
            "depth_csv": str(depth_csv),
        },
        "prior_single_spot_note": (
            "edge-corrected single-spot diagnostic: entrance width ratio 0.967, "
            "peak 0.998; air-MCS lateral RMS ~0.076 mm; residual width develops "
            "downstream → transport over pure source/emittance"
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    md = []
    md.append("# RT07575 入口 / 源相空间残差分层")
    md.append("")
    md.append(f"GPU baseline: `{args.gpu}`")
    md.append(f"掩膜 voxels = **{base['selection_voxels'] if 'selection_voxels' in base else int(np.count_nonzero(selection))}**；NRMSE = **{base['nrmse_pct_dmax']:.3f}%**")
    if gamma:
        md.append(
            f"3%/0mm G/L = **{gamma['gamma_3pct_0mm']['global']:.2f}/{gamma['gamma_3pct_0mm']['local']:.2f}**；"
            f"1%/1mm G/L = **{gamma['gamma_1pct_1mm']['global']:.2f}/{gamma['gamma_1pct_1mm']['local']:.2f}**"
        )
    md.append("")
    md.append(f"深度跨度 patient-X: [{x_min:.2f}, {x_max:.2f}] mm；峰位 mean-dose max @ **{peak_x:.2f} mm**")
    md.append("")
    md.append("## 深度区域残差")
    md.append("")
    md.append("| 区域 | voxels | SSE 占比 | mean ratio GPU/TOPAS | bias %Dmax |")
    md.append("|---|---:|---:|---:|---:|")
    for name, reg in regions.items():
        if reg.get("voxels", 0) == 0:
            continue
        md.append(
            f"| {name} | {reg['voxels']} | {reg['sse_fraction']*100:.1f}% | "
            f"{reg['mean_ratio']:.4f} | {reg['mean_bias_pct_dmax']:+.3f} |"
        )
    md.append("")
    md.append("## 横向 σ_y（GPU−TOPAS，质量加权）")
    md.append("")
    for k, v in sigma_regions.items():
        md.append(f"- {k}: **{v:+.4f} mm**")
    md.append("")
    md.append(
        "先验单点：入口宽比 0.967、峰区 0.998；upstream air MCS 横向 RMS≈0.076 mm "
        "（不足以解释 full-plan local gamma 赤字）。"
    )
    if alt_report:
        md.append("")
        md.append(f"## 对照臂：{alt_report['label']}")
        md.append("")
        am = alt_report["metrics"]
        md.append(
            f"- NRMSE = **{am['nrmse_pct_dmax']:.3f}%**；bias = **{am['bias_pct_dmax']:+.3f}%Dmax**"
        )
        vb = alt_report["vs_baseline"]
        md.append(
            f"- vs baseline 场 RMSE = **{vb['rmse_pct_dmax']:.3f}%Dmax**；"
            f"corr(R,Δ)=**{vb['corr_with_baseline_residual']:.4f}**"
        )
        if "gamma_3pct_0mm" in alt_report:
            md.append(
                f"- 3%/0mm G/L = **{alt_report['gamma_3pct_0mm']['global']:.2f}/"
                f"{alt_report['gamma_3pct_0mm']['local']:.2f}** "
                f"(Δ local **{alt_report.get('delta_local_3pct_0mm_pp', float('nan')):+.2f} pp**)"
            )
    md.append("")
    md.append("## 解读要点")
    md.append("")
    ent = regions.get("entrance_0_20pct_span") or regions.get("entrance_0_30mm_from_min")
    if ent and ent.get("voxels", 0):
        md.append(
            f"- 入口 mean ratio = **{ent['mean_ratio']:.4f}**，SSE 占比 "
            f"**{ent['sse_fraction']*100:.1f}%**"
        )
    peak = regions.get("peak_pm_10mm")
    if peak and peak.get("voxels", 0):
        md.append(
            f"- 峰区 mean ratio = **{peak['mean_ratio']:.4f}**，SSE 占比 "
            f"**{peak['sse_fraction']*100:.1f}%**"
        )
    md.append(
        "- 若入口 SSE 占比低且横向入口差 << 1 mm → 源/emittance 不是 full-plan "
        "local 3%/0mm 主因；残差仍在下游输运/核末态形状。"
    )
    md.append("")
    (args.output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir / 'summary.md'}")
    print(
        f"entrance ratio={regions['entrance_0_20pct_span'].get('mean_ratio')} "
        f"sse%={100*regions['entrance_0_20pct_span'].get('sse_fraction',0):.1f} "
        f"peak ratio={regions['peak_pm_10mm'].get('mean_ratio')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

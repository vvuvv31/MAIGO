#!/usr/bin/env python3
"""Compare single high-weight RT07575 spot: GPU air_loss vs archived TOPAS 100k.

Uses the same mapping/mask conventions as full-plan residual diagnostics.
Reports NRMSE, local/global 3%/0mm & 1%/1mm, lateral width, core residual,
and whether single-spot residual looks like full-plan mottling.
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
from analyze_rt07575_residual_shape_energy import selection_metrics  # noqa: E402
from compare_topas_seed_gamma import load  # noqa: E402


def moments_lateral(
    volume: np.ndarray,
    mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    offset_xyz: tuple[float, float, float],
) -> dict[str, float]:
    w = np.where(mask, volume, 0.0).astype(np.float64)
    total = float(np.sum(w))
    if total <= 0:
        return {}
    nz, ny, nx = volume.shape
    x = offset_xyz[0] + spacing_xyz[0] * np.arange(nx)
    y = offset_xyz[1] + spacing_xyz[1] * np.arange(ny)
    z = offset_xyz[2] + spacing_xyz[2] * np.arange(nz)
    mx = float(np.sum(w.sum(axis=(0, 1)) * x) / total)
    my = float(np.sum(w.sum(axis=(0, 2)) * y) / total)
    mz = float(np.sum(w.sum(axis=(1, 2)) * z) / total)
    # project to y,z
    wy = w.sum(axis=(0, 2))
    wz = w.sum(axis=(1, 2))
    sy = math.sqrt(float(np.sum(wy * (y - my) ** 2) / total))
    sz = math.sqrt(float(np.sum(wz * (z - mz) ** 2) / total))
    sx = math.sqrt(float(np.sum(w.sum(axis=(0, 1)) * (x - mx) ** 2) / total))
    return {
        "com_xyz_mm": [mx, my, mz],
        "sigma_xyz_mm": [sx, sy, sz],
        "total": total,
    }


def core_residual(
    topas: np.ndarray,
    gpu: np.ndarray,
    mask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    offset_xyz: tuple[float, float, float],
    core_radius_mm: float = 10.0,
) -> dict[str, Any]:
    """Residual inside/outside lateral core around TOPAS COM in yz."""
    m = moments_lateral(topas, mask, spacing_xyz, offset_xyz)
    if not m:
        return {}
    _, my, mz = m["com_xyz_mm"]
    nz, ny, nx = topas.shape
    y = offset_xyz[1] + spacing_xyz[1] * np.arange(ny)
    z = offset_xyz[2] + spacing_xyz[2] * np.arange(nz)
    yy, zz = np.meshgrid(y, z, indexing="xy")  # shape (ny,nz)? need (nz,ny)
    # build radius map (nz, ny)
    Zg, Yg = np.meshgrid(z, y, indexing="ij")  # (nz, ny)
    r = np.sqrt((Yg - my) ** 2 + (Zg - mz) ** 2)
    core3 = np.zeros_like(mask, dtype=bool)
    wing3 = np.zeros_like(mask, dtype=bool)
    for ix in range(nx):
        sl = mask[:, :, ix]
        core3[:, :, ix] = sl & (r <= core_radius_mm)
        wing3[:, :, ix] = sl & (r > core_radius_mm)
    def stats(mm):
        if not np.any(mm):
            return {"voxels": 0}
        t = topas[mm]
        g = gpu[mm]
        d = g - t
        dmax = float(np.max(topas[mask]))
        return {
            "voxels": int(np.count_nonzero(mm)),
            "ratio": float(np.mean(g) / max(np.mean(t), 1e-30)),
            "bias_pct_dmax": float(100.0 * np.mean(d) / max(dmax, 1e-30)),
            "sse_frac": float(np.sum(d * d) / max(np.sum((gpu[mask] - topas[mask]) ** 2), 1e-30)),
            "nrmse_pct_dmax": float(
                100.0 * math.sqrt(float(np.mean(d * d))) / max(dmax, 1e-30)
            ),
        }
    return {
        "com_yz_mm": [my, mz],
        "core_radius_mm": core_radius_mm,
        "core": stats(core3),
        "wing": stats(wing3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gpu",
        type=Path,
        default=ROOT
        / "out/ct/RT07575/cascade_secondary_ablation/single_spot_air_loss/dose.mhd",
    )
    ap.add_argument(
        "--topas",
        type=Path,
        default=ROOT
        / "out/ct/RT07575/single_spot_source_diagnostic_edge_corrected/topas_patient.mhd",
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
        / "out/ct/RT07575/cascade_secondary_ablation/single_spot_air_loss_compare",
    )
    ap.add_argument("--skip-gamma", action="store_true")
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
    shape_xyz = tuple(int(v) for v in meta["DimSize"].split())

    # masks
    dmax_body = float(np.max(topas[body])) if np.any(body) else float(np.max(topas))
    # For single spot, BODY∩10% may be small; also report union-10% of each
    sel_body10 = body & (topas >= 0.10 * dmax_body)
    dmax_t = float(np.max(topas))
    dmax_g = float(np.max(gpu))
    sel_union10 = (topas >= 0.10 * dmax_t) | (gpu >= 0.10 * dmax_g)
    sel_t10 = topas >= 0.10 * dmax_t

    reports = {}
    for name, sel in (
        ("topas_ge_10pct_own_dmax", sel_t10),
        ("body_and_topas_ge_10pct_body_dmax", sel_body10),
        ("union_10pct_own_dmax", sel_union10),
    ):
        if not np.any(sel):
            reports[name] = {"voxels": 0}
            continue
        dmax = float(np.max(topas[sel])) if np.any(topas[sel] > 0) else dmax_t
        m = selection_metrics(topas, gpu, sel, dmax)
        entry: dict[str, Any] = {"voxels": int(np.count_nonzero(sel)), "metrics": m}
        mt = moments_lateral(topas, sel, spacing, offset)
        mg = moments_lateral(gpu, sel, spacing, offset)
        entry["moments_topas"] = mt
        entry["moments_gpu"] = mg
        if mt and mg:
            entry["delta_com_xyz_mm"] = [
                mg["com_xyz_mm"][i] - mt["com_xyz_mm"][i] for i in range(3)
            ]
            entry["sigma_ratio_xyz"] = [
                mg["sigma_xyz_mm"][i] / max(mt["sigma_xyz_mm"][i], 1e-30) for i in range(3)
            ]
        entry["core_wing_r10mm"] = core_residual(
            topas, gpu, sel, spacing, offset, core_radius_mm=10.0
        )
        if not args.skip_gamma:
            g = pair_metrics(topas, gpu, sel, shape_xyz, spacing, gamma=True)

            def gp(block, key):
                e = block[key]
                return e["pass_percent"] if isinstance(e, dict) else e

            entry["gamma_3pct_0mm"] = {
                "global": gp(g["gamma_3pct_0mm"], "global"),
                "local": gp(g["gamma_3pct_0mm"], "local"),
            }
            entry["gamma_1pct_1mm"] = {
                "global": gp(g["gamma_1pct_1mm"], "global"),
                "local": gp(g["gamma_1pct_1mm"], "local"),
            }
        reports[name] = entry

    # depth IDD on topas 10% mask
    sel = sel_t10
    xs, idd_t, idd_g = [], [], []
    for ix in range(topas.shape[2]):
        m = sel[:, :, ix]
        if not np.any(m):
            continue
        xs.append(offset[0] + ix * spacing[0])
        idd_t.append(float(np.sum(topas[:, :, ix][m])))
        idd_g.append(float(np.sum(gpu[:, :, ix][m])))
    xs = np.asarray(xs)
    idd_t = np.asarray(idd_t)
    idd_g = np.asarray(idd_g)
    idd = {
        "peak_x_topas_mm": float(xs[int(np.argmax(idd_t))]) if len(xs) else None,
        "peak_x_gpu_mm": float(xs[int(np.argmax(idd_g))]) if len(xs) else None,
        "peak_shift_mm": float(xs[int(np.argmax(idd_g))] - xs[int(np.argmax(idd_t))])
        if len(xs)
        else None,
        "integral_ratio": float(idd_g.sum() / max(idd_t.sum(), 1e-30)),
        "norm_idd_rmse": float(
            np.sqrt(np.mean((idd_g / max(idd_g.max(), 1e-30) - idd_t / max(idd_t.max(), 1e-30)) ** 2))
        )
        if len(xs)
        else None,
    }

    report = {
        "gpu": str(args.gpu),
        "topas": str(args.topas),
        "note": (
            "TOPAS single-spot archive may use slightly different L2 energy "
            "(no air_loss); GPU uses air_loss L2 from full-plan spot 24"
        ),
        "masks": reports,
        "idd_topas10": idd,
        "fullplan_context": {
            "local_3pct_0mm": 43.64,
            "nrmse_pct": 2.280,
            "interpretation_question": (
                "If single-spot local gamma is similarly low and core is under, "
                "residual is single-spot physics; if single-spot is much better, "
                "multi-spot interference/weighting dominates"
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    md = []
    md.append("# RT07575 单高权重 spot（spot 24）air_loss GPU vs TOPAS 100k")
    md.append("")
    md.append(f"GPU: `{args.gpu}`")
    md.append(f"TOPAS: `{args.topas}`")
    md.append(
        "GPU：air_loss 能量的 spot 24，100k hist，seed 20260805，soft400 INCL++ package。"
    )
    md.append("")
    md.append("## 掩膜指标")
    md.append("")
    md.append("| 掩膜 | voxels | NRMSE% | mean E/R | σ ratio y/z | 3%/0mm G/L | 1%/1mm G/L |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, e in reports.items():
        if e.get("voxels", 0) == 0:
            continue
        m = e["metrics"]
        sr = e.get("sigma_ratio_xyz")
        sr_s = f"{sr[1]:.3f}/{sr[2]:.3f}" if sr else "—"
        g30 = e.get("gamma_3pct_0mm", {})
        g11 = e.get("gamma_1pct_1mm", {})
        g30s = (
            f"{g30.get('global', float('nan')):.2f}/{g30.get('local', float('nan')):.2f}"
            if g30
            else "—"
        )
        g11s = (
            f"{g11.get('global', float('nan')):.2f}/{g11.get('local', float('nan')):.2f}"
            if g11
            else "—"
        )
        # mean ratio from moments totals
        mt, mg = e.get("moments_topas") or {}, e.get("moments_gpu") or {}
        mean_ratio = (
            mg.get("total", 0) / max(mt.get("total", 1e-30), 1e-30) if mt and mg else float("nan")
        )
        md.append(
            f"| {name} | {e['voxels']} | {m['nrmse_pct_dmax']:.3f} | "
            f"{mean_ratio:.4f} | {sr_s} | {g30s} | {g11s} |"
        )
    md.append("")
    # core wing for primary mask
    prim = reports.get("topas_ge_10pct_own_dmax", {})
    cw = prim.get("core_wing_r10mm") or {}
    if cw:
        md.append("## Core/wing（TOPAS 10% 掩膜，横向 r≤10 mm）")
        md.append("")
        md.append(
            f"- core ratio = **{cw.get('core', {}).get('ratio', float('nan')):.4f}**，"
            f"SSE% = **{100*cw.get('core', {}).get('sse_frac', 0):.1f}**"
        )
        md.append(
            f"- wing ratio = **{cw.get('wing', {}).get('ratio', float('nan')):.4f}**，"
            f"SSE% = **{100*cw.get('wing', {}).get('sse_frac', 0):.1f}**"
        )
        if prim.get("sigma_ratio_xyz"):
            md.append(
                f"- σ ratio xyz = {[round(v, 4) for v in prim['sigma_ratio_xyz']]}"
            )
        if prim.get("delta_com_xyz_mm"):
            md.append(
                f"- COM GPU−TOPAS mm = {[round(v, 3) for v in prim['delta_com_xyz_mm']]}"
            )
    md.append("")
    md.append("## IDD（TOPAS 10% 掩膜）")
    md.append("")
    md.append(
        f"- 峰位 TOPAS x={idd['peak_x_topas_mm']}，GPU x={idd['peak_x_gpu_mm']}，"
        f"Δ=**{idd['peak_shift_mm']} mm**"
    )
    md.append(
        f"- 积分比 GPU/TOPAS=**{idd['integral_ratio']:.4f}**；"
        f"归一化 IDD RMSE=**{idd['norm_idd_rmse']:.4f}**"
    )
    md.append("")
    md.append("## 与 full-plan 对照")
    md.append("")
    fp_l = 43.64
    ss = reports.get("topas_ge_10pct_own_dmax", {})
    ss_l = (ss.get("gamma_3pct_0mm") or {}).get("local", float("nan"))
    md.append(f"- full-plan local 3%/0mm ≈ **{fp_l:.2f}%**")
    md.append(f"- 单 spot local 3%/0mm ≈ **{ss_l:.2f}%**")
    if isinstance(ss_l, float) and ss_l == ss_l:
        if ss_l < fp_l + 5:
            md.append(
                "- **单野已经很差（与 full-plan 同量级或更差）** → 残差主要是"
                " **单野物理/源形状**，不是多野干涉主导。"
            )
        elif ss_l > 70:
            md.append(
                "- **单野明显好于 full-plan** → 多野叠加/权重/对齐 可能主导 full-plan 赤字。"
            )
        else:
            md.append("- 单野介于中间：单野物理 + 多野叠加 均有贡献。")
    md.append("")
    md.append("脚本: `validation/scripts/analyze_rt07575_single_spot_air_loss_residual.py`")
    (args.output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # print compact
    print(f"Wrote {args.output_dir / 'summary.md'}")
    for name, e in reports.items():
        if e.get("voxels", 0) == 0:
            continue
        g30 = e.get("gamma_3pct_0mm", {})
        print(
            name,
            f"n={e['voxels']}",
            f"NRMSE={e['metrics']['nrmse_pct_dmax']:.3f}",
            f"3/0={g30.get('global', float('nan')):.2f}/{g30.get('local', float('nan')):.2f}",
            f"sig_yz={e.get('sigma_ratio_xyz')}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

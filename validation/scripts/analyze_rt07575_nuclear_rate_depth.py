#!/usr/bin/env python3
"""RT07575 nuclear-interaction rate vs depth diagnostic (no new MC required).

Builds a dose-mask-weighted estimate of primary C-12 inelastic macroscopic
cross section Σ(x) and interaction density λ(x)=Φ(x)Σ(x) along patient beam
depth (patient X), using:

  - CT density + Schneider section IDs
  - Schneider mass XS table (mm^-1 at 1 g/cm3)
  - CSDA energy degradation in water-equivalent depth from water C-12 SP
  - TOPAS dose on the fixed BODY∩≥10% Dmax mask as a primary-path weight

Compares depth profiles of Σ, λ, survival against GPU−TOPAS residual SSE/bias
from air_loss. Also reports expected nuclear interaction probability per
history from a 1-D equivalent path and contrasts with GPU run.log.

Diagnostic only — not a fitted calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_rt07575_equal_history_dose_residual import (  # noqa: E402
    HEADER,
    MAGIC,
    load_gpu_mapped,
    load_patient,
    read_ct_grid,
)
from analyze_rt07575_residual_shape_energy import selection_metrics  # noqa: E402
from compare_topas_seed_gamma import load  # noqa: E402


def load_schneider_mass_xs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return energies [MeV/u] and mass-xs table [nE, nSection] (1/mm at 1 g/cm3)."""
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        sec_cols = [c for c in cols if c.startswith("section_")]
        sec_cols.sort(key=lambda c: int(c.split("_")[1]))
        energies: list[float] = []
        rows: list[list[float]] = []
        for row in reader:
            energies.append(float(row["energy_MeV_per_u"]))
            rows.append([float(row[c]) for c in sec_cols])
    return np.asarray(energies, dtype=np.float64), np.asarray(rows, dtype=np.float64)


def load_c12_water_sp(path: Path) -> tuple[np.ndarray, np.ndarray]:
    energies: list[float] = []
    sp: list[float] = []
    with path.open(encoding="utf-8") as f:
        header = None
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            if line.startswith("atomic_number"):
                header = line.strip().split(",")
                continue
            parts = line.strip().split(",")
            if header is None:
                continue
            z, a = int(float(parts[0])), int(float(parts[1]))
            if z == 6 and a == 12:
                energies.append(float(parts[2]))
                sp.append(float(parts[3]))  # MeV/mm
    e = np.asarray(energies, dtype=np.float64)
    s = np.asarray(sp, dtype=np.float64)
    order = np.argsort(e)
    return e[order], s[order]


def csda_range_table(e_mevu: np.ndarray, sp_mev_per_mm: np.ndarray) -> np.ndarray:
    """CSDA range R(E) in mm water from 0 to E (total KE = 12*E)."""
    r = np.zeros_like(e_mevu)
    for i in range(1, len(e_mevu)):
        de = e_mevu[i] - e_mevu[i - 1]
        if de <= 0:
            r[i] = r[i - 1]
            continue
        r[i] = r[i - 1] + 12.0 * 0.5 * (
            1.0 / max(sp_mev_per_mm[i], 1e-30)
            + 1.0 / max(sp_mev_per_mm[i - 1], 1e-30)
        ) * de
    return r


def energy_from_remaining_range(
    remaining_mm: np.ndarray, e_mevu: np.ndarray, range_mm: np.ndarray
) -> np.ndarray:
    """Invert CSDA range: remaining range → kinetic energy MeV/u."""
    rem = np.clip(remaining_mm, 0.0, float(range_mm[-1]))
    return np.interp(rem, range_mm, e_mevu)


def map_ct_to_patient(
    density_gpu: np.ndarray, section_gpu: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """GPU CCTG (z,y,x)=(nz,ny,nx) → patient (Z,Y,X) same as dose mapping."""
    # density_gpu shaped (nz, ny, nx) with nx=depth along GPU X = patient Y? 
    # From metadata: gpu(x,y,z)=patient(y,z,-x); load_gpu_mapped does
    # transpose(1,2,0) then reverse patient X on GPU dose (depth, Z, Y).
    # CT binary DimSize nx,ny,nz = 505,35,417 with spacing 0.5,2,0.5
    # reshape (nz,ny,nx)=(417,35,505). GPU dose DimSize 505,35,417 =
    # (nx,ny,nz) as (depth, y_gpu, z_gpu) flattened as z-major in load?
    # load_gpu_mapped: reshape(shape[2], shape[1], shape[0]) = (nz,ny,nx)=(417,35,505)
    # then transpose(1,2,0) → (35,505,417) then ::-1 on last axis.
    dens = np.transpose(density_gpu, (1, 2, 0))[:, :, ::-1]
    sec = np.transpose(section_gpu, (1, 2, 0))[:, :, ::-1]
    return dens, sec


def parse_gpu_nuclear(log_path: Path) -> dict[str, float]:
    text = log_path.read_text(encoding="utf-8", errors="replace")

    def num(pat: str, default: float = 0.0) -> float:
        m = re.search(pat, text, re.MULTILINE)
        return float(m.group(1)) if m else default

    return {
        "histories": num(r"^Histories:\s*([0-9.eE+-]+)"),
        "nuclear_interactions": num(r"^Nuclear interactions:\s*([0-9.eE+-]+)"),
        "initial_energy_MeVu": num(r"^Initial energy:\s*([0-9.eE+-]+)\s+MeV/u"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gpu",
        type=Path,
        default=ROOT / "out/ct/RT07575/upstream_air_loss_ablation/air_loss/dose.mhd",
    )
    ap.add_argument(
        "--primary-only",
        type=Path,
        default=ROOT
        / "out/ct/RT07575/cascade_secondary_ablation/phys_primary_only/dose.mhd",
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
        "--ct-grid",
        type=Path,
        default=ROOT / "ct/grid/patient_ct_tps_90_xneg_edge_corrected.bin",
    )
    ap.add_argument(
        "--schneider-xs",
        type=Path,
        default=ROOT
        / "data/c12_inelastic_cross_sections_schneider_geant4_11_1_3.csv",
    )
    ap.add_argument(
        "--water-sp",
        type=Path,
        default=ROOT / "data/ion_stopping_power_water_geant4_11_3_2.csv",
    )
    ap.add_argument(
        "--run-log",
        type=Path,
        default=ROOT / "out/ct/RT07575/upstream_air_loss_ablation/air_loss/run.log",
    )
    ap.add_argument(
        "--e0-MeVu",
        type=float,
        default=204.53,
        help="entrance kinetic energy MeV/u after air loss",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "out/ct/RT07575/cascade_secondary_ablation/nuclear_rate_depth",
    )
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, topas = load_patient(args.topas)
    _, body_v = load_patient(args.body_mask)
    body = body_v > 0.5
    patient_shape = topas.shape
    _, gpu = load_gpu_mapped(args.gpu, patient_shape)
    primary = None
    if args.primary_only.exists():
        _, primary = load_gpu_mapped(args.primary_only, patient_shape)

    meta, _ = load(args.topas)
    spacing = tuple(float(v) for v in meta["ElementSpacing"].split())
    offset = tuple(float(v) for v in meta["Offset"].split())

    dmax_body = float(np.max(topas[body]))
    selection = body & (topas >= 0.10 * dmax_body)
    dmax = float(np.max(topas[selection]))
    base = selection_metrics(topas, gpu, selection, dmax)

    density_gpu, section_gpu = read_ct_grid(args.ct_grid)
    density, section = map_ct_to_patient(density_gpu, section_gpu)
    if density.shape != patient_shape:
        # CT patient mapping may differ by origin/spacing; try alternate reshape
        raise SystemExit(
            f"CT mapped shape {density.shape} != patient dose {patient_shape}"
        )

    e_xs, xs_table = load_schneider_mass_xs(args.schneider_xs)
    n_sec = xs_table.shape[1]
    e_sp, sp = load_c12_water_sp(args.water_sp)
    range_table = csda_range_table(e_sp, sp)
    r0 = float(np.interp(args.e0_MeVu, e_sp, range_table))

    residual = gpu - topas
    nx = patient_shape[2]
    rows: list[dict[str, Any]] = []
    # cumulative water-equivalent depth along +patient X through selection COM path
    # Use dose-weighted mean density per slice for WE progression.
    we_depth = 0.0
    cum_optical = 0.0  # ∫ Σ ds
    for ix in range(nx):
        mask = selection[:, :, ix]
        n = int(np.count_nonzero(mask))
        if n == 0:
            continue
        dens_s = density[:, :, ix][mask].astype(np.float64)
        sec_s = section[:, :, ix][mask].astype(np.int32)
        top_s = topas[:, :, ix][mask].astype(np.float64)
        gpu_s = gpu[:, :, ix][mask].astype(np.float64)
        res_s = residual[:, :, ix][mask].astype(np.float64)
        w = np.clip(top_s, 0.0, None)
        wsum = float(np.sum(w))
        if wsum <= 0:
            w = np.ones_like(top_s)
            wsum = float(np.sum(w))

        mean_rho = float(np.sum(w * dens_s) / wsum)
        # mass XS at current energy for each voxel's section, then density-weighted
        e_now = float(
            energy_from_remaining_range(
                np.array([max(r0 - we_depth, 0.0)]), e_sp, range_table
            )[0]
        )
        # interpolate XS table at e_now for all sections
        xs_e = np.array(
            [np.interp(e_now, e_xs, xs_table[:, s]) for s in range(n_sec)],
            dtype=np.float64,
        )
        sec_clip = np.clip(sec_s, 0, n_sec - 1)
        mass_xs = xs_e[sec_clip]  # 1/mm at 1 g/cm3
        sigma = mass_xs * dens_s  # 1/mm
        mean_sigma = float(np.sum(w * sigma) / wsum)
        # water-only reference at same energy
        # section 07 ~ soft tissue; also water-like: use average of soft sections 3-8
        xs_waterish = float(np.interp(e_now, e_xs, xs_table[:, 7]))  # section 07
        sigma_waterish = xs_waterish * mean_rho

        dx = spacing[0]
        # WE depth step: density-relative to water
        we_step = dx * mean_rho
        # optical depth for survival (use pre-step energy approx)
        optical_step = mean_sigma * dx
        survival = math.exp(-cum_optical)
        # interaction density proxy (relative): Φ0 * survival * Σ
        lambda_rel = survival * mean_sigma

        sse = float(np.sum(res_s * res_s))
        rows.append(
            {
                "ix": ix,
                "x_mm": offset[0] + ix * spacing[0],
                "n": n,
                "mean_rho": mean_rho,
                "E_MeVu": e_now,
                "mean_sigma_per_mm": mean_sigma,
                "sigma_waterish_per_mm": sigma_waterish,
                "we_depth_mm": we_depth,
                "survival": survival,
                "lambda_rel": lambda_rel,
                "mean_topas": float(np.mean(top_s)),
                "mean_gpu": float(np.mean(gpu_s)),
                "mean_R": float(np.mean(res_s)),
                "sse": sse,
                "ratio": float(np.mean(gpu_s) / max(np.mean(top_s), 1e-30)),
            }
        )
        we_depth += we_step
        cum_optical += optical_step

    if not rows:
        raise SystemExit("no selected depth bins")

    # arrays for correlation
    sse_a = np.array([r["sse"] for r in rows], dtype=np.float64)
    r_a = np.array([r["mean_R"] for r in rows], dtype=np.float64)
    lam_a = np.array([r["lambda_rel"] for r in rows], dtype=np.float64)
    sig_a = np.array([r["mean_sigma_per_mm"] for r in rows], dtype=np.float64)
    e_a = np.array([r["E_MeVu"] for r in rows], dtype=np.float64)
    we_a = np.array([r["we_depth_mm"] for r in rows], dtype=np.float64)
    n_a = np.array([r["n"] for r in rows], dtype=np.float64)

    def wcorr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
        ww = w / max(np.sum(w), 1e-30)
        am = np.sum(ww * a)
        bm = np.sum(ww * b)
        av = a - am
        bv = b - bm
        den = math.sqrt(float(np.sum(ww * av * av) * np.sum(ww * bv * bv)))
        if den <= 0:
            return float("nan")
        return float(np.sum(ww * av * bv) / den)

    corrs = {
        "sse_vs_lambda": wcorr(sse_a, lam_a, n_a),
        "sse_vs_sigma": wcorr(sse_a, sig_a, n_a),
        "meanR_vs_lambda": wcorr(r_a, lam_a, n_a),
        "meanR_vs_sigma": wcorr(r_a, sig_a, n_a),
        "sse_vs_E": wcorr(sse_a, e_a, n_a),
        "sse_vs_we_depth": wcorr(sse_a, we_a, n_a),
    }

    # primary-only nuclear dump proxy if available
    primary_corr = {}
    if primary is not None:
        # energy that primary-only deposits beyond full GPU ≈ secondary carry-away
        # map; residual nuclear rate not directly, but primary_only/TOPAS shape
        p_m = selection_metrics(topas, primary, selection, dmax)
        # depth mean primary_only - topas
        prim_R = []
        for r in rows:
            ix = r["ix"]
            mask = selection[:, :, ix]
            prim_R.append(float(np.mean(primary[:, :, ix][mask] - topas[:, :, ix][mask])))
        prim_R_a = np.asarray(prim_R, dtype=np.float64)
        corrs["sse_vs_primaryOnly_minus_topas"] = wcorr(sse_a, prim_R_a, n_a)
        corrs["lambda_vs_primaryOnly_minus_topas"] = wcorr(lam_a, prim_R_a, n_a)
        primary_corr = {
            "primary_only_nrmse_pct_dmax": p_m["nrmse_pct_dmax"],
            "primary_only_bias_pct_dmax": p_m["bias_pct_dmax"],
        }

    # expected nuclear interactions (1D WE path through dose-weighted CT)
    # P_int = 1 - exp(-τ_total); N_exp = histories * P_int
    # Note: multi-spot 3D fluence is not 1D — this is order-of-magnitude only.
    tau_total = float(cum_optical)
    p_int = 1.0 - math.exp(-tau_total)
    gpu_log = parse_gpu_nuclear(args.run_log) if args.run_log.exists() else {}
    histories = gpu_log.get("histories", 0.0)
    n_gpu = gpu_log.get("nuclear_interactions", 0.0)
    n_exp_1d = histories * p_int if histories > 0 else float("nan")

    # region aggregates
    x_min = rows[0]["x_mm"]
    x_max = rows[-1]["x_mm"]
    span = x_max - x_min
    peak_x = max(rows, key=lambda r: r["mean_topas"])["x_mm"]

    def region(pred):
        selr = [r for r in rows if pred(r)]
        if not selr:
            return {}
        n = sum(r["n"] for r in selr)
        sse = sum(r["sse"] for r in selr)
        total_sse = sum(r["sse"] for r in rows)
        return {
            "voxels": n,
            "sse_fraction": sse / max(total_sse, 1e-30),
            "mean_lambda": sum(r["lambda_rel"] * r["n"] for r in selr) / max(n, 1),
            "mean_sigma": sum(r["mean_sigma_per_mm"] * r["n"] for r in selr) / max(n, 1),
            "mean_E_MeVu": sum(r["E_MeVu"] * r["n"] for r in selr) / max(n, 1),
            "mean_ratio": sum(r["ratio"] * r["n"] for r in selr) / max(n, 1),
        }

    regions = {
        "entrance_0_20pct": region(lambda r: r["x_mm"] <= x_min + 0.2 * span),
        "mid_20_60pct": region(
            lambda r: x_min + 0.2 * span < r["x_mm"] <= x_min + 0.6 * span
        ),
        "peak_pm_10mm": region(lambda r: abs(r["x_mm"] - peak_x) <= 10.0),
        "distal": region(lambda r: r["x_mm"] > peak_x + 10.0),
    }

    # write CSV
    csv_path = args.output_dir / "nuclear_rate_depth_profile.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(
            "ix,x_mm,n,mean_rho,E_MeVu,mean_sigma_per_mm,sigma_waterish_per_mm,"
            "we_depth_mm,survival,lambda_rel,mean_topas,mean_gpu,mean_R,sse,ratio\n"
        )
        for r in rows:
            f.write(
                f"{r['ix']},{r['x_mm']},{r['n']},{r['mean_rho']},{r['E_MeVu']},"
                f"{r['mean_sigma_per_mm']},{r['sigma_waterish_per_mm']},"
                f"{r['we_depth_mm']},{r['survival']},{r['lambda_rel']},"
                f"{r['mean_topas']},{r['mean_gpu']},{r['mean_R']},{r['sse']},"
                f"{r['ratio']}\n"
            )

    report = {
        "baseline_dose": base,
        "correlations_voxelWeighted_by_depthBin": corrs,
        "regions": regions,
        "optical_depth_total": tau_total,
        "interaction_probability_1d": p_int,
        "gpu_log": gpu_log,
        "expected_nuclear_1d": n_exp_1d,
        "nuclear_gpu_over_expected_1d": n_gpu / max(n_exp_1d, 1e-30)
        if n_exp_1d == n_exp_1d
        else None,
        "entrance_energy_MeVu": args.e0_MeVu,
        "csda_range_water_mm_at_e0": r0,
        "we_depth_total_mm": we_depth,
        "primary_only": primary_corr,
        "paths": {
            "csv": str(csv_path),
            "gpu": str(args.gpu),
            "ct_grid": str(args.ct_grid),
            "schneider_xs": str(args.schneider_xs),
        },
        "method_notes": [
            "Σ = mass_xs(E, Schneider section) × density; E from water CSDA WE depth",
            "λ_rel = exp(-∫Σ ds) × Σ (relative interaction density)",
            "1-D N_exp = histories × (1-exp(-τ)) is a rough upper/order check; multi-spot 3D fluence is not 1-D",
            "TOPAS dose weights voxels inside the fixed selection mask",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    md: list[str] = []
    md.append("# RT07575 核反应率–深度同构诊断")
    md.append("")
    md.append(
        f"基线 GPU air_loss NRMSE=**{base['nrmse_pct_dmax']:.3f}%**；"
        f"E0=**{args.e0_MeVu:.2f} MeV/u**；水 CSDA R0=**{r0:.1f} mm**；"
        f"掩膜 WE 总深度≈**{we_depth:.1f} mm**。"
    )
    md.append("")
    md.append("## 1. GPU 核反应计数 vs 1D 期望（量级）")
    md.append("")
    md.append(
        f"- GPU nuclear interactions = **{n_gpu:.0f}** / histories **{histories:.0f}** "
        f"（**{n_gpu/max(histories,1):.4f}** /history）"
    )
    md.append(
        f"- 1D 光学厚度 τ=**{tau_total:.3f}** → P_int=**{p_int:.4f}** → "
        f"N_exp≈**{n_exp_1d:.3e}**"
    )
    if n_exp_1d == n_exp_1d and n_exp_1d > 0:
        md.append(
            f"- GPU/N_exp_1d = **{n_gpu/n_exp_1d:.3f}** "
            f"（<<1 预期：3D 多野、侧向漏失、非单能；仅作量级）"
        )
    md.append("")
    md.append("## 2. 深度剖面相关（体素数加权）")
    md.append("")
    md.append("| 相关 | 值 |")
    md.append("|---|---:|")
    for k, v in corrs.items():
        md.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
    md.append("")
    md.append(
        "解读：若 `sse_vs_lambda` 接近 0 → 残差 SSE **不**随核反应密度代理同相；"
        "核反应率空间形状不太可能是主因。若强正相关，则中段核反应过密/过稀值得 A/B。"
    )
    md.append("")
    md.append("## 3. 分区")
    md.append("")
    md.append("| 区域 | SSE% | mean λ_rel | mean Σ [1/mm] | mean E [MeV/u] | ratio |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for name, reg in regions.items():
        if not reg:
            continue
        md.append(
            f"| {name} | {100*reg['sse_fraction']:.1f} | {reg['mean_lambda']:.5f} | "
            f"{reg['mean_sigma']:.5f} | {reg['mean_E_MeVu']:.1f} | {reg['mean_ratio']:.4f} |"
        )
    md.append("")
    md.append("## 4. 结论")
    md.append("")
    s_l = corrs.get("sse_vs_lambda", 0.0)
    if isinstance(s_l, float) and abs(s_l) < 0.25:
        md.append(
            f"1. **sse_vs_lambda={s_l:.3f}（弱）** → 残差 SSE 深度结构与核反应密度代理"
            "不同构；**不宜**把主攻放在整体 XS 标定/反应率缩放。"
        )
    elif isinstance(s_l, float) and s_l >= 0.25:
        md.append(
            f"1. **sse_vs_lambda={s_l:.3f}（正）** → 残差与核反应热点同相；"
            "检查 XS×密度 或反应能谱采样是否在中段过强。"
        )
    else:
        md.append(
            f"1. **sse_vs_lambda={s_l:.3f}（负）** → 残差与核反应代理反相；"
            "可能是次级输运/沉积形状而非反应率本身。"
        )
    md.append(
        "2. 入口/中段/峰区的 Σ 与 E 剖面见 CSV；材料 Schneider XS 与水模 soft 差异 ~1% 量级"
        "（此前已核），不是 39 pp gamma 的来源。"
    )
    md.append(
        "3. **下一杠杆**：中段高梯度的 **H/He 次级横向输运**（步长/MCS 物种差/"
        "secondary condensed history）；或 soft-tissue **200 MeV INCL++ package** 生成。"
    )
    md.append("")
    md.append(f"CSV: `{csv_path}`")
    md.append("脚本: `validation/scripts/analyze_rt07575_nuclear_rate_depth.py`")
    (args.output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_dir / 'summary.md'}")
    print("corrs", corrs)
    print(f"tau={tau_total:.3f} P_int={p_int:.4f} N_gpu={n_gpu:.0f} N_exp1d={n_exp_1d:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

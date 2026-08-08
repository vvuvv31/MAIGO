#!/usr/bin/env python3
"""RT07575 full-plan energy-layer residual attribution (diagnostic only).

Runs GPU dose for each BeamEnergy layer (plan L4 histories only), then
attributes full-plan residual R=GPU−TOPAS via correlation and OLS on the
BODY∩TOPAS≥10% mask.

Does not change production configs. Outputs under
out/ct/RT07575/cascade_secondary_ablation/energy_layer_decomp/.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation/scripts"))
from analyze_rt07575_equal_history_dose_residual import (  # noqa: E402
    load_gpu_mapped,
    load_patient,
)
from compare_topas_seed_gamma import load  # noqa: E402


def parse_channel(text: str, prefix: str, name: str) -> np.ndarray:
    m = re.search(rf"{prefix}:Tf/Scatterer1/{re.escape(name)}\s*=\s*([^\n]+)", text)
    if not m:
        raise RuntimeError(f"missing {prefix} {name}")
    toks = (
        m.group(1)
        .replace("MeV", "")
        .replace("mm", "")
        .replace("deg", "")
        .split()
    )
    n = int(float(toks[0]))
    vals = np.array([float(x) for x in toks[1 : 1 + n]], dtype=np.float64)
    if len(vals) != n:
        raise RuntimeError(f"{name}: expected {n} values, got {len(vals)}")
    return vals


def set_key(cfg: str, key: str, value: str) -> str:
    pat = re.compile(rf"^{re.escape(key)}:.*$", re.M)
    if pat.search(cfg):
        return pat.sub(f"{key}: {value}", cfg)
    return cfg.rstrip() + f"\n{key}: {value}\n"


def write_layer_spots(text: str, E_tot: np.ndarray, L4: np.ndarray, layer_E: float, path: Path) -> int:
    L4_new = L4.copy()
    L4_new[E_tot != layer_E] = 0
    hsum = int(L4_new.sum())
    if hsum == 0:
        return 0
    body = " ".join(str(int(v)) for v in L4_new)
    repl = f"iv:Tf/Scatterer1/L4/Values = {len(L4_new)} {body}"
    new_text, nsub = re.subn(
        r"iv:Tf/Scatterer1/L4/Values\s*=\s*[^\n]+", repl, text, count=1
    )
    if nsub != 1:
        raise RuntimeError("failed to rewrite L4")
    path.write_text(new_text)
    return hsum


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--spots",
        type=Path,
        default=ROOT / "ct/fullplan_result/RT07575/spots_full_plan.txt",
    )
    ap.add_argument(
        "--gpu-full",
        type=Path,
        default=ROOT
        / "out/ct/RT07575/cascade_secondary_ablation/fullplan_mfp0p5_rhs0p9/dose.mhd",
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
        "--base-config",
        type=Path,
        default=ROOT / "config/beam_ct_fullplan_rt07575_let_soft_tissue.yaml",
    )
    ap.add_argument(
        "--gpu-bin",
        type=Path,
        default=ROOT / "build/oneapi-nvidia-release/carbon_mc",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "out/ct/RT07575/cascade_secondary_ablation/energy_layer_decomp",
    )
    ap.add_argument("--skip-run", action="store_true", help="only analyze existing layer doses")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    text = args.spots.read_text()
    E_tot = parse_channel(text, "dv", "L1/Values")
    E_u = E_tot / 12.0
    L4 = parse_channel(text, "iv", "L4/Values").astype(np.int64)
    ids = parse_channel(text, "iv", "L0/Values").astype(np.int64)
    x = parse_channel(text, "dv", "L5/Values")
    y = parse_channel(text, "dv", "L6/Values")
    active = L4 > 0
    layer_keys = sorted(set(E_tot[active].tolist()))
    base_cfg = args.base_config.read_text()

    layer_info = []
    for e in layer_keys:
        mask = active & (E_tot == e)
        h = int(L4[mask].sum())
        layer_info.append(
            dict(
                E_tot=e,
                E_u=e / 12.0,
                n_spots=int(mask.sum()),
                histories=h,
                frac=h / float(L4.sum()),
            )
        )

    layer_dose_paths: dict[float, Path] = {}
    for info in layer_info:
        e = info["E_tot"]
        tag = f"layer_E{int(e)}"
        out_dir = args.out / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        spots_path = out_dir / "spots.txt"
        hsum = write_layer_spots(text, E_tot, L4, e, spots_path)
        dose = out_dir / "dose.mhd"
        raw = out_dir / "dose.raw"
        if args.skip_run or (dose.exists() and raw.exists()):
            if dose.exists():
                layer_dose_paths[e] = dose
            continue
        cfg = base_cfg
        cfg = set_key(cfg, "topas_spots_file", str(spots_path))
        cfg = set_key(cfg, "number_of_histories", str(hsum))
        cfg = set_key(cfg, "nuclear_residual_heat_mfp_mm", "0.5")
        cfg = set_key(cfg, "nuclear_residual_heat_scale", "0.9")
        cfg = set_key(cfg, "electronic_buildup_fraction", "0.0")
        cfg = set_key(cfg, "random_seed", "20260801")
        cfg = set_key(cfg, "voxel_dose_mhd_output_file", str(dose))
        cfg = set_key(cfg, "let_voxel_mhd_output_file", str(out_dir / "letd"))
        (out_dir / "config.yaml").write_text(cfg)
        print(f"RUN {tag} E_u={e/12:.1f} hist={hsum}", flush=True)
        r = subprocess.run(
            [str(args.gpu_bin), "--config", str(out_dir / "config.yaml"), "--device", "cuda"],
            cwd=str(ROOT),
            stdout=open(out_dir / "run.log", "w"),
            stderr=subprocess.STDOUT,
        )
        print(f"  exit={r.returncode}", flush=True)
        if r.returncode == 0 and dose.exists():
            layer_dose_paths[e] = dose

    _, body = load_patient(args.body_mask)
    body = body > 0.5
    _, topas = load_patient(args.topas)
    _, gfull = load_gpu_mapped(args.gpu_full, topas.shape)
    dmeta, _ = load(args.topas)
    sp = tuple(float(v) for v in dmeta["ElementSpacing"].split())
    dmax = float(topas[body].max())
    sel = body & (topas >= 0.10 * dmax)
    R = gfull - topas
    sse = float(np.sum(R[sel] ** 2)) + 1e-30

    rows = []
    Gsum = np.zeros_like(gfull)
    layer_maps = {}
    for info in layer_info:
        e = info["E_tot"]
        path = layer_dose_paths.get(e)
        if path is None:
            continue
        _, gl = load_gpu_mapped(path, topas.shape)
        layer_maps[e] = gl
        Gsum += gl
        gl_s = gl[sel]
        R_s = R[sel]
        corr = (
            0.0
            if gl_s.std() < 1e-30 or R_s.std() < 1e-30
            else float(np.corrcoef(gl_s, R_s)[0, 1])
        )
        thr = np.percentile(np.abs(R_s), 90)
        hot = sel & (np.abs(R) >= thr)
        rows.append(
            dict(
                E_tot=e,
                E_u=e / 12.0,
                n_spots=info["n_spots"],
                histories=info["histories"],
                hist_frac=info["frac"],
                corr_R=corr,
                mass_frac_sel=float(gl[sel].sum()) / max(float(gfull[sel].sum()), 1e-30),
                mass_frac_in_hot=float(gl[hot].sum()) / max(float(gl[sel].sum()), 1e-30),
            )
        )

    keys = [r["E_tot"] for r in rows]
    X = np.column_stack(
        [layer_maps[e][sel].ravel() for e in keys] + [np.ones(int(sel.sum()))]
    )
    y = R[sel].ravel()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    for i, e in enumerate(keys):
        for r in rows:
            if r["E_tot"] == e:
                r["ols_coef"] = float(coef[i])
                pred_i = coef[i] * layer_maps[e][sel]
                r["ols_sse_share"] = float(np.sum(pred_i**2)) / sse
                w = layer_maps[e][sel]
                r["dose_weighted_bias"] = float(np.sum(R[sel] * w) / max(float(w.sum()), 1e-30))
    yhat = X @ coef
    r2 = 1.0 - float(np.sum((y - yhat) ** 2)) / max(
        float(np.sum((y - y.mean()) ** 2)), 1e-30
    )
    sum_layers = float(Gsum[sel].sum())
    full_int = float(gfull[sel].sum())
    rows_sorted = sorted(rows, key=lambda r: -abs(r["corr_R"]))
    order = np.argsort(-L4)

    payload = dict(
        ols_r2=r2,
        intercept=float(coef[-1]),
        sum_layers_over_full=sum_layers / max(full_int, 1e-30),
        layers=rows_sorted,
        top_spots=[
            {
                "id": int(ids[i]),
                "E_u": float(E_u[i]),
                "L4": int(L4[i]),
                "x": float(x[i]),
                "y": float(y[i]),
            }
            for i in order[:15]
        ],
    )
    (args.out / "layer_attribution.json").write_text(json.dumps(payload, indent=2))

    pos = [r for r in rows_sorted if r["corr_R"] > 0.05]
    neg = [r for r in rows_sorted if r["corr_R"] < -0.05]
    pos_s = ", ".join(f"{r['E_u']:.0f}" for r in pos) or "none"
    neg_s = ", ".join(f"{r['E_u']:.0f}" for r in neg) or "none"

    lines = [
        "# RT07575 full-plan energy-layer residual attribution",
        "",
        f"Mask: BODY∩TOPAS≥10% (n={int(sel.sum())}).",
        f"Sum(layer dose)/full GPU on mask = **{sum_layers/max(full_int,1e-30):.4f}**.",
        f"OLS `R ≈ Σ a_i G_i + b`: **R²={r2:.3f}**.",
        "",
        "| E [MeV/u] | spots | hist% | dose% | corr(G,R) | OLS a | hot mass% | bias/Dmax% |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda z: z["E_u"]):
        lines.append(
            f"| {r['E_u']:.1f} | {r['n_spots']} | {100*r['hist_frac']:.1f} | "
            f"{100*r['mass_frac_sel']:.1f} | {r['corr_R']:+.3f} | "
            f"{r.get('ols_coef', 0):+.3g} | {100*r['mass_frac_in_hot']:.1f} | "
            f"{100*r.get('dose_weighted_bias', 0)/dmax:+.3f} |"
        )
    lines += ["", "## Ranked by |corr(G_layer, residual)|", ""]
    for i, r in enumerate(rows_sorted[:8], 1):
        lines.append(
            f"{i}. **{r['E_u']:.1f} MeV/u** — corr={r['corr_R']:+.3f}, "
            f"hist={100*r['hist_frac']:.1f}%, dose={100*r['mass_frac_sel']:.1f}%, "
            f"OLS a={r.get('ols_coef', 0):+.3g}"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        f"- Layers with corr>0.05: {pos_s}.",
        f"- Layers with corr<-0.05: {neg_s}.",
        "- Low OLS R² ⇒ residual is shared multi-layer shape, not one bad bin.",
        "",
        "## Top spots by plan histories",
        "",
    ]
    for i in order[:10]:
        lines.append(
            f"- spot id **{int(ids[i])}**: {E_u[i]:.1f} MeV/u, L4={int(L4[i])}, "
            f"(x,y)=({x[i]:.2f},{y[i]:.2f})"
        )
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("Wrote", args.out / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

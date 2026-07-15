#!/usr/bin/env python3
"""Multi-energy emittance sigma(z) comparison + entrance bias summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_sigma_vs_depth import (  # noqa: E402
    load_sparse_voxels,
    load_topas_raw_dose3d,
    sigma_vs_depth,
)


def main() -> None:
    energies = [150, 200, 250, 350, 400]
    # TOPAS development used 50k; GPU 100k
    topas_histories = 50000
    report: dict[str, object] = {
        "source": {
            "SigmaX_mm": 0.2,
            "SigmaXprime": 0.032,
            "CorrelationX": -0.9411,
            "SigmaY_mm": 0.2,
            "SigmaYprime": 0.032,
            "CorrelationY": 0.9411,
        },
        "histories_gpu": 100000,
        "histories_topas": topas_histories,
        "cases": {},
    }

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.ravel()
    edges = np.arange(0.0, 400.0 + 0.5, 1.0)

    for ax, e in zip(axes, energies):
        gpu_vox = Path(f"out/emittance/e{e}_voxels.csv")
        if not gpu_vox.exists():
            gpu_vox = Path(f"validation/results/emittance_{e}_gpu_voxels.csv")
        topas_dose = Path(
            f"validation/topas/output/emittance_{e}_development_dose_3d.csv"
        )
        if not gpu_vox.exists() or not topas_dose.exists():
            ax.set_title(f"{e} missing")
            ax.axis("off")
            continue

        gx, gy, gz, gw = load_sparse_voxels(gpu_vox)
        tx, ty, tz, tw = load_topas_raw_dose3d(
            topas_dose,
            histories=topas_histories,
            nx=120,
            ny=120,
            nz=400,
            half_xy_mm=30.0,
            half_z_mm=200.0,
        )
        gpu = sigma_vs_depth(gx, gy, gz, gw, edges)
        top = sigma_vs_depth(tx, ty, tz, tw, edges)

        # save per-energy CSVs
        out_dir = Path(f"validation/results/emittance_{e}_sigma")
        out_dir.mkdir(parents=True, exist_ok=True)
        for tag, data in (("gpu", gpu), ("topas", top)):
            with (out_dir / f"{tag}_sigma_vs_depth.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(
                    [
                        "depth_mm",
                        "sigma_x_mm",
                        "sigma_y_mm",
                        "sigma_rms_mm",
                        "integral_MeV_per_primary",
                    ]
                )
                for i in range(len(data["depth_mm"])):
                    writer.writerow(
                        [
                            f"{data['depth_mm'][i]:.6g}",
                            f"{data['sigma_x_mm'][i]:.6g}",
                            f"{data['sigma_y_mm'][i]:.6g}",
                            f"{data['sigma_rms_mm'][i]:.6g}",
                            f"{data['integral_MeV_per_primary'][i]:.6g}",
                        ]
                    )

        mask = (
            np.isfinite(gpu["sigma_rms_mm"])
            & np.isfinite(top["sigma_rms_mm"])
            & (gpu["integral_MeV_per_primary"] > 0)
            & (top["integral_MeV_per_primary"] > 0)
        )
        g_sig = gpu["sigma_rms_mm"][mask]
        t_sig = top["sigma_rms_mm"][mask]
        depth = gpu["depth_mm"][mask]
        rel = 100.0 * (g_sig - t_sig) / np.maximum(t_sig, 1e-12)
        # peak region heuristic: near max of TOPAS integral curve
        peak_idx = int(np.nanargmax(top["integral_MeV_per_primary"]))
        peak_z = float(top["depth_mm"][peak_idx])
        peak_mask = (depth >= peak_z - 15.0) & (depth <= peak_z + 15.0)

        entry = {
            "entrance_sigma_rms_mm": {
                "gpu": float(g_sig[0]) if len(g_sig) else None,
                "topas": float(t_sig[0]) if len(t_sig) else None,
                "source_sigma_mm": 0.2,
            },
            "mean_rel_percent": float(np.mean(rel)) if len(rel) else None,
            "rms_rel_percent": float(np.sqrt(np.mean(rel * rel))) if len(rel) else None,
            "mean_abs_diff_mm": float(np.mean(np.abs(g_sig - t_sig))) if len(g_sig) else None,
            "peak_depth_mm": peak_z,
            "peak_region_mean_rel_percent": float(np.mean(rel[peak_mask]))
            if np.any(peak_mask)
            else None,
        }
        report["cases"][str(e)] = entry

        ax.plot(top["depth_mm"], top["sigma_rms_mm"], lw=2, label="TOPAS")
        ax.plot(gpu["depth_mm"], gpu["sigma_rms_mm"], lw=1.4, label="GPU")
        ax.set_title(
            f"{e} MeV/u  mean rel {entry['mean_rel_percent']:+.1f}%  "
            f"ent {entry['entrance_sigma_rms_mm']['topas']:.2f}/{entry['entrance_sigma_rms_mm']['gpu']:.2f}"
        )
        ax.set_xlabel("Depth (mm)")
        ax.set_ylabel("σ_rms (mm)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    axes[-1].axis("off")
    # entrance bias panel in last axis
    ax = axes[-1]
    es = []
    eg = []
    et = []
    for e in energies:
        c = report["cases"].get(str(e))
        if not c:
            continue
        es.append(e)
        eg.append(c["entrance_sigma_rms_mm"]["gpu"])
        et.append(c["entrance_sigma_rms_mm"]["topas"])
    if es:
        ax.plot(es, et, "o-", label="TOPAS entrance")
        ax.plot(es, eg, "s-", label="GPU entrance")
        ax.axhline(0.2, color="k", ls="--", lw=0.8, label="source σ=0.2 mm")
        ax.set_xlabel("E (MeV/u)")
        ax.set_ylabel("entrance σ_rms (mm)")
        ax.set_title("Entrance dose-weighted σ vs source σ")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    fig.suptitle("Multi-energy emittance σ(z): TOPAS vs GPU")
    fig.tight_layout()
    out_png = Path("validation/results/emittance_multi_sigma_overview.png")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    # entrance bias notes
    report["entrance_bias_notes"] = (
        "Dose-weighted first-bin σ_rms is larger than source geometric σ=0.2 mm because "
        "(1) 0.5 mm lateral voxels and 1 mm depth bin average over path, "
        "(2) angular divergence σ'=0.032 rad immediately spreads dose, "
        "(3) MCS/straggling act in the first bin. GPU systematically slightly narrower "
        "than TOPAS at entrance in prior 200 MeV study."
    )

    out_json = Path("validation/results/emittance_multi_sigma_summary.metrics.json")
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_png}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()

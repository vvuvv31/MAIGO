#!/usr/bin/env python3
"""Summarize multi-energy mode-D neutral GPU vs TOPAS results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    rows = []
    print("E | int% | peak% | NRMSE | midplat% | tail% | g2")
    for e in [100, 150, 200, 250, 300, 350, 400]:
        gp = np.genfromtxt(
            f"validation/results/windows_b580_neutral_mode_d_{e}MeVu_idd_100k.csv",
            delimiter=",",
            names=True,
        )
        tp = np.genfromtxt(
            f"validation/results/topas_{e}MeVu_development.csv",
            delimiter=",",
            names=True,
        )
        gd = gp["depth_mm"]
        gv = gp["energy_deposition_MeV_per_primary"]
        td = tp["depth_mm"]
        tv = tp["energy_deposition_MeV_per_primary"]
        on = np.interp(td, gd, gv)
        gi, ti = float(gv.sum()), float(tv.sum())
        intp = 100.0 * (gi - ti) / ti
        m = json.loads(
            Path(
                f"validation/results/windows_b580_neutral_mode_d_vs_topas_{e}MeVu.metrics.json"
            ).read_text(encoding="utf-8")
        )
        d = m["differences"]
        g = m.get("gamma", {})
        n = tv / tv.max()
        pk = int(np.argmax(n))
        r80 = float("nan")
        for i in range(pk + 1, len(td)):
            if n[i] <= 0.8 < n[i - 1]:
                r80 = float(
                    td[i - 1]
                    + (0.8 - n[i - 1]) * (td[i] - td[i - 1]) / (n[i] - n[i - 1])
                )
                break
        mid = (td >= 0.2 * r80) & (td <= 0.7 * r80)
        midp = float(np.mean(100.0 * (on[mid] - tv[mid]) / tv[mid]))
        print(
            f"{e:3d} | {intp:+6.2f} | {d['peak_value_percent']:+6.2f} | "
            f"{d['NRMSE']:.4f} | {midp:+6.2f} | "
            f"{d['tail_integral_relative_percent']:+6.2f} | "
            f"{g.get('2pct_2mm_pass_rate_percent', float('nan')):5.1f}"
        )
        rows.append(
            {
                "E": e,
                "int_pct": intp,
                "peak_pct": d["peak_value_percent"],
                "NRMSE": d["NRMSE"],
                "midplat": midp,
                "tail": d["tail_integral_relative_percent"],
                "g2": g.get("2pct_2mm_pass_rate_percent"),
                "gpu_int": gi,
                "topas_int": ti,
            }
        )

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.ravel()
    for ax, e in zip(axes, [100, 150, 200, 250, 300, 350, 400]):
        gp = np.genfromtxt(
            f"validation/results/windows_b580_neutral_mode_d_{e}MeVu_idd_100k.csv",
            delimiter=",",
            names=True,
        )
        tp = np.genfromtxt(
            f"validation/results/topas_{e}MeVu_development.csv",
            delimiter=",",
            names=True,
        )
        ax.plot(tp["depth_mm"], tp["energy_deposition_MeV_per_primary"], lw=2, label="TOPAS")
        ax.plot(
            gp["depth_mm"],
            gp["energy_deposition_MeV_per_primary"],
            lw=1.3,
            label="GPU+modeD",
        )
        r = next(x for x in rows if x["E"] == e)
        ax.set_title(f"{e}: int {r['int_pct']:+.2f}% mid {r['midplat']:+.1f}%")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    axes[-1].axis("off")
    fig.suptitle("Mode-D neutral (kerma off) vs TOPAS multi-energy")
    fig.tight_layout()
    out_png = Path("validation/results/windows_b580_neutral_mode_d_multi_energy_overview.png")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    summary = {
        "mode": "first_interaction",
        "kerma": 0.0,
        "neutral_package": "validation/results/topas_200MeVu_neutral_development.bin",
        "neutral_stats": {"interactions": 4798230, "products": 2004174},
        "cases": rows,
        "note": (
            "Mode D recovers charged recoils from first neutral interaction only; "
            "residual continuation not deposited. High-E integral under-recovers "
            "vs interim kerma; peak underdose remains charged-path issue."
        ),
        "compare_to_kerma_0p298": {
            "300_int_pct": {"kerma": -0.27, "mode_d": next(r["int_pct"] for r in rows if r["E"] == 300)},
            "400_int_pct": {"kerma": -1.01, "mode_d": next(r["int_pct"] for r in rows if r["E"] == 400)},
            "200_int_pct": {"kerma": 0.34, "mode_d": next(r["int_pct"] for r in rows if r["E"] == 200)},
        },
    }
    out_json = Path(
        "validation/results/windows_b580_neutral_mode_d_multi_energy_summary.metrics.json"
    )
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose high-energy Bragg-peak underdose (GPU mode-D vs TOPAS)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_idd(path: Path):
    d = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    return d["depth_mm"], d["energy_deposition_MeV_per_primary"]


def load_species(path: Path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=float)


def r80(depth, dose):
    n = dose / np.max(dose)
    pk = int(np.argmax(n))
    for i in range(pk + 1, len(depth)):
        if n[i] <= 0.8 < n[i - 1]:
            return float(
                depth[i - 1]
                + (0.8 - n[i - 1]) * (depth[i] - depth[i - 1]) / (n[i] - n[i - 1])
            )
    return float("nan")


def main() -> None:
    report = {"cases": {}}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for ax_row, e in zip(axes, (300, 400)):
        idd = Path(f"validation/results/windows_b580_neutral_mode_d_{e}MeVu_idd_100k.csv")
        # species from neutral run if present else multi_energy_neutral out
        sp_path = Path(f"out/multi_energy_neutral/e{e}_species.csv")
        topas = Path(f"validation/results/topas_{e}MeVu_development.csv")
        gd, gv = load_idd(idd)
        td, tv = load_idd(topas)
        on = np.interp(td, gd, gv)
        sp = load_species(sp_path)
        sd = sp["depth_mm"]
        primary = sp["primary_c12_MeV_per_primary"]
        frags = sp["total_MeV_per_primary"] - primary
        # peak window ±5 mm around TOPAS peak
        peak_z = float(td[int(np.argmax(tv))])
        win = (td >= peak_z - 5.0) & (td <= peak_z + 5.0)
        mid = (td >= 0.2 * r80(td, tv)) & (td <= 0.7 * r80(td, tv))
        entry = {
            "peak_depth_topas_mm": peak_z,
            "peak_value_pct": 100.0 * (float(np.max(gv)) - float(np.max(tv))) / float(np.max(tv)),
            "peak_window_integ_pct": 100.0
            * (float(np.sum(on[win])) - float(np.sum(tv[win])))
            / float(np.sum(tv[win])),
            "midplat_integ_pct": 100.0
            * (float(np.sum(on[mid])) - float(np.sum(tv[mid])))
            / float(np.sum(tv[mid])),
            "gpu_primary_integral": float(np.sum(primary)),
            "gpu_fragment_integral": float(np.sum(frags)),
            "gpu_primary_at_peak_window": float(
                np.sum(np.interp(td[win], sd, primary))
            ),
            "gpu_fragment_at_peak_window": float(
                np.sum(np.interp(td[win], sd, frags))
            ),
            "topas_peak_window_integral": float(np.sum(tv[win])),
            "species_peak_window_fraction_primary": None,
        }
        g_prim_w = entry["gpu_primary_at_peak_window"]
        g_frag_w = entry["gpu_fragment_at_peak_window"]
        entry["species_peak_window_fraction_primary"] = g_prim_w / (g_prim_w + g_frag_w)
        # relative residual near peak
        peak_band = (td >= peak_z - 15.0) & (td <= peak_z + 15.0)
        entry["peak_band_mean_rel_pct"] = float(
            np.mean(100.0 * (on[peak_band] - tv[peak_band]) / tv[peak_band])
        )
        report["cases"][str(e)] = entry

        ax = ax_row[0]
        ax.plot(td, tv, lw=2, label="TOPAS total")
        ax.plot(gd, gv, lw=1.4, label="GPU total")
        ax.plot(sd, primary, lw=1.0, ls="--", label="GPU primary")
        ax.plot(sd, frags, lw=1.0, ls=":", label="GPU fragments")
        ax.axvspan(peak_z - 5, peak_z + 5, color="0.9")
        ax.set_xlim(peak_z - 25, peak_z + 20)
        ax.set_title(
            f"{e} MeV/u peak zoom  peak%={entry['peak_value_pct']:+.1f}  "
            f"win integ%={entry['peak_window_integ_pct']:+.1f}"
        )
        ax.set_xlabel("Depth (mm)")
        ax.set_ylabel("MeV/primary/bin")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

        ax = ax_row[1]
        rel = 100.0 * (on - tv) / np.maximum(tv, 1e-12)
        ax.plot(td, rel, lw=1.3)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.axvline(peak_z, color="C1", ls="--", lw=0.8)
        ax.set_xlim(peak_z - 40, peak_z + 25)
        ax.set_ylim(-15, 15)
        ax.set_title(f"{e} residual % near peak")
        ax.set_xlabel("Depth (mm)")
        ax.grid(alpha=0.25)

    fig.suptitle("Bragg peak underdose diagnosis (mode-D neutral, no kerma)")
    fig.tight_layout()
    out_dir = Path("validation/results/peak_underdose_diagnosis")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "peak_underdose_300_400.png", dpi=160)
    plt.close(fig)
    (out_dir / "peak_underdose_300_400.metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()

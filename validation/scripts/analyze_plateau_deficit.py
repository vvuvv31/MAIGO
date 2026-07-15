#!/usr/bin/env python3
"""Diagnose multi-energy plateau integral deficit (GPU vs TOPAS)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_idd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    return (
        np.atleast_1d(np.asarray(data["depth_mm"], dtype=float)),
        np.atleast_1d(np.asarray(data["energy_deposition_MeV_per_primary"], dtype=float)),
    )


def load_species(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    return {name: np.atleast_1d(np.asarray(data[name], dtype=float)) for name in data.dtype.names}


def r80(depth: np.ndarray, dose: np.ndarray) -> float:
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
    cases = {
        200: (
            Path("out/multi_energy/e200_species.csv"),
            Path("validation/results/topas_200MeVu_development.csv"),
            Path("validation/results/topas_200MeVu_ancestor_dose_3d_development.idd.csv"),
        ),
        300: (
            Path("out/multi_energy/e300_species.csv"),
            Path("validation/results/topas_300MeVu_development.csv"),
            None,
        ),
        400: (
            Path("out/multi_energy/e400_species.csv"),
            Path("validation/results/topas_400MeVu_development.csv"),
            None,
        ),
    }

    report: dict[str, object] = {"cases": {}}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, (energy, (species_path, topas_path, ancestor_path)) in zip(axes, cases.items()):
        sp = load_species(species_path)
        td, tv = load_idd(topas_path)
        depth = sp["depth_mm"]
        gpu = sp["total_MeV_per_primary"]
        primary = sp["primary_c12_MeV_per_primary"]
        fragments = gpu - primary
        on = np.interp(depth, td, tv)
        r = r80(td, tv)

        # relative curves
        rel = 100.0 * (gpu - on) / np.maximum(on, 1.0e-12)
        m_plat = (depth >= 0.05 * r) & (depth <= 0.80 * r)
        m_mid = (depth >= 0.20 * r) & (depth <= 0.70 * r)

        entry = {
            "R80_mm": r,
            "integral_diff_percent": 100.0 * (float(np.sum(gpu)) - float(np.sum(tv))) / float(np.sum(tv)),
            "plateau_0p05_0p80_integ_diff_percent": 100.0
            * (float(np.sum(gpu[m_plat])) - float(np.sum(on[m_plat])))
            / float(np.sum(on[m_plat])),
            "midplateau_0p20_0p70_mean_rel_percent": float(np.mean(rel[m_mid])),
            "gpu_primary_integral": float(np.sum(primary)),
            "gpu_fragment_integral": float(np.sum(fragments)),
            "gpu_total_integral": float(np.sum(gpu)),
            "topas_total_integral": float(np.sum(tv)),
            "plateau_gpu_primary_fraction": float(np.sum(primary[m_plat]) / np.sum(gpu[m_plat])),
            "plateau_gpu_fragment_fraction": float(np.sum(fragments[m_plat]) / np.sum(gpu[m_plat])),
        }

        if ancestor_path is not None and ancestor_path.exists():
            anc = np.genfromtxt(ancestor_path, delimiter=",", names=True, dtype=float)
            # expected columns include category IDDs
            names = list(anc.dtype.names or ())
            entry["ancestor_columns"] = names
            ad = np.atleast_1d(np.asarray(anc["depth_mm"], dtype=float))
            # common charged categories
            charged_cols = [
                c
                for c in names
                if c
                in {
                    "primary_c12",
                    "secondary_carbon",
                    "boron",
                    "beryllium",
                    "lithium",
                    "helium",
                    "proton",
                    "other_charged",
                    "primary_c12_MeV_per_primary",
                }
            ]
            # try flexible names
            def col(base: str) -> np.ndarray | None:
                for candidate in (base, f"{base}_MeV_per_primary", f"{base}_energy_MeV_per_primary"):
                    if candidate in names:
                        return np.atleast_1d(np.asarray(anc[candidate], dtype=float))
                return None

            t_primary = col("primary_c12")
            t_he = col("helium")
            t_p = col("proton")
            t_n = col("neutron")
            t_g = col("gamma")
            if t_primary is not None:
                tp = np.interp(depth, ad, t_primary)
                entry["topas_primary_integral"] = float(np.sum(tp))
                entry["primary_vs_topas_diff_percent"] = 100.0 * (
                    float(np.sum(primary)) - float(np.sum(tp))
                ) / float(np.sum(tp))
                entry["plateau_primary_vs_topas_percent"] = 100.0 * (
                    float(np.sum(primary[m_plat])) - float(np.sum(tp[m_plat]))
                ) / float(np.sum(tp[m_plat]))
            if t_he is not None:
                th = np.interp(depth, ad, t_he)
                entry["helium_vs_topas_diff_percent"] = 100.0 * (
                    float(np.sum(sp["helium_MeV_per_primary"])) - float(np.sum(th))
                ) / float(np.sum(th))
            if t_p is not None:
                tpr = np.interp(depth, ad, t_p)
                entry["proton_vs_topas_diff_percent"] = 100.0 * (
                    float(np.sum(sp["proton_MeV_per_primary"])) - float(np.sum(tpr))
                ) / float(np.sum(tpr))
            if t_n is not None:
                entry["topas_neutron_integral"] = float(np.sum(t_n))
            if t_g is not None:
                entry["topas_gamma_integral"] = float(np.sum(t_g))

        report["cases"][str(energy)] = entry

        ax.plot(depth, on, label="TOPAS total", lw=2)
        ax.plot(depth, gpu, label="GPU total", lw=1.4)
        ax.plot(depth, primary, label="GPU primary", lw=1.0, ls="--")
        ax.plot(depth, fragments, label="GPU fragments", lw=1.0, ls=":")
        ax.axvspan(0.2 * r, 0.7 * r, color="0.85", label="mid-plateau")
        ax.set_title(
            f"{energy} MeV/u\nmid-plat mean {entry['midplateau_0p20_0p70_mean_rel_percent']:+.1f}%"
        )
        ax.set_xlabel("Depth (mm)")
        ax.set_ylabel("MeV/primary/bin")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    fig.suptitle("Plateau underdose diagnosis: GPU vs TOPAS")
    fig.tight_layout()
    out_png = Path("validation/results/plateau_deficit_diagnosis.png")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    # relative residual plot for 400
    sp = load_species(cases[400][0])
    td, tv = load_idd(cases[400][1])
    depth = sp["depth_mm"]
    on = np.interp(depth, td, tv)
    rel = 100.0 * (sp["total_MeV_per_primary"] - on) / np.maximum(on, 1e-12)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(depth, rel, lw=1.5)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set(xlabel="Depth (mm)", ylabel="(GPU-TOPAS)/TOPAS %", title="400 MeV/u relative residual")
    ax.set_ylim(-15, 15)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    res_png = Path("validation/results/plateau_deficit_400_relative.png")
    fig.savefig(res_png, dpi=160)
    plt.close(fig)

    out_json = Path("validation/results/plateau_deficit_diagnosis.metrics.json")
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_png}")
    print(f"Wrote {res_png}")


if __name__ == "__main__":
    main()

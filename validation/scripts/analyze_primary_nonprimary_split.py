#!/usr/bin/env python3
"""Split multi-energy plateau deficit into primary vs non-primary contributions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_topas_raw(path: Path, histories: int = 100000, bin_w: float = 0.5):
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            parts = [float(x.strip()) for x in stripped.split(",")]
        except ValueError:
            continue
        if len(parts) >= 4:
            rows.append(parts)
    data = np.asarray(rows, dtype=float)
    z = data[:, 2].astype(int)
    val = data[:, 3] / histories
    order = np.argsort(z)
    depth = (z[order] + 0.5) * bin_w
    return depth, val[order]


def load_csv(path: Path, value_col: str = "energy_deposition_MeV_per_primary"):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    return (
        np.atleast_1d(np.asarray(data["depth_mm"], dtype=float)),
        np.atleast_1d(np.asarray(data[value_col], dtype=float)),
    )


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
        100: (
            Path("validation/topas/output/e100_development_primary_c12_energy_deposit.csv"),
            Path("out/multi_energy/e100_species.csv"),
            Path("validation/results/topas_100MeVu_development.csv"),
        ),
        300: (
            Path("validation/topas/output/e300_development_primary_c12_energy_deposit.csv"),
            Path("out/multi_energy/e300_species.csv"),
            Path("validation/results/topas_300MeVu_development.csv"),
        ),
        400: (
            Path("validation/topas/output/e400_development_primary_c12_energy_deposit.csv"),
            Path("out/multi_energy/e400_species.csv"),
            Path("validation/results/topas_400MeVu_development.csv"),
        ),
    }
    report: dict[str, object] = {}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    for ax, (energy, (praw, gsp, ttot)) in zip(axes, cases.items()):
        g = np.genfromtxt(gsp, delimiter=",", names=True, dtype=float)
        gd = np.atleast_1d(np.asarray(g["depth_mm"], dtype=float))
        gtot = np.atleast_1d(np.asarray(g["total_MeV_per_primary"], dtype=float))
        gprim = np.atleast_1d(np.asarray(g["primary_c12_MeV_per_primary"], dtype=float))
        gfrag = gtot - gprim
        td, tv = load_csv(ttot)
        tpd, tpv = load_topas_raw(praw)
        on_tot = np.interp(gd, td, tv)
        on_p = np.interp(gd, tpd, tpv)
        # TOPAS C12-named filter is restricted dE (no delta-rays); not equal to GPU primary.
        r = r80(td, tv)
        m = (gd >= 0.20 * r) & (gd <= 0.70 * r)
        entry = {
            "R80_mm": r,
            "integral_total_diff_percent": 100.0 * (float(gtot.sum()) - float(tv.sum())) / float(tv.sum()),
            "midplateau_total_mean_rel_percent": float(np.mean(100.0 * (gtot[m] - on_tot[m]) / on_tot[m])),
            "midplateau_abs_deficit_MeV_per_primary": float((gtot[m] - on_tot[m]).sum()),
            "gpu_primary_integral": float(gprim.sum()),
            "gpu_fragment_integral": float(gfrag.sum()),
            "topas_total_integral": float(tv.sum()),
            "topas_c12filter_integral": float(tpv.sum()),
            "note": (
                "TOPAS PrimaryC12Energy is OnlyIncludeParticlesNamed GenericIon(6,12) "
                "restricted deposit; GPU primary uses unrestricted CSDA. Compare totals."
            ),
            "entrance_total_rel_percent": float(
                100.0 * (gtot[0] - on_tot[0]) / on_tot[0]
            ),
            "untransported_neutral_hint_MeV_per_primary": {
                100: None,
                300: 154.1,
                400: 286.5,
            }.get(energy),
        }
        report[str(energy)] = entry

        ax.plot(gd, on_tot, label="TOPAS total", lw=2)
        ax.plot(gd, gtot, label="GPU total", lw=1.4)
        ax.plot(gd, gfrag, label="GPU fragments", ls="--", lw=1.0)
        ax.axvspan(0.2 * r, 0.7 * r, color="0.9")
        ax.set_title(
            f"{energy} MeV/u midplat {entry['midplateau_total_mean_rel_percent']:+.1f}%"
        )
        ax.set_xlabel("Depth (mm)")
        ax.set_ylabel("MeV/primary/bin")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    fig.suptitle("High-energy plateau deficit (total IDD)")
    fig.tight_layout()
    out_png = Path("validation/results/plateau_total_deficit.png")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    out_json = Path("validation/results/plateau_total_deficit.metrics.json")
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()

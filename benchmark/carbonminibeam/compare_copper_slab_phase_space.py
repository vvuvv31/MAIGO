#!/usr/bin/env python3
"""Compare pencil-beam C12 slab phase spaces on the physical Copper exit.

TOPAS scores 0.01 mm after the exit; the GPU slab config scores at the exit.
Back-project TOPAS through that vacuum gap before computing spatial moments.
No fit, dose normalization, or angular scale is applied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_topas(path: Path, thickness: float) -> dict[str, np.ndarray]:
    rows = np.atleast_2d(np.loadtxt(path))
    rows = rows[(rows[:, 7] == 1_000_060_120) & (rows[:, 13] == 0)]
    if not len(rows) or np.any(rows[:, 8] != 1):
        raise ValueError("Expected forward parent-0 C12 tracks along TOPAS -Z")
    if not np.allclose(rows[:, 6], 1):
        raise ValueError("This pencil-beam comparison requires unit weights")
    dz = np.sqrt(np.maximum(0, 1 - rows[:, 3]**2 - rows[:, 4]**2))
    if np.any(dz <= 0):
        raise ValueError("Cannot project a grazing track to the exit")
    gap = -10 * rows[:, 2] - thickness / 2
    if np.any(gap < -1e-5) or np.any(gap > 0.02):
        raise ValueError("Phase-space plane disagrees with the supplied slab thickness")
    return dict(
        energy=rows[:, 5],
        x=10 * rows[:, 0] - gap * rows[:, 3] / dz,
        tx=np.arctan2(rows[:, 3], dz),
        ty=np.arctan2(rows[:, 4], dz),
    )


def read_gpu(path: Path) -> dict[str, np.ndarray]:
    rows = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    if not len(rows) or np.any(rows["direction_z"] <= 0):
        raise ValueError("Expected forward GPU primary tracks")
    return dict(
        energy=rows["kinetic_energy_MeV"], x=rows["x_mm"],
        tx=np.arctan2(rows["direction_x"], rows["direction_z"]),
        ty=np.arctan2(rows["direction_y"], rows["direction_z"]),
    )


def summarize(tracks: dict[str, np.ndarray], histories: int) -> dict:
    if any(not np.all(np.isfinite(v)) for v in tracks.values()):
        raise ValueError("Nonfinite phase space")
    x = tracks["x"] - np.mean(tracks["x"])
    tx = tracks["tx"] - np.mean(tracks["tx"])
    x2, xt, t2 = np.mean(x*x), np.mean(x*tx), np.mean(tx*tx)
    radial = np.hypot(tracks["tx"], tracks["ty"]) * 1000
    return {
        "count": len(x), "survival": len(x) / histories,
        "mean_energy_MeV": float(np.mean(tracks["energy"])),
        "std_energy_MeV": float(np.std(tracks["energy"])),
        "x_variance_mm2": float(x2),
        "x_theta_covariance_mm_rad": float(xt),
        "theta_variance_rad2": float(t2),
        "x_theta_correlation": float(xt / np.sqrt(x2*t2)),
        **{f"radial_q{q*100:g}_mrad": float(np.quantile(radial, q))
           for q in (0.68, 0.95, 0.99, 0.999)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topas-full", required=True, type=Path)
    parser.add_argument("--topas-no-elastic", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=Path)
    parser.add_argument("--thickness-mm", required=True, type=float)
    parser.add_argument("--energy-mevu", default=250.0, type=float)
    parser.add_argument("--histories", default=256000, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.thickness_mm <= 0 or args.histories <= 0 or args.energy_mevu <= 0:
        parser.error("thickness, incident energy and histories must be positive")
    tracks = {
        "topas_full": read_topas(args.topas_full, args.thickness_mm),
        "topas_no_ion_elastic": read_topas(args.topas_no_elastic, args.thickness_mm),
        "gpu": read_gpu(args.gpu),
    }
    summary = {k: summarize(v, args.histories) for k, v in tracks.items()}
    ratios = {}
    for num, den in (("gpu", "topas_full"), ("gpu", "topas_no_ion_elastic"),
                     ("topas_no_ion_elastic", "topas_full")):
        ratios[f"{num}_over_{den}"] = {
            k: summary[num][k] / summary[den][k] for k in summary[den]
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "inputs": {k: str(v) for k, v in vars(args).items()},
        "scoring_plane": "Copper exit; TOPAS vacuum gap back-projected",
        "summary": summary, "ratios": ratios,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for label, rows in tracks.items():
        r = np.sort(np.hypot(rows["tx"], rows["ty"]) * 1000)
        indices = np.unique(np.rint(np.linspace(0, len(r)-1, 5000)).astype(int))
        axes[0].semilogy(r[indices], 1 - indices/len(r), label=label)
        axes[1].hist(rows["x"], bins=200, density=True, histtype="step", label=label)
        axes[2].hist(rows["energy"], bins=200, density=True, histtype="step", label=label)
    axes[0].set(xlabel="Radial projected angle (mrad)", ylabel="Survival function",
                ylim=(1e-4, 1), xlim=(0, max(np.quantile(
                    np.hypot(v["tx"], v["ty"])*1000, .9999) for v in tracks.values())))
    axes[1].set(xlabel="Exit x (mm)", ylabel="Probability density (1/mm)")
    axes[2].set(xlabel="Exit kinetic energy (MeV)", ylabel="Probability density (1/MeV)",
                yscale="log")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{args.energy_mevu:g} MeV/u C12, {args.thickness_mm:g} mm Copper, {args.histories:,} histories")
    fig.savefig(args.output_dir / "phase_space.png", dpi=170)
    plt.close(fig)
    print(args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Cu-slab gate: MAIGO exit rows vs executed G4 Cu-slab oracle (10 mm Cu,
250 MeV/u, 0.25 mm maxstep). Moments with batch SEs, JSON verdict.

Usage: cu_slab_gate.py --maigo CSV --g4 PREFIX --out JSON [--batches 8]
G4 prefix files: <prefix>_theta.csv (space angle mrad col0? see oracle),
<step>_steplen etc. The oracle StepAct writes theta CSVs with the space
angle in mrad as col 0, x/y mm, E.
"""
import argparse
import hashlib
import json
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def batch_stats(vals, batches):
    idx = np.array_split(np.arange(len(vals)), batches)
    ms = np.array([vals[i].mean() for i in idx if len(i)])
    m = vals.mean()
    se = ms.std(ddof=1) / np.sqrt(len(ms)) if len(ms) > 1 else np.nan
    return m, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maigo", required=True)
    ap.add_argument("--g4", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, default=8)
    a = ap.parse_args()
    m = np.loadtxt(a.maigo, delimiter=",")
    gth = np.loadtxt(a.g4 + "_theta.csv", delimiter=",")
    # G4 summary aggregates (meanExitE, rmsE, varThx, varThy, varX, cov)
    summ = {}
    with open(a.g4 + "_summary.csv") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            vals = [float(v) for v in line.split(",")]
            if len(vals) == 3:
                summ["events"], summ["exitN"], summ["exitFrac"] = vals
            elif len(vals) == 6:
                (summ["meanE"], summ["rmsE"], summ["varThx"],
                 summ["varThy"], summ["varX"], summ["cov"]) = vals
    print("g4 summary:", summ)
    mth, mx, my, mE = m[:, 0], m[:, 1], m[:, 2], m[:, 3]
    gE_mean, gE_rms = summ["meanE"], summ["rmsE"]
    gx_rms = np.sqrt(summ["varX"])
    res = {"inputs": {"maigo": a.maigo, "maigo_sha": sha(a.maigo),
                      "g4": a.g4 + "_theta.csv",
                      "g4_sha": sha(a.g4 + "_theta.csv")}}
    am, ase = batch_stats(mth ** 2, a.batches)
    xm, xse = batch_stats(mx, a.batches)
    em, ese = batch_stats(mE, a.batches)
    res["maigo"] = {"ang_mean": am, "ang_se": ase, "x_mean": xm,
                    "x_se": xse, "E_mean": em, "E_se": ese,
                    "q999": float(np.quantile(mth, 0.999)), "n": len(mth)}
    gam, gase = batch_stats(gth ** 2, a.batches)
    res["g4"] = {"ang_mean": gam, "ang_se": gase, "x_rms": gx_rms,
                 "E_mean": gE_mean, "E_rms": gE_rms,
                 "q999": float(np.quantile(gth, 0.999)), "n": len(gth),
                 "exitFrac": summ["exitN"] / summ["events"]}
    M, G = res["maigo"], res["g4"]
    # MAIGO space-angle E[th^2] vs G4 projected varThx+varThy (equal for the
    # mrad-scale forward peak to the comparison precision).
    res["ratio_ang"] = M["ang_mean"] / (summ["varThx"] + summ["varThy"])
    res["ratio_q999"] = M["q999"] / G["q999"]
    res["rel_E"] = M["E_mean"] / G["E_mean"] - 1
    res["ratio_xrms"] = float(np.sqrt((mx ** 2).mean()) / gx_rms)
    print(f"maigo n={M['n']} ang={M['ang_mean']:.4f}±{M['ang_se']:.4f} "
          f"q999={M['q999']:.2f} E={M['E_mean']:.1f}±{M['E_se']:.1f}")
    print(f"g4    n={G['n']} ang={G['ang_mean']:.4f}±{G['ang_se']:.4f} "
          f"q999={G['q999']:.2f} E={G['E_mean']:.1f} rms={G['E_rms']:.1f}")
    print(f"ratios: ang={res['ratio_ang']:.4f} q999={res['ratio_q999']:.4f} "
          f"dE={res['rel_E']:+.4f}")
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

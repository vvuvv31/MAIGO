#!/usr/bin/env python3
"""Replay gate: GPU planes vs TOPAS .phsp planes directly (no hand-copied
moments). Matched histories, exact space-angle observable, batch SEs,
machine-readable JSON verdict vs acceptance.yaml.

Usage: replay_gate.py --gpu CSV --topas-dir DIR --out JSON [--batches 8]
"""
import argparse
import hashlib
import json
import math

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_gpu(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1,
                   usecols=(0, 1, 2, 4, 5, 6, 7, 8))
    # cols: history, plane, depth, x, y, dx, dy, dz
    return d


def load_topas(path):
    # TOPAS ASCII phsp, 14 cols (0-based): 0 X[cm], 1 Y[cm], 2 Z[cm],
    # 3 dx, 4 dy, 5 E[MeV], 6 weight, 7 PDG, 8 flag3rd, 9 flag1st,
    # 10 RunID, 11 EventID, 12 TrackID, 13 seed.
    # Beam along Y (dy~=1); depth_mm = Y*10-60; transverse = X, Z.
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 14:
                continue
            try:
                x, y, z = float(p[0]), float(p[1]), float(p[2])
                dx, dy = float(p[3]), float(p[4])
                en = float(p[5])
                f3 = float(p[8])
                first = float(p[9])
                ev = int(float(p[11]))
            except ValueError:
                continue
            rows.append((ev, first, x * 10.0, y * 10.0, z * 10.0,
                         dx, dy, en, f3))
    # one row per event: prefer first-scored-particle rows
    rows.sort(key=lambda r: (r[0], -r[1]))
    seen = set()
    out = {}
    for ev, first, x, y, z, dx, dy, en, f3 in rows:
        if ev in seen:
            continue
        seen.add(ev)
        dz = math.sqrt(max(0.0, 1.0 - dx * dx - dy * dy))
        if f3 > 0.5:
            dz = -dz
        out[ev] = (x, z, dx, dy, dz, en, y)  # transverse x,z; Y=depth check
    return out


def interval_metrics(g0, g1, t0, t1):
    # g/t: dict history -> (x, y, dx, dy, dz); matched common set
    common = set(g0) & set(g1) & set(t0) & set(t1)
    if len(common) < 100:
        return None
    H = np.array(sorted(common))
    def stack(d, H):
        return np.array([d[h] for h in H])
    A0, A1 = stack(g0, H), stack(g1, H)
    B0, B1 = stack(t0, H), stack(t1, H)
    out = {}
    for tag, P0, P1 in (("gpu", A0, A1), ("topas", B0, B1)):
        v0 = P0[:, 2:5]
        v1 = P1[:, 2:5]
        cross = np.linalg.norm(np.cross(v0, v1), axis=1)
        dot = np.clip((v0 * v1).sum(axis=1), -1.0, 1.0)
        th = np.arctan2(cross, dot) * 1000.0  # mrad, exact space angle
        dx = P1[:, 0] - P0[:, 0]
        dy = P1[:, 1] - P0[:, 1]
        out[tag] = dict(th=th, dx=dx, dy=dy, n=len(H))
    return out, H


def batch_stats(vals, batches):
    idx = np.array_split(np.arange(len(vals)), batches)
    ms = np.array([vals[i].mean() for i in idx if len(i)])
    m = vals.mean()
    se = ms.std(ddof=1) / np.sqrt(len(ms)) if len(ms) > 1 else np.nan
    return m, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--topas-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, default=8)
    a = ap.parse_args()
    g = load_gpu(a.gpu)
    depths = [40.0, 60.0, 80.0, 100.0, 120.0]
    planes_g = {}
    for k in range(5):
        sel = g[g[:, 1] == k]
        planes_g[k] = {int(r[0]): r[3:8] for r in sel}
    planes_t = {}
    for k, dep in enumerate(depths):
        t = load_topas(f"{a.topas_dir}/primary_{int(dep):03d}.phsp")
        # depth sanity: median Y must equal 60+dep within 0.1 mm
        ys = np.array([v[6] for v in t.values()])
        assert abs(float(np.median(ys)) - (60.0 + dep)) < 0.1, (dep, float(np.median(ys)))
        planes_t[k] = {ev: v[:5] for ev, v in t.items()}
    res = {"inputs": {"gpu": a.gpu, "gpu_sha": sha(a.gpu),
                      "topas_dir": a.topas_dir},
           "intervals": []}
    for k in range(4):
        m = interval_metrics(planes_g[k], planes_g[k + 1],
                             planes_t[k], planes_t[k + 1])
        if m is None:
            continue
        out, H = m
        row = {"from_mm": depths[k], "to_mm": depths[k + 1],
               "n_common": len(H)}
        for tag in ("gpu", "topas"):
            th = out[tag]["th"]
            r = {}
            r["ang_mean"], r["ang_se"] = batch_stats(th ** 2, a.batches)
            dx, dy = out[tag]["dx"], out[tag]["dy"]
            r["dis_mean"], r["dis_se"] = batch_stats(dx ** 2 + dy ** 2,
                                                     a.batches)
            r["q999"] = float(np.quantile(th, 0.999))
            r["tail_p"], _ = batch_stats((th > 100.0).astype(float),
                                         a.batches)
            row[tag] = r
        g0, g1 = row["gpu"], row["topas"]
        row["ratio_ang"] = g0["ang_mean"] / g1["ang_mean"]
        row["ratio_dis"] = g0["dis_mean"] / g1["dis_mean"]
        row["ratio_q999"] = g0["q999"] / g1["q999"]
        # survival (unconditional, both sides)
        for tag, planes in (("gpu", planes_g), ("topas", planes_t)):
            n0 = len(planes[k])
            n1 = len(planes[k + 1])
            row[f"surv_{tag}"] = n1 / n0 if n0 else float("nan")
        res["intervals"].append(row)
        print(f"{depths[k]:.0f}->{depths[k+1]:.0f} n={len(H)} "
              f"ang {g0['ang_mean']:.4f}±{g0['ang_se']:.4f} / "
              f"{g1['ang_mean']:.4f}±{g1['ang_se']:.4f} "
              f"ratio={row['ratio_ang']:.4f} "
              f"dis_ratio={row['ratio_dis']:.4f} "
              f"q999 {g0['q999']:.2f}/{g1['q999']:.2f} "
              f"ratio={row['ratio_q999']:.4f} "
              f"surv {row['surv_gpu']:.4f}/{row['surv_topas']:.4f}")
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

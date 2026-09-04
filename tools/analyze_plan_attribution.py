#!/usr/bin/env python3
"""Reproduce Step-30 plan-attribution key numbers from frozen artifacts.

Reads GPU/TOPAS dose artifacts + phase-space summaries and emits a JSON
with every number cited in the diagnosis. No simulation; pure analysis.
Exit nonzero on any mismatch beyond the stated gates.
"""
import csv
import glob
import json
import struct
import sys

import numpy as np

REPO = "/mnt/sdb/wuwei/MAIGO"
SRCVAL = "/mnt/sda/wuwei/srcval"
GROUPS = "/mnt/sda/wuwei/groups_topas"
AIRTEST = "/mnt/sda/wuwei/airtest"

failures = []


def check(name, value, lo, hi):
    ok = lo <= value <= hi
    print(("PASS " if ok else "FAIL ") + f"{name} = {value:.6f}  gate=[{lo},{hi}]")
    if not ok:
        failures.append(name)


def load_cctg():
    with open(f"{REPO}/benchmark/topas10x/RT06423/patient_ct_tps_90_xneg.bin", "rb") as f:
        d = f.read()
    nx, ny, nz = struct.unpack("<III", d[8:20])
    n = nx * ny * nz
    dens = np.frombuffer(d[44:44 + 4 * n], dtype=np.float32).reshape((nz, ny, nx))
    mat = np.frombuffer(d[44 + 4 * n:44 + 5 * n], dtype=np.uint8).reshape((nz, ny, nx))
    Mc = np.flip(np.transpose(mat, (1, 2, 0)), axis=2)
    Dc = np.flip(np.transpose(dens, (1, 2, 0)), axis=2)
    return Mc, Dc


def load_gpu_group(g):
    v = np.fromfile(f"{REPO}/out/rt06423_group_{g}/dose_group_{g}.raw",
                    dtype=np.float32).reshape((440, 34, 440))
    return np.flip(np.transpose(v, (1, 2, 0)), axis=2)


def load_topas_group(g):
    return np.fromfile(f"{GROUPS}/dose_{g}.bin", dtype=np.float64).reshape((34, 440, 440))


def main():
    out = {}
    Mc, Dc = load_cctg()
    V_cm3 = 0.05 * 0.2 * 0.05
    mass_kg = Dc * V_cm3 / 1000.0
    air = Mc == 0

    # 1. GPU superposition linearity
    ref = np.fromfile(
        f"{REPO}/out/rt06423_schneider_v2_1_match/dose_match_01_s1.raw",
        dtype=np.float32)
    gsum = np.zeros_like(ref)
    for g in ["g1", "g2", "g3", "g4", "g5"]:
        v = np.fromfile(f"{REPO}/out/rt06423_group_{g}/dose_group_{g}.raw",
                        dtype=np.float32)
        gsum += v
    out["superposition_ratio"] = float(gsum.sum() / ref.sum())
    out["superposition_max_abs_diff"] = float(np.abs(gsum - ref).max())
    check("superposition_ratio", out["superposition_ratio"], 0.999999, 1.000001)

    # 2. Per-group totals + air fractions (MeV via shared grid mass)
    out["groups"] = {}
    for g in ["g1", "g2", "g3", "g4", "g5"]:
        G = load_gpu_group(g)
        T = load_topas_group(g)
        MeV_T = float((T * mass_kg).sum() * 6.242e12)
        MeV_G = float((G * mass_kg).sum() * 6.242e12)
        air_T = float((T[air] * mass_kg[air]).sum() * 6.242e12)
        air_G = float((G[air] * mass_kg[air]).sum() * 6.242e12)
        out["groups"][g] = {
            "MeV_T": MeV_T, "MeV_G": MeV_G, "total_T_over_G": MeV_T / MeV_G,
            "air_T_over_G": air_T / air_G,
        }
        print(f"{g}: total T/G={MeV_T / MeV_G:.4f} air T/G={air_T / air_G:.4f}")
    for g in ["g1", "g2", "g3", "g4", "g5"]:
        check(f"{g}_total", out["groups"][g]["total_T_over_G"], 0.97, 1.03)

    # 3. Uniform air slab EM + nuclear
    out["air_slab"] = {}
    for mode, nh, gpath in [("em", 200000, f"{AIRTEST}/em/dose.raw"),
                            ("full", 200000, f"{AIRTEST}/full/dose.raw")]:
        d = np.genfromtxt(f"{AIRTEST}/topas_dose3d_air_{mode}.csv", delimiter=",")
        T = np.zeros((100, 100, 440))
        T[d[:, 0].astype(int), d[:, 1].astype(int), d[:, 2].astype(int)] = d[:, 3]
        g = np.fromfile(gpath, dtype=np.float32).reshape((440, 100, 100))
        g = g.transpose((2, 1, 0))
        ratio = (T.sum() / nh) / (g.sum() / 500000)
        out["air_slab"][mode] = {"T_over_G": float(ratio)}
        print(f"air-slab {mode}: T/G={ratio:.4f}")
    check("air_slab_em", out["air_slab"]["em"]["T_over_G"], 0.995, 1.005)
    check("air_slab_full", out["air_slab"]["full"]["T_over_G"], 0.75, 0.90)

    # 4. Sandwich sections
    def sandwich_load(f, nx, ny, nz):
        d = np.genfromtxt(f, delimiter=",")
        T = np.zeros((nx, ny, nz))
        T[d[:, 0].astype(int), d[:, 1].astype(int), d[:, 2].astype(int)] = d[:, 3]
        return T

    g = np.fromfile(f"{AIRTEST}/sandwich/dose.raw", dtype=np.float32)
    g = g.reshape((440, 100, 100)).transpose((2, 1, 0))
    Vcm3 = 0.2 * 0.2 * 0.05
    out["sandwich"] = {}
    for tag, csv, nz, rho, zoff in [("tisA", "topas_nested_tisa", 120, 1.0788, 0),
                                    ("air", "topas_nested_air", 100, 0.0393235, 120),
                                    ("tisB", "topas_nested_tisb", 220, 1.0788, 220)]:
        T = sandwich_load(f"{AIRTEST}/{csv}.csv", 100, 100, nz)
        m = rho * Vcm3 / 1000.0
        MeV_T = float(T.sum() / 200000 * m * 6.242e12)
        MeV_G = float(g[:, :, zoff:zoff + nz].sum() / 500000 * m * 6.242e12)
        out["sandwich"][tag] = {"MeV_T": MeV_T, "MeV_G": MeV_G,
                                "T_over_G": MeV_T / MeV_G}
        print(f"sandwich {tag}: T/G={MeV_T / MeV_G:.4f}")
    check("sandwich_tisA", out["sandwich"]["tisA"]["T_over_G"], 0.98, 1.03)
    check("sandwich_air", out["sandwich"]["air"]["T_over_G"], 0.15, 0.30)
    check("sandwich_tisB", out["sandwich"]["tisB"]["T_over_G"], 1.08, 1.20)

    # 5. Source truth acceptance (counts + moments already in summaries)
    out["source"] = {}
    tags = ["low_center", "low_maxx", "low_maxy", "mid_center", "mid_maxx",
            "mid_maxy", "high_center", "high_maxx", "high_maxy"]
    for tag in tags:
        s = json.load(open(f"{SRCVAL}/gpu_{tag}_summary.json"))
        box = s[0]
        n_hist = box["histories"]
        planes = [r for r in s if "plane_y" in r]
        out["source"][tag] = {"histories": n_hist,
                              "planes": {r["plane_y"]: r for r in planes}}
    print(f"source summaries loaded for {len(tags)} spots")

    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", sys.argv[1])
    if failures:
        print("FAILURES:", failures)
        return 1
    print("ALL GATES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

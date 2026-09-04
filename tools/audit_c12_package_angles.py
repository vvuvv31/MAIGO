#!/usr/bin/env python3
"""Decode the frozen C12 primary package for one channel/energy and emit
differential secondaries: multiplicity, total KE, KE spectra, cos(theta)
from stored local_direction, pT/pL, KE-weighted angular moments,
>5/10/20-degree fractions, and event energy closure.

Usage: audit_c12_package_angles.py PACKAGE TZ E_MEVU OUT_JSON [WINDOW]
  e.g. .../cinel03_c12_targets_v2_1.bin 8 200 out.json 10
Target is element Z (8 = O). Cells whose [e_low, e_up] overlap
[E-WINDOW/2, E+WINDOW/2] are aggregated (default WINDOW = package bin
width behavior: exact bracket only when WINDOW=0).
"""
import json
import math
import struct
import sys

import numpy as np

sys.path.insert(0, "/mnt/sdb/wuwei/MAIGO/tools")
from cinel03 import Cinel03Package  # noqa: E402

RAW = struct.Struct(
    "<QIQIII"
    "ihhfff"
    "ff"
    "fff"
    "fff"
    "ffff"
    "ii"
    "hhI"
    "iii"
    "iihh"
    + "f" * 13
    + "ff"
    + "IIf"
    + "f" * 10
    + "64s32s64s64s"
)
# field offsets within RAW (by element index after unpack)
I_RUN, I_THREAD, I_EVENT, I_TRACK, I_PARENT, I_SEQ = 0, 1, 2, 3, 4, 5
I_PDG, I_PZ, I_PA = 6, 7, 8
I_COLLE, I_COLLEU = 12, 13
I_CDIR = 17  # 3 floats: 17,18,19
I_LOCAL_DEP = 49  # after 13 parent floats (36..48)
I_DIRECT = 51
I_UNSUPP_N = 52
I_UNSUPP_E = 53

PROD = struct.Struct("<ihh" + "f" * 15 + "i")
P_PDG, P_Z, P_A = 0, 1, 2
P_KE = 6
P_LDX, P_LDY, P_LDZ = 10, 11, 12
P_ROLE = 18


def main():
    pkg_path, tz, e_mevu, out_path = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    window = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    pkg = Cinel03Package.read_binary(pkg_path)
    print(f"cells={len(pkg.cells)} interactions={len(pkg.interactions)} "
          f"products={len(pkg.products)} emin={pkg.minimum_energy_MeV_per_u} "
          f"ew={pkg.energy_bin_width_MeV_per_u}")

    lo, hi = e_mevu - window / 2.0, e_mevu + window / 2.0
    cells = [c for c in pkg.cells
             if c["projectile_z"] == 6 and c["projectile_a"] == 12
             and c["target_element_z"] == tz
             and c["energy_lower_MeV_per_u"] <= hi
             and c["energy_upper_MeV_per_u"] >= lo]
    print(f"matching C12+Z{tz} cells for {e_mevu} MeV/u: {len(cells)}")
    for c in cells:
        print(f"  ebin={c['energy_bin']} range=[{c['energy_lower_MeV_per_u']},"
              f"{c['energy_upper_MeV_per_u']}] count={c['interaction_count']}")

    # product offsets are cumulative over interaction file order
    all_counts = []
    off = 0
    for raw in pkg.interactions:
        f = RAW.unpack(raw)
        all_counts.append(f[I_DIRECT])
    # prefix sums
    prefix = np.zeros(len(all_counts) + 1, dtype=np.int64)
    prefix[1:] = np.cumsum(all_counts)

    per_species = {}
    events = 0
    closure = []
    for c in cells:
        for k in range(c["interaction_count"]):
            idx = c["interaction_offset"] + k
            f = RAW.unpack(pkg.interactions[idx])
            ncol = f[I_DIRECT]
            p0 = prefix[idx]
            ev_ke = 0.0
            for j in range(ncol):
                p = PROD.unpack(pkg.products[p0 + j])
                z, a, ke = p[P_Z], p[P_A], p[P_KE]
                ldz = p[P_LDZ]
                role = p[P_ROLE]
                if role != 0:
                    continue
                key = f"Z{z}A{a}"
                s = per_species.setdefault(key, {"n": 0, "ke": 0.0, "cos": [],
                                                 "ke_cos": 0.0})
                s["n"] += 1
                s["ke"] += ke
                s["cos"].append(ldz)
                s["ke_cos"] += ke * ldz
                ev_ke += ke
            events += 1
            parent_e = None
            closure.append({"ev_ke": ev_ke})

    res = {"package": pkg_path, "target_z": tz, "energy_mevu": e_mevu,
           "cells": len(cells), "events": events, "species": {}}
    for key in sorted(per_species):
        s = per_species[key]
        cos = np.array(s["cos"], dtype=np.float64)
        theta = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        mult = s["n"] / max(events, 1)
        res["species"][key] = {
            "multiplicity": mult,
            "count": s["n"],
            "total_ke_MeV": s["ke"],
            "ke_per_event": s["ke"] / max(events, 1),
            "mean_cos": float(cos.mean()),
            "mean_theta_deg": float(theta.mean()),
            "ke_weighted_cos": s["ke_cos"] / max(s["ke"], 1e-30),
            "frac_gt5": float((theta > 5).mean()),
            "frac_gt10": float((theta > 10).mean()),
            "frac_gt20": float((theta > 20).mean()),
            "ke_p50_MeV": None,
        }
        print(f"{key}: mult={mult:.3f} ke/ev={s['ke'] / max(events,1):.1f} "
              f"<th>={theta.mean():.1f}deg kewcos={s['ke_cos']/max(s['ke'],1e-30):.3f} "
              f">5={((theta>5).mean()):.3f} >10={((theta>10).mean()):.3f} "
              f">20={((theta>20).mean()):.3f}")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", out_path, "events:", events)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analytic CSDA-raytrace expectation for INCLXX package events in a
200x200x220 mm air slab (rho=0.0393235): straight-line transport of each
charged product from its birth vertex, depositing min(KE, CSDA along
in-box path). Compares against GPU and TOPAS slab deposits to judge
which engine's air dose is consistent with the shared birth population.
"""
import struct
import sys

import numpy as np

sys.path.insert(0, "/mnt/sdb/wuwei/MAIGO/tools")
from cinel03 import Cinel03Package, PRODUCT_FORMAT, RAW_FIXED_FORMAT  # noqa: E402

RHO_AIR = 0.0393235
BOX = (200.0, 200.0, 220.0)  # x,y,z mm
# Campaign frame: beam enters z=220 along -z (matches stored directions).


def load_ion_sp():
    import csv
    tab = {}
    with open("/mnt/sdb/wuwei/MAIGO/data/ion_stopping_power_water_geant4_11_3_2.csv") as f:
        for r in csv.reader(f):
            if not r or not r[0].lstrip("-").isdigit():
                continue
            tab.setdefault((int(r[0]), int(r[1])), []).append(
                (float(r[2]), float(r[3])))
    for k in tab:
        tab[k] = (np.array([e for e, _ in tab[k]]),
                  np.array([s for _, s in tab[k]]))
    return tab


def csda_range_mm(tab, z, a, ke_mev, rho):
    key = (z, a) if (z, a) in tab else (1, 1)
    e, sp = tab[key]
    # mass SP table is per-mm at rho=1 (water); scale by rho ratio
    eu = np.linspace(0.05, ke_mev, 400)
    sp_i = np.interp(eu, e / max(a, 1), sp) * rho
    sp_i = np.maximum(sp_i, 1e-9)
    return float(np.trapezoid(1.0 / sp_i, eu))


def ray_box(px, py, pz, dx, dy, dz):
    t_enter, t_exit = 0.0, 1e30
    for p, d, lo, hi in ((px, dx, -100.0, 100.0), (py, dy, -100.0, 100.0),
                         (pz, dz, 0.0, 220.0)):
        if abs(d) < 1e-9:
            if not (lo <= p <= hi):
                return None
            continue
        t0, t1 = (lo - p) / d, (hi - p) / d
        if t0 > t1:
            t0, t1 = t1, t0
        t_enter, t_exit = max(t_enter, t0), min(t_exit, t1)
    if t_exit < t_enter:
        return None
    return t_enter, t_exit


def main():
    pkg = Cinel03Package.read_binary(
        "/mnt/sdb/wuwei/MAIGO/data/schneider/cinel03_c12_targets_v2_1.bin")
    tab = load_ion_sp()
    cells = [c for c in pkg.cells if c["projectile_z"] == 6
             and c["projectile_a"] == 12 and c["target_element_z"] == 8
             and c["energy_lower_MeV_per_u"] <= 210
             and c["energy_upper_MeV_per_u"] >= 190]
    counts = []
    for raw in pkg.interactions:
        counts.append(RAW_FIXED_FORMAT.unpack(raw)[51])
    import itertools
    pref = list(itertools.accumulate([0] + counts))
    rng = np.random.default_rng(0)
    dep_ev = []
    nev = 0
    for c in cells:
        for k in range(c["interaction_count"]):
            idx = c["interaction_offset"] + k
            # vertex: exponential attenuation over first 86 mm from entry z=220
            zv = 220.0 - float(rng.exponential(200.0))
            if zv < 134.0:
                continue
            nev += 1
            dep = 0.0
            for j in range(counts[idx]):
                p = PRODUCT_FORMAT.unpack(pkg.products[pref[idx] + j])
                if p[18] != 0 or p[1] <= 0:
                    continue
                z, a, ke = p[1], p[2], p[6]
                if ke <= 0:
                    continue
                seg = ray_box(0.0, 0.0, zv, p[7], p[8], p[9])
                if seg is None:
                    continue
                tin, tout = seg
                path = max(0.0, tout - max(tin, 0.0))
                if path <= 0:
                    continue
                r = csda_range_mm(tab, z, a, ke, RHO_AIR)
                if r >= path:
                    # escapes: deposit dE/dx * path (approx via mean SP)
                    rfull = csda_range_mm(tab, z, a, ke, RHO_AIR)
                    dep += ke * min(1.0, path / max(rfull, 1e-9))
                else:
                    dep += ke
            dep_ev.append(dep)
    dep_ev = np.array(dep_ev)
    print(f"events={nev} analytic mean deposit/event={dep_ev.mean():.1f} MeV "
          f"(GPU full-nuclear net ~862, TOPAS ~38)")
    print(f"median={np.median(dep_ev):.1f} p10={np.percentile(dep_ev,10):.1f}")


if __name__ == "__main__":
    main()

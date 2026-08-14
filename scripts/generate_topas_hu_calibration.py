#!/usr/bin/env python3
"""Print the moquimc-style C-12 mass SPR(rho, E) knots used by the GPU.

Runtime transport builds the same curve from:
  data/stopping_power_{air,lung,water,bone}_geant4_11_3_2.csv
This script is a host-side check, not a required build step.
"""
import math

def load_sp(path):
    energies, values = [], []
    with open(path, encoding="ascii") as handle:
        for line in handle:
            line = line.strip()
            if not line or line[0] in "#abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
                continue
            energy, stopping = line.split(",")[:2]
            energies.append(float(energy))
            values.append(float(stopping))
    return energies, values


def interp(energies, values, energy):
    if energy <= energies[0]:
        return values[0]
    if energy >= energies[-1]:
        return values[-1]
    lo, hi = 0, len(energies) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if energies[mid] <= energy:
            lo = mid
        else:
            hi = mid
    span = energies[hi] - energies[lo]
    frac = 0.0 if span <= 0.0 else (energy - energies[lo]) / span
    return values[lo] + frac * (values[hi] - values[lo])


def lerp(x, x0, x1, y0, y1):
    if x1 <= x0:
        return y1
    return y0 + (x - x0) / (x1 - x0) * (y1 - y0)


def main():
    water_e, water_s = load_sp("data/stopping_power_water_geant4_11_3_2.csv")
    air_e, air_s = load_sp("data/stopping_power_air_geant4_11_3_2.csv")
    lung_e, lung_s = load_sp("data/stopping_power_lung_geant4_11_3_2.csv")
    bone_e, bone_s = load_sp("data/stopping_power_bone_geant4_11_3_2.csv")
    rho_air, rho_lung_table, rho_bone = 0.00120479, 1.04, 1.85

    def fs(table_e, table_s, rho_ref, energy):
        return interp(table_e, table_s, energy) / (
            rho_ref * interp(water_e, water_s, energy)
        )

    def mass_spr(rho, energy):
        fs_air = fs(air_e, air_s, rho_air, energy)
        fs_lung = fs(lung_e, lung_s, rho_lung_table, energy)
        fs_bone = fs(bone_e, bone_s, rho_bone, energy)
        if rho <= 0.26:
            return lerp(rho, rho_air, 0.26, fs_air, fs_lung)
        if rho < 0.90:
            return lerp(rho, 0.26, 0.90, fs_lung, 1.0)
        if rho <= 1.20:
            return 1.0
        if rho < rho_bone:
            return lerp(rho, 1.20, rho_bone, 1.0, fs_bone)
        return fs_bone

    print("rho     E=10     E=100    E=200    E=300")
    for rho in (0.0012, 0.05, 0.15, 0.26, 0.40, 0.70, 1.00, 1.20, 1.50, 1.85):
        vals = [mass_spr(rho, energy) for energy in (10.0, 100.0, 200.0, 300.0)]
        print(f"{rho:5.3f}  " + "  ".join(f"{v:7.4f}" for v in vals))


if __name__ == "__main__":
    main()

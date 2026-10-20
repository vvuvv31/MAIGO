#!/usr/bin/env python3
"""Assemble the proton runtime bank with explicit reuse provenance."""
import argparse, shutil, struct
from pathlib import Path
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--reference-bank', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    data = a.output.resolve() / 'data'
    out = data/'elastic/schneider/proton'
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Refuse to overwrite nonempty proton elastic output')
    (data/'stopping/water').mkdir(parents=True,exist_ok=True)
    work = a.work.resolve()
    old = a.reference_bank.resolve()
    raw = work / 'elastic/raw/z1a1.bin'
    new = out / 'all_ion.bin'
    with open(old, 'rb') as f:
        magic = f.read(8)
        ver, np_, ns, nt, ne, nq = struct.unpack('<6I', f.read(24))
        if magic != b'ELBANK01' or (ver, np_, ns, nt) != (1, 18, 26, 13):
            raise ValueError('Elastic schema')
        masses = np.fromfile(f, '<f8', 18)
        grid = np.fromfile(f, '<f4', ne)
    dtype = np.dtype([('fraction', '<f4'), ('mass', '<f4'), ('a', '<u4')])
    with open(raw, 'rb') as f:
        if f.read(8) != b'ELRAW001':
            raise ValueError('Elastic raw magic')
        if struct.unpack('<6I', f.read(24)) != (1, 1, ns, nt, ne, nq):
            raise ValueError('Elastic raw dimensions')
        mass = struct.unpack('<d', f.read(8))[0]
        eg = np.fromfile(f, '<f8', ne)
        rates = np.fromfile(f, '<f8', ns * nt * ne).reshape(ns, nt, ne)
        samples = np.fromfile(f, dtype, nt * ne * nq).reshape(nt, ne, nq)
        if f.read(1) or not np.array_equal(grid, eg.astype('<f4')):
            raise ValueError('Elastic grid/size')
    if not np.all(np.isfinite(rates) & (rates >= 0)) or not np.all(np.isfinite(samples['fraction']) & (samples['fraction'] >= 0) & (samples['fraction'] <= 1)):
        raise ValueError('Elastic nonfinite values')
    for ti in range(nt):
        for ei in range(ne):
            if np.any(rates[:, ti, ei] > 0) and np.any(samples['a'][ti, ei] == 0):
                raise ValueError('Elastic rate without target samples')
    shutil.copy2(old, new)
    with open(new, 'r+b') as f:
        f.seek(32)
        f.write(struct.pack('<d', mass))
        f.seek(32 + 18 * 8 + ne * 4)
        f.write(rates.astype('<f4').tobytes())
        f.seek(32 + 18 * 8 + ne * 4 + 18 * ns * nt * ne * 4)
        f.write(samples.tobytes())
    recoil = work / 'recoil/raw/recoil_stopping.bin'
    dest = out / 'recoil_stopping.bin'
    shutil.copy2(recoil, dest)
    with open(dest, 'rb') as f:
        if f.read(8) != b'ELRSP001':
            raise ValueError('Recoil format')
        ni, nm, nenergy = struct.unpack('<3I', f.read(12))
        f.read(ni * 8 + nenergy * 4)
        vals = np.fromfile(f, '<f4')
        if nm != 26 or len(vals) != ni * nm * nenergy or (not np.all(np.isfinite(vals) & (vals > 0))):
            raise ValueError('Recoil payload')
    raw_water = work / 'tables/proton_water.phsp'
    values = np.loadtxt(raw_water)
    if values.shape != (4001, 5) or not np.isfinite(values).all() or (not np.all(values[:, 2] > 0)):
        raise ValueError('Water stopping raw')
    np.savetxt(data / 'stopping/water/proton.csv', values[:, [0, 2]], delimiter=',', fmt='%.12g', header='Water_75eV; proton unrestricted electronic stopping\nenergy_MeVu,stopping_power_MeV_per_mm')
    print(out)
if __name__ == '__main__':
    main()

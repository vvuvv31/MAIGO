#!/usr/bin/env python3
"""Compile CarbonLossRangeNtuple output into the device loss-range table.

Dedups MT worker repeats, validates strict monotonicity of E and R,
compares restricted DEDX against the compiled unrestricted stopping CSV,
and writes data/urban/c12_copper_loss_range_g4_11_3_2.csv + metadata JSON.
"""
import csv, hashlib, json, sys
from pathlib import Path

SRC = Path('/tmp/opencode/lossrange/output/loss_range.phsp')
DST = Path('/mnt/sdb/wuwei/MAIGO_pristine/data/urban/c12_copper_loss_range_g4_11_3_2.csv')
META = Path('/mnt/sdb/wuwei/MAIGO_pristine/data/urban/c12_copper_loss_range_g4_11_3_2.metadata.json')
SP_CSV = Path('/mnt/sda/wuwei/minibeam_copper_extract_e250/compiled/c12_copper_stopping_geant4_11_3_2.csv')

seen = {}
with open(SRC) as f:
    for line in f:
        c = line.split()
        if len(c) < 6:
            continue
        eu = float(c[0])
        if eu in seen:
            continue
        seen[eu] = (float(c[0]), float(c[1]), float(c[2]), float(c[3]),
                    float(c[4]), float(c[5]))

rows = [seen[k] for k in sorted(seen)]
print(f'nodes: {len(rows)} E/u: {rows[0][0]}..{rows[-1][0]}')
assert len(rows) == 4001, len(rows)
for i in range(1, len(rows)):
    assert rows[i][1] > rows[i-1][1] and rows[i][2] > rows[i-1][2], i

maxres = max(abs(r[5]) for r in rows)
print(f'max |inverse residual| = {maxres:.6f} MeV')

# restricted vs compiled (unrestricted total) stopping at a few energies
sp = {}
with open(SP_CSV) as f:
    for line in f:
        c = line.split(',')
        if len(c) < 2:
            continue
        try:
            sp[float(c[0])] = float(c[1])
        except ValueError:
            pass
print('E/u  restricted  compiled(total)  ratio')
for eu in (20.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0):
    r = min(rows, key=lambda x: abs(x[0] - eu))
    s = min(sp.items(), key=lambda kv: abs(kv[0] - eu))
    print(f'{eu:5.1f}  {r[4]:9.3f}  {s[1]:9.3f}  {r[4]/s[1]:.4f}   '
          f'R_loss={r[2]:.3f}mm')

DST.parent.mkdir(parents=True, exist_ok=True)
with open(DST, 'w') as f:
    f.write('# C12 in Cu restricted loss-range table, Geant4 11.3.2 opt4, '
            'default-region 0.05mm cuts couple (theRangeTableForLoss).\n'
            '# energy_MeVu,energy_total_MeV,loss_range_mm,'
            'restricted_dedx_MeV_per_mm,inverse_residual_MeV\n')
    for eu, et, r, csda, d, res in rows:
        f.write(f'{eu:.6f},{et:.6f},{r:.6f},{d:.6f},{res:.6f}\n')

h = hashlib.sha256(DST.read_bytes()).hexdigest()
META.write_text(json.dumps({
    'source_phsp': str(SRC),
    'physics': 'g4em-standard_opt4 + g4decay, CutForAllParticles=0.05mm, '
               'default region couple, G4_Cu',
    'projectile': {'z': 6, 'a': 12},
    'nodes': len(rows),
    'energy_range_MeVu': [rows[0][0], rows[-1][0]],
    'max_inverse_residual_MeV': maxres,
    'sha256': h,
    'note': 'CSDA column is 0 in raw (CSDA tables disabled); omitted. '
            'R(E) is the restricted-loss range used as Urban currentRange; '
            'E(R) its inverse; DEDX is GetDEDX with the production couple.',
}, indent=2) + '\n')
print(f'wrote {DST} sha={h[:16]}...')

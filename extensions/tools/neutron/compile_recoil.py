#!/usr/bin/env python3
"""Package native recoil stopping extension, retaining explicit zero values."""
from pathlib import Path
import csv, json, sys, math
root=Path(sys.argv[1]);out=root/'package.json'
if out.exists():raise RuntimeError('manifest exists')

requested={tuple(map(int,l.split())) for l in (root/'species.txt').read_text().splitlines()}
materials=[]
for name in ['Water_75eV', 'G4_Cu', 'G4_AIR']:
    p = root / name
    if not (p / 'COMPLETE').is_file():
        raise RuntimeError('incomplete extraction')
    with (p / 'stopping.csv').open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != int((p / 'COMPLETE').read_text()):
        raise RuntimeError('row count mismatch')
    groups = {}
    zero = 0
    for r in rows:
        za = (int(r['atomic_number']), int(r['mass_number']))
        e = float(r['energy_MeVu'])
        s = float(r['stopping_power_MeV_per_mm'])
        n = float(r['nuclear_stopping_power_MeV_per_mm'])
        if not all((math.isfinite(v) for v in (e, s, n))) or e <= 0 or min(s, n) < 0:
            raise RuntimeError('invalid stopping')
        groups.setdefault(za, []).append(e)
        zero += s + n == 0
    if set(groups) != requested or any((len(v) != 769 or any((b <= a for a, b in zip(v, v[1:]))) for v in groups.values())):
        raise RuntimeError('species/energy coverage mismatch')
    prov = dict((line.split(' ', 1) for line in (p / 'provenance.txt').read_text().splitlines() if line.startswith(('density_g_cm3 ', 'excitation_eV ', 'material '))))
    if prov['material'] != name or (name == 'Water_75eV' and float(prov['excitation_eV']) != 75):
        raise RuntimeError('material mismatch')
    materials.append(dict(material=name, file=f'{name}/stopping.csv', excitation_eV=float(prov['excitation_eV']), density_g_cm3=float(prov['density_g_cm3']), species=len(groups), rows=len(rows), zero_total_rows=zero, minimum_energy_MeVu=min((min(v) for v in groups.values())), maximum_energy_MeVu=max((max(v) for v in groups.values()))))
m = dict(schema='maigo.neutron.recoil-stopping.v1', runtime_validated=False, materials=materials, method='G4EmCalculator electronic plus nuclear stopping, Geant4 11.3.2 EM opt4; used only after existing species tables', notes=['Straight CSDA transport; MCS, nuclear descendants and photons/electrons remain incomplete.', 'Ground-state ZA stopping; native excited-fragment masses still used for elastic kinematics.'])
out.write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))

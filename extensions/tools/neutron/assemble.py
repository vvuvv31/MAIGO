"""Attach independently refined macroscopic rates to a draft collision bank."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import struct
MATERIALS = ('Water_75eV', 'G4_Cu', 'G4_AIR')
PROCESSES = (111, 121, 131, 141)
STATUS = {'pointwise_bound_met': 1, 'zero_process': 2, 'adjacent_float32': 3, 'statistics_limited': 4, 'refine': 5}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))

def f32(value):
    import math
    x = float(value)
    require(math.isfinite(x) and x >= 0, 'invalid nonnegative coefficient')
    y = struct.unpack('<f', struct.pack('<f', x))[0]
    require(math.isfinite(y) and (x == 0 or y > 0), 'FP32 coefficient out of range')
    return y

def assemble(bank, directories, output):
    require(not output.exists(), 'output already exists')
    manifest = json.loads((bank / 'package.json').read_text())
    require(manifest['schema'] == 'maigo.neutron.native-collisions.v1', 'expected v1 bank')
    require(manifest['runtime_validated'] is False, 'expected draft collision bank')
    binary = Path(manifest['binary'])
    require(binary.name == str(binary), 'invalid collision filename')
    require(tuple((m['name'] for m in manifest['materials'])) == MATERIALS, 'material order mismatch')
    packed = bytearray()
    sources = []
    for mi, (name, directory) in enumerate(zip(MATERIALS, directories)):
        require((directory / 'COMPLETE').is_file(), 'incomplete rate extraction')
        meta = json.loads((directory / 'manifest.json').read_text())
        require(meta['schema'] == 'maigo.neutron.rate-refinement.v1' and meta['material'] == name, 'rate schema/material mismatch')
        require(meta['runtime_validated'] is False, 'expected draft rate table')
        rates = {}
        for row in rows(directory / 'rates.csv'):
            energy = f32(row['energy_MeV'])
            require(energy > 0 and energy == float(row['energy_MeV']), 'energy must already be FP32')
            subtype = int(row['subtype'])
            queries = int(row['rate_queries'])
            require(subtype in PROCESSES and 1 < queries < 2 ** 32, 'invalid process/query count')
            key = (energy, subtype)
            require(key not in rates, 'duplicate rate')
            rates[key] = (f32(row['rate_per_mm']), f32(row['rate_sem_per_mm']), queries)
        energies = sorted({key[0] for key in rates})
        require(len(energies) == meta['rate_nodes'] and len(rates) == 4 * len(energies), 'rate grid incomplete')
        bank_material = manifest['materials'][mi]
        require(energies[0] <= f32(bank_material['energy_min_MeV']) and energies[-1] >= f32(bank_material['energy_max_MeV']), 'rates do not cover event bank')
        audits = {}
        counts = Counter()
        for row in rows(directory / 'interval_audit.csv'):
            key = (float(row['low_MeV']), float(row['high_MeV']), int(row['subtype']))
            require(key not in audits and row['status'] in STATUS, 'invalid/duplicate interval audit')
            audits[key] = STATUS[row['status']]
            counts[row['status']] += 1
        require(dict(counts) == meta['interval_status_counts'], 'interval status count mismatch')
        for subtype in PROCESSES:
            for i, energy in enumerate(energies):
                state = 0 if i == len(energies) - 1 else audits.pop((energy, energies[i + 1], subtype))
                rate, sem, queries = rates[energy, subtype]
                packed.extend(struct.pack('<4I3f', mi, subtype, queries, state, energy, rate, sem))
        require(not audits, 'unreferenced interval audits')
        sources.append(dict(manifest=str((directory / 'manifest.json').resolve()), **meta))
    output.mkdir(parents=True)
    shutil.copyfile(bank / binary, output / binary)
    count = len(packed) // 28
    require(count < 2 ** 32, 'too many rate records')
    (output / 'macroscopic.bin').write_bytes(struct.pack('<8s4I', b'NEUTR001', 1, 0, count, 0) + packed)
    manifest.update(schema='maigo.neutron.native-collisions.v2', status='independent_rate_grid_draft', runtime_validated=False, macroscopic=dict(binary='macroscopic.bin', count=count, record_bytes=28, sources=sources, interval_status_codes=STATUS, interpolation='log-energy/linear-rate; no extrapolation', final_state_interpolation='unimplemented'))
    (output / 'package.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), macroscopic_records=count, runtime_validated=False)))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bank', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--water', type=Path, required=True)
    parser.add_argument('--copper', type=Path, required=True)
    parser.add_argument('--air', type=Path, required=True)
    args = parser.parse_args()
    assemble(args.bank, (args.water, args.copper, args.air), args.output)

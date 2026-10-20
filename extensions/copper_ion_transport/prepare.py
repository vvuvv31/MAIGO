#!/usr/bin/env python3
"""Extract and validate native Cu transport data for charged CINPKG04 products."""
import argparse
import json
import struct
import subprocess
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
KIT = HERE.parent

def required_species(urban_path, nuclear_path):
    with urban_path.open('rb') as stream:
        magic = stream.read(8)
        version, nm, ns, nn, nf, ng, validated = struct.unpack('<7I', stream.read(28))
        if magic != b'URBANV22' or version != 2 or validated != 1 or not 0 < ns <= 512:
            raise ValueError('Expected current validated Urban v2 package')
        stream.seek(36+20*nm)
        species = {struct.unpack('<IIf', stream.read(12))[:2] for _ in range(ns)}
    header_format = struct.Struct('<8sIIIIIIIQQQQffQQ40sI')
    with nuclear_path.open('rb') as stream:
        h = header_format.unpack(stream.read(header_format.size))
    if h[0] != b'CINPKG04' or h[1] != 4 or h[2] != header_format.size or h[3] != 0x01020304 or h[6] != 72 or h[11] != nuclear_path.stat().st_size:
        raise ValueError('Expected current CINPKG04 nuclear product layout')
    offset = h[2] + h[8]*h[4] + h[9]*h[5]
    dtype = np.dtype(dict(names=['z', 'a', 'energy', 'role'],
        formats=['<i2', '<i2', '<f4', '<i4'], offsets=[4, 6, 20, 68], itemsize=72))
    products = np.memmap(nuclear_path, mode='r', offset=offset, shape=(h[10],), dtype=dtype)
    selected = ((products['role'] == 0) | (products['role'] == 1)) & (products['z'] > 0) & (products['a'] > 0) & (products['energy'] > 0)
    species.update((int(z), int(a)) for z, a in np.unique(np.stack(
        [products['z'][selected], products['a'][selected]], axis=1), axis=0))
    if len(species) > 512:
        raise ValueError('Cu species exceeds the current package bound')
    return sorted(species)

def write_metadata(output, species, args):
    metadata = dict(schema=1, data_filename='fragments.csv', geant4_version='11.3.2',
        physics=['QGSP_BIC_HP', 'G4EmStandardPhysics_option4', 'G4IonINCLXXPhysics'],
        material='G4_Cu', density_g_per_cm3=8.96, random_seed=10070917,
        projectiles=[dict(z=z, a=a) for z, a in species], nodes_per_projectile=101,
        samples_per_node=128, energy_grid='0.001 * 10**(k/15) MeV/u, k=0..100', target_isotopes=[63, 65],
        sampler='Species-specific nearest energy node; natural Cu isotope and two-body CM transfer fraction',
        limitations=['Finite event-bank convergence pending', 'Carbon minibeam dose acceptance pending',
                    'Native ground-state ions only; Cu recoil energy remains a beamline sink'])
    (output/'fragments.metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    metadata = dict(schema=1, data_filename='copper_fragments.bin', geant4_version='11.3.2',
        physics='G4EmStandardPhysics_option4; all transported charged ions use Urban',
        material='G4_Cu', density_g_per_cm3=8.96, production_cut_mm=.05,
        projectiles=[dict(z=z, a=a) for z, a in species], dense_nodes=args.dense_nodes,
        validation_nodes=args.validation_nodes, minimum_energy_MeV=.05, maximum_energy_MeV=6500,
        heldout_grid_offset=.37, fp32_maximum_relative_error_gate=.0005,
        validation_log=(output/'urban_validation.log').read_text())
    (output/'copper_fragments.metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geant4-dir', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--urban-package', type=Path, required=True)
    parser.add_argument('--nuclear-package', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--dense-nodes', type=int, default=8193)
    parser.add_argument('--validation-nodes', type=int, default=1025)
    args = parser.parse_args()
    build, output = args.build_dir.resolve(), args.output_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=False)
    species = required_species(args.urban_package, args.nuclear_package)
    species_file = output/'species.txt'
    species_file.write_text(''.join(f'{z} {a}\n' for z, a in species))
    species_json = output/'species.json'
    species_json.write_text(json.dumps(species)+'\n')
    subprocess.run(['cmake', '-S', str(HERE), '-B', str(build),
                    f'-DGeant4_DIR={args.geant4_dir.resolve()}'], check=True)
    subprocess.run(['cmake', '--build', str(build), '-j', str(args.jobs)], check=True)
    campaign, packed = output/'urban_campaign', output/'urban_packed'
    subprocess.run(['python3', str(KIT/'tools/urban/run_campaign.py'), '--binary', str(build/'extract_urban'),
        '--out', str(campaign), '--medium', 'copper', '--species', str(species_json), '--workers', str(args.jobs),
        '--dense-nodes', str(args.dense_nodes), '--validation-nodes', str(args.validation_nodes)], check=True)
    # Each native job exports exact MFP edge probes; use the independently
    # exported validation edge rows, without another equivalent edge campaign.
    subprocess.run(['python3', str(KIT/'tools/urban/pack_reference.py'), str(campaign), str(packed),
        '--edges', str(campaign), '--materials', 'G4_Cu', '--species-count', str(len(species))], check=True)
    with (output/'urban_validation.log').open('w') as log:
        subprocess.run([str(build/'validate_urban'), str(packed/'urban_mcs_candidate.bin'),
            str(packed/'probes.bin'), str(packed/'mfp_probes.bin')], stdout=log, stderr=subprocess.STDOUT, check=True)
    (output/'copper_fragments.bin').write_bytes((packed/'urban_mcs_candidate.bin').read_bytes())
    subprocess.run([str(build/'extract'), str(species_file), str(output/'fragments.csv')], check=True)
    write_metadata(output, species, args)

if __name__ == '__main__':
    main()

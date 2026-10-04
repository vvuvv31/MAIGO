#!/usr/bin/env python3
"""Export the compact native elastic formula corrections with paired checks."""
import argparse, concurrent.futures, json, os, subprocess, struct
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'out/electron_standalone_20261003'

def one(task):
    out, node, binary, samples = task
    d = out / f'node{node:02d}'
    d.mkdir()
    source = BASE / 'photon_conditional_bank512' / f'node{node:02d}'
    text = (source / 'topas.txt').read_text().replace('PhotonConditionalDump', 'ElectronElasticDump')
    (d / 'topas.txt').write_text(text)
    (d / 'elastic_samples.txt').write_text(str(samples) + '\n')
    env = dict(os.environ, TOPAS_G4_DATA_DIR='/software/geant4-11.3.2/share/Geant4/data', LD_LIBRARY_PATH='/software/topas/lib:/software/geant4-11.3.2/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))
    with (d / 'topas.log').open('x') as log:
        r = subprocess.run([str(binary), 'topas.txt'], cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT)
    (d / 'execution.json').write_text(json.dumps(dict(returncode=r.returncode)) + '\n')
    if r.returncode or 'ELECTRON_ELASTIC_DUMP=PASS' not in (d / 'topas.log').read_text():
        raise RuntimeError(f'elastic export failed: {d}')
    artifact = d / 'elastic_kernel.bin'
    raw = artifact.read_bytes()
    assert len(raw) == 9464
    magic, version, count, mass, bc, xc2 = struct.unpack_from('<8sIIddd', raw)
    assert (magic, version, count) == (b'MAIELS01', 1, 31) and min(mass, bc, xc2) > 0
    rows = np.frombuffer(raw, dtype='<f8', offset=40).reshape(31, 38)
    assert np.isfinite(rows).all() and (rows[:, :3] > 0).all()
    audit = json.loads((d / 'elastic_audit.json').read_text())
    assert audit['samples'] == samples and audit['paired_random_streams_equal']
    assert (d / 'elastic_checks.bin').stat().st_size == 32 * samples
    identity = json.loads((BASE / 'birth_bank64' / f'node{node:02d}' / 'complete.json').read_text())['material_identity']
    result = dict(node=node, material_identity=identity, bytes=len(raw), audit=audit)
    (d / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(node, audit, flush=True)
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--nodes', nargs='+', type=int, default=[57, 12, 32])
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--samples', type=int, default=100000)
    p.add_argument('--binary', type=Path, default=BASE / 'electron_elastic_native_v2/topas')
    a = p.parse_args()
    assert len(set(a.nodes)) == len(a.nodes) and all((0 <= n < 82 for n in a.nodes))
    out = a.out.resolve()
    out.mkdir(exist_ok=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        r = list(pool.map(one, [(out, n, a.binary.resolve(), a.samples) for n in a.nodes]))
    index = out / 'index.txt'
    index.write_text(f'MAIELB02 {len(r)}\n' + ''.join((f'node{x['node']:02d}/elastic_kernel.bin node{x['node']:02d}/elastic_checks.bin {x['audit']['samples']}\n' for x in r)))
    (out / 'report.json').write_text(json.dumps(dict(status='NATIVE_ELASTIC_PRIMITIVE_NOT_COMPLETE_BOUNDARY_TRANSPORT', total_bytes=sum((x['bytes'] for x in r)), results=r), indent=2) + '\n')
if __name__ == '__main__':
    main()

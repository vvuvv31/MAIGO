#!/usr/bin/env python3
"""Read-only integrity and partition verification; NOT physics acceptance."""
import argparse
import json
import struct
from pathlib import Path

import numpy as np

from compile_water_electron_segments import sha
from water_electron_path_graph import NODE_DTYPE, NONE


def verify(path):
    path = Path(path)
    m = json.loads(path.with_suffix('.metadata.json').read_text())
    nominal = m['nominal_energy_max_MeVu']
    if ((m['schema_version'], nominal) not in ((2,300),(3,450)) or m['status'] != 'UNVALIDATED_WATER_EM_DIAGNOSTIC'
            or m['material'] != 'Water_75eV' or m['density_g_cm3'] != 1
            or m['production_cut_mm'] != .05 or m['projectile'] != [6, 12]
            or not nominal <= m['actual_energy_max_MeVu'] <= nominal+.0001):
        raise ValueError('Wrong scope/schema/material')
    if (path.name != m['data_filename'] or path.stat().st_size != m['data_size_bytes']
            or sha(path) != m['data_sha256']):
        raise ValueError('Data identity mismatch')
    if not m['pins']:
        raise ValueError('Missing provenance')
    for filename, pin in m['pins'].items():
        if sha(filename) != pin:
            raise ValueError('Provenance changed: ' + filename)
    with path.open('rb') as f:
        magic, version, nc, ns = struct.unpack('<8sIII', f.read(20))
        if magic != b'WELSEG01' or version != (1 if m['schema_version']==2 else 2) or nc != nominal//5 or len(m['channels']) != nc:
            raise ValueError('Header/channel count')
        if path.stat().st_size != 20+40*nc+56*ns:
            raise ValueError('Payload byte count')
        descriptors = [struct.unpack('<4dII', f.read(40)) for _ in range(nc)]
    samples = np.memmap(path, mode='r', dtype='<f8', offset=20+40*nc, shape=(ns, 7))
    if not np.isfinite(samples).all():
        raise ValueError('Nonfinite sample')
    next_offset = 0
    for b, (c, binary) in enumerate(zip(m['channels'], descriptors)):
        low, high, fraction, escape, offset, count = binary
        loss, birth, dep = c['parent_loss_MeV'], c['birth_MeV'], c['deposited_MeV']
        if loss <= 0 or min(birth, dep, c['photon_escape_MeV']) < 0:
            raise ValueError('Invalid energy partition')
        expected = (c['energy_min_MeVu'], c['energy_max_MeVu'], c['deposited_fraction'],
                    c['photon_escape_MeV']/loss, c['offset'], c['count'])
        if binary != expected or offset != next_offset or offset+count > ns or c['energy_bin'] != b:
            raise ValueError('Channel/payload mismatch')
        if low != b*5 or high != (m['actual_energy_max_MeVu'] if b == nc-1 else (b+1)*5):
            raise ValueError('Energy gap/alias')
        residual = c['raw_local_MeV'] + birth - loss
        if abs(residual-c['terminal_local_excess_MeV']) > 1e-10*loss:
            raise ValueError('Unreported raw residual')
        if (b != 0 and abs(residual) > 1e-7*loss) or abs(residual) > 1e-3*loss:
            raise ValueError('Primary partition failure')
        if (abs(birth-dep-c['photon_escape_MeV']) > 1e-7*loss
                or abs(fraction-dep/loss) > 1e-12 or not 0 <= fraction+escape <= 1):
            raise ValueError('Electron partition/normalization failure')
        if count:
            cdf = samples[offset:offset+count, 0]
            if np.any(cdf <= 0) or np.any(np.diff(cdf) <= 0) or cdf[-1] != 1 or fraction <= 0:
                raise ValueError('Invalid energy-weighted CDF')
        elif fraction != 0:
            raise ValueError('Missing response samples')
        next_offset += count
    if next_offset != ns:
        raise ValueError('Unreferenced payload')
    filename = m['path_filename']
    if Path(filename).name != filename:
        raise ValueError('Path graph must be a companion file')
    companion = path.parent/filename
    if sha(companion) != m['path_sha256'] or companion.stat().st_size != m['path_size_bytes']:
        raise ValueError('Path graph identity mismatch')
    with companion.open('rb') as f:
        magic, version, count, nodes = struct.unpack('<8sIII', f.read(20))
    if (magic != b'WELPTH01' or version != 1 or count != ns or nodes != m['path_nodes']
            or companion.stat().st_size != 20+4*ns+32*nodes):
        raise ValueError('Path graph header/size mismatch')
    heads = np.memmap(companion, mode='r', dtype='<u4', offset=20, shape=(ns,))
    graph = np.memmap(companion, mode='r', dtype=NODE_DTYPE, offset=20+4*ns, shape=(nodes,))
    if (np.any(heads >= nodes) or np.any(graph['reserved'] != 0)
            or not np.isfinite(graph['point']).all()
            or np.any((graph['previous'] != NONE) & (graph['previous'] >= np.arange(nodes)))):
        raise ValueError('Invalid/cyclic path graph')
    roots = graph['previous'] == NONE
    if not np.any(roots) or np.any(graph['point'][roots] != 0):
        raise ValueError('Invalid electron path origin')
    # Noncollapsed samples are continuous charged deposits. Their graph head
    # must be the precise beginning of the final depositing step. Collapsed
    # photon interactions intentionally retain a preceding flight in the graph.
    for start in range(0, ns, 65536):
        s = samples[start:start+65536]
        continuous = np.any(s[:, 1:4] != s[:, 4:7], axis=1)
        points = graph['point'][heads[start:start+len(s)]]
        if np.any(np.linalg.norm(points[continuous]-s[continuous, 1:4], axis=1) > 1e-6):
            raise ValueError('Sample/ancestor frame mismatch')
    return dict(channels=nc, segments=ns, path_nodes=nodes, data_sha256=m['data_sha256'],
                integrity_passed=True, physics_validated=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('data', type=Path)
    print(json.dumps(verify(p.parse_args().data), indent=2))

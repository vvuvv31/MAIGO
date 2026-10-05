#!/usr/bin/env python3
"""Verify delta-moment schema, finite moments and alignment with the EM source."""
import argparse
import json
from pathlib import Path
import struct
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def verify(path, source=None, *, metadata=None):
    path = Path(path)
    meta = metadata if metadata is not None else json.loads(path.with_suffix('.json').read_text())
    if path.stat().st_size != meta['bytes']:
        raise ValueError('Delta moments size/manifest mismatch')
    with path.open('rb') as stream:
        magic, version, nodes = struct.unpack('<8sII', stream.read(16))
        if magic == b'EMDFAC01':
            records = nodes
            nodes, = struct.unpack('<I', stream.read(4))
            if version != 1 or meta['schema'] != 'EMDFAC01' or nodes != meta['nodes'] or records != meta['records'] or not 26*18 <= records <= 256*18:
                raise ValueError('Factorized delta header/manifest mismatch')
            from verify_density_encoding import verify_delta
            verify_delta(stream, records, nodes)
        else:
            if (magic, version) != (b'EMDMOMT2', 2) or nodes != meta['nodes'] or path.stat().st_size != 16 + 8 * nodes:
                raise ValueError('Delta moments header/layout mismatch')
            moments = np.memmap(path, dtype='<f4', offset=16, mode='r', shape=(nodes, 2))
            for start in range(0, nodes, 1000000):
                block = moments[start:start + 1000000]
                if not np.all(np.isfinite(block) & (block >= 0)):
                    raise ValueError('Nonfinite or negative delta moments')
    if source is not None:
        with Path(source).open('rb') as stream:
            header = struct.unpack('<8s6I', stream.read(32))
        if header[0] not in (b'EMJOINT1',b'EMFAC001',b'EMCNT001') or header[1] != 1 or header[5] != nodes:
            raise ValueError('Delta node count differs from EM source')
    print(f'OK: {nodes} delta nodes; stored layout/fields and source dimensions checked')
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('table', nargs='?', type=Path, default=ROOT/'data/em/unified_em_delta_moments_v2.bin')
    parser.add_argument('--source', type=Path, default=ROOT/'data/em/unified_em_v1.bin')
    args = parser.parse_args()
    verify(args.table, args.source)

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
    with path.open('rb') as stream:
        magic, version, nodes = struct.unpack('<8sII', stream.read(16))
    if (magic, version) != (b'EMDMOMT2', 2) or nodes != meta['nodes']:
        raise ValueError('Delta moments header/manifest mismatch')
    if path.stat().st_size != 16 + 8 * nodes or path.stat().st_size != meta['bytes']:
        raise ValueError('Invalid delta moments layout')
    moments = np.memmap(path, dtype='<f4', offset=16, mode='r', shape=(nodes, 2))
    for start in range(0, nodes, 1000000):
        block = moments[start:start + 1000000]
        if not np.all(np.isfinite(block) & (block >= 0)):
            raise ValueError('Nonfinite or negative delta moments')
    if source is not None:
        with Path(source).open('rb') as stream:
            header = struct.unpack('<8s6I', stream.read(32))
        if header[:2] != (b'EMJOINT1', 1) or header[5] != nodes:
            raise ValueError('Delta node count differs from EM source')
    print(f'OK: {nodes} delta nodes; layout, moments and source dimensions checked')
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('table', nargs='?', type=Path, default=ROOT/'data/em/unified_em_delta_moments_v2.bin')
    parser.add_argument('--source', type=Path, default=ROOT/'data/em/unified_em_v1.bin')
    args = parser.parse_args()
    verify(args.table, args.source)

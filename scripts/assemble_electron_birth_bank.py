#!/usr/bin/env python3
"""Combine current projectile-tagged birth banks without resampling any node."""
import argparse
import json
import math
from pathlib import Path
import struct
import numpy as np

CHANNEL = np.dtype([('low','<f8'), ('high','<f8'), ('fraction','<f8'), ('offset','<u8'), ('count','<u8')])
SAMPLE = np.dtype([('cdf','<f8'), ('energy','<f8'), ('cosine','<f8')])


def inspect(path, projectile):
    result = []
    with path.open('rb') as stream:
        magic, version, count, z, a = struct.unpack('<8s4I', stream.read(24))
        if (magic, version) != (b'MAIBIR02', 2) or (z,a) != projectile or not 1 <= count <= 4096:
            raise ValueError(f'Birth header/projectile mismatch: {path}')
        for _ in range(count):
            begin = stream.tell()
            section, density, samples, bins = struct.unpack('<idQI', stream.read(24))
            expected_bins = (60 if a == 12 else 50) if section == 25 else (51 if a == 12 else 50)
            if not 0 <= section <= 25 or not math.isfinite(density) or density <= 0 or bins != expected_bins:
                raise ValueError('Material/density/energy bins')
            if section == 25 and density != 1:
                raise ValueError('Water_75eV density')
            channels = np.frombuffer(stream.read(40*bins), CHANNEL)
            end = stream.tell() + 24*samples
            if len(channels) != bins or end > path.stat().st_size:
                raise ValueError('Truncated node')
            states = np.memmap(path, SAMPLE, mode='r', offset=stream.tell(), shape=(samples,)) if samples else np.empty(0, SAMPLE)
            offset = 0
            for i,c in enumerate(channels):
                n = int(c['count']);fraction = float(c['fraction'])
                if c['low'] != 5*i or c['high'] != 5*(i+1) or int(c['offset']) != offset or offset+n > samples:
                    raise ValueError('Channel domain/offset')
                if not math.isfinite(fraction) or not 0 <= fraction <= 1 or bool(n) != (fraction > 0):
                    raise ValueError('Birth yield/support')
                block = states[offset:offset+n]
                ceiling = .8 if section == 25 else .65
                if n and (not all(np.isfinite(block[k]).all() for k in SAMPLE.names)
                          or block['cdf'][0] <= 0 or np.any(np.diff(block['cdf']) <= 0)
                          or block['cdf'][-1] != 1 or np.any(block['energy'] <= 0)
                          or np.any(block['energy'] > ceiling+1e-12) or np.any(abs(block['cosine']) > 1+1e-12)):
                    raise ValueError('Invalid birth samples')
                offset += n
            if offset != samples:
                raise ValueError('Unreferenced samples')
            result.append(dict(path=str(path), offset=begin, bytes=end-begin, section=section,
                               density_g_cm3=density, samples=int(samples), channels=bins))
            del states
            stream.seek(end)
        if stream.read(1):
            raise ValueError('Trailing birth bytes')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--projectile', choices=['proton','carbon'], required=True)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    projectile = (1,1) if args.projectile == 'proton' else (6,12)
    nodes = [node for path in args.inputs for node in inspect(path.resolve(), projectile)]
    if len({(n['section'],n['density_g_cm3']) for n in nodes}) != len(nodes):
        raise ValueError('Duplicate material/density node')
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise ValueError('Output already exists')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as dest:
        dest.write(struct.pack('<8s4I', b'MAIBIR02', 2, len(nodes), *projectile))
        for node in nodes:
            with Path(node['path']).open('rb') as source:
                source.seek(node['offset']);remaining=node['bytes']
                while remaining:
                    data = source.read(min(remaining, 8<<20))
                    if not data:
                        raise ValueError('Source truncated during assembly')
                    dest.write(data);remaining-=len(data)
    args.output.with_suffix('.json').write_text(json.dumps(dict(schema='MAIBIR02',projectile=projectile,
        bytes=args.output.stat().st_size,nodes=nodes,status='STRUCTURE_AND_SUPPORT_CHECKED_NOT_DOSE_ACCEPTANCE'),indent=2)+'\n')
    print(args.output, len(nodes), args.output.stat().st_size)


if __name__ == '__main__':
    main()

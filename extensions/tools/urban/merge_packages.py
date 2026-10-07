#!/usr/bin/env python3
"""Merge validated CT and copper Urban packages with matching projectile registries."""
import argparse
from pathlib import Path
import struct


def read(path):
    payload = path.read_bytes()
    if payload[:8] != b'URBANV22':
        raise ValueError('Urban format')
    version, nm, ns, nn, nf, ng, gate = struct.unpack_from('<7I', payload, 8)
    if version != 2 or gate != 1:
        raise ValueError('Urban version/context gate')
    offset = 36
    materials = [struct.unpack_from('<i4f', payload, offset+i*20) for i in range(nm)]
    offset += nm*20
    species = [struct.unpack_from('<IIf', payload, offset+i*12) for i in range(ns)]
    offset += ns*12
    records = [list(struct.unpack_from('<2f10I', payload, offset+i*48)) for i in range(nm*ns)]
    offset += nm*ns*48
    nodes = bytearray(payload[offset:offset+nn*8]); offset += nn*8
    mfps = bytearray(payload[offset:offset+nf*8]); offset += nf*8
    segments = bytearray(payload[offset:])
    if len(segments) != ng*24:
        raise ValueError('Urban size')
    return materials, species, records, nodes, mfps, segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ct', type=Path, required=True)
    parser.add_argument('--copper', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    materials, species, records, nodes, mfps, segments = read(args.ct)
    cm, cs, cr, cn, cf, cg = read(args.copper)
    mi = next(i for i, material in enumerate(cm) if material[0] == -2)
    if any(material[0] == -2 for material in materials):
        raise ValueError('CT package already contains copper')
    for ion in species:
        si = next(i for i, candidate in enumerate(cs) if candidate[:2] == ion[:2])
        if abs(cs[si][2]-ion[2]) > 1e-4:
            raise ValueError('Projectile mass differs')
        record = cr[mi*len(cs)+si].copy()
        for offset, count, source, target, stride in [(2, 3, cn, nodes, 8), (4, 5, cf, mfps, 8),
                (6, 9, cg, segments, 24), (7, 10, cg, segments, 24), (8, 11, cg, segments, 24)]:
            start = record[offset]
            record[offset] = len(target)//stride
            target.extend(source[start*stride:(start+record[count])*stride])
        records.append(record)
    materials.append(cm[mi])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        stream.write(b'URBANV22'+struct.pack('<7I', 2, len(materials), len(species),
                                          len(nodes)//8, len(mfps)//8, len(segments)//24, 1))
        for row in materials: stream.write(struct.pack('<i4f', *row))
        for row in species: stream.write(struct.pack('<IIf', *row))
        for row in records: stream.write(struct.pack('<2f10I', *row))
        for array in (nodes, mfps, segments): stream.write(array)


if __name__ == '__main__':
    main()

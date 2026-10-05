"""Validate stored compact EM fields; full reconstruction is checked by the C++ loader."""
import struct
import numpy as np


def read(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError('Truncated density-factorized package')
    return data


def runs(stream, count, maximum):
    n, = struct.unpack('<I', read(stream, 4))
    if not 0 < n <= count:
        raise ValueError('Invalid density run count')
    a = np.frombuffer(read(stream, 8*n), '<u4').reshape(n, 2)
    if a[0, 0] != 0 or np.any(a[:, 0] >= count) or np.any(a[1:, 0] <= a[:-1, 0]) or np.any(a[:, 1] > maximum):
        raise ValueError('Invalid density runs')


def field(stream, count):
    n, = struct.unpack('<I', read(stream, 4))
    if not 2 <= n <= count:
        raise ValueError('Invalid density knot count')
    a = np.frombuffer(read(stream, 8*n), dtype=[('index', '<u4'), ('coefficient', '<f4')])
    if a['index'][0] != 0 or a['index'][-1] != count-1 or np.any(a['index'][1:] <= a['index'][:-1]) or not np.isfinite(a['coefficient']).all():
        raise ValueError('Invalid density knots')
    runs(stream, count, 1)


def end(stream):
    if stream.read(1):
        raise ValueError('Trailing density payload')


def verify_em(stream, nm, ns, nr, nn, nv):
    # Header/material/species verification has already consumed those directories.
    records = np.frombuffer(read(stream, 104*nr), dtype=[('f', '<f4', (14,)), ('u', '<u4', (12,))])
    if not np.isfinite(records['f']).all():
        raise ValueError('Nonfinite EM records')
    splines = np.frombuffer(read(stream, 24*nv), '<f4').reshape(nv, 6)
    if not np.isfinite(splines).all() or np.any(splines[:, 1] <= splines[:, 0]):
        raise ValueError('Invalid native splines')
    bases = []
    offset = 0
    for i, r in enumerate(records):
        base, ratio = struct.unpack('<If', read(stream, 8))
        count = int(r['u'][1])
        if not (2 <= count <= nn and r['u'][0] == offset and offset+count <= nn and base < nr and base % ns == i % ns and records[base]['u'][1] == count and np.isfinite(ratio) and ratio > 0):
            raise ValueError('Invalid EM density directory')
        bases.append(base)
        offset += count
        if base == i:
            a = np.frombuffer(read(stream, 52*count), '<f4').reshape(count, 13)
            if not np.isfinite(a).all() or np.any(a[:, 0] <= 0) or np.any(a[1:, 0] <= a[:-1, 0]) or not np.isin(a[:, 11], [0, 1, 2, 3]).all() or not np.isin(a[:, 12], [0, 1]).all():
                raise ValueError('Invalid EM density base')
        else:
            for _ in range(10):
                field(stream, count)
            runs(stream, count, 3)
            runs(stream, count, 1)
    if offset != nn or any(bases[b] != b for b in bases):
        raise ValueError('Incomplete/chained EM density bases')
    end(stream)


def verify_delta(stream, nr, nn):
    bases, counts = [], []
    offset = 0
    for i in range(nr):
        begin, count, identity, base, ratio = struct.unpack('<IIIIf', read(stream, 20))
        if not (identity == i and begin == offset and 2 <= count <= nn and offset+count <= nn and base < nr and base % 18 == i % 18 and np.isfinite(ratio) and ratio > 0):
            raise ValueError('Invalid delta density directory')
        bases.append(base)
        counts.append(count)
        offset += count
        if base == i:
            a = np.frombuffer(read(stream, 8*count), '<f4')
            if not np.all(np.isfinite(a) & (a >= 0)):
                raise ValueError('Invalid delta density base')
        else:
            field(stream, count)
            field(stream, count)
    if offset != nn or any(bases[b] != b or counts[b] != counts[i] for i, b in enumerate(bases)):
        raise ValueError('Incomplete/chained delta density bases')
    end(stream)

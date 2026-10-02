"""File formats and frozen inputs for the RT07575 minibeam GPU plan."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
CASE = ROOT.parent / 'RT07575_minibeam'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_mhd(path):
    path = Path(path)
    fields = {}
    for line in path.read_text().splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            fields[key.strip()] = value.strip()
    assert fields['ElementType'] == 'MET_FLOAT'
    dimensions = tuple(map(int, fields['DimSize'].split()))
    raw = path.parent / fields['ElementDataFile']
    assert raw.stat().st_size == int(np.prod(dimensions)) * 4
    return np.fromfile(raw, '<f4').reshape(dimensions[::-1]), fields


def write_mhd(path, array_zyx, reference_fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(array_zyx, dtype='<f4')
    assert np.isfinite(array).all() and np.all(array >= 0)
    raw = path.with_suffix('.raw')
    temporary = raw.with_suffix('.raw.tmp')
    array.tofile(temporary)
    temporary.replace(raw)
    fields = dict(reference_fields)
    fields.update(DimSize=' '.join(map(str, array.shape[::-1])),
                  ElementType='MET_FLOAT', ElementDataFile=raw.name,
                  DoseUnits='Gy')
    path.write_text(''.join(f'{key} = {value}\n' for key, value in fields.items()))


def packed_to_patient(array):
    # Inverse of the historical, verified RT07575 x-negative beam packing.
    # Packed GPU [z,y,x] = [patient X descending, patient Z, patient Y].
    return np.flip(array.transpose(1, 2, 0), axis=2)

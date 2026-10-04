#!/usr/bin/env python3
"""Sum accepted shards without changing the patient axes or fitting dose."""
import json
import h5py
import numpy as np
from common import ROOT, save_json, read_mhd, write_mhd, packed_to_patient


def roi_indices():
    result = {}
    with h5py.File(ROOT/'inputs/patient.mat', 'r') as patient:
        cells = patient['cst'][()]
        for column in range(cells.shape[1]):
            name = ''.join(chr(int(v)) for v in patient[cells[1,column]][()].reshape(-1))
            container = patient[cells[3,column]]
            if container.size == 0:
                continue
            data = patient[container[()].reshape(-1)[0]]
            if data.attrs.get('MATLAB_empty', False):
                continue
            result[name] = np.unique(data[()].astype(np.int64).reshape(-1))-1
    assert len(result['GTV']) == 31184
    return result


def dose_metrics(dose, rois):
    # MATLAB linear index is Y-fastest for a YXZ cube.
    flat = dose.transpose(0,2,1).reshape(-1)
    result = {}
    for name, indices in rois.items():
        values = flat[indices]
        if not len(values):
            continue
        item = dict(voxels=len(values), mean_Gy=float(values.mean()), max_Gy=float(values.max()))
        if name == 'GTV':
            item.update(D95_Gy=float(np.percentile(values,5)),
                        D98_Gy=float(np.percentile(values,2)),
                        D50_Gy=float(np.percentile(values,50)),
                        D2_Gy=float(np.percentile(values,98)),
                        V4Gy_percent=float(100*np.mean(values>=4)))
        result[name] = item
    return result


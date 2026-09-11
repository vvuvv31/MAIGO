"""Exact-isocenter profiles/planes for electron-ON full20 (float march).

Frozen dirs supply manifest/geometry/TOPAS reference; electron dirs supply
gpu_sum.raw + execution.json + gamma_coarse.json. Same plots/CSVs as the
frozen profiles directory; method strings mark the electron physics.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pydicom

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'benchmark/benchmark20260909'))
sys.path.insert(0, str(REPO / 'tools'))
from run import sha  # noqa: E402
from plot_topas10x_isocenter_profiles import (  # noqa: E402
    isocenter_index, extract_profile)
import plot_ct_full20_isocenter as frozen_plot  # noqa: E402

CASES = ('RT06423', 'RT07575', '20022516')
FROZEN = Path('/mnt/sda/wuwei/ct_previous_full20_20260909')
ELEC_BASE = Path('/mnt/sda/wuwei/electron_full20_float_20260910')
ELEC_LEAN_OLD = Path('/mnt/sda/wuwei/electron_full20_lean_20260909')
OUTPUT = REPO / 'benchmark/topas10x/gpu_electron_20260910_profiles'

# RT07575 float run lives under the lean (double-run) base? No: float rerun
# went to electron_full20_float_20260910. Resolve per case.
ELEC = {'RT06423': ELEC_BASE / 'RT06423',
        'RT07575': ELEC_BASE / 'RT07575',
        '20022516': ELEC_BASE / '20022516'}


def plot(case):
    fz = FROZEN / case
    src = ELEC[case]
    m = json.loads((fz / 'manifest.json').read_text())
    s = json.loads((src / 'execution.json').read_text())
    gamma = json.loads((src / 'gamma_coarse.json').read_text())
    if s['status'] != 'complete_experiment' or \
            sum(r['histories'] for r in s['completed']) != m['histories']:
        raise ValueError('Incomplete electron fullplan ' + case)
    if s['overflow_excluded']:
        raise ValueError('Overflow excluded in ' + case)
    if sha(src / 'gpu_sum.raw') != s['aggregate_sha256']:
        raise ValueError('Changed electron dose')
    if sha(fz / 'topas_sum.raw') != m['reference_sum_sha256']:
        raise ValueError('Changed reference')
    g = np.fromfile(src / 'gpu_sum.raw', '<f4').reshape(m['gpu_shape_zyx'])
    if m['mapping'] == 'packed_xneg':
        g = np.flip(g.transpose(1, 2, 0), axis=2)
    elif m['mapping'] != 'native':
        raise ValueError('Unknown mapping')
    t = np.fromfile(fz / 'topas_sum.raw', '<f4').reshape(m['topas_shape_zyx'])
    if g.shape != t.shape or not np.isfinite(g).all() or not np.isfinite(t).all():
        raise ValueError('Invalid dose')
    dims = np.array(t.shape[::-1])
    spacing = np.array(m['spacing_zyx'][::-1])
    parameter = Path(m['replicas'][0]['path']) / 'run_full_plan.txt'
    iso, local = isocenter_index(parameter.read_text(), dims, spacing)
    ct, ct_pins = frozen_plot.load_ct(case, dims, spacing)
    dest = OUTPUT / case
    dest.mkdir(parents=True, exist_ok=True)
    audit = dict(
        isocenter_index_xyz=iso.tolist(),
        isocenter_centered_patient_mm_xyz=local.tolist(),
        parameter_file=str(parameter), parameter_sha256=sha(parameter),
        gpu_sha256=sha(src / 'gpu_sum.raw'),
        reference_sha256=sha(fz / 'topas_sum.raw'),
        electron_execution_sha256=sha(src / 'execution.json'),
        electron_gamma_sha256=sha(src / 'gamma_coarse.json'),
        electron_binary_sha256=s['binary_sha256'],
        electron_joint_sha256=s['joint_sha256'],
        method='Patient XYZ; exact isocenter interpolation; absolute Gy; '
               'no fitted scale or registration; ELECTRON joint r3 ON '
               '(float march production build)',
        orientation='Coordinate axes increasing, not radiological display convention',
        difference_definition='100*(GPU-TOPAS)/TOPAS whole-volume maximum',
        display_limits_percent=[-5, 5], ct_pins=ct_pins,
        ct_window_HU=[-1000, 1000], overlay_colormap='jet', overlay_alpha=0.5,
        files={})
    fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharex='col', layout='constrained')
    for a, label in enumerate('XYZ'):
        x, tt = extract_profile(t, iso, spacing, a)
        _, gg = extract_profile(g, iso, spacing, a)
        delta = gg.astype(float) - tt
        np.savetxt(dest / f'profile_{label}.csv',
                   np.column_stack((x, tt, gg, delta, 100 * delta / t.max())),
                   delimiter=',',
                   header='offset_from_isocenter_mm,TOPAS_Gy,GPU_Gy,'
                          'GPU_minus_TOPAS_Gy,difference_percent_reference_Dmax',
                   comments='')
        axes[0, a].plot(x, tt, label='TOPAS')
        axes[0, a].plot(x, gg, '--', label='GPU electron')
        axes[0, a].set(title=f'Patient {label} axis', ylabel='Dose (Gy)')
        axes[0, a].legend()
        axes[1, a].plot(x, 100 * delta / t.max())
        axes[1, a].axhline(0, color='gray', lw=.6)
        axes[1, a].set(xlabel=f'{label} - isocenter (mm)',
                       ylabel='GPU - TOPAS (% Dmax)', ylim=(-5, 5))
        for ax in axes[:, a]:
            ax.axvline(0, color='gray', lw=.5)
            ax.grid(alpha=.2)
    fig.suptitle(case + ' | electron full20 | exact-isocenter line profiles')
    fig.savefig(dest / 'isocenter_profiles.png', dpi=170)
    plt.close(fig)
    planes = []
    for name, h, v, fixed in [('Axial', 0, 1, 2), ('Coronal', 0, 2, 1), ('Sagittal', 1, 2, 0)]:
        xx, yy = np.meshgrid(np.arange(dims[h], dtype=float),
                             np.arange(dims[v], dtype=float))
        xyz = np.empty((3, *xx.shape))
        xyz[h] = xx
        xyz[v] = yy
        xyz[fixed] = iso[fixed]
        tt = map_coordinates(t, xyz[::-1], order=1, mode='constant', cval=np.nan,
                             prefilter=False)
        gg = map_coordinates(g, xyz[::-1], order=1, mode='constant', cval=np.nan,
                             prefilter=False)
        if not np.isfinite(tt).all() or not np.isfinite(gg).all():
            raise ValueError('Slice extrapolation')
        delta = gg.astype(float) - tt
        extent = [(-.5 - iso[h]) * spacing[h], (dims[h] - .5 - iso[h]) * spacing[h],
                  (-.5 - iso[v]) * spacing[v], (dims[v] - .5 - iso[v]) * spacing[v]]
        ct_plane = map_coordinates(ct, xyz[::-1], order=1, mode='constant', cval=np.nan,
                                   prefilter=False)
        if not np.isfinite(ct_plane).all():
            raise ValueError('CT slice extrapolation')
        planes.append((name, h, v, tt, gg, delta, extent, ct_plane))
        np.savez_compressed(dest / (name.lower() + '.npz'), topas_Gy=tt, gpu_Gy=gg,
                            difference_Gy=delta,
                            difference_percent_reference_Dmax=100 * delta / t.max(),
                            ct_HU=ct_plane,
                            x_mm=(np.arange(dims[h]) - iso[h]) * spacing[h],
                            y_mm=(np.arange(dims[v]) - iso[v]) * spacing[v])
    dose_max = max(max(float(p[3].max()), float(p[4].max())) for p in planes)
    fig, axes = plt.subplots(3, 3, figsize=(17, 15), layout='constrained')
    for row, (name, h, v, tt, gg, delta, extent, ct_plane) in enumerate(planes):
        for col, (values, title, cmap, lo, hi, unit) in enumerate([
                (tt, 'TOPAS', 'jet', 0, dose_max, 'Gy'),
                (gg, 'GPU electron', 'jet', 0, dose_max, 'Gy'),
                (100 * delta / t.max(), 'GPU - TOPAS', 'jet', -5, 5,
                 '% reference Dmax')]):
            ax = axes[row, col]
            ax.imshow(ct_plane, origin='lower', extent=extent, aspect='equal',
                      interpolation='nearest', cmap='gray', vmin=-1000, vmax=1000)
            im = ax.imshow(values, origin='lower', extent=extent, aspect='equal',
                           interpolation='nearest', cmap=cmap, vmin=lo, vmax=hi,
                           alpha=0.5)
            ax.axhline(0, color='cyan', lw=.4)
            ax.axvline(0, color='cyan', lw=.4)
            ax.set(title=name + ' | ' + title,
                   xlabel=f'{"XYZ"[h]} - isocenter (mm)',
                   ylabel=f'{"XYZ"[v]} - isocenter (mm)')
            fig.colorbar(im, ax=ax, label=unit, shrink=.8,
                         extend='both' if col == 2 else 'neither')
    fig.suptitle(case + ' | electron full20 | exact-isocenter anatomical planes')
    fig.savefig(dest / 'isocenter_planes.png', dpi=160)
    plt.close(fig)
    for f in dest.iterdir():
        if f.name != 'manifest.json':
            audit['files'][f.name] = sha(f)
    (dest / 'manifest.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(case, 'electron isocenter profiles and planes complete', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wait', action='store_true')
    args = p.parse_args()
    deadline = time.monotonic() + 8 * 3600
    for case in CASES:
        while not (ELEC[case] / 'gamma_coarse.json').exists():
            if not args.wait or time.monotonic() > deadline:
                raise RuntimeError('Electron fullplan/Gamma not ready: ' + case)
            time.sleep(60)
        plot(case)
    lines = ['# Electron full20 20260910：等中心图（电子包，float march）', '',
             '与 `gpu_current_20260909_profiles` 同口径：2D图以原始DICOM HU灰度图为底，'
             'jet色图50%透明度叠加。CT窗为−1000～1000 HU；逐层方向、位置、间距和尺寸与剂量网格验证一致。', '',
             '二维图为3×3：行=轴位/冠状位/矢状位，列=TOPAS剂量(Gy)/GPU电子剂量(Gy)/百分比差值。'
             '双方剂量共用色标。1D误差与2D差值均为100×(GPU−TOPAS)/TOPAS全体积峰值，'
             '显示范围−5%～+5%，保留正负。CSV/NPZ保留未截断值。无配准/拟合归一化。', '',
             'GPU为电子 joint r3 ON（float march 生产构建），同源同步长；剂量绝对Gy。', '']
    for case in CASES:
        lines += [f'- {case}：[三轴 profile及百分比误差]({case}/isocenter_profiles.png)、'
                  f'[三解剖面百分比差值]({case}/isocenter_planes.png)。']
    (OUTPUT / 'README.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()

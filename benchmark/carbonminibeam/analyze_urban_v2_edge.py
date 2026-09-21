"""Fixed-slit edge benchmark: single 0.5mm vacuum slit at x=0 in Cu block.
TOPAS (two-box CuL/CuR + vacuum gap, opt4+decay) vs GPU urban_v2 (cylindrical
Cu + slit air, endpoint safety, exact acceptance gate), C12 250MeV/u pencil,
EM-only, matched 0.05mm external step, mcs_scale=1, 20k histories/point.
Reports P(exit|d), P(touch and exit|d), Cu path, angles, branch fractions,
and the ACTUAL sampled entries (nomcs control proves entry==exit).
"""
import csv, math, statistics, sys, glob, os

TOPAS_DIR = '/tmp/opencode/edge_topas'
GPU_DIR = '/tmp/opencode/edge_gpu'
TAGS = ['negp25', 'negp20', 'negp05', 'negp01', 'negp001', 'posp001', 'posp01', 'posp05']
D = {'negp25': -0.25, 'negp20': -0.20, 'negp05': -0.05, 'negp01': -0.01,
     'negp001': -0.001, 'posp001': 0.001, 'posp01': 0.01, 'posp05': 0.05}
NINC = 20000

def load_topas(tag):
    xs, txs, es = [], [], []
    with open(f'{TOPAS_DIR}/{tag}/output/slab_exit.phsp') as f:
        for line in f:
            c = line.split()
            if len(c) < 14:
                continue
            if c[7] != '1000060120' or c[9] != '1':
                continue
            dx, dz = float(c[3]), 1.0
            xs.append(float(c[0]) * 10.0 - dx / dz * 0.01)
            txs.append(dx / dz)
            es.append(float(c[5]))
    return xs, txs, es

def load_gpu(tag):
    rows = list(csv.DictReader(open(f'{GPU_DIR}/ph_{tag}_urbanv2.csv')))
    xs = [float(r['x_mm']) for r in rows]
    txs = [float(r['direction_x']) / float(r['direction_z']) for r in rows]
    es = [float(r['kinetic_energy_MeV']) for r in rows]
    return xs, txs, es, rows

def q(v, p):
    s = sorted(abs(t) for t in v)
    return s[min(len(s) - 1, int(p * len(s)))] if s else float('nan')

print(f"{'d':>7} {'src':>5} {'Nexit':>6} {'Pexit':>7} {'Ptouch&exit':>11} "
      f"{'meanCupath':>10} {'xmean':>9} {'xstd':>8} {'th_q50':>9} {'th_q95':>9} {'th_q99':>9}")
out = {}
for tag in TAGS:
    d = D[tag]
    for src, loader in (('TOPAS', load_topas), ('GPUv2', load_gpu)):
        if src == 'TOPAS':
            xs, txs, es = loader(tag)
            rows = None
        else:
            xs, txs, es, rows = loader(tag)
        n = len(xs)
        pexit = n / NINC
        if rows is None:
            ptouch = float('nan')
            cup = float('nan')
            br = ''
        else:
            t = sum(1 for r in rows if r['ever_in_copper'] == '1')
            ptouch = t / NINC
            cup = statistics.fmean([float(r['cumulative_cu_true_path_mm']) for r in rows]) if rows else 0.0
            b = [sum(int(r[k]) for r in rows) for k in
                 ('disp_below_min', 'disp_accept', 'disp_reduce', 'disp_cancel')]
            st = sum(int(r['cu_steps']) for r in rows)
            br = (f"acc={b[1]/st:.3f} red={b[2]/st:.4f} can={b[3]/st:.2e} "
                  f"below={b[0]/st:.2e}" if st else 'nosteps')
        xm = statistics.fmean(xs) if xs else float('nan')
        xs_ = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        print(f"{d:>7.3f} {src:>5} {n:>6} {pexit:>7.4f} {ptouch:>11.4f} "
              f"{cup:>10.3f} {xm:>9.4f} {xs_:>8.4f} {q(txs,0.5):>9.2e} "
              f"{q(txs,0.95):>9.2e} {q(txs,0.99):>9.2e}  {br}")
        out[(tag, src)] = (pexit, ptouch)

print('\nP(exit|d)  [TOPAS vs GPUv2]:')
for tag in TAGS:
    print(f"  d={D[tag]:+.3f}: {out[(tag,'TOPAS')][0]:.4f} vs {out[(tag,'GPUv2')][0]:.4f}")
print('P(touch&exit|d) [GPUv2 only; TOPAS needs per-step ntuple]:')
for tag in TAGS:
    print(f"  d={D[tag]:+.3f}: {out[(tag,'GPUv2')][1]:.4f}")

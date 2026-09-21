"""250MeV/u single-center EM-only water-entry: TOPAS (2M rerun, Water_75eV,
opt4+decay, exact saved inputs) vs GPU (10M: old-urban, old-v2 fe4be,
new-v2 c025/c005). Primary C12 only (TOPAS: PDG 1000060120, first-scored).
"""
import csv, math, statistics, sys

N_TOPAS = 2000000
N_GPU = 10000000

def load_topas(path):
    xs, txs, es = [], [], []
    n0 = 0
    with open(path) as f:
        for line in f:
            c = line.split()
            if len(c) < 14:
                continue
            if c[7] != '1000060120' or c[9] != '1':
                continue
            n0 += 1
            xs.append(float(c[0]) * 10.0)
            txs.append(float(c[3]))
            es.append(float(c[5]))
    return xs, txs, es, n0

def load_gpu(path):
    rows = list(csv.DictReader(open(path)))
    xs = [float(r['x_mm']) for r in rows]
    txs = [float(r['direction_x']) / float(r['direction_z']) for r in rows]
    es = [float(r['kinetic_energy_MeV']) for r in rows]
    return xs, txs, es, rows

def q(v, p):
    s = sorted(abs(t) for t in v)
    return s[min(len(s) - 1, int(p * len(s)))] if s else float('nan')

def stats(name, xs, txs, es, ninc, rows=None):
    n = len(xs)
    mx, mt = statistics.fmean(xs), statistics.fmean(txs)
    a0 = sum((t - mt) ** 2 for t in txs) / n
    cov = sum((x - mx) * (t - mt) for x, t in zip(xs, txs)) / n
    print(f"--- {name} ---")
    print(f"yield={n/ninc:.5f} (N={n})")
    print(f"x: mean={mx:.4f} RMS={math.sqrt(sum((x-mx)**2 for x in xs)/n):.4f} "
          f"q95={q(xs,0.95):.4f} q99={q(xs,0.99):.4f}")
    print(f"theta: RMS={math.sqrt(a0):.6f} q50={q(txs,0.5):.6f} "
          f"q68={q(txs,0.68):.6f} q95={q(txs,0.95):.6f} q99={q(txs,0.99):.6f} "
          f"q999={q(txs,0.999):.6f}")
    print(f"Cov(x,theta)={cov:.6e}")
    print(f"E: mean={statistics.fmean(es):.2f} std={statistics.pstdev(es):.2f}")
    if rows is not None:
        te = sum(1 for r in rows if r['ever_in_copper'] == '1')
        print(f"ever_in_copper frac={te/n:.4f} "
              f"mean Cu path={statistics.fmean([float(r['cumulative_cu_true_path_mm']) for r in rows]):.3f}")
    return {'yield': n / ninc, 'xrms': math.sqrt(sum((x-mx)**2 for x in xs)/n),
            'trms': math.sqrt(a0), 'q95': q(txs, 0.95), 'q99': q(txs, 0.99),
            'cov': cov, 'emean': statistics.fmean(es),
            'estd': statistics.pstdev(es)}

tx, tt, te, tn = load_topas('/tmp/opencode/mb_topas/we2m/output/water_entrance.phsp')
r = {'TOPAS': stats('TOPAS', tx, tt, te, N_TOPAS)}
import glob
for tag, path in (('old-urban', '/tmp/opencode/mb10m/ph_oldurban.csv'),
                  ('old-v2', '/tmp/opencode/mb10m/ph_oldv2.csv'),
                  ('new-v2-c025', '/tmp/opencode/mb10m/ph_newv2_c025.csv'),
                  ('new-v2-c005', '/tmp/opencode/mb10m/ph_newv2_c005.csv')):
    try:
        xs, txs, es, rows = load_gpu(path)
    except FileNotFoundError:
        print(f"--- {tag}: NOT_RUN ---")
        continue
    r[tag] = stats(tag, xs, txs, es, N_GPU, rows)

"""1mm homogeneous Cu slab, C12 250MeV/u EM-only, matched 0.05mm step.
TOPAS (opt4+decay, box 120x120x1mm, pencil 3000MeV) vs GPU (same entrance,
slit-offset solid-Cu block, phase record at Cu exit z=0).
Outputs A0/A1/A2, quantiles, transmission, E, urban_v2 step diagnostics.
PASS->exit 0 only if the script itself runs; physics verdicts are printed,
not asserted (no fake gates).
"""
import csv, math, statistics, sys

def load_topas(path):
    xs, txs, es = [], [], []
    n = 0
    with open(path) as f:
        for line in f:
            c = line.split()
            if len(c) < 14 or c[0].startswith('#'):
                continue
            pdg = c[7]
            first = c[9]
            if pdg != '1000060120' or first != '1':
                continue
            n += 1
            x, dx, dz, e = float(c[0]) * 10.0, float(c[3]), 1.0, float(c[5])
            # back-project 0.01mm drift to Cu face (z=-0.5mm)
            xs.append(x - dx / dz * 0.01)
            txs.append(dx / dz)
            es.append(e)
    return xs, txs, es, n

def load_gpu(path):
    xs, txs, es, diag = [], [], [], []
    n = 0
    with open(path) as f:
        for r in csv.DictReader(f):
            n += 1
            xs.append(float(r['x_mm']))
            dz = float(r['direction_z'])
            txs.append(float(r['direction_x']) / dz)
            es.append(float(r['kinetic_energy_MeV']))
            diag.append(r)
    return xs, txs, es, n, diag

def quantiles(v, qs=(0.5, 0.68, 0.95, 0.99, 0.999)):
    s = sorted(abs(t) for t in v)
    n = len(s)
    return {q: s[min(n - 1, int(q * n))] for q in qs}

def moments(xs, txs):
    n = len(xs)
    mx, mt = statistics.fmean(xs), statistics.fmean(txs)
    a0 = sum((t - mt) ** 2 for t in txs) / n
    a2 = sum((x - mx) ** 2 for x in xs) / n
    a1 = sum((x - mx) * (t - mt) for x, t in zip(xs, txs)) / n
    return a0, a1, a2, mx, mt

topas = sys.argv[1] if len(sys.argv) > 1 else \
    '/tmp/opencode/slab_topas/t1_prod/output/slab_exit.phsp'
gurban = sys.argv[2] if len(sys.argv) > 2 else \
    '/tmp/opencode/urbanv2_smoke/ph_urban.csv'
gv2 = sys.argv[3] if len(sys.argv) > 3 else \
    '/tmp/opencode/urbanv2_smoke/ph_urbanv2.csv'

NINC = 256000
tx, tt, te, tn = load_topas(topas)
ux, ut, ue, un, _ = load_gpu(gurban)
vx, vt, ve, vn, vd = load_gpu(gv2)

def report(name, xs, ts, es, n):
    a0, a1, a2, mx, mt = moments(xs, ts)
    q = quantiles(ts)
    print(f"--- {name} ---")
    print(f"N_exit={n} transmission={n/NINC:.5f}")
    print(f"A0=Var(tx)={a0:.6e} A1=Cov(x,tx)={a1:.6e} A2=Var(x)={a2:.6e}")
    print(f"x RMS={math.sqrt(a2):.6e} xmean={mx:.3e} txmean={mt:.3e}")
    print("q50/68/95/99/99.9(|tx>)=" +
          "/".join(f"{q[k]:.4e}" for k in (0.5, 0.68, 0.95, 0.99, 0.999)))
    print(f"E mean/std={statistics.fmean(es):.4f}/{statistics.pstdev(es):.4f}")

report('TOPAS', tx, tt, te, tn)
report('old-urban', ux, ut, ue, un)
report('urban_v2', vx, vt, ve, vn)

t0, t1, t2, _, _ = moments(tx, tt)
for name, xs, ts in (('old-urban', ux, ut), ('urban_v2', vx, vt)):
    a0, a1, a2, _, _ = moments(xs, ts)
    q, tq = quantiles(ts), quantiles(tt)
    print(f"--- ratio {name}/TOPAS ---")
    print(f"A0={a0/t0:.4f} A1={a1/t1:.4f} A2={a2/t2:.4f} "
          f"xRMS={math.sqrt(a2/t2):.4f} q99={q[0.99]/tq[0.99]:.4f} "
          f"q999={q[0.999]/tq[0.999]:.4f} trans={(len(xs)/NINC)/(tn/NINC):.5f}")

# urban_v2 step diagnostics
steps = sum(int(r['cu_steps']) for r in vd)
below = sum(int(r['disp_below_min']) for r in vd)
acc = sum(int(r['disp_accept']) for r in vd)
red = sum(int(r['disp_reduce']) for r in vd)
can = sum(int(r['disp_cancel']) for r in vd)
cone = sum(int(r['cth_eq_one']) for r in vd)
gs = sum(float(r['cu_g_sum_mm']) for r in vd)
ts_ = sum(float(r['cu_t_sum_mm']) for r in vd)
ds = sum(float(r['cu_delta_sum_mm']) for r in vd)
r2 = sum(float(r['cu_raw_disp_sum2_mm2']) for r in vd)
a2s = sum(float(r['cu_acc_disp_sum2_mm2']) for r in vd)
zero = steps - (below + acc + red + can)
print("--- urban_v2 Cu-step diagnostics ---")
print(f"steps={steps} mean_g={gs/steps:.6f} mean_t={ts_/steps:.6f} "
      f"mean_delta={ds/steps:.3e}")
print(f"raw_disp_RMS={math.sqrt(r2/steps):.6e} acc_disp_RMS={math.sqrt(a2s/steps):.6e}")
print(f"zero_raw={zero/steps:.4e} below_min={below/steps:.4e} accept={acc/steps:.4f} "
      f"reduce={red/steps:.4e} cancel={can/steps:.4e} cth==1={cone/steps:.4f}")

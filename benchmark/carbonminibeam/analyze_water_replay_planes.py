"""Same-source water replay planes: FE-tail vs Water-Urban (frozen entrance).
Per plane: theta RMS/q68/q95/q99/q99.9, pitch-folded x, Cov, fixed-region
fluence (peak/shoulder/valley on 3.6mm pitch fold).
"""
import csv, math, statistics

PITCH = 3.6

def load(path):
    planes = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            d = round(float(r['depth_mm']), 1)
            planes.setdefault(d, []).append(r)
    return planes

def fold(x):
    u = (x + PITCH / 2) % PITCH - PITCH / 2
    return u

def q(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(p * len(s)))] if s else float('nan')

for tag in ('fe', 'urban'):
    planes = load(f'/tmp/opencode/replay/planes_{tag}.csv')
    print(f"===== {tag} =====")
    for d in sorted(planes):
        rows = planes[d]
        tx = [float(r['direction_x']) for r in rows]
        xs = [fold(float(r['x_mm'])) for r in rows]
        n = len(rows)
        mt = statistics.fmean(tx)
        rms = math.sqrt(sum((t - mt) ** 2 for t in tx) / n)
        atx = [abs(t) for t in tx]
        cov = sum((x - statistics.fmean(xs)) * (t - mt) for x, t in zip(xs, tx)) / n
        # fixed regions on folded pitch: peak |u|<0.25, shoulder 0.9-1.4, valley 1.55-1.8
        pk = sum(1 for x in xs if abs(x) < 0.25)
        sh = sum(1 for x in xs if 0.9 < abs(x) < 1.4)
        va = sum(1 for x in xs if 1.55 < abs(x) < 1.8)
        print(f"d={d:6.1f} N={n:7d} thRMS={rms:.6f} q68={q(atx,0.68):.6f} "
              f"q95={q(atx,0.95):.6f} q99={q(atx,0.99):.6f} q999={q(atx,0.999):.6f} "
              f"Cov={cov:.4e} flu P/S/V={pk}/{sh}/{va}")

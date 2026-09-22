#!/usr/bin/env python3
"""Dose gate: 3-seed GPU cuwater EM-only vs TOPAS dose.bin, frozen
acceptance.yaml. Fixed ROI, no registration/renormalization. JSON verdict.

Usage: dose_gate.py --gpu DIR_WITH_s1_s2_s3 --topas BIN --accept YAML --out JSON
Expects <dir>/cuwater_C_s{1,2,3}/dose.raw (float32 1000x1000).
"""
import argparse
import hashlib
import json
import numpy as np
import yaml

NX, NZ = 1000, 1000
DX, DZ = 0.1, 0.25
X0, Z0 = -50.0, 0.0


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--topas", required=True)
    ap.add_argument("--accept", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    acc = yaml.safe_load(open(a.accept))["carbon_em_only_dose"]
    seeds = []
    files = []
    for s in (1, 2, 3):
        p = f"{a.gpu}/cuwater_C_s{s}/dose.raw"
        files.append(p)
        seeds.append(np.fromfile(p, dtype="<f4").reshape((NZ, NX)))
    topas = np.fromfile(a.topas, dtype="<f8").reshape((NZ, NX))
    x = X0 + (np.arange(NX) + 0.5) * DX
    z = Z0 + (np.arange(NZ) + 0.5) * DZ
    peak_m = np.abs(x) < 0.25
    valley_m = np.abs(np.abs(x) - 1.8) < 0.45
    res = {"inputs": {"seeds": files, "shas": [sha(p) for p in files],
                      "topas": a.topas, "topas_sha": sha(a.topas)},
           "rows": []}
    allpass = True
    print(f"{'depth':>8} {'tot':>8} {'peak':>8} {'val':>8} {'pvdr':>8} "
          f"{'peak_se%':>9} {'val_se%':>9}")
    for dep in acc["depths_mm"]:
        iz = int(np.argmin(np.abs(z - dep)))
        T = topas[iz]
        S = np.array([s[iz] for s in seeds])
        row = {"depth_mm": dep}
        # totals (same aperture = whole grid here)
        tm, tse = S.sum(axis=1).mean(), S.sum(axis=1).std(ddof=1) / 3 ** 0.5
        row["total"] = {"gpu": float(tm), "ref": float(T.sum()),
                        "rel": float(tm / T.sum() - 1)}
        # peak / valley row means, seed SE
        pm = S[:, peak_m].mean(axis=1)
        vm = S[:, valley_m].mean(axis=1)
        pm_m, pm_se = pm.mean(), pm.std(ddof=1) / 3 ** 0.5
        vm_m, vm_se = vm.mean(), vm.std(ddof=1) / 3 ** 0.5
        Pm, Vm = T[peak_m].mean(), T[valley_m].mean()
        pv = pm / vm
        pv_m, pv_se = pv.mean(), pv.std(ddof=1) / 3 ** 0.5
        Pv = Pm / Vm
        # reference SE by parity assumption (same N=10M; documented)
        row["peak"] = {"gpu": float(pm_m), "se": float(pm_se),
                       "ref": float(Pm), "rel": float(pm_m / Pm - 1)}
        row["valley"] = {"gpu": float(vm_m), "se": float(vm_se),
                         "ref": float(Vm), "rel": float(vm_m / Vm - 1)}
        row["pvdr"] = {"gpu": float(pv_m), "se": float(pv_se),
                       "ref": float(Pv), "rel": float(pv_m / Pv - 1)}
        # equivalence: |rel| + 2*SE_ratio < gate (ref SE = gpu SE parity)
        def verdict(rel, se_ratio, gate):
            return abs(rel) + 2 * se_ratio < gate
        se_r_peak = pm_se / Pm
        se_r_val = vm_se / Vm
        se_r_pvdr = pv_se / Pv
        se_r_tot = tse / T.sum()
        v_tot = verdict(row["total"]["rel"], se_r_tot, acc["total_deposit_rel"])
        v_peak = verdict(row["peak"]["rel"], se_r_peak, acc["peak_mean_rel"])
        # valley: relative gate + absolute floor for near-zero reference
        v_val = verdict(row["valley"]["rel"], se_r_val, acc["valley_mean_rel"])
        if Vm < acc["valley_abs_floor_Gy_per_history"]:
            v_val = abs(vm_m - Vm) < acc["valley_abs_floor_Gy_per_history"]
        v_pvdr = verdict(row["pvdr"]["rel"], se_r_pvdr, acc["pvdr_rel"])
        row["pass"] = {"total": bool(v_tot), "peak": bool(v_peak),
                       "valley": bool(v_val), "pvdr": bool(v_pvdr)}
        row["pass_all"] = all(row["pass"].values())
        allpass &= row["pass_all"]
        res["rows"].append(row)
        print(f"{dep:8.0f} {row['total']['rel']:+.4f} {row['peak']['rel']:+.4f} "
              f"{row['valley']['rel']:+.4f} {row['pvdr']['rel']:+.4f} "
              f"{100*se_r_peak:9.3f} {100*se_r_val:9.3f} "
              f"{'PASS' if row['pass_all'] else 'FAIL'}")
    # FWHM / R80 on the central axis (depth profiles)
    cx = int(np.argmin(np.abs(x)))
    gp = np.array([s[:, cx] for s in seeds]).mean(axis=0)
    tp = topas[:, cx]
    hm = gp.max() / 2
    above = np.where(gp >= hm)[0]
    fwhm = (above[-1] - above[0]) * DZ if len(above) else float("nan")
    th = np.where(tp >= tp.max() / 2)[0]
    fwhm_r = (th[-1] - th[0]) * DZ if len(th) else float("nan")
    # R80 distal: last z where profile >= 80% max
    g80 = z[np.where(gp >= 0.8 * gp.max())[0][-1]] if len(gp) else float("nan")
    t80 = z[np.where(tp >= 0.8 * tp.max())[0][-1]] if len(tp) else float("nan")
    res["fwhm"] = {"gpu_mm": float(fwhm), "ref_mm": float(fwhm_r),
                   "rel": float(fwhm / fwhm_r - 1)}
    res["r80"] = {"gpu_mm": float(g80), "ref_mm": float(t80),
                  "abs_mm": float(g80 - t80)}
    v_fwhm = abs(res["fwhm"]["rel"]) < acc["fwhm_rel"]
    v_r80 = abs(res["r80"]["abs_mm"]) < acc["r80_abs_mm"]
    res["pass"] = {"fwhm": bool(v_fwhm), "r80": bool(v_r80),
                   "rows": bool(allpass),
                   "all": bool(allpass and v_fwhm and v_r80)}
    print(f"FWHM gpu={fwhm:.2f} ref={fwhm_r:.2f} rel={res['fwhm']['rel']:+.4f} "
          f"{'PASS' if v_fwhm else 'FAIL'}")
    print(f"R80 gpu={g80:.2f} ref={t80:.2f} d={res['r80']['abs_mm']:+.3f} "
          f"{'PASS' if v_r80 else 'FAIL'}")
    print("OVERALL:", "PASS" if res["pass"]["all"] else "FAIL")
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

"""Offline fixed-parameter mass-thickness hypothesis, not GPU physics.
R~Exp(lambda), deposition uniform on [0,R]. Finite-volume convolution of
the baseline 3-D lateral sums. No fitting, dose normalization, or table writes.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.special import exp1
from analyze_longitudinal_holdout import gpu_curve, sha
from compile_schneider_delta_longitudinal import lateral_integral
from verify_schneider_longitudinal_candidate import verify

RHO_REF = 0.01131606474518776

def integrated_cdf(s, lam):
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("Invalid range")
    s = np.asarray(s, dtype=float)
    x = np.maximum(s, 0)/lam
    term = np.zeros_like(x)
    np.multiply(x*x, exp1(x), out=term, where=x>0)
    h = lam*(x + .5*(1-x)*np.exp(-x) + .5*term - .5)
    return np.where(s>0, h, 0.)

def weights(n, pitch, lam):
    if n < 1 or not np.isfinite(pitch) or pitch <= 0:
        raise ValueError("Invalid geometry")
    h = integrated_cdf(np.arange(n+1)*pitch, lam)
    w = np.empty(n)
    w[0] = h[1]/pitch
    if n > 1:
        w[1:] = (h[2:]-2*h[1:-1]+h[:-2])/pitch
    if np.min(w) < -1e-10 or np.sum(w) > 1+1e-10:
        raise ValueError("Nonphysical kernel weights")
    return np.maximum(w, 0.)  # negative cancellation at machine precision only

def predict(source, fraction, lam, pitch=.5):
    source=np.asarray(source,dtype=float)
    if not np.isfinite(source).all() or np.any(source<0) or not 0<=fraction<=.5:
        raise ValueError("Invalid source/fraction")
    w=weights(len(source),pitch,lam)
    moved=np.convolve(source*fraction,w)[:len(source)]
    result=source*(1-fraction)+moved
    escape=float((source*fraction).sum()-moved.sum())
    if escape < -1e-9*max(1.,source.sum()):
        raise ValueError("Negative escape")
    return result, escape

def metrics(curve, reference):
    windows=[]
    for lo,hi in [(2,20),(20,50),(50,100),(100,120)]:
        sl=slice(int(2*lo),int(2*hi))
        windows.append(dict(depth_mm=[lo,hi],relative_error=float(curve[sl].sum()/reference[sl].sum()-1)))
    rebin=[]
    for first in range(4,240,10):
        end=min(first+10,240)
        rebin.append(dict(depth_mm=[first*.5,end*.5],
                          relative_error=float(curve[first:end].sum()/reference[first:end].sum()-1)))
    return dict(windows=windows,rebinned_5mm=rebin,
                max_window_error=max(abs(x["relative_error"]) for x in windows),
                max_rebin_error=max(abs(x["relative_error"]) for x in rebin))

def audit(repo, reference_root, density_root):
    table=repo/"data/schneider/schneider_section0_c12_delta_longitudinal_v1.csv"
    verify(table)
    rows=list(csv.DictReader(table.open()))
    energy=np.array([float(r["energy_MeV_per_u"]) for r in rows])
    fraction=float(np.interp(175,energy,[float(r["forward_fraction"]) for r in rows]))
    lam_ref=float(np.interp(175,energy,[float(r["lambda_mm"]) for r in rows]))
    cases={}
    for name,root in [("hu1000",reference_root),("hu975",density_root/"hu975"),("hu951",density_root/"hu951")]:
        config=json.loads((root/"campaign.json").read_text())
        if config["energy_MeV_u"]!=175 or config["histories_per_run"]!=20000:
            raise ValueError("Unexpected source")
        rho=config["density_g_cm3"]
        base=gpu_curve(root/"gpu_base")/20000
        truth=(lateral_integral(root/"topas_s1/dose.csv")+lateral_integral(root/"topas_s2/dose.csv"))/40000
        pred,escape=predict(base,fraction,lam_ref*RHO_REF/rho)
        # Refinement check preserves the assumed uniform source inside each bin.
        fine,escfine=predict(np.repeat(base/2,2),fraction,lam_ref*RHO_REF/rho,.25)
        fine=fine.reshape(-1,2).sum(axis=1)
        m=metrics(pred,truth)
        m.update(baseline=metrics(base,truth),density_g_cm3=rho,
                 lambda_mm=lam_ref*RHO_REF/rho,
                 escaped_Gy_sum_per_primary=escape,
                 conservation_residual=float(pred.sum()+escape-base.sum()),
                 refinement_max_abs=float(np.max(np.abs(pred-fine))),
                 input_hashes={str(root/p):sha(root/p) for p in [
                     "campaign.json","gpu_base/dose.mhd","gpu_base/dose.raw",
                     "topas_s1/dose.csv","topas_s2/dose.csv"]})
        if name=="hu1000":
            runtime=gpu_curve(root/"gpu_long")/20000
            m["offline_vs_runtime"]=metrics(pred,runtime)
        cases[name]=m
    surrogate=cases["hu1000"]["offline_vs_runtime"]
    surrogate_ok=surrogate["max_window_error"]<=.005 and surrogate["max_rebin_error"]<=.005
    error_ok=all(c["max_window_error"]<=.01 and c["max_rebin_error"]<=.02 for c in cases.values())
    return dict(status=("SURROGATE_INCONCLUSIVE" if not surrogate_ok else
                        "PASS_OFFLINE_HYPOTHESIS_ONLY" if error_ok else "FAIL_OFFLINE_HYPOTHESIS"),
                fraction=fraction,lambda_ref_mm=lam_ref,scale=1,
                table_sha256=sha(table),surrogate_gate=surrogate_ok,prediction_gate=error_ok,cases=cases,
                limitations=["Constant 175 MeV/u parameters along depth; not a per-track replay.",
                             "Uniform per-bin source approximates step midpoints.",
                             "Baseline includes transverse migration/escape; is not raw local ionization.",
                             "No angular/joint or interface validation. No GPU density rule changed.",
                             "Conservation is the convolution accounting identity, not independent physics evidence."])

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--density",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(audit(Path(__file__).resolve().parents[1],a.reference,a.density),indent=2))

#!/usr/bin/env python3
"""Nominal-total-Gy gamma pass rates on full 10%-reference mask.

Trilinear evaluation on a 0.5 mm spherical search lattice, not an exact
continuous minimizer. Zero-DTA is a same-voxel dose difference test.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from run_topas10x_gpu_benchmark import sha, save


def pass_mask(evaluation, reference, pts, spacing, dose_percent, dta, local, step=.5):
    rr=reference[tuple(pts.T)]
    tol=dose_percent*.01*(rr if local else float(reference.max()))
    initial=evaluation[tuple(pts.T)]
    passed=np.square((initial-rr)/tol)<=1
    if dta==0: return passed
    k=int(np.floor(dta/step)); offsets=[]
    for z in range(-k,k+1):
        for y in range(-k,k+1):
            for x in range(-k,k+1):
                dist=step*np.sqrt(x*x+y*y+z*z)
                if 0<dist<=dta+1e-12:
                    offsets.append((dist,np.array([z,y,x])*step/spacing))
    indices=np.flatnonzero(~passed)
    for dist,offset in sorted(offsets,key=lambda v:v[0]):
        if not len(indices): break
        coords=pts[indices].T.astype(float)+offset[:,None]
        valid=np.all((coords>=0)&(coords<=np.array(evaluation.shape)[:,None]-1),axis=0)
        values=map_coordinates(evaluation,coords,order=1,mode="constant",cval=np.nan,prefilter=False)
        denom=tol[indices] if local else tol
        hit=valid & ((dist/dta)**2+np.square((values-rr[indices])/denom)<=1)
        passed[indices[hit]]=True
        indices=indices[~hit]
    return passed


def evaluate(folder):
    m=json.loads((folder/"manifest.json").read_text())
    s=json.loads((folder/"execution.json").read_text())
    if s["status"]!="complete": raise ValueError("Incomplete case")
    if sha(folder/"gpu_sum.raw")!=s["aggregate_sha256"]:raise ValueError("GPU sum SHA mismatch")
    if sha(folder/"topas_sum.raw")!=m["reference_sum_sha256"]:raise ValueError("Reference SHA mismatch")
    if sum(x["histories"] for x in s["completed"])!=m["histories"]:raise ValueError("Incomplete histories")
    g=np.fromfile(folder/"gpu_sum.raw",dtype="<f4").reshape(m["gpu_shape_zyx"])
    if m["mapping"]=="packed_xneg":g=np.flip(g.transpose(1,2,0),axis=2)
    elif m["mapping"]!="native":raise ValueError("Unknown geometry mapping")
    r=np.fromfile(folder/"topas_sum.raw",dtype="<f4").reshape(m["topas_shape_zyx"])
    if r.shape!=g.shape:raise ValueError("Shape mismatch")
    mask=r>=.1*r.max(); pts=np.argwhere(mask)
    if not len(pts): raise ValueError("Empty mask")
    sample=np.random.default_rng(42).choice(len(pts),min(50000,len(pts)),replace=False)
    results={}
    for dd,dta in [(3,3),(2,2),(1,1),(3,0)]:
        for local in [False,True]:
            key=f"{'local' if local else 'global'}_{dd}pct_{dta}mm"
            passed=pass_mask(g,r,pts,np.array(m["spacing_zyx"]),dd,dta,local)
            results[key]={"full_mask_percent":100*float(passed.mean()),"passed":int(passed.sum()),
                          "evaluated":len(pts),"sample50k_percent":100*float(passed[sample].mean())}
            print(m["case"],key,results[key],flush=True)
    out={"case":m["case"],"histories_gpu":m["histories"],"histories_topas":sum(x["histories"] for x in m["replicas"]),
        "dose_scale":1.0,"reference":"sum of raw TOPAS float64 DoseToMedium Sum replicas; float32 export",
        "method":{"threshold":"reference >= 10% maximum","distance_search_step_mm":.5,
                  "interpolation":"trilinear","domain":"inside evaluated volume only; no boundary extrapolation",
                  "global_normalization":"reference maximum","local_normalization":"reference dose at query voxel",
                  "sample_seed":42,"sampling":"all mask voxels, plus frozen default_rng(42) 50k subset",
                  "limitation":"finite search lattice; not exact continuous minimization","mapping":m["mapping"],
                  "zero_dta":"same-voxel dose-only comparison"},
        "gamma":results,"dose_sum_ratio":float(g.sum(dtype=np.float64)/r.sum(dtype=np.float64)),
        "in_mask_dose_sum_ratio":float(g[mask].sum(dtype=np.float64)/r[mask].sum(dtype=np.float64)),
        "high_dose_correlation":float(np.corrcoef(g[mask],r[mask])[0,1]),
        "physics_scope":m["physics"],"queue_overflow_accepted_shards":0,
        "overflow_attempts_excluded":s["overflow_attempts"],
        "source_sha256":sha(__file__),"gpu_sha256":sha(folder/"gpu_sum.raw"),"reference_sha256":sha(folder/"topas_sum.raw")}
    if (folder/"gamma.json").exists():raise ValueError("Refuse overwrite gamma")
    save(folder/"gamma.json",out)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("folder",type=Path)
    evaluate(p.parse_args().folder)

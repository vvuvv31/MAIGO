"""Read-only bidirectional interface baseline audit from 3-D DoseToMedium."""
import argparse,json,re
from pathlib import Path
import numpy as np
from analyze_longitudinal_holdout import sha

def topas(path,nz):
    with path.open() as f:header="".join(next(f) for _ in range(8))
    for axis,n,pitch in [("X",100,.2),("Y",100,.2),("Z",nz,.05)]:
        m=re.search(rf"# {axis} in (\d+) bins of ([\d.]+) cm",header)
        if not m or int(m[1])!=n or float(m[2])!=pitch:raise ValueError("Geometry mismatch")
    if "DoseToMedium ( Gy ) : Sum" not in header:raise ValueError("Wrong quantity")
    x=np.loadtxt(path,delimiter=",",comments="#")
    if x.shape!=(10000*nz,4) or not np.isfinite(x).all() or np.any(x<0):raise ValueError("Invalid scorer")
    xyz=x[:,:3].astype(int)
    if np.any(xyz!=x[:,:3]) or np.any(xyz>=np.array([100,100,nz])):raise ValueError("Invalid indices")
    ids=(xyz[:,2]*100+xyz[:,1])*100+xyz[:,0]
    if np.unique(ids).size!=len(ids):raise ValueError("Duplicate voxel")
    return np.bincount(xyz[:,2],weights=x[:,3],minlength=nz)

def gpu(path,nz=90):
    fields=dict(x.split("=",1) for x in path.read_text().splitlines() if "=" in x)
    fields={k.strip():v.strip() for k,v in fields.items()}
    if fields["DimSize"]!=f"100 100 {nz}" or fields["ElementSpacing"]!="2 2 0.5" or fields["DoseUnits"]!="Gy":raise ValueError("GPU geometry/units")
    if fields["ElementType"]!="MET_FLOAT":raise ValueError("GPU payload type")
    x=np.fromfile(path.parent/fields["ElementDataFile"],dtype="<f4")
    if len(x)!=10000*nz or not np.isfinite(x).all() or np.any(x<0):raise ValueError("GPU payload")
    return x.reshape(nz,100,100).sum(axis=(1,2),dtype=float)

def analyze(root):
    manifest=json.loads((root/"manifest.json").read_text())
    for path,expected in manifest["inputs"].items():
        if sha(root/path)!=expected:raise ValueError("Changed prepared input")
    reports={}
    for name,c in manifest["cases"].items():
        case=root/name
        nz=sum(seg["length_mm"]*2 for seg in c["segments"])
        histories=manifest["histories"]
        refs=[]
        for seed in ("s1","s2"):
            refs.append(np.concatenate([topas(case/seed/f"dose{i}.csv",seg["length_mm"]*2) for i,seg in enumerate(c["segments"])])/histories)
        truth=(refs[0]+refs[1])/2
        actual=gpu(case/"gpu/dose.mhd",nz)/histories
        rho=np.concatenate([np.full(seg["length_mm"]*2,manifest["materials"][str(seg["hu"])]["rho"]) for seg in c["segments"]])
        factor=rho*2e-6/1.602176634e-13
        interface=c["interface_mm"];z=(np.arange(nz)+.5)*.5
        windows=[]
        for lo,hi in [(-5,-2),(-2,0),(0,2),(2,5)]:
            take=(z>=interface+lo)&(z<interface+hi)
            ref=truth[take].sum()
            windows.append(dict(relative_depth_mm=[lo,hi],
                topas_Gy_per_primary=float(ref),gpu_Gy_per_primary=float(actual[take].sum()),
                gpu_relative_error=float(actual[take].sum()/ref-1),
                topas_replica_relative_difference=float(abs(refs[0][take].sum()-refs[1][take].sum())/ref)))
        q=json.loads((case/"gpu/out/run/quality_report.json").read_text())
        inputs=[case/"gpu/dose.raw",case/"gpu/dose.mhd",case/"gpu/out/run/quality_report.json"]
        inputs += [case/s/f"dose{i}.csv" for s in ("s1","s2") for i in (0,1)]
        reports[name]=dict(windows=windows,quality_accepted=q["accepted"],
            interface_profiles=dict(z_relative_mm=(z[(z>=interface-5)&(z<interface+5)]-interface).tolist(),
                topas_Gy_per_primary=truth[(z>=interface-5)&(z<interface+5)].tolist(),
                gpu_Gy_per_primary=actual[(z>=interface-5)&(z<interface+5)].tolist()),
            reference_consistent=all(w["topas_replica_relative_difference"]<=.01 for w in windows),
            total_MeV_per_primary=dict(topas=float((truth*factor).sum()),gpu=float((actual*factor).sum())),
            artifacts={str(p):sha(p) for p in inputs})
    consistent=all(c["reference_consistent"] for c in reports.values())
    return dict(status="BASELINE_INTERFACE_DIAGNOSIS_ONLY" if consistent else "REFERENCE_INCONSISTENT",cases=reports,
                longitudinal_candidate_tested=False,patient_gamma_improvement_proven=False)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);a=p.parse_args()
    print(json.dumps(analyze(a.root),indent=2))

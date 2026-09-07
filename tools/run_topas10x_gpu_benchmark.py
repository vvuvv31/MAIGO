#!/usr/bin/env python3
"""Fresh local GPU benchmark; exact replica histories, fail-closed shard merge."""
import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time
import shutil

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
EXE = REPO / "build/oneapi-nvidia-release/carbon_mc"
BASE = Path("/mnt/sda/wuwei/rt06423_delta_tail_escape_full20/configs/rt06423_delta_tail_01.yaml")


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024**2),b""): h.update(b)
    return h.hexdigest()


def save(p, x):
    Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")


def config_write(p,c):
    # Native parser accepts simple scalar YAML, not Python/null spellings.
    # yaml.safe_load turns an intentionally empty optional output into None.
    # Preserve the empty value on a second write, never a literal file "None".
    p.write_text("\n".join(f"{k}: {'' if v is None else str(v).lower() if isinstance(v,bool) else v}" for k,v in c.items())+"\n")


def header(p):
    with Path(p).open("rb") as f: h=struct.unpack("<5I6f",f.read(44))
    if h[0]!=0x47544343 or h[1]!=3: raise ValueError("CCTG must use Schneider v3")
    return h


def prepare(root,case):
    folder=root/case
    if folder.exists(): raise ValueError("Refuse existing case "+str(folder))
    folder.mkdir(parents=True)
    common=yaml.safe_load(BASE.read_text())
    bench=REPO/"benchmark/topas10x"
    rebase_mm=[0.0,0.0,0.0]
    if case=="RT06423":
        source=bench/case; replicas=[source/"replicas"/f"s{i}" for i in range(1,5)]
        grid=source/"patient_ct_tps_90_xneg.bin"; mapping="packed_xneg"
    elif case=="20022516":
        source=bench/case; replicas=[source/"replicas"/f"s{i}" for i in range(1,5)]
        native=source/"patient_ct.bin"; mapping="native"
        h=list(header(native));rebase_mm[2]=-h[7];h[7]=0.0
        grid=root/"20022516_origin0.bin"
        if grid.exists(): raise ValueError("Refuse rebased CT overwrite")
        with native.open("rb") as src, grid.open("xb") as dst:
            src.read(44);dst.write(struct.pack("<5I6f",*h));shutil.copyfileobj(src,dst)
        # Rigid coordinate relabelling only: move CT and all source points by
        # the same +1 mm in Z. Dose/material arrays and relative poses unchanged.
        common.update(tps_apply_topas_patient_placement=False,spots_patient_rot_z_deg=0.0,
            spots_patient_trans_x_mm=-69.6605,spots_patient_trans_y_mm=9.3887,
            spots_patient_trans_z_mm=.0819,
            tps_isocenter_mm=f"[69.6605, -9.3887, {40.9181+rebase_mm[2]}]")
    elif case=="RT07575":
        source=bench/"RT07575_pbs_s1"
        replicas=[bench/f"RT07575_pbs_s{i}" for i in range(1,6)]
        grid=root/"rt07575_packed.bin"; mapping="packed_xneg"
        common.update(spots_patient_trans_x_mm=-42.8515,spots_patient_trans_y_mm=-12.7636,
            spots_patient_trans_z_mm=1.3617,spots_patient_rot_z_deg=90.0,
            spots_ct_axis_min_mm=-104.25)
    else: raise ValueError(case)
    # Exact counts actually requested of TOPAS, not rounded once after multiplying by 10.
    rows=list(csv.DictReader((source/"spots.csv").open()))
    counts=np.zeros(len(rows),dtype=np.int64)
    ref=None; replica_records=[]; pins={}
    import re
    for rep in replicas:
        text=(rep/"run_full_plan.txt").read_text()
        scale=Decimal(re.search(r"u:So/CarbonPBS/HistoriesScale\s*=\s*([0-9.]+)",text)[1])
        rr=list(csv.DictReader((rep/"spots.csv").open()))
        if rr!=rows or sha(rep/"beam_model.csv")!=sha(source/"beam_model.csv"):
            raise ValueError("Replica sources differ")
        n=np.array([int((Decimal(r["weight"])*scale).to_integral_value(rounding=ROUND_HALF_UP)) for r in rr])
        counts+=n
        raw=rep/"OSMK_Dtotal_full_plan.bin"
        bh=(rep/"OSMK_Dtotal_full_plan.binheader").read_text()
        dims=[int(re.search(rf"# {a} in (\d+) bins",bh)[1]) for a in "XYZ"]
        spacing=[10*float(re.search(rf"# {a} in \d+ bins of ([0-9.eE+-]+) cm",bh)[1]) for a in "XYZ"]
        # TOPAS binary Sum is float64; exported MHD dose.raw is float32.
        if raw.stat().st_size != int(np.prod(dims))*8:
            raise ValueError("Unexpected TOPAS Sum binary byte count")
        data=np.fromfile(raw,dtype="<f8").reshape(tuple(reversed(dims)))
        if not np.isfinite(data).all() or np.any(data<0): raise ValueError("Bad reference")
        if ref is None: ref=data.astype(np.float64); ref_dims=dims; ref_spacing=spacing
        else:
            if dims!=ref_dims or spacing!=ref_spacing: raise ValueError("Replica grids differ")
            ref+=data
        replica_records.append({"path":str(rep),"scale":str(scale),"histories":int(n.sum())})
        for p in [raw,rep/"OSMK_Dtotal_full_plan.binheader",rep/"run_full_plan.txt",rep/"spots.csv",rep/"beam_model.csv"]:
            pins[str(p)]=sha(p)
    ref.astype("<f4").tofile(folder/"topas_sum.raw")
    gh=header(grid); nx,ny,nz=gh[2:5]
    if mapping=="packed_xneg":
        if [nz,nx,ny]!=ref_dims: raise ValueError("Packed shape mismatch")
        if not np.allclose([gh[10],gh[8],gh[9]],ref_spacing): raise ValueError("Packed spacing mismatch")
    elif [nx,ny,nz]!=ref_dims: raise ValueError("Native shape mismatch")
    # Header owns volume dimensions; preserve source placement, no dose-based registration.
    for k in list(common):
        if k.startswith("voxel_bins_") or k.startswith("voxel_size_") or k in ("depth_bin_width_mm","phantom_length_mm","scorer_area_mm2"):
            del common[k]
    common.update(ct_grid_file=str(grid),tps_beam_model_file=str(source/"beam_model.csv"),
                  tps_histories_scale=1.0,LET=False)
    common.pop("LET_type",None);common.pop("LET_name",None)
    # Resolve all repository data inputs before placing configs outside the repository.
    for k,v in list(common.items()):
        if isinstance(v,str) and (v.startswith("data/") or v.startswith("benchmark/")):
            common[k]=str(REPO/v)
    for k,v in common.items():
        if isinstance(v,str) and Path(v).is_file(): pins[v]=sha(v)
    pins[str(EXE)]=sha(EXE)
    pins[str(REPO/"src/transport_sycl.cpp")]=sha(REPO/"src/transport_sycl.cpp")
    tasks=[]
    for i in range(20):
        # Rotate remainder recipients per spot to balance shards; exact integer conservation.
        alloc=counts//20+(((i-np.arange(len(counts)))%20)<counts%20)
        d=folder/f"shard_{i+1:02d}";d.mkdir()
        spots=d/"spots.csv"
        with spots.open("w") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
            for r,n in zip(rows,alloc):
                if n: w.writerow(dict(r,weight=int(n)))
        c=dict(common,number_of_histories=int(alloc.sum()),random_seed=9000000000+i*1000003,
               tps_spots_file=str(spots),dose_to_medium_name="dose")
        config_write(d/"config.yaml",c)
        tasks.append({"directory":str(d),"histories":int(alloc.sum())})
        for p in [spots,d/"config.yaml"]: pins[str(p)]=sha(p)
    assert sum(t["histories"] for t in tasks)==int(counts.sum())
    save(folder/"manifest.json",{"case":case,"histories":int(counts.sum()),"mapping":mapping,
        "gpu_shape_zyx":[nz,ny,nx],"topas_shape_zyx":list(ref.shape),"spacing_zyx":ref_spacing[::-1],
        "replicas":replica_records,"shards":tasks,"input_sha256":pins,
        "reference_sum_sha256":sha(folder/"topas_sum.raw"),"physics":"current v2.1 plus existing entrance-mask candidate; research only",
        "source_dirty":True,"dose_scale":1.0,"coordinate_rebase_mm":rebase_mm,
        "coordinate_rebase_reason":"CT and source translate together; no relative displacement or dose registration"})
    print("prepared",case,int(counts.sum()),flush=True)


def run(root,case,resume=False):
    folder=root/case;m=json.loads((folder/"manifest.json").read_text())
    if (folder/"execution.json").exists():
        if not resume: raise ValueError("Refuse duplicate execution without --resume")
        old=json.loads((folder/"execution.json").read_text())
        if old["status"]=="complete": raise ValueError("Case already complete")
        shutil.copy2(folder/"execution.json",folder/f"execution_previous_{time.time_ns()}.json")
    subprocess.run(["python3","tools/verify_schneider_v2_1_data.py"],cwd=REPO,check=True)
    for p,h in m["input_sha256"].items():
        if sha(p)!=h: raise ValueError("Input changed: "+p)
    state={"status":"running","case":case,"completed":[],"overflow_attempts":[]}
    save(folder/"execution.json",state)
    total=np.zeros(m["gpu_shape_zyx"],dtype=np.float64)
    def task(t,depth=0):
        d=Path(t["directory"]);started=time.monotonic()
        recovered=resume and all((d/x).exists() for x in ("dose.raw","dose.mhd","quality_report.json","energy_ledger.json"))
        if not recovered:
            # Production ledger location is out/<config stem>, independent of
            # --voxel-dose-mhd. Use a unique stem and collect those records.
            unique="bench_"+root.name+"_"+case+"_"+"_".join(d.relative_to(folder).parts)
            runtime_config=d/(unique+".yaml")
            qdir=REPO/"out"/unique
            if qdir.exists(): raise ValueError("Refuse existing quality directory "+str(qdir))
            shutil.copy2(d/"config.yaml",runtime_config)
            with (d/"gpu.log").open("x") as log:
                proc=subprocess.run([str(EXE),"--config",str(runtime_config),"--device","cuda",
                                     "--voxel-dose-mhd",str(d/"dose.mhd")],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            if proc.returncode: raise RuntimeError("GPU exited nonzero: "+str(d))
            for output in qdir.glob("*.json"): shutil.copy2(output,d/output.name)
        q=json.loads((d/"quality_report.json").read_text());l=json.loads((d/"energy_ledger.json").read_text())
        if q["queue_overflow_count"] or q["queue_overflow_energy_MeV"]:
            state["overflow_attempts"].append(str(d));save(folder/"execution.json",state)
            if depth>=5: raise RuntimeError("Overflow persists after 5 splits")
            c=yaml.safe_load((d/"config.yaml").read_text()); rr=list(csv.DictReader((d/"spots.csv").open()))
            for half in range(2):
                sub=d/f"split_{half}";sub.mkdir()
                counts=[int(r["weight"])//2+(int(r["weight"])%2 if half else 0) for r in rr]
                if sum(counts)==0: continue
                with (sub/"spots.csv").open("w") as f:
                    w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader()
                    for r,n in zip(rr,counts):
                        if n:w.writerow(dict(r,weight=n))
                cc=dict(c,number_of_histories=sum(counts),tps_spots_file=str(sub/"spots.csv"),
                        random_seed=c["random_seed"]+1000000007+half,dose_to_medium_name="dose")
                config_write(sub/"config.yaml",cc)
                task({"directory":str(sub),"histories":sum(counts)},depth+1)
            return
        if not q["accepted"] or q["failures"]: raise RuntimeError("Quality gate rejected: "+str(d)+" "+str(q["failures"]))
        if l["histories"]!=t["histories"]: raise RuntimeError("Actual histories mismatch")
        a=np.fromfile(d/"dose.raw",dtype="<f4").reshape(total.shape)
        if not np.isfinite(a).all() or np.any(a<0): raise ValueError("Invalid dose")
        total[:]+=a
        state["completed"].append(dict(t,seconds=time.monotonic()-started,recovered_existing=recovered,dose_sha256=sha(d/"dose.raw"),
                                      quality_sha256=sha(d/"quality_report.json"),ledger_sha256=sha(d/"energy_ledger.json")))
        save(folder/"execution.json",state)
        print(case,"validated",d.name,len(state["completed"]),"histories",sum(x["histories"] for x in state["completed"]),flush=True)
    try:
        for t in m["shards"]: task(t)
        if sum(x["histories"] for x in state["completed"])!=m["histories"]:raise ValueError("Incomplete aggregate")
        total.astype("<f4").tofile(folder/"gpu_sum.raw")
        first=Path(state["completed"][0]["directory"])/"dose.mhd"
        text=first.read_text()
        import re
        (folder/"gpu_sum.mhd").write_text(re.sub(r"ElementDataFile\s*=.*","ElementDataFile = gpu_sum.raw",text))
        state.update(status="complete",aggregate_sha256=sha(folder/"gpu_sum.raw"))
        save(folder/"execution.json",state)
    except BaseException as e:
        state.update(status="failed",error=str(e));save(folder/"execution.json",state);raise


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=["prepare","run"])
    p.add_argument("--root",type=Path,required=True);p.add_argument("--case",required=True,choices=["RT06423","20022516","RT07575"])
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    if a.mode=="run":run(a.root.resolve(),a.case,a.resume)
    else:prepare(a.root.resolve(),a.case)

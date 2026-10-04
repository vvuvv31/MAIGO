#!/usr/bin/env python3
"""Matched RT07575 birth-only ablation; original interface fallback remains."""
import argparse,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'benchmark/protonCT/RT07575_minibeam_GPU_schneider_joint_20261003'
sys.path.insert(0,str(BASE));from common import sha,save_json,read_mhd,packed_to_patient
from metrics import roi_indices,dose_metrics
sys.path.insert(0,str(ROOT/'scripts'));from compare_minibeam import AUDITS,integer_fields

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'out/electron_standalone_20261003/rt07575_birth_screen')
    p.add_argument('--shards',type=int,default=2)
    a=p.parse_args();assert 1<=a.shards<=118
    out=a.out.resolve();out.mkdir(exist_ok=False)
    build=ROOT/'out/electron_standalone_20261003/gpu_build.log';deadline=time.time()+1800
    while '[100%] Built target carbon_mc' not in build.read_text():
        if time.time()>deadline:raise RuntimeError('Build did not complete; no run')
        time.sleep(5)
    cache=(ROOT/'build/CMakeCache.txt').read_text();assert 'CARBON_DOSE_FP32:BOOL=ON' in cache and 'CARBON_DOSE_FP64:BOOL=OFF' in cache
    shutil.copy2(ROOT/'build/carbon_mc',out/'carbon_mc');shutil.copy2(build,out/'build.log');(out/'data').symlink_to(ROOT/'data')
    source_files=['src/transport_sycl.cpp','include/carbon/electron_birth_bank.hpp','include/carbon/electron_birth_bank_device.hpp','include/carbon/material_electron_birth_prepare.hpp','src/config.cpp','src/run_quality.cpp']
    save_json(out/'build.json',dict(binary_sha256=sha(out/'carbon_mc'),source_sha256={f:sha(ROOT/f) for f in source_files},FP32=True,FP64=False))
    package=ROOT/'out/electron_standalone_20261003/birth_bank64/birth_bank.bin';meta=json.loads(package.with_name('manifest.json').read_text());assert sha(package)==meta['sha256']
    names=[f'{arm}{i:04d}' for i in range(a.shards) for arm in (['control','fresh'] if i%2==0 else ['fresh','control'])];manifest=[]
    for name in names:
        index=int(name[-4:]);config=(BASE/f'batch{index:04d}.yaml').read_text()
        if name.startswith('fresh'):config+=f'\nmaterial_electron_birth_file: {package}\nmaterial_electron_birth_sha256: {meta["sha256"]}\n'
        (out/(name+'.yaml')).write_text(config);manifest.append(name+'.yaml')
    (out/'plan.manifest').write_text('\n'.join(manifest)+'\n')
    env=dict(os.environ,LD_LIBRARY_PATH='/home/wuwei/sycl_workspace/llvm/build/install/lib:'+os.environ.get('LD_LIBRARY_PATH',''),CARBON_RUNTIME_BREAKDOWN='1')
    start=time.perf_counter()
    with (out/'gpu.log').open('x') as log:r=subprocess.run([str(out/'carbon_mc'),'--plan-manifest',str(out/'plan.manifest')],cwd=out,stdout=log,stderr=subprocess.STDOUT,env=env)
    save_json(out/'execution.json',dict(returncode=r.returncode,wall_seconds=time.perf_counter()-start))
    assert r.returncode==0,('GPU screen failed',r.returncode)
    text=(out/'gpu.log').read_text();chunks=text.split('[plan-shard] ')[1:];assert len(chunks)==len(names)
    quality={};work={};timing={}
    for name,chunk in zip(names,chunks):
        q=json.loads((out/'out'/name/'quality_report.json').read_text());quality[name]=q;assert q['accepted'] and q['queue_overflow_count']==0 and not q['failures']
        line=next(l for l in chunk.splitlines() if l.startswith('[electron-joint-pilot-work]'));w=dict((k,int(v)) for k,v in re.findall(r'(\w+)=(\d+)',line));assert w['invalid']==0;work[name]=w
        if name.startswith('fresh'):assert '[independent-electron-birth]' in chunk and 'legacy_birth_reads=0' in chunk
        timing[name]=dict(transport_seconds=float(re.search(r'\[runtime-stage\] transport_loop_including_scoring_and_queue_transfers seconds=(\S+)',chunk)[1]),electron_seconds=float(re.search(r'\[electron-queue-time\] seconds=(\S+)',chunk)[1]))
    def exact_audits(a,b):
        da=json.loads((out/'out'/a/'energy_ledger.json').read_text());db=json.loads((out/'out'/b/'energy_ledger.json').read_text())
        ia=integer_fields(da);ib=integer_fields(db)
        ca=chunks[names.index(a)];cb=chunks[names.index(b)]
        return dict(primary_audits_equal=[l for l in ca.splitlines() if l.startswith(AUDITS)]==[l for l in cb.splitlines() if l.startswith(AUDITS)],integer_deltas={k:[ia.get(k),ib.get(k)] for k in ia.keys()|ib.keys() if ia.get(k)!=ib.get(k)})
    audits=[exact_audits(f'control{i:04d}',f'fresh{i:04d}') for i in range(a.shards)]
    assert all(a['primary_audits_equal'] for a in audits)
    dose={name:read_mhd(out/'out'/name/'dose.mhd')[0] for name in names}
    x=sum(dose[f'control{i:04d}'].astype(float) for i in range(a.shards));y=sum(dose[f'fresh{i:04d}'].astype(float) for i in range(a.shards))
    previous=read_mhd(BASE/'out/batch0000/dose.mhd')[0];regression=float(abs(dose['control0000'].astype(float)-previous).sum()/previous.sum());assert regression<1e-5,regression
    rois=roi_indices();body=np.zeros((35,417,505),bool);body.reshape(-1)[np.union1d(rois['BODY'],rois['GTV'])]=True;body=np.flip(body.transpose(1,0,2),axis=0)
    xx=-169.1015+(np.arange(505)+.5)*.5;yy=-33.6383+(np.arange(35)+.5)*2;zz=(np.arange(417)+.5)*.5;phase=(xx+1.8)%3.6-1.8
    domain=body&(abs(xx)[None,None,:]<=14.4)&(abs(yy)[None,:,None]<=10);pv={};pv_se={}
    for lo,hi in [(20,40),(40,80),(80,120),(120,160)]:
        for label,sel in [('peak',abs(phase)<=.25),('valley',abs(abs(phase)-1.8)<=.25)]:
            mask=domain&((zz>=lo)&(zz<hi))[:,None,None]&sel[None,None,:]
            pv[f'{lo}-{hi}mm_{label}']=float(100*(y[mask].sum()/x[mask].sum()-1))
            paired=[100*(dose[f'fresh{i:04d}'][mask].sum(dtype=float)/dose[f'control{i:04d}'][mask].sum(dtype=float)-1) for i in range(a.shards)]
            pv_se[f'{lo}-{hi}mm_{label}']=float(np.std(paired,ddof=1)/np.sqrt(a.shards)) if a.shards>1 else None
    local={}
    for threshold in [.01,.1]:
        mask=body&(x>=threshold*x[body].max());r=abs(y[mask]-x[mask])/x[mask]
        local[str(threshold)]=dict(voxels=int(mask.sum()),pass_1percent=float(np.mean(r<=.01)),pass_3percent=float(np.mean(r<=.03)),maximum_percent=float(100*r.max()),percentiles_percent=(100*np.quantile(r,[.5,.95,.99])).tolist())
    report=dict(status='FRESH_BIRTH_ABLATION_WITH_LEGACY_INTERFACE_TRANSPORT',histories_per_arm=1983920*a.shards,quality=quality,work=work,timing=timing,primary_audits=audits,disabled_regression_relative_L1=regression,whole_grid_relative_L1=float(abs(x-y).sum()/x.sum()),BODY_relative_L1=float(abs(x[body]-y[body]).sum()/x[body].sum()),peak_valley_change_percent=pv,peak_valley_sampling_SE_percent=pv_se,local_change=local,scope='New82-node birth bank plus Schneider25 joint transport. Old birth samples/track binding bypassed; old library retained for boundary continuation. Preliminary paired screen, not full independence or1% acceptance. SE measures runtime sampling only, not finite training uncertainty.')
    save_json(out/'report.json',report);print(json.dumps({k:report[k] for k in ['status','timing','whole_grid_relative_L1','BODY_relative_L1','peak_valley_change_percent']},indent=2),flush=True)
if __name__=='__main__':main()

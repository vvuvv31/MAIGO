#!/usr/bin/env python3
"""Checkpointed group32 full plan; stop on any rejected shard, never renormalize to dose."""
import argparse, fcntl, json, os, re, subprocess, time
from pathlib import Path
import numpy as np
from common import ROOT, sha, save_json, read_mhd, write_mhd, packed_to_patient
from metrics import roi_indices, dose_metrics


def aggregate(jobs, manifest, name, wall):
    values=np.zeros((417,35,505),dtype=np.float64)
    for job in jobs:
        dose,fields=read_mhd(ROOT/'out'/job['name']/'dose.mhd');assert dose.shape==values.shape
        values+=dose
    histories=sum(j['histories'] for j in jobs)
    reference,ref_fields=read_mhd(ROOT/'inputs/physical_dose.mhd')
    patient=packed_to_patient(values)*manifest['protons_4Gy']/histories
    directory=ROOT/name;directory.mkdir(exist_ok=True)
    write_mhd(directory/'dose_4Gy.mhd',patient,ref_fields)
    rois=roi_indices();gm=dose_metrics(patient,rois);rm=dose_metrics(reference,rois)
    selected=np.union1d(rois['BODY'],rois['GTV'])
    g=patient.transpose(0,2,1).reshape(-1)[selected];r=reference.transpose(0,2,1).reshape(-1)[selected]
    report=dict(state='complete' if histories==manifest['histories'] else 'partial',histories=histories,requested_histories=manifest['histories'],parts=len(jobs),wall_seconds=wall,histories_per_second=histories/wall if wall else None,
        scale=manifest['protons_4Gy']/histories,normalization='incident optimized protons / simulated protons; no fit',gpu_metrics=gm,reference_metrics=rm,body_relative_L1=float(abs(g-r).sum()/r.sum()),dose_difference_0mm={},scope=manifest['scope'])
    cut=r>=reference.max()*.1
    for pct in [1,2,3]:
        report['dose_difference_0mm'][f'local_{pct}pct']=float(100*np.mean(abs(g[cut]-r[cut])<=pct/100*r[cut]))
        report['dose_difference_0mm'][f'global_{pct}pct']=float(100*np.mean(abs(g[cut]-r[cut])<=pct/100*reference.max()))
    report['dose_difference_cutoff']='BODY union GTV; reference >= 10% whole-reference maximum'
    save_json(directory/'summary.json',report)
    return report


def validate(job,log):
    directory=ROOT/'out'/job['name'];q=json.loads((directory/'quality_report.json').read_text())
    assert q.get('accepted') and q.get('queue_overflow_count')==0,(job['name'],q)
    assert any(v['code']=='unvalidated_material_electron_joint_pilot' for v in q['approximations'])
    dose,fields=read_mhd(directory/'dose.mhd')
    assert dose.shape==(417,35,505) and np.isfinite(dose).all() and dose.min()>=0 and dose.max()>0
    assert f"Histories: {job['histories']}" in log
    return dict(name=job['name'],histories=job['histories'],quality=q,dose_sha256=sha(directory/'dose.raw'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',choices=['100k','2m']);p.add_argument('--group',type=int,default=4);args=p.parse_args()
    lock=(ROOT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((ROOT/'manifest.json').read_text());binary=ROOT/'inputs/carbon_mc';identity=json.loads((ROOT/'build.json').read_text());assert sha(binary)==identity['binary_sha256']
    env=os.environ.copy();env['LD_LIBRARY_PATH']='/home/wuwei/sycl_workspace/llvm/build/install/lib:'+env.get('LD_LIBRARY_PATH','');env['CARBON_RUNTIME_BREAKDOWN']='1'
    assert not any(k.startswith('CARBON_') and k!='CARBON_RUNTIME_BREAKDOWN' for k in env),'Unexpected kernel override'
    jobs=manifest['jobs'] if not args.pilot else [j for j in manifest['pilots'] if j['name']=='pilot_'+args.pilot]
    accepted=[];wall=0.;statuspath=ROOT/('campaign_status.json' if not args.pilot else 'pilot_'+args.pilot+'_status.json')
    for job in jobs:
        assert sha(ROOT/(job['name']+'.yaml'))==job['config_sha256']
        assert sha(ROOT/'sources'/(job['name']+'.txt'))==job['source_sha256']
    start=time.time()
    try:
        for offset in range(0,len(jobs),args.group):
            group=jobs[offset:offset+args.group];name=('pilot_'+args.pilot if args.pilot else f'group{offset//args.group:03d}')
            recordpath=ROOT/(name+'_accepted.json')
            if recordpath.exists():
                record=json.loads(recordpath.read_text())
                assert record['names']==[j['name'] for j in group]
                for j,a in zip(group,record['accepted']):assert sha(ROOT/'out'/j['name']/'dose.raw')==a['dose_sha256']
                accepted+=group;wall+=record['wall_seconds'];continue
            batch=ROOT/(name+'.manifest');batch.write_text('\n'.join(j['name']+'.yaml' for j in group)+'\n')
            logpath=ROOT/(name+'.log');t=time.perf_counter()
            save_json(statuspath,dict(state='running',completed_histories=sum(j['histories'] for j in accepted),requested_histories=manifest['histories'],current_group=name,pid=os.getpid(),started_unix=start))
            with logpath.open('w') as log:
                proc=subprocess.run([str(binary),'--plan-manifest',str(batch)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            elapsed=time.perf_counter()-t;text=logpath.read_text()
            assert proc.returncode==0,(name,proc.returncode)
            work=[dict((k,int(v)) for k,v in re.findall(r'(\w+)=(\d+)',line)) for line in text.splitlines() if line.startswith('[electron-joint-pilot-work]')]
            assert len(work)==len(group) and all(w['started']>0 and w['invalid']==0 for w in work),work
            save_json(ROOT/(name+'_joint_work.json'),work)
            assert [int(n) for n in re.findall(r'^Histories: (\d+)$',text,re.M)]==[j['histories'] for j in group]
            audits=[validate(j,text) for j in group]
            save_json(recordpath,dict(names=[j['name'] for j in group],accepted=audits,wall_seconds=elapsed,binary_sha256=identity['binary_sha256']))
            accepted+=group;wall+=elapsed
            summary=aggregate(accepted,manifest,'pilot_'+args.pilot+'_results' if args.pilot else 'results',wall)
            save_json(statuspath,dict(state='pilot_complete' if args.pilot else ('transport_complete' if summary['state']=='complete' else 'running'),completed_histories=summary['histories'],requested_histories=manifest['histories'],completed_parts=len(accepted),wall_seconds=wall,histories_per_second=summary['histories_per_second'],remaining_seconds=(manifest['histories']-summary['histories'])/summary['histories_per_second'],pid=os.getpid()))
            print(json.dumps(dict(group=name,histories=summary['histories'],speed=summary['histories_per_second'],GTV=summary['gpu_metrics']['GTV'])),flush=True)
        if not args.pilot:
            summary=aggregate(accepted,manifest,'results',wall)
            subprocess.run([os.sys.executable,str(ROOT/'compare.py')],cwd=ROOT,check=True)
            save_json(statuspath,dict(state='complete',completed_histories=summary['histories'],requested_histories=manifest['histories'],completed_parts=len(accepted),wall_seconds=wall,histories_per_second=summary['histories_per_second']))
    except BaseException as error:
        save_json(statuspath,dict(state='failed',error=str(error),completed_histories=sum(j['histories'] for j in accepted),pid=os.getpid()))
        raise
if __name__=='__main__':main()

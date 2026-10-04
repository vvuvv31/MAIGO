#!/usr/bin/env python3
"""Finish the already-running bank, then run full-response GPU checks and comparison.

Never advances on missing/failed tables, mismatched projectile, or failed GPU quality.
"""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'out/carbon_ct_minibeam_20261002'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
    p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bank-pids',nargs='+',type=int,required=True);args=parser.parse_args()
    env=os.environ.copy();env['LD_LIBRARY_PATH']='/home/wuwei/sycl_workspace/llvm/build/install/lib:'+env.get('LD_LIBRARY_PATH','')
    env['PYTHONPATH']='/mnt/sda/wuwei/maigo_gamma_python_deps';env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
    frozen=OUT/'candidate_carbon_mc'
    import shutil
    if frozen.exists():raise RuntimeError('Refuse to overwrite frozen campaign binary')
    shutil.copy2(ROOT/'build/carbon_mc',frozen)
    cache=(ROOT/'build/CMakeCache.txt').read_text();assert 'CARBON_DOSE_FP32:BOOL=ON' in cache and 'CARBON_DOSE_FP64:BOOL=OFF' in cache
    save(OUT/'campaign_binary.json',dict(binary_sha256=sha(frozen),transport_source_sha256=sha(ROOT/'src/transport_sycl.cpp'),
        build_cache_sha256=sha(ROOT/'build/CMakeCache.txt')))
    jobs=json.loads((OUT/'bank_jobs.json').read_text());deadline=time.monotonic()+6*3600
    while True:
        done=sum((Path(j['path'])/'complete.json').exists() for j in jobs)
        save(OUT/'campaign_status.json',dict(stage='building_c12_bank',compiled=done,total=len(jobs),accepted=False))
        if done==len(jobs):break
        alive=[]
        for pid in args.bank_pids:
            try:os.kill(pid,0);alive.append(pid)
            except ProcessLookupError:pass
        if not alive:raise RuntimeError('Bank workers stopped with incomplete tables')
        if time.monotonic()>deadline:raise RuntimeError('Bank completion timeout')
        time.sleep(20)
    subprocess.run([sys.executable,str(HERE/'prepare_bank.py'),'--workers','1'],cwd=ROOT,env=env,check=True)
    subprocess.run([sys.executable,str(HERE/'prepare_gpu.py')],cwd=ROOT,env=env,check=True)
    base=yaml.safe_load((ROOT/'config/carbon_ct_minibeam.yaml').read_text())
    for k,v in base.items():
        if isinstance(v,str) and (v.startswith('data/') or v.startswith('inputs/')):base[k]=str(ROOT/v)
    for mode in ['combined','compact']:
        save(OUT/'campaign_status.json',dict(stage=f'gpu_full_response_handoff_{mode}',histories=1000,accepted=False))
        c=dict(base,minibeam_primary_pipeline=mode);config=OUT/f'full_response_{mode}.yaml';config.write_text(yaml.safe_dump(c,sort_keys=False))
        d=OUT/f'full_response_{mode}'
        with (OUT/f'full_response_{mode}.log').open('x') as log:
            subprocess.run([str(frozen),'--config',str(config),'--histories-per-batch','1000','--batch-count','1',
                '--batch-output',str(d)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    ds=[OUT/f'full_response_{mode}'/'batch000000' for mode in ['combined','compact']]
    q=[json.loads((d/'quality_report.json').read_text()) for d in ds]
    assert all(a['accepted'] and a['queue_overflow_count']==0 for a in q)
    a,b=[np.fromfile(d/'dose.raw',np.float32).astype(float) for d in ds]
    l1=float(abs(a-b).sum()/max(a.sum(),1e-30));peak=float(abs(a-b).max()/max(a.max(),1e-30))
    assert l1<1e-6 and peak<1e-6
    save(OUT/'full_response_handoff.json',dict(histories=1000,dose_relative_l1=l1,peak_normalized_max_difference=peak,quality=q))
    save(OUT/'campaign_status.json',dict(stage='gpu_full_response_2x100k',accepted=False,binary_sha256=sha(frozen)))
    with (OUT/'gpu_batches.log').open('x') as log:
        subprocess.run([str(frozen),'--config',str(ROOT/'config/carbon_ct_minibeam.yaml'),
            '--histories-per-batch','100000','--batch-count','2','--batch-output',str(OUT/'gpu_batches')],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    for i in [1,2]:
        s=json.loads((OUT/f'topas_b{i:02d}'/'status.json').read_text());assert s['accepted']
    with (OUT/'comparison.log').open('x') as log:
        subprocess.run([sys.executable,str(HERE/'compare.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    save(OUT/'campaign_status.json',dict(stage='initial_comparison_complete',accepted=False,
        note='Initial 200k per engine comparison; finite-statistics dose acceptance remains to be assessed',
        binary_sha256=sha(frozen),comparison=str(OUT/'comparison.json')))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(OUT/'campaign_status.json',dict(stage='failed',accepted=False,error=str(e)))
        raise

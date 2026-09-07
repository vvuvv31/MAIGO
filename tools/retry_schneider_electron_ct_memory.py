"""Retry only confirmed OOM jobs in new directories; keep all original evidence."""
import argparse,json,re,subprocess
from pathlib import Path
from analyze_longitudinal_holdout import sha
from execute_electron_response_campaign import queue_usage

def retry(root,out):
    m=json.loads((root/'manifest.json').read_text());failed=[]
    for section,c in m['cases'].items():
        directory=Path(c['path']).parent;logs=list(directory.glob('job_*.log'))
        if len(logs)!=1:raise ValueError('Ambiguous job identity')
        jid=re.fullmatch(r'job_(\d+)\.log',logs[0].name)[1]
        rows=subprocess.check_output(['sacct','-j',jid,'--format=JobIDRaw,State','--noheader','--parsable2'],text=True).splitlines()
        state=next(row.split('|')[1] for row in rows if row.split('|')[0]==jid)
        if state=='OUT_OF_MEMORY':failed.append((section,c,directory,jid))
        elif state not in ('COMPLETED','RUNNING','PENDING'):raise ValueError('Unexpected failure: '+state)
    if not failed:raise ValueError('No confirmed OOM jobs')
    q=subprocess.check_output(['squeue','-u','wuwei','-h','-o','%i|%C|%D|%m'],text=True)
    cpu,mem=queue_usage(q)
    if cpu+4*len(failed)>192 or mem+8*len(failed)*1024**3>160*1024**3:raise ValueError('Retry budget unavailable')
    out.mkdir(exist_ok=False);jobs=[]
    for section,c,old,jid in failed:
        dest=out/old.parent.name/old.name;dest.mkdir(parents=True)
        for name in ('run.txt','run.slurm'):
            text=(old/name).read_text().replace(str(old),str(dest))
            if name=='run.slurm':text=text.replace('--mem=4G','--mem=8G')
            (dest/name).write_text(text)
        m['cases'][section]=dict(c,path=str(dest/'steps.phsp'),slurm_file=str(dest/'run.slurm'),mem_GiB=8,replaces_failed_job=jid)
        jobs.append(dest/'run.slurm')
    m['inputs'].update({str(p):sha(p) for p in out.rglob('*') if p.is_file()})
    m['inputs'][str(root/'manifest.json')]=sha(root/'manifest.json')
    m['retry_submission']=dict(jobs=len(jobs),existing_cpus=cpu,existing_mem_bytes=mem,total_cpus=cpu+4*len(jobs),total_mem_bytes=mem+8*len(jobs)*1024**3)
    with (out/'manifest.json').open('x') as f:json.dump(m,f,indent=2)
    for job in jobs:print(subprocess.check_output(['sbatch',str(job)],text=True).strip(),job.parent.parent.name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();retry(a.root.resolve(),a.out.resolve())

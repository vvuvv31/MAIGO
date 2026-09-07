"""Independent G4_AIR entrance-moment campaign, local Slurm, bounded lanes."""
import argparse,csv,json,re,subprocess
from pathlib import Path
import numpy as np
from run_topas10x_gpu_benchmark import sha

ROOT=Path('/mnt/sda/wuwei/upstream_air_moments_r2_20260906')

def prepare():
    ROOT.mkdir(exist_ok=False)
    source=Path('/mnt/sda/wuwei/ct_entrance_phase_r2_20260906/run.txt')
    t=source.read_text()
    t='\n'.join(l for l in t.splitlines() if 'Ge/Patient/' not in l and not l.startswith('includeFile'))+'\n'
    assert 'Ge/Patient/' not in t
    bm=ROOT/'beam_model.csv'
    bm.write_text('energy_MeV,sigma_x_mm,sigma_xp_rad,corr_x,sigma_y_mm,sigma_yp_rad,corr_y,energy_spread_percent\n'+''.join(f'{12*E},0,0,0,0,0,0,0\n' for E in range(110,211,10)))
    cases=[]
    for E in range(110,211,10):
        for L in (300,315):
            d=ROOT/f'e{E}_l{L}';d.mkdir();threads=4*round(E/35)
            (d/'spots.csv').write_text(f'spot_id,x_mm,y_mm,energy_MeV,weight\n1,0,0,{12*E},50000\n')
            text=re.sub(r'i:Ts/NumberOfThreads = \d+',f'i:Ts/NumberOfThreads = {threads}',t)
            text=re.sub(r'i:Ts/Seed = \d+',f'i:Ts/Seed = {9280000+E*10+L}',text)
            text=re.sub(r's:So/CarbonPBS/SpotPlanFile = .*',f's:So/CarbonPBS/SpotPlanFile = "{d}/spots.csv"',text)
            text=re.sub(r's:So/CarbonPBS/BeamModelFile = .*',f's:So/CarbonPBS/BeamModelFile = "{bm}"',text)
            text=re.sub(r'd:Ge/Entrance/TransY = .*',f'd:Ge/Entrance/TransY = {-450+L-.001} mm',text)
            text=re.sub(r's:Sc/Entrance/OutputFile = .*',f's:Sc/Entrance/OutputFile = "{d}/entrance"',text)
            (d/'run.txt').write_text(text)
            (d/'run.slurm').write_text(f'#!/bin/bash\n#SBATCH --job-name=air_{E}_{L}\n#SBATCH --partition=compute\n#SBATCH --nodes=1\n#SBATCH --cpus-per-task={threads}\n#SBATCH --mem=8G\n#SBATCH --time=00:30:00\n#SBATCH --output={d}/job_%j.log\n#SBATCH --error={d}/job_%j.err\nset -euo pipefail\ncd {d}\n/home/wuwei/topas/topas-build/topas {d}/run.txt\n')
            cases.append(dict(energy=E,length=L,directory=str(d),threads=threads))
    pins={str(p):sha(p) for p in ROOT.rglob('*') if p.is_file()}
    pins[str(source)]=sha(source);exe=Path('/home/wuwei/topas/topas-build/topas');pins[str(exe)]=sha(exe)
    m=dict(cases=cases,inputs=pins,status='PREPARED_EXPERIMENT',physics='G4_AIR, g4em-standard_opt4 + g4decay; no CT in measurement world',max_concurrent_jobs=7)
    (ROOT/'manifest.json').write_text(json.dumps(m,indent=2))
    lanes=[None]*7
    for i,c in enumerate(cases):
        args=['sbatch','--parsable']
        if lanes[i%7]:args+=['--dependency=afterok:'+lanes[i%7]]
        job=subprocess.check_output(args+[str(Path(c['directory'])/'run.slurm')],text=True).strip().split(';')[0]
        if not job.isdigit():raise ValueError(job)
        c['job_id']=job;lanes[i%7]=job
        (ROOT/'manifest.json').write_text(json.dumps(m,indent=2))
        print(job,c['energy'],c['length'],flush=True)

def collect():
    m=json.loads((ROOT/'manifest.json').read_text());records=[]
    for p,h in m['inputs'].items():
        if sha(p)!=h:raise ValueError('Changed input '+p)
    for c in m['cases']:
        d=Path(c['directory']);job=c['job_id']
        status=subprocess.check_output(['sacct','-j',job,'--format=JobIDRaw,State,ExitCode','-n','-P'],text=True)
        if f'{job}|COMPLETED|0:0' not in status.splitlines():raise ValueError('Incomplete '+job)
        a=np.loadtxt(d/'entrance.phsp');a=a[a[:,7]==1000060120]
        if len(a)!=50000 or np.any(a[:,4]<.99):raise ValueError('Primary count/direction')
        if np.max(np.abs(a[:,1]*10-(-450+c['length'])))>1e-3:raise ValueError('Wrong plane')
        x=a[:,0]*10;z=a[:,2]*10;angle=a[:,3]/a[:,4]
        xx=(np.var(x)+np.var(z))/2;aa=np.var(angle);xa=np.mean((x-x.mean())*(angle-angle.mean()))
        if xx<=0 or aa<=0 or xx*aa<=xa*xa or max(abs(x.mean()),abs(z.mean()))>.005:raise ValueError('Invalid phase moments')
        c.update(phase_sha256=sha(d/'entrance.phsp'),header_sha256=sha(d/'entrance.header'),xx=float(xx),xa=float(xa),aa=float(aa),primaries=len(a))
        records.append([c['energy'],c['length'],xx,xa,aa])
    dest=ROOT/'upstream_air_moments_v1.csv'
    with dest.open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['energy_MeVu','length_mm','position_variance_mm2','position_angle_cov_mm_rad','angle_variance_rad2']);w.writerows(records)
    m.update(status='COLLECTED_NOT_PATIENT_VALIDATED',table_sha256=sha(dest),holdout='155 MeV/u, 307.6287 mm not used in grid; ct_entrance_phase_r2_20260906')
    (ROOT/'collected_manifest.json').write_text(json.dumps(m,indent=2))
    print('Collected',len(records),'rows',sha(dest),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collect',action='store_true');a=p.parse_args()
    collect() if a.collect else prepare()

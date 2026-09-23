from pathlib import Path
import csv,fcntl,hashlib,json,os,re,shutil,subprocess,sys,time
import numpy as np
R=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923')
O=R/'evidence/field3cm_emonly_20260923'
M=json.loads((O/'manifest.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for f in M['files']:assert sha(f['path'])==f['sha256'],f['path']
assert sha(O/'bin/carbon_mc')==M['gpu_binary_sha256']

def dump(path,obj):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def topas(name,n):
    d=Path(M['cases'][name]);start=time.monotonic()
    with (d/'run.log').open('x') as log:
        p=subprocess.run([str(O/'bin/topas'),'run.txt'],cwd=d,stdout=log,stderr=subprocess.STDOUT)
    text=(d/'run.log').read_text()
    if p.returncode:raise RuntimeError((name,p.returncode,text[-4000:]))
    assert 'History-to-spot map is event-ID based (MT-safe).' in text
    assert 'geant4-11-03-patch-02' in text
    assert f'Particle source CarbonPBS: Total number of histories: {n}\n' in text
    if name!='source_audit':
        a=np.fromfile(d/'dose.bin',dtype='<f8')
        assert len(a)==1000000 and np.isfinite(a).all() and a.min()>=0 and a.max()>0
    q=dict(name=name,histories=n,threads=128,wall_s=time.monotonic()-start,accepted=True,
           topas_binary_sha256=sha(O/'bin/topas'),slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    dump(d/'quality.json',q);print('TOPAS',json.dumps(q),flush=True)
    return q

def gpu(key,n):
    cp=Path(M['cases'][key]);name=cp.stem;dest=R/'out'/name
    assert not dest.exists(),str(dest)
    start=time.monotonic()
    with (cp.parent/'run.log').open('x') as log:
        p=subprocess.run([str(O/'bin/carbon_mc'),'--config',str(cp)],cwd=R,stdout=log,stderr=subprocess.STDOUT)
    text=(cp.parent/'run.log').read_text()
    if p.returncode:raise RuntimeError((key,p.returncode,text[-4000:]))
    assert f'spots: 256/256 active; total histories: {n}' in text
    q=json.loads((dest/'quality_report.json').read_text())
    u=json.loads(re.search(r'URBAN_RUN_QUALITY (\{[^\n]+\})',text).group(1))
    assert q['accepted'] and not q['failures'] and not any(u[k] for k in ['fatal','cap','guard','subulp'])
    a=np.fromfile(dest/'dose.raw',dtype='<f4');assert len(a)==1000000 and np.isfinite(a).all() and a.min()>=0 and a.max()>0
    result=dict(name=name,histories=n,accepted=True,wall_s=time.monotonic()-start,
                relative_energy_residual=q['relative_energy_residual'],urban=u,dose_path=str(dest/'dose.raw'),
                gpu_binary_sha256=M['gpu_binary_sha256'],slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    dump(cp.parent/'quality.json',result);print('GPU',json.dumps(result),flush=True)
    return result

def gpu_lock():
    f=open('/tmp/maigo-review-gpu.lock','a');fcntl.flock(f,fcntl.LOCK_EX)
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,capture_output=True,check=True)
    assert not p.stdout.strip(),'GPU is occupied: '+p.stdout
    return f

mode=sys.argv[1]
if mode=='preflight':
    source=O/'topas_build/topas'
    assert source.exists()
    shutil.copy2(source,O/'bin/topas')
    assert b'History-to-spot map is event-ID based (MT-safe).' in source.read_bytes()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=O/'topas_tps_source_extension',text=True).strip()==M['extension_commit']
    for sub in ['src','include']:
        for f in (O/'topas_tps_source_extension'/sub).iterdir():
            if f.suffix in ['.cc','.hh']:assert sha(f)==sha(O/'topas_build/extensions'/f.name)
    q=topas('source_audit',M['source_audit']['histories'])
    d=Path(M['cases']['source_audit']);a=np.loadtxt(d/'source_audit.phsp',ndmin=2)
    n=M['source_audit']['histories'];weights=np.asarray(M['source_audit']['weights'])
    assert a.shape==(n,14),a.shape
    ids=a[:,11].astype(int);assert np.array_equal(np.sort(ids),np.arange(n))
    header=(d/'source_audit.header').read_text()
    for label in ['7: Weight','8: Particle Type (in PDG Format)','12: Event ID','13: Track ID','14: Parent ID']:
        assert label in header,label
    assert np.all(a[:,6]==1) and np.all(a[:,7]==1000060120)
    assert np.all(a[:,9]==1) and np.all(a[:,10]==0) and np.all(a[:,12]==1) and np.all(a[:,13]==0)
    spot=np.searchsorted(np.cumsum(weights),ids,side='right')
    counts=np.bincount(spot,minlength=256);assert np.array_equal(counts,weights)
    plan=list(csv.DictReader((O/'inputs/source_audit_spots.csv').open()))
    iso_x=np.array([float(x['x']) for x in plan])[spot];iso_y=np.array([float(x['y']) for x in plan])[spot]
    plane=M['source_audit']['plane_world_y_mm']
    ex=iso_x*(1+plane/6228.28);ez=iso_y*(1+plane/7007.64)
    errors=dict(x_mm=float(np.max(abs(a[:,0]*10-ex))),y_mm=float(np.max(abs(a[:,1]*10-plane))),z_mm=float(np.max(abs(a[:,2]*10-ez))))
    assert max(errors.values())<2e-4,errors
    assert np.all(abs(a[:,5]-3000)<.001)
    audit=dict(accepted=True,spots=256,histories=n,threads=128,each_event_once=True,spot_counts=counts.tolist(),max_position_error=errors,scope='Zero-emittance vacuum source audit only; not dose evidence.')
    dump(O/'source_audit.json',audit)
    topas('topas_smoke',25600)
    lock=gpu_lock();gpu('gpu_smoke',25600);lock.close()
    dump(O/'preflight.json',dict(accepted=True,extension_commit=M['extension_commit'],topas_binary_sha256=sha(O/'bin/topas'),gpu_binary_sha256=M['gpu_binary_sha256'],source_audit=audit,fullchain_smoke_histories_per_engine=25600))
elif mode=='topas':
    p=json.loads((O/'preflight.json').read_text());assert p['accepted'] and sha(O/'bin/topas')==p['topas_binary_sha256']
    i=int(sys.argv[2]);assert i in range(1,5);topas(f'topas_b{i}',M['histories_per_batch'])
elif mode in ['gpu','gpu_batch']:
    assert json.loads((O/'preflight.json').read_text())['accepted']
    indices=range(1,5) if mode=='gpu' else [int(sys.argv[2])]
    assert all(i in range(1,5) for i in indices)
    lock=gpu_lock()
    for i in indices:gpu(f'gpu_b{i}',M['histories_per_batch'])
else:raise ValueError(mode)

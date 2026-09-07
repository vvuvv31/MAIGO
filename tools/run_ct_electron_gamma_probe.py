"""One local patient shard, explicit unvalidated B or frozen A. No production promotion."""
import argparse,json,os,subprocess,time,shutil,hashlib
from pathlib import Path
import numpy as np
import yaml
from run_topas10x_gpu_benchmark import sha,config_write

def validate_experiment_exit(returncode,log_text,experimental):
    errors=[line for line in log_text.splitlines() if line.startswith('carbon_mc: ')]
    if experimental:
        # Both a quality refusal and an I/O exception use EXIT_FAILURE. A
        # pre-written quality JSON alone cannot distinguish them.
        if returncode!=1 or len(errors)!=1 or not errors[0].startswith('carbon_mc: run quality fail (mode=smoke, failures=1,'):
            raise RuntimeError('Unexpected process/output failure: '+str(errors))
    elif returncode!=0 or errors:
        raise RuntimeError('Unexpected process failure: '+str(errors))

def run(source,out,binary,joint=None):
    repo=Path(__file__).resolve().parents[1]
    subprocess.run(['python3',str(repo/'tools/verify_schneider_v2_1_data.py')],check=True)
    cfg=yaml.safe_load(source.read_text());cfg.update(run_mode='smoke',dose_to_medium_name='dose')
    # Explicit YAML OFF wins over the runner's --joint convenience argument.
    if cfg.get('ct_electron_segment_transport') is False:
        joint=None
    if joint:
        cfg.update(ct_electron_joint_patient_experiment=True,ct_electron_joint_response_diagnostic_file=str(joint),
                   ct_electron_joint_response_sha256=sha(joint),ct_electron_joint_response_metadata_sha256=sha(joint.with_suffix('.metadata.json')))
    out.mkdir(parents=True,exist_ok=False);config_write(out/'run.yaml',cfg)
    runtime=out/('electron_'+hashlib.sha256(str(out).encode()).hexdigest()[:16]+'.yaml')
    shutil.copy2(out/'run.yaml',runtime)
    qdir=repo/'out'/runtime.stem
    if qdir.exists():raise ValueError('Refuse existing quality directory '+str(qdir))
    pins={str(source):sha(source),str(binary):sha(binary)}
    for key,value in cfg.items():
        if isinstance(value,str) and Path(value).is_file():pins[value]=sha(value)
    env=dict(os.environ,ONEAPI_DEVICE_SELECTOR='cuda:*')
    env['LD_LIBRARY_PATH']='/home/wuwei/sycl_workspace/llvm/build/install/lib:'+env.get('LD_LIBRARY_PATH','')
    started=time.monotonic()
    with (out/'run.log').open('x') as log:
        proc=subprocess.run([str(binary),'--config',str(runtime),'--device','cuda','--voxel-dose-mhd',str(out/'dose.mhd')],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    if not (qdir/'quality_report.json').is_file():
        raise RuntimeError(f'GPU exited {proc.returncode} without quality output; inspect {out / "run.log"}')
    for output in qdir.glob('*.json'):shutil.copy2(output,out/output.name)
    q=json.loads((out/'quality_report.json').read_text());ledger=json.loads((out/'energy_ledger.json').read_text())
    if q['queue_overflow_count'] or q['queue_overflow_energy_MeV']:raise RuntimeError('OVERFLOW: exclude this dose and split BOTH paired inputs before retry')
    expected=['unvalidated_electron_joint_response'] if joint else []
    if [f['code'] for f in q['failures']]!=expected or q['accepted']==bool(joint) or (proc.returncode==0)==bool(joint):
        raise RuntimeError('Unexpected quality/exit: '+str(q['failures']))
    validate_experiment_exit(proc.returncode,(out/'run.log').read_text(),bool(joint))
    for file in ('dose.raw','dose.mhd'):
        if not (out/file).is_file() or not (out/file).stat().st_size:
            raise RuntimeError('Required dose output missing: '+file)
    if ledger['histories']!=cfg['number_of_histories']:raise ValueError('History mismatch')
    if joint:
        d=ledger['electron_joint_response']
        if not d['patient_experiment'] or d['queries']<=0 or d['ordered_path_replays']<=0 or d['redistributed_MeV']<=0:
            raise ValueError('Patient response was not exercised')
    for path,pin in pins.items():
        if sha(path)!=pin:raise ValueError('Input changed while running: '+path)
    report=dict(role='B' if joint else 'A',status='EXPERIMENT_ONLY',histories=ledger['histories'],
        seconds=time.monotonic()-started,inputs=pins,binary_sha256=sha(binary),config_sha256=sha(out/'run.yaml'),
        dose_sha256=sha(out/'dose.raw'),quality=q)
    with (out/'run_report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(report['role'],'completed',report['histories'],'histories; production not enabled',flush=True)

def gamma(root,reference):
    from evaluate_topas10x_gpu_gamma import pass_mask
    m=json.loads((reference/'manifest.json').read_text())
    if sha(reference/'topas_sum.raw')!=m['reference_sum_sha256']:raise ValueError('Reference pin')
    ref=np.fromfile(reference/'topas_sum.raw','<f4').reshape(m['topas_shape_zyx'])
    pts=np.argwhere(ref>=.1*ref.max());reports={};results={}
    for role in ('A','B'):
        d=root/role;r=json.loads((d/'run_report.json').read_text());reports[role]=r
        if sha(d/'dose.raw')!=r['dose_sha256']:raise ValueError('Dose pin')
        a=np.fromfile(d/'dose.raw','<f4').reshape(m['gpu_shape_zyx'])
        if m['mapping']=='packed_xneg':a=np.flip(a.transpose(1,2,0),axis=2)
        elif m['mapping']!='native':raise ValueError('Mapping')
        a=a.astype(float)*(m['histories']/r['histories'])
        if not np.isfinite(a).all() or np.any(a<0):raise ValueError('Invalid dose')
        results[role]={}
        for dd,dta in ((3,3),(2,2),(1,1),(3,0)):
            for local in (False,True):
                key=f'{"local" if local else "global"}_{dd}{dta}'
                results[role][key]=100*float(pass_mask(a,ref,pts,np.array(m['spacing_zyx']),dd,dta,local).mean())
    if reports['A']['binary_sha256']!=reports['B']['binary_sha256'] or reports['A']['histories']!=reports['B']['histories']:raise ValueError('Unpaired runs')
    configs=[]
    experimental={'ct_electron_joint_patient_experiment','ct_electron_joint_response_diagnostic_file',
                  'ct_electron_joint_response_sha256','ct_electron_joint_response_metadata_sha256'}
    for role in ('A','B'):
        config=root/role/'run.yaml'
        if sha(config)!=reports[role]['config_sha256']:raise ValueError('Config pin')
        configs.append({k:v for k,v in yaml.safe_load(config.read_text()).items() if k not in experimental})
    if configs[0]!=configs[1]:raise ValueError('A/B source or transport configuration differs')
    for path,pin in reports['A']['inputs'].items():
        if reports['B']['inputs'].get(path)!=pin:raise ValueError('A/B input provenance differs: '+path)
    result=dict(status='PATIENT_GAMMA_EXPERIMENT_NOT_PRODUCTION',results=results,
        delta_percentage_points={k:results['B'][k]-results['A'][k] for k in results['A']},
        reference_sha256=m['reference_sum_sha256'],mask_voxels=len(pts),method='Frozen >=10% full mask; 0.5mm search lattice; nominal exact-history scale; no fit')
    with (root/'gamma.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--binary',type=Path);p.add_argument('--joint',type=Path);p.add_argument('--reference',type=Path);a=p.parse_args()
    if a.reference:gamma(a.out.resolve(),a.reference.resolve())
    else:run(a.source.resolve(),a.out.resolve(),a.binary.resolve(),a.joint.resolve() if a.joint else None)

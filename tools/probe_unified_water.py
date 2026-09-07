"""Local GPU safety probe for native water + shared CINEL03 framework."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
import yaml
from run_topas10x_gpu_benchmark import config_write, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path('config/unified_water_cinel03_smoke.yaml'))
    parser.add_argument('--histories',type=int,default=10000)
    parser.add_argument('--generations',type=int,default=2)
    parser.add_argument('--em-only',action='store_true',help='Diagnostic: disable nuclear and secondary transport')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--origin',action='store_true',help='Diagnostic 3D primary/charged-species scoring')
    parser.add_argument('--binary',type=Path,default=Path('build/oneapi-nvidia-unified-default/carbon_mc'))
    args=parser.parse_args()
    if args.histories<=0 or args.generations<0:raise ValueError('Invalid counts')
    repo=Path(__file__).resolve().parents[1]
    subprocess.run(['python3',str(repo/'tools/verify_schneider_v2_1_data.py')],check=True,cwd=repo)
    out=args.out.resolve();binary=args.binary.resolve()
    cfg=yaml.safe_load(args.config.resolve().read_text())
    cfg.update(number_of_histories=args.histories,cinel02_max_secondary_inelastic_generations=args.generations)
    if args.seed is not None:cfg['random_seed']=args.seed
    if args.em_only:cfg.update(enable_inelastic=False,enable_secondary_transport=False)
    if args.origin:
        cfg.update(enable_charged_origin_voxel_scoring=True,
                   charged_origin_voxel_mhd_output_prefix=str(out/'origin'),
                   charged_origin_voxel_output_file='',charged_origin_voxel_dose_Gy_output_file='')
    for key,value in list(cfg.items()):
        if isinstance(value,str) and (repo/value).is_file():cfg[key]=str((repo/value).resolve())
    out.mkdir(parents=True,exist_ok=False)
    source=out/(out.name+'.yaml');config_write(source,cfg)
    qdir=repo/'out'/source.stem
    if qdir.exists():raise ValueError('Quality path already exists')
    pins={str(binary):sha(binary),str(source):sha(source)}
    for value in cfg.values():
        if isinstance(value,str) and Path(value).is_file():pins[value]=sha(Path(value))
    env=dict(os.environ,ONEAPI_DEVICE_SELECTOR='cuda:*',LD_LIBRARY_PATH='/home/wuwei/sycl_workspace/llvm/build/install/lib:'+os.environ.get('LD_LIBRARY_PATH',''))
    with (out/'run.log').open('x') as log:
        proc=subprocess.run([str(binary),'--config',str(source),'--device','cuda','--voxel-dose-mhd',str(out/'dose.mhd')],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    for path in qdir.glob('*.json'):shutil.copy2(path,out/path.name)
    if not (out/'quality_report.json').is_file():
        raise RuntimeError(f'GPU exited {proc.returncode} before quality output: '+(out/'run.log').read_text()[-3000:])
    quality=json.loads((out/'quality_report.json').read_text())
    ledger=json.loads((out/'energy_ledger.json').read_text())
    if quality['failures'] or not quality['accepted']:
        raise RuntimeError('Unexpected quality failures: '+str(quality['failures']))
    errors=[line for line in (out/'run.log').read_text().splitlines() if line.startswith('carbon_mc: ')]
    if proc.returncode!=0 or errors:
        raise RuntimeError('Unexpected process or output failure: '+str(errors))
    if quality['queue_overflow_count'] or quality['queue_overflow_energy_MeV']:raise RuntimeError('Overflow: split and rerun')
    if ledger['material_physics_mode']!='Water' or not ledger['unified_water_nuclear_transport']:
        raise RuntimeError('Wrong material/nuclear route')
    if ledger['histories']!=args.histories:raise RuntimeError('History mismatch')
    d=ledger['schneider_diagnostics']
    active_stopping=[line for line in (out/'run.log').read_text().splitlines()
                     if line.startswith('Unified water active ion stopping table: ')]
    expected_ion_path=Path(cfg['particle_stopping_power_file'])
    if not args.em_only and (len(active_stopping)!=1 or str(expected_ion_path) not in active_stopping[0] or \
            'SHA256='+sha(expected_ion_path) not in active_stopping[0]):
        raise RuntimeError('Actual uploaded ion stopping table was not verified')
    if args.em_only:
        if any(d[k]!=0 for k in ('primary_hazards','primary_events_replayed','secondary_hazards','secondary_events_replayed')):
            raise RuntimeError('Nuclear events present in EM-only control')
    elif d['primary_hazards']<=0 or d['primary_events_replayed']<=0:raise RuntimeError('No primary CINEL03 replay exercised')
    if not args.em_only and args.generations>0 and (d['secondary_hazards']<=0 or d['secondary_events_replayed']<=0):
        raise RuntimeError('No secondary CINEL03 replay exercised')
    if args.generations==0 and d['secondary_hazards']!=0:raise RuntimeError('Generation cap ignored')
    provenance=ledger['schneider_physics_provenance']
    for label,key in [('primary_rate','ct_schneider_primary_rate_file'),('secondary_rate','ct_schneider_secondary_rate_file'),
                      ('primary_package','ct_schneider_c12_cinel03_file'),('secondary_package','ct_schneider_secondary_cinel03_file')]:
        if provenance[label]['sha256']!=sha(Path(cfg[key])):raise RuntimeError('Loaded provenance mismatch '+label)
    if not (out/'dose.raw').is_file() or not (out/'dose.mhd').is_file():raise RuntimeError('Missing 3D dose')
    for path,h in pins.items():
        if sha(Path(path))!=h:raise RuntimeError('Input changed: '+path)
    with (out/'manifest.json').open('x') as stream:
        json.dump(dict(status='SMOKE_ONLY_NOT_PRODUCTION',em_only=args.em_only,histories=args.histories,generations=args.generations,
            pins=pins,dose_sha256=sha(out/'dose.raw'),quality=quality),stream,indent=2)
    print('Unified native water smoke gate PASS; production remains refused:',out,flush=True)


if __name__=='__main__':main()

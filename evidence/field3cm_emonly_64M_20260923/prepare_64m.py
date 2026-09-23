from pathlib import Path
import copy,datetime,hashlib,json,re,shutil
import yaml
R=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923')
OLD=R/'evidence/field3cm_emonly_20260923'
O=R/'evidence/field3cm_emonly_64M_20260923'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
assert not O.exists(),str(O)
old=json.loads((OLD/'manifest.json').read_text())
status=json.loads((OLD/'serial_status.json').read_text());assert status['status']=='COMPLETE'
pre=json.loads((OLD/'preflight.json').read_text());assert pre['accepted']
for item in old['files']:assert sha(item['path'])==item['sha256'],item['path']
assert sha(OLD/'bin/topas')==pre['topas_binary_sha256']
assert sha(OLD/'bin/carbon_mc')==old['gpu_binary_sha256']
u=json.loads((OLD/'results/uncertainty_details.json').read_text())
for item in u['dose_files']:assert sha(item['path'])==item['sha256'],item['path']
cfg=yaml.safe_load(Path(old['cases']['gpu_b1']).read_text())
dependencies=[]
started=datetime.datetime.fromisoformat(status['started_at']).timestamp()
for k,v in cfg.items():
    if isinstance(v,str) and v.startswith('/'):
        p=Path(v);s=p.stat();assert s.st_mtime<started,(k,'data modified since original run')
        dependencies.append(dict(config_key=k,path=str(p),sha256=sha(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns))
assert next(v['sha256'] for v in dependencies if v['config_key']=='em_package_file')==cfg['em_package_sha256']
assert next(v['sha256'] for v in dependencies if v['config_key']=='unified_water_material_file')==cfg['unified_water_material_sha256']
for sub in ['cases','bin','logs','results']: (O/sub).mkdir(parents=True,exist_ok=True)
for fn in ['topas','carbon_mc']:shutil.copy2(OLD/'bin'/fn,O/'bin'/fn)
for fn in ['preflight.json','source_audit.json']:shutil.copy2(OLD/fn,O/fn)
manifest=copy.deepcopy(old)
manifest.update(batches=20,histories_per_engine=64000000,base_evidence=str(OLD),
    reused_batches=[1,2,3,4],new_batches=list(range(5,21)),
    additional_histories_per_engine=51200000,mode='direct_serial',memory_max_bytes=30000000000,
    reused_dose_files=u['dose_files'],runtime_dependencies=dependencies)
manifest['plan']['per_spot_histories_total']=250000
manifest['cases']={k:v for k,v in old['cases'].items() if re.fullmatch(r'(topas|gpu)_b[1-4]',k)}
seeds={'TOPAS':[],'GPU':[]}
new_files=[]
base_td=Path(old['cases']['topas_b1'])
for i in range(1,21):
    seeds['TOPAS'].append(2026101000+i);seeds['GPU'].append(2026102000+i)
    if i<=4:continue
    td=O/'cases'/f'topas_b{i}';td.mkdir()
    for fn in ['run1.txt','run_energy_scan.txt','aperture.txt']:shutil.copy2(base_td/fn,td/fn)
    original=(base_td/'run.txt').read_text()
    txt,count=re.subn(r'(i:Ts/Seed\s*=\s*)\d+',lambda m:m[1]+str(seeds['TOPAS'][-1]),original)
    assert count==1
    assert re.sub(r'i:Ts/Seed\s*=\s*\d+','SEED',txt)==re.sub(r'i:Ts/Seed\s*=\s*\d+','SEED',original)
    (td/'run.txt').write_text(txt)
    name=f'field3cm_emonly_64m_b{i}';gd=O/'cases'/name;gd.mkdir()
    assert not (R/'out'/name).exists()
    gc=copy.deepcopy(cfg);gc['random_seed']=seeds['GPU'][-1]
    assert {k:v for k,v in gc.items() if k!='random_seed'}=={k:v for k,v in cfg.items() if k!='random_seed'}
    cp=gd/(name+'.yaml');cp.write_text(yaml.safe_dump(gc,sort_keys=False))
    manifest['cases'][f'topas_b{i}']=str(td);manifest['cases'][f'gpu_b{i}']=str(cp)
    new_files.extend([*td.iterdir(),cp])
assert len(set(seeds['TOPAS']+seeds['GPU']))==40
manifest['random_seeds']=seeds
manifest['files']=old['files']+[dict(path=str(p),sha256=sha(p)) for p in new_files]
manifest['baseline_batch_wall_s']={e:sum(s['wall_s'] for s in status['completed_stages'] if s['name'].startswith(e+'_b'))/4 for e in ['topas','gpu']}
dump(O/'manifest.json',manifest)
dump(O/'preparation.json',dict(status='PREPARED',created_at=datetime.datetime.now().astimezone().isoformat(),
    binary_hashes=dict(TOPAS=sha(O/'bin/topas'),GPU=sha(O/'bin/carbon_mc')),
    all_old_input_and_dose_hashes_verified=True,only_random_seed_changes_within_each_engine=True,
    dependency_note='Current dependency hashes recorded; all dependency mtimes predate the original run. Explicit EM package and material hashes match original configuration.',
    planned_histories_per_engine=64000000,additional_histories_per_engine=51200000,
    expected_additional_wall_hours=sum(manifest['baseline_batch_wall_s'].values())*16/3600))
print((O/'preparation.json').read_text(),flush=True)

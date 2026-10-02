#!/usr/bin/env python3
"""Independent 0.1% full-plan replica, exact integer spot weights, same frozen physics."""
import csv,json,math,shutil
from pathlib import Path
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'RT07575_minibeam_GPU'

def main():
    if (ROOT/'manifest.json').exists():raise SystemExit('Already prepared')
    for name in ['common.py','metrics.py','run_gpu.py','compare.py','plots.py','uncertainty.py','peak_valley.py','build.json']:
        shutil.copy2(PREVIOUS/name,ROOT/name)
    (ROOT/'data').symlink_to(ROOT.parents[2]/'data',target_is_directory=True)
    (ROOT/'inputs').symlink_to(PREVIOUS/'inputs',target_is_directory=True)
    from common import sha,save_json
    original=json.loads((PREVIOUS/'manifest.json').read_text());identity=json.loads((ROOT/'build.json').read_text())
    assert identity['dose_fp32'] and not identity['dose_fp64']
    assert sha(ROOT/'inputs/carbon_mc')==identity['binary_sha256']
    assert sha(ROOT.parents[2]/'build/carbon_mc')==identity['binary_sha256'],'Current binary differs from last validated code'
    rows=list(csv.DictReader((ROOT/'inputs/spots_optimized.csv').open()));weights=np.array([float(r['weight']) for r in rows]);total=round(weights.sum()*.001)
    assert total==236086432
    exact=total*weights/weights.sum();counts=np.floor(exact).astype(np.int64);counts[np.argsort(-(exact-counts),kind='stable')[:total-int(counts.sum())]]+=1
    parts=math.ceil(total/2000000);split=np.repeat((counts//parts)[None,:],parts,axis=0);cursor=0
    for spot,n in enumerate(counts%parts):
        split[(cursor+np.arange(n))%parts,spot]+=1;cursor=(cursor+int(n))%parts
    assert np.array_equal(split.sum(axis=0),counts) and split.sum()==total and split.sum(axis=1).max()<=2000000
    with (ROOT/'spot_history_allocation.csv').open('w') as stream:
        w=csv.writer(stream);w.writerow(['plan_index','spot_id','protons_4Gy','histories_0p1pct']);w.writerows((i,r['spot_id'],weights[i],counts[i]) for i,r in enumerate(rows))
    base=yaml.safe_load((PREVIOUS/'batch0000.yaml').read_text());source=(PREVIOUS/'sources/batch0000.txt').read_text();jobs=[]
    assert base['material_electron_group_size']==32
    assert sha(base['material_electron_response_index_file'])==base['material_electron_response_index_sha256']
    (ROOT/'sources').mkdir()
    for i,ns in enumerate(split):
        name=f'batch{i:04d}';path=ROOT/'sources'/f'{name}.txt'
        lines=[f'dv:Tf/Scatterer1/L4/Values = {len(ns)} '+' '.join(map(str,ns)) if line.startswith('dv:Tf/Scatterer1/L4/Values') else line for line in source.splitlines()]
        path.write_text('\n'.join(lines)+'\n');seed=7000000001+i*1000000000
        cfg=dict(base,number_of_histories=int(ns.sum()),topas_spots_file=str(path),random_seed=seed)
        config=ROOT/(name+'.yaml');config.write_text(yaml.safe_dump(cfg,sort_keys=False))
        assert {k:v for k,v in cfg.items() if k not in ['number_of_histories','topas_spots_file','random_seed']}=={k:v for k,v in base.items() if k not in ['number_of_histories','topas_spots_file','random_seed']}
        jobs.append(dict(name=name,histories=int(ns.sum()),seed=seed,config_sha256=sha(config),source_sha256=sha(path)))
    manifest=dict(original,histories=total,parts=parts,scale_4Gy=float(weights.sum()/total),jobs=jobs,pilots=[],fraction_of_4Gy=.001,previous_campaign=str(PREVIOUS),independent_of_previous=True,previous_dose_reused=False)
    save_json(ROOT/'manifest.json',manifest)
    p=ROOT/'compare.py';s=p.read_text().replace("summary['histories']==23608643","summary['histories']==json.loads((ROOT/'manifest.json').read_text())['histories']");p.write_text(s)
    print(json.dumps(dict(histories=total,parts=parts,min_shard=int(split.sum(axis=1).min()),max_shard=int(split.sum(axis=1).max()),active_spots=int((counts>0).sum()))))
if __name__=='__main__':main()

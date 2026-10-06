#!/usr/bin/env python3
"""Explicit experimental RT07575 head hybrid26 campaign; FP32 dose only."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import yaml
import validate_em_density_factorization as v

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['smoke','full','analyze'],required=True)
    p.add_argument('--tags',nargs='+',choices=['s1','s2','h01'],default=['s1','s2','h01'])
    a=p.parse_args()
    root=v.ROOT/'out/electron_composition26_20261006'
    v.OUT=root/'gpu_hybrid26_v2';v.OUT.mkdir(parents=True,exist_ok=True)
    binary=v.OUT/'candidate_carbon_mc'
    control=v.ROOT/'out/electron_cut_conditioned26_20261005/runs_validation'
    updates={'material_electron_birth_file':str(v.ROOT/'out/electron_cut_conditioned26_20261005/package/birth26.bin')}
    env={'CARBON_ELECTRON_RATE26_FILE':str(root/'package/rates26.bin'),'CARBON_ELECTRON_MOLLER26':'1'}
    case='proton_head_minibeam';arm='hybrid26'
    if a.stage=='smoke':
        if binary.exists():raise ValueError('Frozen experiment binary already exists')
        shutil.copy2(v.ROOT/'build/carbon_mc',binary)
        v.run(case,arm,'smoke',2,50000,1001904402,binary,config_updates=updates,runtime_env=env)
    elif a.stage=='full':
        smoke=json.loads((v.OUT/'runs'/f'{case}_{arm}_smoke/execution.json').read_text())
        assert smoke['runtime_pass']
        for tag in ['s1','s2','h01']:
            completed=v.OUT/'runs'/f'{case}_{arm}_{tag}/execution.json'
            if completed.exists():
                assert json.loads(completed.read_text())['runtime_pass'];continue
            rec=json.loads((control/'runs'/f'{case}_birth26v3_{tag}/execution.json').read_text())
            v.run(case,arm,tag,rec['n'],rec['h'],rec['seed'],binary,config_updates=updates,runtime_env=env)
    else:
        tags=a.tags
        report_dir=v.OUT/('full_0p1_analysis' if tags==['s1','s2','h01'] else 'subset_'+'_'.join(tags)+'_analysis');report_dir.mkdir(exist_ok=False)
        members=[]
        for tag in tags:
            dirs=[control/'runs'/f'{case}_birth26v3_{tag}',v.OUT/'runs'/f'{case}_{arm}_{tag}']
            records=[json.loads((d/'execution.json').read_text()) for d in dirs]
            members.append(dict(tag=tag,control=records[0],candidate=records[1]))
            assert all(r['runtime_pass'] for r in records)
            for key in ['seed','n','h','histories','spot_allocation']:
                if records[0][key]!=records[1][key]:raise ValueError(f'Matched campaign differs: {key}')
            configs=[yaml.safe_load((d/'case.yaml').read_text()) for d in dirs]
            for key in configs[0].keys()|configs[1].keys():
                if key=='spot_weights_file':
                    assert np.array_equal(*[np.loadtxt(c[key]) for c in configs])
                elif configs[0].get(key)!=configs[1].get(key):
                    raise ValueError(f'Matched configuration differs: {key}')
        new_report,new,mask,spacing=v.analyze(case,arm,tags,report_dir)
        v.OUT=control
        old_report,old,old_mask,old_spacing=v.analyze(case,'birth26v3',tags,report_dir)
        assert np.array_equal(mask,old_mask) and np.array_equal(spacing,old_spacing)
        hit=v.evaluator()(new,old,np.argwhere(mask),spacing,3,.2,True,step=.02)
        delta=new[mask]-old[mask];local=100*abs(delta)/old[mask]
        report=dict(status='EXPERIMENTAL_HYBRID26_SHARED83_LOW_IONIZATION_RATES_TERMINALS_AND_JOINT',default_promoted=False,
                    preliminary=tags!=['s1','s2','h01'],
                    complete_electron26_qualified=False,case_identity='RT07575 head',
                    control_TOPAS=old_report,candidate_TOPAS=new_report,
                    against_control_pass_percent=float(100*hit.mean()),
                    relative_L1=float(abs(delta).sum()/old[mask].sum()),
                    dose_sum_change_percent=float(100*delta.sum()/old[mask].sum()),
                    local_percent_quantiles={str(q):float(np.percentile(local,q)) for q in [50,95,99,99.9,100]})
        report['members']=members
        report['timing']={}
        for label in ['control','candidate']:
            rows=[m[label] for m in members]
            report['timing'][label]=dict(wall_seconds=sum(r['wall_seconds'] for r in rows),
                resident_primary_s=sum((r['n']-1)*r['h'] for r in rows)/sum(sum(r['batch_wall_seconds'][1:]) for r in rows),
                max_energy_residual=max(r['max_energy_residual'] for r in rows))
        report['timing']['controls_reused_not_interleaved']=True
        v.save(report_dir/'comparison.json',report)
        np.savez_compressed(report_dir/'dose_arrays.npz',old=old.astype('<f4'),new=new.astype('<f4'),mask=mask,spacing=spacing)
        print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()

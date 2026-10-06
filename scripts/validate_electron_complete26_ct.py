#!/usr/bin/env python3
"""CT-only independent26 runtime/dose screens; carbon minibeam accuracy deferred."""
import argparse,json,filecmp
import yaml
from pathlib import Path
import numpy as np
import validate_em_density_factorization as v

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['smoke','broadbeam','analyze'],required=True)
    p.add_argument('--cases',nargs='+',choices=['proton_broadbeam','carbon_broadbeam'],default=['proton_broadbeam','carbon_broadbeam'])
    a=p.parse_args();root=v.ROOT/'out/electron_complete26_20261006';v.OUT=root/'gpu_complete26_v2';v.OUT.mkdir(parents=True,exist_ok=True)
    binary=v.OUT/'candidate_carbon_mc';assert binary.exists()
    pkg=root/'gpu_package_v1'
    updates=dict(material_electron_birth_file=str(pkg/'birth26.bin'),material_electron_joint_pilot_file=str(pkg/'joint.bin'),material_electron_charged_index_file=str(pkg/'charged.index'),material_photon_response_index_file=str(pkg/'photon.index'))
    control=v.ROOT/'out/em_continuous_density_20261005/broadbeam_final'
    if a.stage=='smoke':
        for case in ['proton_broadbeam','carbon_broadbeam','carbon_minibeam']:
            done=v.OUT/'runs'/f'{case}_complete26_smoke/execution.json'
            if done.exists():assert json.loads(done.read_text())['runtime_pass'];continue
            v.run(case,'complete26','smoke',2,50000,1001904402,binary,config_updates=updates)
    elif a.stage=='broadbeam':
        for case in a.cases:
            for tag in ['c1','c2']:
                done=v.OUT/'runs'/f'{case}_complete26_{tag}/execution.json'
                if done.exists():assert json.loads(done.read_text())['runtime_pass'];continue
                rec=json.loads((control/'runs'/f'{case}_continuous_{tag}/execution.json').read_text())
                v.run(case,'complete26',tag,rec['n'],rec['h'],rec['seed'],binary,config_updates=updates)
    else:
        for case in a.cases:
            report_dir=v.OUT/(case+'_analysis');report_dir.mkdir(exist_ok=False)
            candidate_root=v.OUT
            for tag in ['c1','c2']:
                dirs=[control/'runs'/f'{case}_continuous_{tag}',v.OUT/'runs'/f'{case}_complete26_{tag}']
                records=[json.loads((d/'execution.json').read_text()) for d in dirs]
                assert all(r['runtime_pass'] for r in records)
                for key in ['n','h','seed','histories']:assert records[0][key]==records[1][key]
                configs=[yaml.safe_load((d/'case.yaml').read_text()) for d in dirs]
                for key in configs[0].keys()|configs[1].keys():
                    if key in updates:continue
                    if key in ['em_package_file','em_delta_moments_file']:
                        assert filecmp.cmp(configs[0][key],configs[1][key],shallow=False)
                    else:assert configs[0].get(key)==configs[1].get(key),key
            new_report,new,mask,spacing=v.analyze(case,'complete26',['c1','c2'],report_dir)
            v.OUT=control;old_report,old,old_mask,old_spacing=v.analyze(case,'continuous',['c1','c2'],report_dir)
            v.OUT=candidate_root;assert np.array_equal(mask,old_mask) and np.array_equal(spacing,old_spacing)
            hit=v.evaluator()(new,old,np.argwhere(mask),spacing,1,1,True,step=.1)
            delta=new[mask]-old[mask]
            r=dict(status='DOSE_FIRST_COMPLETE26_NOT_DEFAULT',control_TOPAS=old_report,candidate_TOPAS=new_report,against_control_pass_percent=float(100*hit.mean()),relative_L1=float(abs(delta).sum()/old[mask].sum()),dose_sum_change_percent=float(100*delta.sum()/old[mask].sum()),accuracy_accepted=bool(new_report['passed']))
            v.save(report_dir/'comparison.json',r);print(json.dumps(r),flush=True)
if __name__=='__main__':main()

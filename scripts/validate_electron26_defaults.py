#!/usr/bin/env python3
"""Smoke all eight default26 presets using their installed package paths."""
import argparse,json,shutil
import yaml
import validate_em_density_factorization as v

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--histories',type=int,default=10000);a=p.parse_args()
    v.OUT=v.ROOT/'out/electron_complete26_20261006/default_promotion_smoke_v2';v.OUT.mkdir(exist_ok=False)
    binary=v.OUT/'carbon_mc';shutil.copy2(v.ROOT/'build/carbon_mc',binary)
    def preset(case):return v.absolute_paths(yaml.safe_load((v.ROOT/f'config/{case}.yaml').read_text()))
    v.config=preset;reports=[]
    for species in ['proton','carbon']:
        for geometry in ['ct','water']:
            for beam in ['broadbeam','minibeam']:
                case=f'{species}_{geometry}_{beam}'
                _,rec=v.run(case,'default26','smoke',2,a.histories,1001904402,binary)
                reports.append(rec)
    v.save(v.OUT/'summary.json',dict(passed=True,default_promoted=True,presets=8,reports=reports,water_dose_acceptance=False,carbon_minibeam_dose_acceptance=False))
if __name__=='__main__':main()

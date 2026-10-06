#!/usr/bin/env python3
"""Separate native history noise from Born birth-moment differences.

Independent primary histories are the uncertainty units; individual weighted
electrons are not treated as independent observations. No yield or dose gate.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from extract_ion_electron_birth import DTYPE
from probe_electron_birth_analytic26 import born_joint,compare,cuts,native_nodes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extraction',type=Path,required=True)
    p.add_argument('--charged',type=Path,default=Path('data/archive/electron_density_nodes_20261006/shared/charged'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();lookup=cuts(args.charged);rows=[]
    completed=[];incomplete=[];projectiles=set()
    for directory in sorted(args.extraction.glob('node*')):
        if not (directory/'manifest.json').is_file() or not (directory/'birth_bank.bin').is_file():
            incomplete.append(str(directory));continue
        metadata=json.loads((directory/'manifest.json').read_text())
        if metadata['status']!='NATIVE_BIRTH_ENERGY_AUDITED_NOT_DOSE_ACCEPTANCE':
            raise ValueError('Unaudited extraction')
        projectiles.add((metadata['projectile_z'],metadata['projectile_a']))
        completed.append(str(directory))
        raw=np.fromfile(directory/'birth.phsp',DTYPE)
        summary=raw[raw['kind']==0];emitted=raw[raw['kind']==1]
        history_count=len(np.unique(summary['event']))
        if history_count<2:raise ValueError('Need independent histories')
        for s,r,z,a,channels,states in native_nodes(directory/'birth_bank.bin'):
            cut=lookup[s,r]
            for i,ch in enumerate(channels):
                if ch['low']<50 or not ch['count']:continue
                e,mu,w=born_joint(float(ch['low']),float(ch['high']),cut,a)
                block=states[int(ch['offset']):int(ch['offset']+ch['count'])]
                metrics=compare(block,e,mu,w)
                births=emitted[emitted['bin']==i]
                weight=births['birth'];events=births['event']
                measured_energy=np.sum(weight**2)/weight.sum()
                measured_cosine=np.sum(weight*births['mu'])/weight.sum()
                errors={}
                for key,values,mean,predicted in [
                    ('mean_energy',weight,measured_energy,float(w@e)),
                    ('mean_cosine',births['mu'],measured_cosine,float(w@mu))]:
                    residual=np.bincount(events,weights=weight*(values-mean),minlength=history_count)
                    se=float(np.sqrt(history_count/(history_count-1)*np.sum(residual**2))/weight.sum())
                    errors[key]=dict(native=mean,analytic=predicted,history_standard_error=se,
                                     difference_in_standard_errors=float((predicted-mean)/se) if se else None)
                rows.append(dict(section=s,density=r,channel=i,histories=history_count,
                                 **metrics,moment_uncertainty=errors))
    if not rows or len(projectiles)!=1:raise ValueError('Missing/mixed completed projectile probes')
    projectile={ (1,1):'proton',(6,12):'carbon'}[next(iter(projectiles))]
    summaries={k:dict(maximum=max(abs(x[k]) for x in rows),median=float(np.median([abs(x[k]) for x in rows])))
               for k in ['energy_cdf_max','joint_cdf_max','mean_energy_relative','mean_cosine_difference']}
    for key in ['mean_energy','mean_cosine']:
        deviations=[abs(x['moment_uncertainty'][key]['difference_in_standard_errors']) for x in rows
                    if x['moment_uncertainty'][key]['difference_in_standard_errors'] is not None]
        summaries[key+'_z_score']=dict(maximum=max(deviations),over_three=sum(x>3 for x in deviations),
                                        tests=len(deviations),multiple_comparison_correction=False)
    report=dict(status='NATIVE_COMPONENT_PROBE_NOT_YIELD_OR_DOSE_ACCEPTANCE',projectile=projectile,
                completed_nodes=completed,incomplete_nodes=incomplete,summaries=summaries,rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()

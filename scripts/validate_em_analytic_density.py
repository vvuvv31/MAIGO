"""Explicit A/B validation of the offline 26-base analytic ion EM experiment."""
from pathlib import Path
import argparse,json
import numpy as np
import validate_em_density_factorization as v

def compare_results(case,current,candidate,reference_dose,evaluation,mask,spacing,out):
    pts=np.argwhere(mask)
    hit=v.evaluator()(evaluation,reference_dose,pts,spacing,current['dose_percent'],current['dta_mm'],True,step=current['search_step_mm'])
    difference=evaluation[mask]-reference_dose[mask]
    relative=100*abs(difference)/reference_dose[mask]
    record=dict(case=case,current_TOPAS=current,analytic_TOPAS=candidate,
         against_current_pass_percent=float(100*hit.mean()),
         relative_L1=float(abs(difference).sum()/reference_dose[mask].sum()),
         dose_sum_change_percent=float(100*difference.sum()/reference_dose[mask].sum()),
         local_percent_quantiles={str(p):float(np.percentile(relative,p)) for p in [50,95,99,99.9,100]},
         accuracy_accepted=bool(candidate['passed']),default_promoted=False)
    v.save(out/f'{case}_comparison.json',record);print(json.dumps(record),flush=True)
    return record

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['smoke','broadbeam','head-0p1','analyze-broadbeam','analyze-head'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--em',type=Path,required=True);p.add_argument('--delta',type=Path,required=True)
    p.add_argument('--binary',type=Path,default=v.ROOT/'build/carbon_mc')
    p.add_argument('--head-reference-root',type=Path,default=v.ROOT/'out/em_density_factorization_20261005')
    a=p.parse_args();v.OUT=a.output.resolve();v.OUT.mkdir(parents=True,exist_ok=True)
    pair=(a.em.resolve(),a.delta.resolve());assert all(f.is_file() for f in pair)
    def run(case,arm,tag,n,h,seed):
        directory=v.OUT/'runs'/f'{case}_{arm}_{tag}'
        if directory.exists():
            rec=json.loads((directory/'execution.json').read_text());assert rec['runtime_pass'] and rec['n']==n and rec['h']==h and rec['seed']==seed
            # Reuse completed evidence only; incomplete runs require fresh dirs.
            if arm=='analytic':assert (rec['em_package_file'],rec['em_delta_moments_file'])==tuple(map(str,pair))
            print('Reusing completed '+directory.name,flush=True);return
        v.run(case,arm,tag,n,h,seed,a.binary.resolve(),package_pair=pair if arm=='analytic' else None)
    if a.phase=='smoke':
        for case in ['proton_broadbeam','carbon_broadbeam','proton_minibeam','carbon_minibeam','proton_head_minibeam']:
            run(case,'analytic','smoke',2,100000,10579001)
        return
    if a.phase in ['broadbeam','analyze-broadbeam']:
        results=[]
        for case,n,h in [('proton_broadbeam',10,1253150),('carbon_broadbeam',7,925987)]:
            if a.phase=='broadbeam':
                for tag,seed,arms in [('s1',1001904502,['current','analytic']),('s2',1001904602,['analytic','current'])]:
                    for arm in arms:run(case,arm,tag,n,h,seed)
            current,ref,mask,spacing=v.analyze(case,'current',['s1','s2'])
            candidate,ev,_,_=v.analyze(case,'analytic',['s1','s2'])
            results.append(compare_results(case,current,candidate,ref,ev,mask,spacing,v.OUT))
        v.save(v.OUT/'broadbeam_report.json',results);return
    case='proton_head_minibeam';tags=['s1','s2','h01']
    if a.phase=='head-0p1':
        for tag,n,seed in [('s1',12,1001904502),('s2',12,1001904602),('h01',95,1001904702)]:
            # Check reused control physics before launching a costly campaign.
            control=a.head_reference_root.resolve()/'runs'/f'{case}_factorized_{tag}'
            oldcfg=__import__('yaml').safe_load((control/'case.yaml').read_text())
            newcfg=v.config(case);newcfg['random_seed']=seed
            ignored={'em_package_file','em_delta_moments_file','spot_weights_file'}
            differences={k:(oldcfg.get(k),newcfg.get(k)) for k in oldcfg.keys()|newcfg.keys() if k not in ignored and oldcfg.get(k)!=newcfg.get(k)}
            if differences:raise RuntimeError(f'Existing control physics differs: {differences}')
            run(case,'analytic',tag,n,1983920,seed)
    # Existing control has equal histories/seeds/spot allocations. Its old
    # gamma report is recomputed with the exact same evaluator and mask.
    candidate_out=v.OUT;v.OUT=a.head_reference_root.resolve()
    current,ref,mask,spacing=v.analyze(case,'factorized',tags,report_dir=candidate_out)
    v.OUT=candidate_out
    candidate,ev,_,_=v.analyze(case,'analytic',tags)
    compare_results(case,current,candidate,ref,ev,mask,spacing,v.OUT)

if __name__=='__main__':main()

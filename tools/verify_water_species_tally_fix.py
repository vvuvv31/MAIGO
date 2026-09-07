"""Verify diagnostic-only tally change on frozen water and CT GPU runs."""
import argparse,json
from pathlib import Path
import numpy as np
from run_topas10x_gpu_benchmark import sha

def check_pins(pins):
    for path,h in pins.items():
        if sha(Path(path))!=h:raise ValueError('Changed input: '+path)

def energy_closure(ledger):
    layout=ledger['cinel02_species_transport_ledger_layout']
    values=np.array(ledger['cinel02_species_transport_ledger_MeV']).reshape(layout['shape'])
    names=layout['metric'];j=names.index('queued_birth_kinetic')
    sinks=['continuous_deposit_all','nuclear_local_deposit_all','terminal_deposit_all',
           'boundary_escape_kinetic','reaction_export_kinetic','step_limit_escape_kinetic']
    residual=values[:,[names.index(k) for k in sinks]].sum(axis=1)-values[:,j]
    return {s:float(residual[i]/max(values[i,j],1.)) for i,s in enumerate(layout['species'])}

def pair(before,after,ct=False):
    ledgers=[];artifacts=[];failure_codes=[]
    for root in [before,after]:
        report=root/('run_report.json' if ct else 'manifest.json')
        m=json.loads(report.read_text());check_pins(m['inputs'] if ct else m['pins'])
        if sha(root/'dose.raw')!=m['dose_sha256']:raise ValueError('Changed dose')
        q=json.loads((root/'quality_report.json').read_text())
        failure_codes.append([v['code'] for v in q['failures']])
        if q['queue_overflow_count'] or q['queue_overflow_energy_MeV']:raise ValueError('Overflow')
        if not ct and [v['code'] for v in q['failures']]!=['unvalidated_unified_water_transport']:
            raise ValueError('Unexpected quality failure')
        ledgers.append(json.loads((root/'energy_ledger.json').read_text()))
        artifacts.extend([report,root/'dose.raw',root/'quality_report.json',root/'energy_ledger.json'])
    if failure_codes[0]!=failure_codes[1]:raise ValueError('Changed quality failures')
    a=np.fromfile(before/'dose.raw','<f4');b=np.fromfile(after/'dose.raw','<f4')
    if a.shape!=b.shape or not np.isfinite(a).all() or a.max()<=0 or not np.array_equal(a,b):
        raise ValueError('Diagnostic edit changed dose')
    x,y=[v['schneider_diagnostics'] for v in ledgers]
    for k,v in x.items():
        if isinstance(v,int) and not k.startswith('E_') and 'MeV' not in k and v!=y[k]:
            raise ValueError('Changed nuclear count: '+k)
    if ct and ledgers[0]['electron_joint_response']['ordered_path_replays']!=ledgers[1]['electron_joint_response']['ordered_path_replays']:
        raise ValueError('Changed electron replay count')
    closures=[energy_closure(l) for l in ledgers]
    if not ct and max(abs(v) for v in closures[1].values())>1e-3:
        raise ValueError('Species energy closure failed')
    return dict(bitwise_dose_equal=True,integer_nuclear_counts_equal=True,
        species_relative_closure_before=closures[0],species_relative_closure_after=closures[1],
        pins={str(f):sha(f) for f in artifacts})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    dest=p.parse_args().out;root=Path('/mnt/sda/wuwei')
    cases={
        'water_025':pair(root/'unified_water300_origin_20260907',root/'unified_water300_ledger_fix_20260907'),
        'water_00625':pair(root/'unified_water300_step_convergence_20260907/quarter',root/'unified_water300_ledger_fix_quarter_20260907'),
        'ct_50k':pair(root/'unified_water_stoppingfix_ct50k_20260907/after',root/'unified_water_ledgerfix_ct50k_20260907',True)}
    result=dict(status='DIAGNOSTIC_TALLY_FIX_VALIDATED_NOT_PHYSICS_ACCEPTANCE',cases=cases,
        verifier_sha256=sha(Path(__file__).resolve()))
    with dest.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:{x:y for x,y in v.items() if x!='pins'} for k,v in cases.items()},indent=2))

if __name__=='__main__':main()

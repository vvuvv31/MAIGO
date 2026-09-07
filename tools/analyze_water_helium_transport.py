"""Diagnostic helium nuclear-energy audit and fixed-binary step refinement."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_water_helium_birth import rows,key
from compare_unified_water_reference import depth_metrics
from run_topas10x_gpu_benchmark import sha

def verify(pins):
    for path,h in pins.items():
        if sha(Path(path))!=h:raise ValueError('Changed artifact: '+path)

def nuclear_summary(path,n):
    events={};result={a:dict(count=0,pre_step_KE=0.,step_local=0.,continuation_KE=0.,continuation_count=0) for a in [3,4,6]}
    for r in rows(path):
        if r[0]!='interaction' or int(r[10])!=2 or int(r[11]) not in result:continue
        k=key(r)
        if k in events:raise ValueError('Duplicate interaction')
        if float(r[25])!=1:raise ValueError('Unexpected weight')
        mass=int(r[11]);events[k]=(int(r[5]),mass)
        d=result[mass];d['count']+=1;d['pre_step_KE']+=float(r[13])/n;d['step_local']+=float(r[15])/n
    seen=set()
    for r in rows(path):
        k=key(r)
        if r[0]!='product' or k not in events:continue
        track,mass=events[k]
        if int(r[6])!=track or (int(r[10]),int(r[11]))!=(2,mass):continue
        if k in seen:raise ValueError('Duplicate same-track continuation')
        seen.add(k);d=result[mass];d['continuation_count']+=1;d['continuation_KE']+=float(r[13])/n
    for d in result.values():d['pre_step_minus_continuation_KE']=d['pre_step_KE']-d['continuation_KE']
    return result

def gpu_summary(root):
    m=json.loads((root/'manifest.json').read_text());verify(m['pins'])
    if sha(root/'dose.raw')!=m['dose_sha256']:raise ValueError('Changed GPU dose')
    if m['histories']!=50000 or m['generations']!=2:raise ValueError('Unpaired histories/generations')
    q=json.loads((root/'quality_report.json').read_text())
    if [v['code'] for v in q['failures']]!=['unvalidated_unified_water_transport'] or q['queue_overflow_count']:
        raise ValueError('Unexpected quality failure')
    l=json.loads((root/'energy_ledger.json').read_text());layout=l['cinel02_species_transport_ledger_layout']
    values=np.array(l['cinel02_species_transport_ledger_MeV']).reshape(layout['shape'])/m['histories']
    species={s:dict(zip(layout['metric'],values[layout['species'].index(s)])) for s in ['3He','4He','6He']}
    factor=2e-6*6.241509074e12/m['histories']
    dose=np.fromfile(root/'dose.raw','<f4').astype(float).reshape(800,64,64)
    helium=np.fromfile(root/'origin_secondary_helium.raw','<f4').astype(float)
    profile=dose.sum(axis=(1,2))*factor
    return dict(total_MeV_per_primary=float(profile.sum()),helium_origin_MeV_per_primary=float(helium.sum()*factor),
        depth=depth_metrics(profile),species=species)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    root=p.parse_args().out.resolve();verify(json.loads((root/'manifest.json').read_text())['pins'])
    topas=Path('/mnt/sda/wuwei/unified_water300_birth_roi_20260907')
    m=json.loads((topas/'analysis.json').read_text());verify(m['pins'])
    verify(json.loads((topas/'manifest.json').read_text())['pins'])
    cases={'step_0.25':Path('/mnt/sda/wuwei/unified_water300_origin_20260907'),'step_0.125':root/'half','step_0.0625':root/'quarter'}
    result=dict(status='DIAGNOSTIC_NOT_PRODUCTION',topas_roi_nuclear=nuclear_summary(topas/'cascade.phsp',50000),
        gpu={k:gpu_summary(v) for k,v in cases.items()},limitations=[
            'TOPAS nuclear energies are pre-step and ROI-only; GPU export is post-EM and all transported space.',
            'Pre-step minus same-species continuation is not a full nuclear energy balance.',
            'Step changes alter random streams; a single run per step is not a confidence interval.',
            'Nuclear input gap alone does not prove incorrect rate; compare path exposure and spectra.',
            'No production step tuning; generation limit remains unchanged.'])
    artifacts=[topas/'analysis.json',topas/'cascade.phsp',Path(__file__).resolve()]
    for r in cases.values():artifacts.extend([r/'manifest.json',r/'energy_ledger.json',r/'dose.raw',r/'origin_secondary_helium.raw'])
    result['pins']={str(f):sha(f) for f in artifacts}
    with (root/'analysis.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='pins'},indent=2))

if __name__=='__main__':main()

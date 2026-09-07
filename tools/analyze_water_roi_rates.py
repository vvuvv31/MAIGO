"""Same-ROI replay-energy diagnostic and independent G4_WATER rate comparison."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from run_topas10x_gpu_benchmark import sha
from verify_water_species_tally_fix import pair,check_pins

def compare_rates(reference,runtime):
    if reference.ndim!=2 or reference.shape[1]!=5 or not np.isfinite(reference).all():
        raise ValueError('Invalid reference rate payload')
    if np.any(reference[:,3]<0):raise ValueError('Negative reference rates')
    result={}
    for z,a in sorted(set(zip(runtime['Z'].astype(int),runtime['A'].astype(int)))):
        ref=reference[(reference[:,0]==z)&(reference[:,1]==a)]
        rows=runtime[(runtime['Z']==z)&(runtime['A']==a)]
        if len(ref)!=401 or len(rows)!=401 or not np.allclose(ref[:,2],np.arange(401)+.01,rtol=0,atol=.001):
            raise ValueError('Incomplete/duplicate reference energy grid')
        if not np.allclose(rows['E_MeVu'],ref[:,2],rtol=0,atol=.001):raise ValueError('Energy mismatch')
        for key in ['total_per_mm','H_per_mm','O_per_mm']:
            if not np.isfinite(rows[key]).all() or np.any(rows[key]<0):raise ValueError('Invalid runtime rates')
        if not np.allclose(rows['total_per_mm'],rows['H_per_mm']+rows['O_per_mm'],rtol=1e-12,atol=1e-15):
            raise ValueError('Partial sum mismatch')
        covered=(rows['HO_in_domain']==1)&(ref[:,3]>0)
        ratios=rows['total_per_mm'][covered]/ref[covered,3]
        result[f'{z},{a}']=dict(fully_covered_points=int(covered.sum()),
            masked_or_reference_zero_points=int((~covered).sum()),
            minimum_ratio=float(ratios.min()) if len(ratios) else None,
            maximum_ratio=float(ratios.max()) if len(ratios) else None,
            maximum_relative_difference=float(np.max(np.abs(ratios-1))) if len(ratios) else None)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rate-job',type=int,required=True);a=p.parse_args()
    base=Path('/mnt/sda/wuwei');root=base/'unified_water_ion_rate_probe_20260907'
    state=subprocess.check_output(['sacct','-j',str(a.rate_job),'--format=JobIDRaw,State,ExitCode','-n','-P'],text=True)
    if f'{a.rate_job}|COMPLETED|0:0' not in state.splitlines():raise ValueError('Unfinished rate probe')
    check_pins(json.loads((root/'manifest.json').read_text())['pins'])
    topas=base/'unified_water300_step_convergence_20260907/analysis.json'
    t=json.loads(topas.read_text());check_pins(t['pins'])
    gpu=base/'unified_water300_roi_replay_20260907'
    regressions=dict(water=pair(base/'unified_water300_ledger_fix_20260907',gpu),
        ct=pair(base/'unified_water_ledgerfix_ct50k_20260907',base/'unified_water_roi_ct50k_20260907',True))
    l=json.loads((gpu/'energy_ledger.json').read_text());layout=l['cinel02_species_transport_ledger_layout']
    values=np.array(l['cinel02_species_transport_ledger_MeV']).reshape(layout['shape'])/l['histories']
    replay={}
    for s,mass in [('3He','3'),('4He','4'),('6He','6')]:
        v=dict(zip(layout['metric'],values[layout['species'].index(s)]))
        incoming=v['cinel03_replay_input_fov'];all_input=v['reaction_export_kinetic']
        if incoming>all_input*1.0001 or incoming<0:raise ValueError('Invalid ROI subset')
        replay[s]=dict(gpu_all_post_em_input=all_input,gpu_roi_post_em_input=incoming,
            gpu_roi_collision_step_loss=v['cinel03_replay_step_dE_fov'],
            topas_roi_pre_step_input=t['topas_roi_nuclear'][mass]['pre_step_KE'],
            topas_roi_same_track_continuation=t['topas_roi_nuclear'][mass]['continuation_KE'])
    rates=compare_rates(np.loadtxt(root/'rates.phsp'),np.genfromtxt(root/'runtime_rates.csv',delimiter=',',names=True))
    files=[root/'rates.phsp',root/'rates.header',root/'runtime_rates.csv',topas,gpu/'energy_ledger.json',Path(__file__).resolve(),
        Path('tools/dump_water_ion_rates.cpp').resolve(),Path('build/oneapi-nvidia-water-roi/dump_water_ion_rates').resolve(),
        Path('build/oneapi-nvidia-water-roi/libcarbon_core.a').resolve(),Path('src/material_nuclear_rates.cpp').resolve(),
        Path('include/carbon/material_nuclear_rates.hpp').resolve()]
    result=dict(status='DIAGNOSTIC_ONLY_NOT_PRODUCTION',rate_job=a.rate_job,replay_energy_MeV_per_primary=replay,
        direct_rates=rates,regressions=regressions,pins={str(f):sha(f) for f in files},limitations=[
            'TOPAS interaction inputs are pre-step; GPU replay inputs are post-EM. Step loss is reported, not silently corrected.',
            'Rate comparison uses production adapter host evaluation; host/device lookup tests are separate.',
            'Domain-masked or zero-reference points are excluded from ratio summary and counted explicitly.',
            'Rate extraction ASCII precision and interpolation limit direct equality.',
            'Agreement of rate values does not prove exposure or stochastic transport agreement.'])
    with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(replay=result['replay_energy_MeV_per_primary'],helium_rates={k:rates[k] for k in ['2,3','2,4']},
        bitwise_dose_equal={k:v['bitwise_dose_equal'] for k,v in regressions.items()}),indent=2))

if __name__=='__main__':main()

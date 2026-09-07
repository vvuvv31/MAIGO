"""Seal matched half-plan comparison after both full 10-shard runs finish."""
import json,time
from pathlib import Path
import yaml
from run_topas10x_gpu_benchmark import sha

def main():
    base=Path('/mnt/sda/wuwei/electron_ct_half10_20260906/20022516')
    new=Path('/mnt/sda/wuwei/electron_ct_half10_step025_20260906/20022516')
    deadline=time.monotonic()+4*3600
    while not (new/'gamma.json').exists():
        if (new/'execution.json').exists():
            state=json.loads((new/'execution.json').read_text())
            if state['status']=='failed_experiment':raise RuntimeError(state.get('error'))
        if time.monotonic()>deadline:raise TimeoutError('Half-plan comparison deadline')
        time.sleep(15)
    gammas=[];pins={}
    for d in (base,new):
        state=json.loads((d/'execution.json').read_text())
        if state['status']!='complete_experiment' or state['overflow_excluded']:
            raise ValueError('Require complete unsplit paired runs')
        if sum(c['histories'] for c in state['completed'])!=88586520:raise ValueError('History count')
        if sha(d/'gpu_sum.raw')!=state['aggregate_sha256']:raise ValueError('Aggregate pin')
        gammas.append(json.loads((d/'gamma.json').read_text()))
        for file in ('gamma.json','execution.json','gpu_sum.raw','inputs.json'):
            pins[str(d/file)]=sha(d/file)
    for i in range(1,11):
        configs=[]
        for d in (base,new):
            c=yaml.safe_load((d/'half_sources'/f'config_{i:02d}.yaml').read_text())
            c['tps_spots_file']=sha(Path(c['tps_spots_file']));configs.append(c)
        if configs[0].pop('maximum_step_mm')!=.5 or configs[1].pop('maximum_step_mm')!=.25:raise ValueError('Step settings')
        if configs[0]!=configs[1]:raise ValueError('Unpaired config/seed/source')
        reports=[json.loads((d/f'shard_{i:02d}'/'run_report.json').read_text()) for d in (base,new)]
        if reports[0]['binary_sha256']!=reports[1]['binary_sha256']:raise ValueError('Unpaired binary')
    for key in ('histories','reference_sha256','mask_voxels','scale','method'):
        if gammas[0][key]!=gammas[1][key]:raise ValueError('Gamma protocol mismatch '+key)
    a=gammas[0]['results']['new_half10'];b=gammas[1]['results']['new_half10']
    result=dict(status='MATCHED_STEP_REFINEMENT_EXPERIMENT_NOT_PRODUCTION',histories=88586520,
        step05=a,step025=b,delta_percentage_points={k:b[k]-a[k] for k in a},pins=pins,
        limitation='Electron response remains experimental; numerical step sensitivity does not identify the complete microscopic cause')
    target=new/'paired_half10_comparison.json'
    with target.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()

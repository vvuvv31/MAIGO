"""One matched shard at 0.125 mm; frozen binary, no production promotion."""
import json
from pathlib import Path
import numpy as np
import yaml
from run_ct_electron_gamma_probe import run
from run_topas10x_gpu_benchmark import sha, config_write
from evaluate_topas10x_gpu_gamma import pass_mask


def main():
    data=Path('/mnt/sda/wuwei')
    base=data/'electron_ct_half10_20260906/20022516/shard_01'
    fine=data/'ct_step025_shard01_20260906/run'
    out=data/'ct_step0125_shard01_20260907'
    binary=Path('build/oneapi-nvidia-electron-bounds/carbon_mc').resolve()
    joint=data/'schneider_electron_ct_runtime_r3_20260906/joint_response.csv'
    if sha(binary)!='253e9ce8e3a5b15cafd111013438573d12cccf9d7b7d23b5167e47be71fca4fc':
        raise ValueError('Frozen binary changed')
    cfg=yaml.safe_load((fine/'run.yaml').read_text())
    if cfg['maximum_step_mm']!=.25:raise ValueError('Unexpected baseline')
    cfg['maximum_step_mm']=.125
    out.mkdir(exist_ok=False)
    config_write(out/'source.yaml',cfg)
    run(out/'source.yaml',out/'run',binary,joint)
    dirs=[base,fine,out/'run']
    reports=[json.loads((d/'run_report.json').read_text()) for d in dirs]
    configs=[]
    for d,r,step in zip(dirs,reports,(.5,.25,.125)):
        if sha(d/'dose.raw')!=r['dose_sha256'] or sha(d/'run.yaml')!=r['config_sha256']:
            raise ValueError('Dose/config pin mismatch')
        if r['histories']!=8858906 or r['binary_sha256']!=sha(binary):
            raise ValueError('Unmatched histories/binary')
        q=r['quality']
        if q['queue_overflow_count'] or q['queue_overflow_energy_MeV']:
            raise ValueError('Overflow: exclude run')
        if [f['code'] for f in q['failures']]!=['unvalidated_electron_joint_response']:
            raise ValueError('Unexpected quality failure')
        c=yaml.safe_load((d/'run.yaml').read_text())
        if c.pop('maximum_step_mm')!=step:raise ValueError('Step mismatch')
        c['tps_spots_file']=sha(Path(c['tps_spots_file']))
        configs.append(c)
    if not all(c==configs[0] for c in configs):raise ValueError('Unpaired source/config')
    refdir=data/'topas10x_threecase_20260905_r3/20022516'
    m=json.loads((refdir/'manifest.json').read_text())
    if m['mapping']!='native' or sha(refdir/'topas_sum.raw')!=m['reference_sum_sha256']:
        raise ValueError('Reference mapping/pin')
    ref=np.fromfile(refdir/'topas_sum.raw','<f4').reshape(m['gpu_shape_zyx'])
    mask=ref>=.1*ref.max();pts=np.argwhere(mask)
    results={};previous=None;differences={}
    for d,r,step in zip(dirs,reports,('.5','.25','.125')):
        dose=np.fromfile(d/'dose.raw','<f4').reshape(ref.shape).astype(float)*(m['histories']/r['histories'])
        if not np.isfinite(dose).all() or np.any(dose<0):raise ValueError('Invalid dose')
        metrics={}
        for dd,dta in ((3,3),(2,2),(1,1),(3,0)):
            for local in (False,True):
                key=f'{"local" if local else "global"}_{dd}{dta}'
                metrics[key]=100*float(pass_mask(dose,ref,pts,np.array(m['spacing_zyx']),dd,dta,local).mean())
        results[step]=metrics
        if previous is not None:
            differences[step]=float(np.sqrt(np.mean(((dose[mask]-previous)/ref.max())**2))*100)
        previous=dose[mask].copy()
    result=dict(status='EXPERIMENT_NOT_PRODUCTION',histories=8858906,results=results,
        successive_dose_rms_pct_peak=differences,mask_voxels=len(pts),
        reference_sha256=m['reference_sum_sha256'],binary_sha256=sha(binary),
        dose_pins={str(d/'dose.raw'):r['dose_sha256'] for d,r in zip(dirs,reports)},
        limitation='Single-shard numerical convergence; different step counts change RNG sequences and MCS. Not a proof of microscopic cause.')
    with (out/'gamma.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

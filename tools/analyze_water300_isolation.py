"""Summarize EM, repeat seeds, and cascade-cap control without fitting scale."""
import argparse,json
from pathlib import Path
from run_topas10x_gpu_benchmark import sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();base=Path('/mnt/sda/wuwei')
    names={'full_seed1':'unified_water78_e300_50k_20260907',
           'full_seed2':'unified_water300_seed2_20260907',
           'em':'unified_water300_em_20260907'}
    results={};pins={}
    for key,name in names.items():
        path=base/name/'comparison.json';r=json.loads(path.read_text())
        for f,h in r['dose_pins'].items():
            if sha(Path(f))!=h:raise ValueError('Changed dose '+f)
        pins[str(path)]=sha(path);results[key]=r['metrics']
    g1=results['full_seed1']['gpu_MeV_per_primary'];g2=results['full_seed2']['gpu_MeV_per_primary']
    t1=results['full_seed1']['topas_MeV_per_primary'];t2=results['full_seed2']['topas_MeV_per_primary']
    report=dict(status='DIAGNOSTIC_NOT_PRODUCTION',metrics=results,pins=pins,
        gpu_seed_difference_percent=100*(g2-g1)/((g1+g2)/2),
        topas_seed_difference_percent=100*(t2-t1)/((t1+t2)/2),
        mean_full_gpu_over_topas=(g1+g2)/(t1+t2),
        generation3_control='REJECTED_AT_CONFIG: maximum allowed generation cap is2; not measured',
        em_gpu_over_topas=results['em']['gpu_over_topas'],
        limitations=['Two seeds are a repeatability check, not a precise confidence interval.',
                     'Changing generation cap is diagnostic only; do not promote by dose agreement.',
                     'EM excludes nuclear interactions; residual full-physics gap still needs origin/process attribution.'])
    with a.out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

"""Diagnostic-only water step refinement; never migrate physics defaults."""
import argparse,json,subprocess,sys
from pathlib import Path
import yaml
from run_topas10x_gpu_benchmark import sha,config_write

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=a.out.resolve();root.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[1]
    baseline=Path('/mnt/sda/wuwei/unified_water300_origin_20260907/unified_water300_origin_20260907.yaml')
    cfg=yaml.safe_load(baseline.read_text())
    if cfg['maximum_step_mm']!=0.25:raise ValueError('Wrong baseline step')
    binary=repo/'build/oneapi-nvidia-unified-water/carbon_mc'
    if sha(binary)!='77166cf4d1247700b17c1b858b8c2479d3e398aba0046eda02f965bc22429135':
        raise ValueError('Unpaired GPU binary')
    for label,step in [('half',0.125),('quarter',0.0625)]:
        case=dict(cfg,maximum_step_mm=step);config=root/(label+'.yaml');config_write(config,case)
        subprocess.run([sys.executable,str(repo/'tools/probe_unified_water.py'),'--config',str(config),
            '--out',str(root/label),'--histories','50000','--seed','202619071','--generations','2','--origin',
            '--binary',str(binary)],check=True)
    files=[baseline,binary,Path(__file__).resolve(),repo/'tools/probe_unified_water.py']
    (root/'manifest.json').write_text(json.dumps(dict(status='DIAGNOSTIC_ONLY',
        pins={str(f):sha(f) for f in files},steps_mm=[0.25,0.125,0.0625],
        limitations=['Step change also changes random streams; no same-track attribution.',
                    'Not a production step recommendation or independent seed confidence interval.']),indent=2))

if __name__=='__main__':main()

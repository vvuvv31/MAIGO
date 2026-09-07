"""Prepare 100/300 MeV/u water candidates, without changing frozen physics."""
import argparse,json
from pathlib import Path
import yaml
from run_topas10x_gpu_benchmark import config_write,sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    root=p.parse_args().out.resolve();root.mkdir(parents=True,exist_ok=False)
    source=Path('/mnt/sda/wuwei/unified_water78_stopping_20260907/compiled_v2/candidate.yaml')
    cfg=yaml.safe_load(source.read_text())
    for energy,length in ((100,200),(300,400)):
        c=dict(cfg,initial_energy_MeVu=energy,phantom_length_mm=length,
               voxel_bins_z=2*length,number_of_histories=50000)
        config_write(root/f'e{energy}.yaml',c)
    files=[source,*root.glob('*.yaml'),Path(__file__).resolve()]
    (root/'manifest.json').write_text(json.dumps(dict(status='DIAGNOSTIC_ONLY',
        pins={str(f):sha(f) for f in files}),indent=2))

if __name__=='__main__':main()

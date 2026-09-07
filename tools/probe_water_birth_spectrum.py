"""Read actual queued GPU birth spectra using the existing diagnostic switch."""
import argparse,subprocess,sys
from pathlib import Path
import yaml
from run_topas10x_gpu_benchmark import config_write

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    root=p.parse_args().out.resolve();root.mkdir(parents=True,exist_ok=False)
    source=Path('/mnt/sda/wuwei/unified_water300_roi_replay_20260907/unified_water300_roi_replay_20260907.yaml')
    cfg=yaml.safe_load(source.read_text());cfg['fragment_birth_spectrum_output_file']=str(root/'run/birth')
    config_write(root/'input.yaml',cfg)
    subprocess.run([sys.executable,str(Path(__file__).with_name('probe_unified_water.py')),
        '--config',str(root/'input.yaml'),'--out',str(root/'run'),'--histories','50000','--seed','202619071',
        '--generations','2','--origin','--binary','build/oneapi-nvidia-water-roi/carbon_mc'],check=True)

if __name__=='__main__':main()

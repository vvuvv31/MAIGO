"""Reuse the existing dose scoring copy, rather than add a Water parallel world."""
import argparse,json,subprocess,sys
from pathlib import Path
from run_topas10x_gpu_benchmark import sha

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    root=p.parse_args().out.resolve()
    prepare=Path(__file__).with_name('prepare_water_birth_isolation.py')
    subprocess.run([sys.executable,str(prepare),'--out',str(root),'--mode','birth'],check=True)
    config=root/'topas.txt';text=config.read_text()
    old='s:Sc/Cascade/Component = "Water"\nb:Sc/Cascade/PropagateToChildren = "True"'
    if text.count(old)!=1:raise ValueError('Unexpected scorer configuration')
    text=text.replace(old,'s:Sc/Cascade/Component = "ROI"\ni:Sc/Cascade/XBins = 64\ni:Sc/Cascade/YBins = 64\ni:Sc/Cascade/ZBins = 800')
    config.write_text(text)
    path=root/'manifest.json';m=json.loads(path.read_text())
    m['pins'][str(config)]=sha(config);m['pins'][str(Path(__file__).resolve())]=sha(Path(__file__).resolve())
    m['limitations']+=['ROI-only first-tracked census: parents or births outside ROI are not fully covered.',
        'Scorer shares existing ROI 64x64x800 parallel copy; no extra Water propagation copy.']
    m['recording_scope']='ROI';path.write_text(json.dumps(m,indent=2))

if __name__=='__main__':main()

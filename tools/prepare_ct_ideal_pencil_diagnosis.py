"""Ideal monoenergetic pencil in real CT: EM versus full nuclear transport."""
import csv,json,re
from pathlib import Path
import yaml
from run_topas10x_gpu_benchmark import config_write,sha

def main():
    source=Path('/mnt/sda/wuwei/ct_single_spot_residual_20260906/mid')
    root=Path('/mnt/sda/wuwei/ct_ideal_pencil_20260906');root.mkdir(exist_ok=False)
    cfg=yaml.safe_load((source/'gpu.yaml').read_text())
    beam=Path(cfg['tps_beam_model_file'])
    with beam.open() as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    for row in rows:
        for key in fields:
            if key!='energy_MeV':row[key]='0'
    with (root/'beam_model.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    cases=[]
    for label in ('em','nuclear'):
        d=root/label;d.mkdir()
        with (source/'spots.csv').open() as f:spots=list(csv.DictReader(f))
        spots[0]['weight']='50000'
        with (d/'spots.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(spots[0]));w.writeheader();w.writerows(spots)
        c=dict(cfg,number_of_histories=50000,tps_spots_file=str(d/'spots.csv'),tps_beam_model_file=str(root/'beam_model.csv'),random_seed=9186301,
            enable_inelastic=label=='nuclear',enable_secondary_transport=label=='nuclear')
        config_write(d/'gpu.yaml',c);config_write(d/'gpu_step025.yaml',dict(c,maximum_step_mm=.25))
        t=(source/'topas.txt').read_text().replace(str(source),str(d)).replace(str(beam),str(root/'beam_model.csv'))
        t=t.replace('NumberOfThreads = 12','NumberOfThreads = 48').replace('Seed = 9186202','Seed = 9186302')
        if label=='em':t=re.sub(r'^sv:Ph/Default/Modules = .*$', 'sv:Ph/Default/Modules = 2 "g4em-standard_opt4" "g4decay"',t,flags=re.M)
        (d/'topas.txt').write_text(t)
        (d/'run.slurm').write_text(f'#!/bin/bash\n#SBATCH --job-name=ideal_ct_{label}\n#SBATCH --partition=compute\n#SBATCH --nodes=1\n#SBATCH --cpus-per-task=48\n#SBATCH --mem=32G\n#SBATCH --time=01:00:00\n#SBATCH --output={d}/job_%j.log\n#SBATCH --error={d}/job_%j.err\nset -euo pipefail\ncd {d}\n/home/wuwei/topas/topas-build/topas {d}/topas.txt\n')
        cases.append(dict(label=label,directory=str(d),histories=50000,energy_MeVu=155,mode=label))
    pins={str(p):sha(p) for p in root.rglob('*') if p.is_file()}
    for p in (source/'gpu.yaml',source/'topas.txt',beam,Path('/home/wuwei/topas/topas-build/topas')):pins[str(p)]=sha(p)
    (root/'manifest.json').write_text(json.dumps(dict(cases=cases,inputs=pins,status='ISOLATION_DIAGNOSTIC_NOT_PRODUCTION'),indent=2))

if __name__=='__main__':main()

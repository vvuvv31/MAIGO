"""Three clinical single-energy-layer spots in the original patient CT.

Retains clinical 1% energy spread/emittance: this is not an ideal zero-width
monoenergetic beam. TOPAS runs only through local Slurm; GPU is local only.
"""
import csv
import json
import re
from pathlib import Path
import yaml
from run_topas10x_gpu_benchmark import config_write, sha


def main():
    repo=Path(__file__).resolve().parents[1]
    root=Path('/mnt/sda/wuwei/ct_single_spot_residual_20260906')
    root.mkdir(exist_ok=True)
    if any(root.iterdir()):raise FileExistsError(root)
    source=Path('/mnt/sda/wuwei/electron_ct_half10_20260906/20022516/half_sources/config_01.yaml')
    cfg=yaml.safe_load(source.read_text())
    with Path(cfg['tps_spots_file']).open() as f:rows=list(csv.DictReader(f))
    energies=sorted({float(r['energy_MeV']) for r in rows})
    base=repo/'benchmark/topas10x/20022516'
    template=base/'replicas/s1/run_full_plan.txt'
    schneider=repo/'data/HUtoMaterialSchneider.txt'
    assert schneider.is_file()
    cases=[]
    for label,index,threads in [('low',len(energies)//4,8),('mid',len(energies)//2,12),('high',3*len(energies)//4,16)]:
        e=energies[index]
        row=min((r for r in rows if float(r['energy_MeV'])==e),key=lambda r:float(r['x_mm'])**2+float(r['y_mm'])**2)
        d=root/label;d.mkdir()
        row=dict(row,weight=300000)
        with (d/'spots.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
        c=dict(cfg,number_of_histories=300000,tps_spots_file=str(d/'spots.csv'),random_seed=9186201,history_chunk_size=131072)
        config_write(d/'gpu.yaml',c)
        config_write(d/'gpu_step025.yaml',dict(c,maximum_step_mm=.25))
        t=template.read_text()
        t='\n'.join(line for line in t.splitlines() if 'Sc/AllHadronLETd/' not in line and 'Sc/PrimaryC12LETd/' not in line)+'\n'
        t=re.sub(r'i:Ts/NumberOfThreads = \d+',f'i:Ts/NumberOfThreads = {threads}',t)
        t=re.sub(r'i:Ts/Seed = \d+','i:Ts/Seed = 9186202',t)
        t=t.replace('includeFile = HUtoMaterialSchneider.txt',f'includeFile = {schneider}')
        t=t.replace('= "dicom"',f'= "{base / "dicom"}"')
        t=t.replace('= "spots.csv"',f'= "{d / "spots.csv"}"')
        t=t.replace('= "beam_model.csv"',f'= "{Path(c["tps_beam_model_file"])}"')
        t=t.replace('HistoriesScale = 2.5','HistoriesScale = 1.0')
        t=t.replace('= "OSMK_Dtotal_full_plan"',f'= "{d / "dose_topas"}"')
        t=t.replace('= "Overwrite"','= "Exit"')
        (d/'topas.txt').write_text(t)
        (d/'run.slurm').write_text(f'#!/bin/bash\n#SBATCH --job-name=ct_one_{label}\n#SBATCH --partition=compute\n#SBATCH --nodes=1\n#SBATCH --cpus-per-task={threads}\n#SBATCH --mem=16G\n#SBATCH --time=02:00:00\n#SBATCH --output={d}/job_%j.log\n#SBATCH --error={d}/job_%j.err\nset -euo pipefail\ncd {d}\n/home/wuwei/topas/topas-build/topas {d}/topas.txt\n')
        cases.append(dict(label=label,energy_MeVu=e/12,spot=row,threads=threads,mem_GiB=16,directory=str(d)))
    pins={str(p):sha(p) for p in [source,template,schneider,Path(cfg['ct_grid_file']),Path(cfg['tps_beam_model_file']),Path('/home/wuwei/topas/topas-build/topas')]}
    pins.update({str(p):sha(p) for p in root.rglob('*') if p.is_file()})
    (root/'manifest.json').write_text(json.dumps(dict(status='DIAGNOSTIC_NOT_PRODUCTION',cases=cases,inputs=pins,scope='Original CT, clinical single spot with original 1% spread; no source or scale fitting'),indent=2))
    print(json.dumps(cases,indent=2))


if __name__=='__main__':main()

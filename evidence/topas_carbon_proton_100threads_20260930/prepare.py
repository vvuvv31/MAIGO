#!/usr/bin/env python3
"""Matched local TOPAS plan smoke; optimized weights, 100 physical cores."""
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
import numpy as np

REPO=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923')
HERE=Path(__file__).resolve().parent
CASE=REPO/'benchmark/protonCT/RT07575_proton'
TOPAS=Path('/mnt/sda/wuwei/rt07575_proton_dij_20260929/build/topas')
RAW=Path('/mnt/sda/wuwei/topas_carbon_proton_100threads_20260930')
CARBON=Path('/mnt/sda/wuwei/carbon_fe_gamma_20260929/latest_6481909/spots.csv')
CARBON_BEAM=Path('/mnt/sdb/wuwei/MAIGO/benchmark/topas10x/RT07575_pbs_s1/beam_model.csv')
PROTON=REPO/'benchmark/protonCT/RT07575_proton_GPU/inputs/spots_optimized.csv'
N=200000

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

RAW.mkdir(exist_ok=False)
manifest=dict(histories_per_species=N,threads=100,slurm_cpus=100,memory_GiB=24,
              topas=str(TOPAS),topas_sha256=sha(TOPAS),
              comparison='single allocation, sequential carbon then proton, same core affinity, same binary/CT/physics/scorer; distinct optimized source plans',
              source_allocation='largest remainder over full optimized source weights; no spot cherry picking',
              CT=str(CASE/'dicom'),HU=sha(CASE/'HUtoMaterialSchneider.txt'),
              physics='native full 7 modules incl opt4, QGSP_BIC_HP, ion-INCLXX, elastic_HP; no GPU FE substitutions',
              scoring='DoseToMedium Sum only,417x505x35; no LET, no DIJ',species={})
template=(CASE/'full_dij_parts/runs/spot_000001.txt').read_text()
for particle,source,beam in [('carbon',CARBON,CARBON_BEAM),('proton',PROTON,CASE/'beam_model.csv')]:
    directory=RAW/particle
    directory.mkdir()
    (directory/'dicom').symlink_to(CASE/'dicom',target_is_directory=True)
    shutil.copyfile(CASE/'HUtoMaterialSchneider.txt',directory/'HUtoMaterialSchneider.txt')
    shutil.copyfile(beam,directory/'beam_model.csv')
    with source.open() as f:
        reader=csv.DictReader(f);fields,rows=reader.fieldnames,list(reader)
    weights=np.array([float(r['weight']) for r in rows]);exact=weights/weights.sum()*N
    counts=np.floor(exact).astype(np.int64)
    counts[np.argsort(-(exact-counts),kind='stable')[:N-int(counts.sum())]]+=1
    assert counts.sum()==N and np.all(counts[weights==0]==0)
    with (directory/'spots.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows(dict(row,weight=int(count)) for row,count in zip(rows,counts))
    text=template
    text=re.sub(r'^#.*\n','',text,flags=re.M)
    text=re.sub(r'^i:Ts/Seed = .*$', 'i:Ts/Seed = 930075851',text,flags=re.M)
    text=re.sub(r'^i:Ts/NumberOfThreads = .*$', 'i:Ts/NumberOfThreads = 100',text,flags=re.M)
    text=text.replace('"dose/spot_000001_dose"','"dose_plan"')
    text=re.sub(r'^.*So/ProtonPBS/(?:DijMode|DijHistoriesPerSpot|FirstSpot|LastSpot).*$\n?','',text,flags=re.M)
    text+='b:So/ProtonPBS/DijMode = "False"\nb:So/ProtonPBS/SkipZeroWeightSpots = "True"\nu:So/ProtonPBS/HistoriesScale = 1.0\n'
    text+='d:Ph/Default/EMRangeMax = 10 GeV\n'
    if particle=='carbon':
        text=text.replace('So/ProtonPBS/','So/CarbonPBS/').replace('BeamParticle = "proton"','BeamParticle = "GenericIon(6,12,6)"')
    (directory/'run.txt').write_text(text)
    manifest['species'][particle]=dict(directory=str(directory),original_source=str(source),original_source_sha256=sha(source),
             original_weight_sum=float(weights.sum()),original_rows=len(rows),original_nonzero_spots=int((weights>0).sum()),
             allocated_histories=int(counts.sum()),allocated_nonzero_spots=int((counts>0).sum()),
             source_sha256=sha(directory/'spots.csv'),beam_sha256=sha(directory/'beam_model.csv'),config_sha256=sha(directory/'run.txt'),
             energy_MeV_min=min(float(r['energy_MeV']) for r in rows),energy_MeV_max=max(float(r['energy_MeV']) for r in rows))
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(RAW/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))

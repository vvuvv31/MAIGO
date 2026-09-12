#!/usr/bin/env python3
"""Prepare local-only TOPAS matched elastic references; never submits jobs itself."""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--baseline', type=Path, default=Path('/mnt/sda/wuwei/ct_previous_full20_20260909'))
    ap.add_argument('--topas', type=Path, default=Path('/home/wuwei/topas/all-ion-elastic-build-20260911/topas'))
    args = ap.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    binary = root/'topas-frozen'
    if binary.exists():
        raise FileExistsError(binary)
    shutil.copy2(args.topas, binary)
    campaign = {'status': 'prepared_not_accepted', 'node': 'ps', 'cpu_limit': 192, 'memory_limit_GiB': 144, 'topas_sha256': sha(binary), 'cases': []}
    for case in ['RT07575', 'RT06423', '20022516']:
        baseline = json.loads((args.baseline/case/'manifest.json').read_text())
        directory = root/case
        directory.mkdir()
        replicas = []
        for index, old in enumerate(baseline['replicas']):
            source = Path(old['path'])
            run = directory/f'replica_{index+1:02d}'
            run.mkdir()
            for name in ['dicom', 'spots.csv', 'beam_model.csv', 'HUtoMaterialSchneider.txt']:
                target = source/name
                if not target.exists() and name == 'dicom':
                    target = Path(__file__).resolve().parents[1]/'benchmark/topas10x'/case/'dicom'
                if not target.exists() and name == 'HUtoMaterialSchneider.txt':
                    target = Path(__file__).resolve().parents[1]/'data/HUtoMaterialSchneider.txt'
                    if sha(target) != '5022cd89617b28dbd8ee8bf8b095ea20cfd99f6405218693c0df238b3617a139':
                        raise ValueError('Schneider material input pin mismatch')
                target = target.resolve(strict=True)
                (run/name).symlink_to(target)
            original = source/'run_full_plan.txt'
            text = original.read_text()
            pattern = r'(sv:Ph/Default/Modules\s*=\s*)7([^\n]*)'
            match = re.search(pattern, text)
            if not match or 'CarbonIonElasticPhysics' in match.group(2):
                raise ValueError(f'Unexpected physics list: {original}')
            text = re.sub(pattern, lambda m:m.group(1)+'8'+m.group(2)+' "CarbonIonElasticPhysics"', text)
            text = re.sub(r'i:Ts/NumberOfThreads\s*=\s*\d+', 'i:Ts/NumberOfThreads = 64', text)
            # Dose-only: preserve the existing 3D dose grid and source settings.
            text = '\n'.join(line for line in text.splitlines() if not re.match(r'\w+:Sc/(AllHadronLETd|PrimaryC12LETd)/', line))+'\n'
            (run/'run.txt').write_text(text)
            replicas.append({'directory':str(run), 'expected_histories':old['histories'], 'original_config':str(original), 'original_config_sha256':sha(original), 'config_sha256':sha(run/'run.txt'), 'dicom_directory':str((run/'dicom').resolve()), 'inputs':{n:sha(run/n) for n in ['spots.csv','beam_model.csv','HUtoMaterialSchneider.txt']}})
        script = f'''#!/bin/bash
#SBATCH --job-name=elastic_ref_{case}
#SBATCH --partition=compute
#SBATCH --nodelist=ps
#SBATCH --nodes=1
#SBATCH --array=1-{len(replicas)}%1
#SBATCH --cpus-per-task=64
#SBATCH --mem=48G
#SBATCH --output={directory}/job_%A_%a.log
#SBATCH --error={directory}/job_%A_%a.err
set -euo pipefail
source /software/geant4-11.3.2/bin/geant4.sh
replica_name=$(printf 'replica_%02d' "$SLURM_ARRAY_TASK_ID")
cd "{directory}/$replica_name"
"{binary}" run.txt
'''
        (directory/'submit.slurm').write_text(script)
        campaign['cases'].append({'case':case, 'histories':baseline['histories'], 'replicas':replicas, 'submit_script':str(directory/'submit.slurm')})
    (root/'campaign.json').write_text(json.dumps(campaign,indent=2)+'\n')
    print(root/'campaign.json')


if __name__ == '__main__':
    main()

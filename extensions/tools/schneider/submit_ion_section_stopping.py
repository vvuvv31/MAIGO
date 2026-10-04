#!/usr/bin/env python3
"""Prepare/submit a local single-thread TOPAS material stopping extraction.
Requires an independently built extractor binary. Never edits a reference build.
"""
import argparse, json, os, shlex, subprocess
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
RUN_FILE = 'includeFile = {hu_file}\n\ns:Ge/World/Material = "Vacuum"\nd:Ge/World/HLX = 1.0 m\nd:Ge/World/HLY = 1.0 m\nd:Ge/World/HLZ = 1.0 m\nb:Ge/World/Invisible = "TRUE"\n\ns:Ge/Patient/Parent = "World"\ns:Ge/Patient/Material = "G4_WATER"\ns:Ge/Patient/Type = "TsDicomPatient"\ns:Ge/Patient/DicomDirectory = "{dicom_dir}"\nb:Ge/Patient/PreLoadAllMaterials = "True"\n\ni:Ts/NumberOfThreads = 1\ni:Ts/Seed = 918711\nsv:Ph/Default/Modules = 7 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4decay" "g4ion-inclxx" "g4h-elastic_HP" "g4stopping" "g4radioactivedecay"\ns:Ge/Probe/Type = "TsBox"\ns:Ge/Probe/Parent = "World"\ns:Ge/Probe/Material = "G4_WATER"\nd:Ge/Probe/HLX = 1 mm\nd:Ge/Probe/HLY = 1 mm\nd:Ge/Probe/HLZ = 1 mm\nd:Ge/Probe/TransZ = -800 mm\ns:Ge/BeamPosition/Type = "Group"\nd:Ge/BeamPosition/TransZ = -800 mm\ns:Ge/BeamPosition/Parent = "World"\n\ns:So/Demo/Type = "Beam"\ns:So/Demo/Component = "BeamPosition"\ns:So/Demo/BeamParticle = "GenericIon(6,12)"\nd:So/Demo/BeamEnergy = 200.0 MeV\nu:So/Demo/BeamEnergySpread = 0.0\ns:So/Demo/BeamPositionDistribution = "None"\ns:So/Demo/BeamAngularDistribution = "None"\ni:So/Demo/NumberOfHistoriesInRun = 1\n\ns:Sc/IonStoppingDump/Quantity = "IonSchneiderStoppingPowerDump"\ns:Sc/IonStoppingDump/Component = "Probe"\ns:Sc/IonStoppingDump/OutputJsonFile = "{out}/raw/topas_ion_schneider_stopping.json"\ns:Sc/IonStoppingDump/OutputCsvFile = "{out}/raw/topas_ion_schneider_stopping.csv"\nd:Sc/IonStoppingDump/MinEnergyMeVu = 0.01 MeV\nd:Sc/IonStoppingDump/MaxEnergyMeVu = 6000.11 MeV\nd:Sc/IonStoppingDump/EnergyStepMeVu = 0.1 MeV\ns:Sc/IonStoppingDump/IfOutputFileAlreadyExists = "Exit"\n'

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--topas', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--dicom-dir', required=True, type=Path)
    ap.add_argument('--submit', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if not a.topas.is_file():
        raise ValueError('Missing extraction binary')
    if a.dry_run:
        print(RUN_FILE.format(out=a.out, hu_file=(REPO / 'data/HUtoMaterialSchneider.txt').resolve(), dicom_dir=a.dicom_dir.resolve()))
        return
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'raw').mkdir()
    run = a.out / 'run.txt'
    run.write_text(RUN_FILE.format(out=a.out, hu_file=(REPO / 'data/HUtoMaterialSchneider.txt').resolve(), dicom_dir=a.dicom_dir.resolve()))
    sb = a.out / 'submit.slurm'
    sb.write_text('#!/bin/bash\n#SBATCH --job-name=ion_schneider_sp\n#SBATCH --partition=compute\n#SBATCH --nodes=1\n#SBATCH --cpus-per-task=1\n#SBATCH --mem=10G\n#SBATCH --output=' + str(a.out / 'job_%j.log') + '\n#SBATCH --error=' + str(a.out / 'job_%j.err') + '\nset -euo pipefail\ncd ' + shlex.quote(str(a.out)) + '\nsource /software/geant4-11.3.2/bin/geant4.sh\n' + shlex.quote(str(a.topas)) + ' ' + shlex.quote(str(run)) + '\n')
    pins = {'binary': str(a.topas)}
    dicom = a.dicom_dir
    pins['dicom'] = [str(p) for p in sorted(dicom.rglob('*')) if p.is_file()]
    if not pins['dicom']:
        raise ValueError('No DICOM inputs')
    (a.out / 'manifest.json').write_text(json.dumps(pins, indent=2) + '\n')
    if a.submit:
        rows = subprocess.check_output(['squeue', '-u', os.environ.get('USER', 'wuwei'), '-h', '-o', '%C|%m'], text=True)
        cpu = 0
        mem = 0
        for line in rows.splitlines():
            c, m = line.split('|')
            cpu += int(c)
            unit = m[-1]
            v = float(m[:-1]) if unit in 'KMG' else float(m)
            mem += v * {'K': 1 / 1048576, 'M': 1 / 1024, 'G': 1}.get(unit, 1 / 1024)
        if cpu + 1 > 192 or mem + 10 > 160:
            raise RuntimeError('Insufficient job resource headroom')
        pins['job_id'] = subprocess.check_output(['sbatch', '--parsable', str(sb)], text=True).strip()
        (a.out / 'manifest.json').write_text(json.dumps(pins, indent=2) + '\n')
        print(pins['job_id'])
    else:
        print(sb)
if __name__ == '__main__':
    main()

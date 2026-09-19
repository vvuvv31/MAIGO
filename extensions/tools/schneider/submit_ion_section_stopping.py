#!/usr/bin/env python3
"""Prepare/submit a local single-thread TOPAS material stopping extraction.
Requires an independently built extractor binary. Never edits a reference build.
"""
import argparse, hashlib, json, os, shlex, subprocess
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

RUN_FILE = """includeFile = {hu_file}

s:Ge/World/Material = "Vacuum"
d:Ge/World/HLX = 1.0 m
d:Ge/World/HLY = 1.0 m
d:Ge/World/HLZ = 1.0 m
b:Ge/World/Invisible = "TRUE"

s:Ge/Patient/Parent = "World"
s:Ge/Patient/Material = "G4_WATER"
s:Ge/Patient/Type = "TsDicomPatient"
s:Ge/Patient/DicomDirectory = "{dicom_dir}"
b:Ge/Patient/PreLoadAllMaterials = "True"

i:Ts/NumberOfThreads = 1
i:Ts/Seed = 918711
sv:Ph/Default/Modules = 7 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4decay" "g4ion-inclxx" "g4h-elastic_HP" "g4stopping" "g4radioactivedecay"
s:Ge/Probe/Type = "TsBox"
s:Ge/Probe/Parent = "World"
s:Ge/Probe/Material = "G4_WATER"
d:Ge/Probe/HLX = 1 mm
d:Ge/Probe/HLY = 1 mm
d:Ge/Probe/HLZ = 1 mm
d:Ge/Probe/TransZ = -800 mm
s:Ge/BeamPosition/Type = "Group"
d:Ge/BeamPosition/TransZ = -800 mm
s:Ge/BeamPosition/Parent = "World"

s:So/Demo/Type = "Beam"
s:So/Demo/Component = "BeamPosition"
s:So/Demo/BeamParticle = "GenericIon(6,12)"
d:So/Demo/BeamEnergy = 200.0 MeV
u:So/Demo/BeamEnergySpread = 0.0
s:So/Demo/BeamPositionDistribution = "None"
s:So/Demo/BeamAngularDistribution = "None"
i:So/Demo/NumberOfHistoriesInRun = 1

s:Sc/IonStoppingDump/Quantity = "IonSchneiderStoppingPowerDump"
s:Sc/IonStoppingDump/Component = "Probe"
s:Sc/IonStoppingDump/OutputJsonFile = "{out}/raw/topas_ion_schneider_stopping.json"
s:Sc/IonStoppingDump/OutputCsvFile = "{out}/raw/topas_ion_schneider_stopping.csv"
d:Sc/IonStoppingDump/MinEnergyMeVu = 0.01 MeV
d:Sc/IonStoppingDump/MaxEnergyMeVu = 6000.11 MeV
d:Sc/IonStoppingDump/EnergyStepMeVu = 0.1 MeV
s:Sc/IonStoppingDump/IfOutputFileAlreadyExists = "Exit"
"""

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--topas', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--dicom-dir', required=True, type=Path)
    ap.add_argument('--submit', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a=ap.parse_args()
    if not a.topas.is_file(): raise ValueError('Missing extraction binary')
    if a.dry_run:
        print(RUN_FILE.format(out=a.out, hu_file=(REPO/'data/HUtoMaterialSchneider.txt').resolve(), dicom_dir=a.dicom_dir.resolve())); return
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'raw').mkdir()
    run=a.out/'run.txt'; run.write_text(RUN_FILE.format(out=a.out, hu_file=(REPO/'data/HUtoMaterialSchneider.txt').resolve(), dicom_dir=a.dicom_dir.resolve()))
    sb=a.out/'submit.slurm'
    sb.write_text('#!/bin/bash\n#SBATCH --job-name=ion_schneider_sp\n#SBATCH --partition=compute\n#SBATCH --nodes=1\n#SBATCH --cpus-per-task=1\n#SBATCH --mem=10G\n#SBATCH --output='+str(a.out/'job_%j.log')+'\n#SBATCH --error='+str(a.out/'job_%j.err')+'\nset -euo pipefail\ncd '+shlex.quote(str(a.out))+'\nsource /software/geant4-11.3.2/bin/geant4.sh\n'+shlex.quote(str(a.topas))+' '+shlex.quote(str(run))+'\n')
    pins={'binary':str(a.topas),'binary_sha256':sha(a.topas),'run_sha256':sha(run),
          'hu_sha256':sha(REPO/'data/HUtoMaterialSchneider.txt'),
          'source_sha256':sha(REPO/'extensions/topas/common/IonSchneiderStoppingPowerDump.cc')}
    # Freeze all external DICOM inputs as well as executable and run file.
    dicom=a.dicom_dir
    pins['dicom']={str(p):sha(p) for p in sorted(dicom.rglob('*')) if p.is_file()}
    if not pins['dicom']: raise ValueError('No DICOM inputs')
    (a.out/'manifest.json').write_text(json.dumps(pins,indent=2)+'\n')
    if a.submit:
        rows=subprocess.check_output(['squeue','-u',os.environ.get('USER','wuwei'),'-h','-o','%C|%m'],text=True)
        cpu=0; mem=0
        for line in rows.splitlines():
            c,m=line.split('|');cpu+=int(c); unit=m[-1];v=float(m[:-1]) if unit in 'KMG' else float(m)
            mem+=v*({'K':1/1048576,'M':1/1024,'G':1}.get(unit,1/1024))
        if cpu+1>192 or mem+10>160: raise RuntimeError('Insufficient job resource headroom')
        pins['job_id']=subprocess.check_output(['sbatch','--parsable',str(sb)],text=True).strip()
        (a.out/'manifest.json').write_text(json.dumps(pins,indent=2)+'\n')
        print(pins['job_id'])
    else: print(sb)
if __name__=='__main__':main()

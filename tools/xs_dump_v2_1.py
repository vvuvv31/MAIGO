#!/usr/bin/env python3
"""Rate v2.1 XS dump wave: deterministic Geant4 XS over [0.1, 460.1] step 0.5.

921 grid points per projectile (vs v1 860 over [0.5,430]). 13 local sbatch
jobs (12 CPUs/8G each, deterministic 1-history dumps, minutes each).
Work dir /mnt/sda/wuwei/secondary-rates-v2-1/ (v1 dir untouched).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/mnt/sdb/wuwei/MAIGO")
WORK = Path("/mnt/sda/wuwei/secondary-rates-v2-1")
TOPAS_BIN = "/home/wuwei/topas/topas-build/topas"
TOPAS_G4_DATA = "/software/geant4-11.3.2/share/Geant4/data"
TOPAS_LD_PATH = "/software/topas/lib:/software/geant4-11.3.2/lib"

PROJS = [
    ("b11", "B11", 5, 11, "GenericIon(5,11)"),
    ("b10", "B10", 5, 10, "GenericIon(5,10)"),
    ("be9", "Be9", 4, 9, "GenericIon(4,9)"),
    ("be7", "Be7", 4, 7, "GenericIon(4,7)"),
    ("be10", "Be10", 4, 10, "GenericIon(4,10)"),
    ("li7", "Li7", 3, 7, "GenericIon(3,7)"),
    ("li6", "Li6", 3, 6, "GenericIon(3,6)"),
    ("he4", "He4", 2, 4, "alpha"),
    ("he3", "He3", 2, 3, "He3"),
    ("h1", "H1", 1, 1, "proton"),
    ("h2", "H2", 1, 2, "deuteron"),
    ("h3", "H3", 1, 3, "triton"),
    ("c11", "C11", 6, 11, "GenericIon(6,11)"),
]
EMIN, EMAX, ESTEP = 0.1, 460.1, 0.5


def main():
    submit = "--submit" in sys.argv
    WORK.mkdir(parents=True, exist_ok=True)
    raw = WORK / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    jids = []
    for pid, sym, z, a, part in PROJS:
        param = WORK / f"{pid}_param.txt"
        param.write_text(f"""includeFile = {REPO}/data/HUtoMaterialSchneider.txt

s:Ge/World/Material = "Vacuum"
d:Ge/World/HLX = 100.0 m
d:Ge/World/HLY = 100.0 m
d:Ge/World/HLZ = 100.0 m
b:Ge/QuitIfOverlapDetected = "False"

s:Ge/Patient/Parent = "World"
s:Ge/Patient/Material = "G4_WATER"
s:Ge/Patient/Type = "TsDicomPatient"
s:Ge/Patient/DicomDirectory = "/mnt/sda/wuwei/maigo-ct-schneider/step-03/DICOM_Box/DICOM_Box"
b:Ge/Patient/PreLoadAllMaterials = "True"
d:Ge/Patient/TransX = -45.0 m
b:Ge/Patient/Invisible = "TRUE"

s:Ge/Box/Parent = "World"
s:Ge/Box/Type = "TsBox"
s:Ge/Box/Material = "G4_WATER"
d:Ge/Box/HLX = 10.0 mm
d:Ge/Box/HLY = 10.0 mm
d:Ge/Box/HLZ = 10.0 mm

sv:Ph/Default/Modules = 6 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4decay" "g4ion-inclxx" "g4h-elastic_HP" "g4stopping"

s:So/Source/Type = "Beam"
s:So/Source/Component = "Box"
s:So/Source/BeamParticle = "{part}"
d:So/Source/BeamEnergy = {100.0 * a:.1f} MeV
u:So/Source/BeamEnergySpread = 0.0
s:So/Source/BeamPositionDistribution = "None"
s:So/Source/BeamAngularDistribution = "None"
i:So/Source/NumberOfHistoriesInRun = 1

s:Sc/XsDump/Quantity = "CarbonSchneiderInelasticXsDump"
s:Sc/XsDump/Component = "Box"
s:Sc/XsDump/OutputFile = "{raw}/{pid}_dump.json"
s:Sc/XsDump/OutputCsvFile = "{raw}/{pid}_dump.csv"
i:Sc/XsDump/ProjectileZ = {z}
i:Sc/XsDump/ProjectileA = {a}
d:Sc/XsDump/MinEnergyMeVu = {EMIN} MeV
d:Sc/XsDump/MaxEnergyMeVu = {EMAX} MeV
d:Sc/XsDump/EnergyStepMeVu = {ESTEP} MeV
""")
        slurm = WORK / f"run_{pid}.sh"
        slurm.write_text(f"""#!/bin/bash
#SBATCH --job-name=xs21_{pid}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=/mnt/sda/wuwei/job_%j.log
#SBATCH --error=/mnt/sda/wuwei/job_%j.err

export TOPAS_G4_DATA_DIR={TOPAS_G4_DATA}
export LD_LIBRARY_PATH={TOPAS_LD_PATH}:$LD_LIBRARY_PATH

{TOPAS_BIN} {param}
""")
        slurm.chmod(0o755)
        if submit:
            r = subprocess.run(["sbatch", str(slurm)], capture_output=True,
                               text=True, check=True)
            jid = r.stdout.strip().split()[-1]
            jids.append(jid)
            print(f"  {pid}: job {jid}")
    if submit:
        print(f"submitted {len(jids)} jobs; 13x4=52 CPUs")
    else:
        print("generated (dry run, use --submit)")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Scheme-2 data campaign: per-ion per-Schneider-section Geant4 stopping.

Chain (all steps pinned, fail-closed):
  1. TOPAS extension IonSchneiderStoppingPowerDump (already placed at
     /home/wuwei/topas/extensions/): 25 sections x 18 species x 4302
     energies, UNRESTRICTED electronic dE/dx, DICOM_Box Schneider patient.
  2. Register scorer: TsExtensionManager.cc if-chain +
     topas-build/extensions/CMakeLists.txt lists (see --apply-topas-registration).
  3. Rebuild TOPAS extension lib (never touch the frozen binary in place;
     build a campaign binary or verify extension reload path first).
  4. sbatch 1-history calculator job (G4EmCalculator, seconds-minutes).
  5. tools/compile_schneider_ion_stopping.py -> SCHNIOSP v1 + manifest
     (C12 cross-check vs validated Schneider table, E>=5 MeV/u, tol 2%).
  6. tools/verify_schneider_v2_1_data.py extension + host/device lookup
     tests + 50k closure gates before any production use.

Usage:
  python3 tools/submit_ion_section_stopping.py --dry-run   # print all steps
  python3 tools/submit_ion_section_stopping.py --apply-topas-registration
  python3 tools/submit_ion_section_stopping.py --submit    # sbatch the job
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXT_SRC = Path("/home/wuwei/topas/extensions")
BUILD_EXT = Path("/home/wuwei/topas/topas-build/extensions")
OUT = Path("/mnt/sda/wuwei/ion_section_stopping")

RUN_FILE = """includeFile = /mnt/sdb/wuwei/MAIGO/data/HUtoMaterialSchneider.txt

s:Ge/World/Material = "Vacuum"
d:Ge/World/HLX = 1.0 m
d:Ge/World/HLY = 1.0 m
d:Ge/World/HLZ = 1.0 m
b:Ge/World/Invisible = "TRUE"

s:Ge/Patient/Parent = "World"
s:Ge/Patient/Material = "G4_WATER"
s:Ge/Patient/Type = "TsDicomPatient"
s:Ge/Patient/DicomDirectory = "/mnt/sda/wuwei/maigo-ct-schneider/step-03/DICOM_Box/DICOM_Box"
b:Ge/Patient/PreLoadAllMaterials = "True"

sv:Ph/Default/Modules = 6 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4decay" "g4ion-binarycascade" "g4h-elastic_HP" "g4stopping"

s:So/Demo/Type = "Beam"
s:So/Demo/Component = "BeamPosition"
s:So/Demo/BeamParticle = "GenericIon(6,12)"
d:So/Demo/BeamEnergy = 200.0 MeV
u:So/Demo/BeamEnergySpread = 0.0
s:So/Demo/BeamPositionDistribution = "None"
s:So/Demo/BeamAngularDistribution = "None"
i:So/Demo/NumberOfHistoriesInRun = 1

s:Sc/IonStoppingDump/Quantity = "IonSchneiderStoppingPowerDump"
s:Sc/IonStoppingDump/Component = "Patient"
s:Sc/IonStoppingDump/OutputJsonFile = "{out}/raw/topas_ion_schneider_stopping.json"
s:Sc/IonStoppingDump/OutputCsvFile = "{out}/raw/topas_ion_schneider_stopping.csv"
d:Sc/IonStoppingDump/MinEnergyMeVu = 0.01 MeV
d:Sc/IonStoppingDump/MaxEnergyMeVu = 430.11 MeV
d:Sc/IonStoppingDump/EnergyStepMeVu = 0.1 MeV
s:Sc/IonStoppingDump/IfOutputFileAlreadyExists = "Overwrite"
"""

SBATCH = """#!/bin/bash
#SBATCH --job-name=ion_schneider_sp
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G
#SBATCH --output=/mnt/sda/%u/job_%j.log
#SBATCH --error=/mnt/sda/%u/job_%j.err
{topas} --config {runfile}
"""


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def registration_edits():
    mgr = BUILD_EXT / "TsExtensionManager.cc"
    text = mgr.read_text()
    anchor = ('\t\t\t\t\tif (quantityNameLower=="carbonschneiderstoppingpowerdump")\n'
              '\t\t\t\t\treturn new CarbonSchneiderStoppingPowerDump(pM, mM, gM, scM, this, '
              'currentScorerName, quantityName, outFileName, isSubScorer);\n')
    assert anchor in text, "manager anchor changed"
    add = anchor + ('\t\t\t\t\tif (quantityNameLower=="ionschneiderstoppingpowerdump")\n'
                    '\t\t\t\t\treturn new IonSchneiderStoppingPowerDump(pM, mM, gM, scM, this, '
                    'currentScorerName, quantityName, outFileName, isSubScorer);\n')
    include = '#include "CarbonSchneiderStoppingPowerDump.hh"\n'
    assert include in text, "manager include anchor changed"
    return [(str(mgr), include, include + '#include "IonSchneiderStoppingPowerDump.hh"\n'),
            (str(mgr), anchor, add)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply-topas-registration", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--topas", default="/home/wuwei/topas/topas-frozen-d50f90504eafe2b2")
    a = ap.parse_args()

    for f in ("IonSchneiderStoppingPowerDump.cc", "IonSchneiderStoppingPowerDump.hh"):
        assert (EXT_SRC / f).exists(), f"missing extension source {f}"
        print(f"extension source present: {f} sha={sha(EXT_SRC / f)[:12]}")

    steps = [
        "1. copy IonSchneiderStoppingPowerDump.{cc,hh} into topas-build/extensions/",
        "2. TsExtensionManager.{cc,hh}: include + if-chain entry",
        "3. topas-build/extensions/CMakeLists.txt: header+source lists",
        "4. rebuild TOPAS extension lib (frozen binary untouched)",
        "5. sbatch calculator job -> raw CSV+JSON",
        "6. tools/compile_schneider_ion_stopping.py -> SCHNIOSP v1 + manifest",
        "7. extend manifest verifier + lookup tests + 50k gates",
    ]
    print("\n".join(steps))

    if a.dry_run and not (a.apply_topas_registration or a.submit):
        print("dry-run only; no changes made")
        return

    if a.apply_topas_registration:
        for name in ("IonSchneiderStoppingPowerDump.cc", "IonSchneiderStoppingPowerDump.hh"):
            dst = BUILD_EXT / name
            if dst.exists():
                assert sha(dst) == sha(EXT_SRC / name), f"build copy differs: {name}"
                print(f"build copy already in sync: {name}")
            else:
                shutil.copy2(EXT_SRC / name, dst)
                print(f"copied to build tree: {name}")
        for path, old, new in registration_edits():
            text = Path(path).read_text()
            if new in text:
                print(f"registration already present in {path}")
            else:
                Path(path).write_text(text.replace(old, new))
                print(f"patched registration in {path}")
        print("CMakeLists list entries still required (manual review): "
              "IonSchneiderStoppingPowerDump.hh/.cc")
        return

    if a.submit:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "raw").mkdir(exist_ok=True)
        runfile = OUT / "run_ion_schneider_stopping_dump.txt"
        runfile.write_text(RUN_FILE.format(out=OUT))
        sb = OUT / "submit.slurm"
        sb.write_text(SBATCH.format(topas=a.topas, runfile=runfile))
        print(f"wrote {runfile} and {sb}")
        print("NOTE: build the extension into the campaign binary BEFORE submitting.")
        reply = input("Submit now with sbatch? [y/N] ")
        if reply.strip().lower() != "y":
            print("not submitted")
            return
        subprocess.run(["sbatch", str(sb)], check=True)


if __name__ == "__main__":
    main()

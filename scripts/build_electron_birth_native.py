#!/usr/bin/env python3
"""Isolated native birth-only scorer build; official TOPAS reference unchanged."""
import json,subprocess,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    old=ROOT/'out/electron_joint_family_20261003/gs_displacement';spec=json.loads((old/'build_manifest.json').read_text());out=ROOT/'out/electron_standalone_20261003/native_build';out.mkdir(exist_ok=True)
    for name in ['ProtonElectronBirthNtuple.cc','ProtonElectronBirthNtuple.hh']:shutil.copy2(ROOT/'benchmark/electron_response_model/native'/name,out/name)
    manager=(old/'TsExtensionManager.cc').read_text().replace('#include "ElectronFamilyProfile.hh"','#include "ProtonElectronBirthNtuple.hh"')
    manager=manager.replace('quantityNameLower=="electronfamilyprofile"','quantityNameLower=="protonelectronbirthntuple"').replace('new ElectronFamilyProfile(','new ProtonElectronBirthNtuple(')
    (out/'TsExtensionManager.cc').write_text(manager)
    commands=[]
    for name in ['ProtonElectronBirthNtuple','TsExtensionManager']:
        cmd=[v.replace(str(old),str(out)).replace('ElectronFamilyProfile',name) for v in spec['commands'][0]]
        commands.append(cmd);subprocess.run(cmd,check=True,cwd=spec['base_build'])
    cmd=[]
    for arg in spec['commands'][-1]:
        if arg==str(old/'G4GoudsmitSaundersonMscModel.o'):continue # birth extraction uses original native EM implementation
        cmd.append(arg.replace(str(old),str(out)).replace('ElectronFamilyProfile.o','ProtonElectronBirthNtuple.o'))
    commands.append(cmd);subprocess.run(cmd,check=True,cwd=spec['base_build'])
    record=dict(binary_sha256=sha(out/'topas'),source_sha256={n:sha(out/n) for n in ['ProtonElectronBirthNtuple.cc','ProtonElectronBirthNtuple.hh','TsExtensionManager.cc']},commands=commands,base_build=spec['base_build'],native_inputs=spec['native_inputs'],physics_changes='none relative to base extraction binary; no GS override; scorer consumes no RNG and never kills a track')
    (out/'build_manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(record['binary_sha256'])
if __name__=='__main__':main()

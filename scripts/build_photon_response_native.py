#!/usr/bin/env python3
"""Build isolated read-only photon response recorder; no physics override."""
import json,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    old=ROOT/'out/electron_standalone_20261003/native_build';spec=json.loads((old/'build_manifest.json').read_text());out=ROOT/'out/electron_standalone_20261003/photon_native';out.mkdir(exist_ok=False)
    for n in ['PhotonResponseNtuple.cc','PhotonResponseNtuple.hh','ElectronExtractionRange.hh']:shutil.copy2(ROOT/'benchmark/electron_response_model/native'/n,out/n)
    manager=(old/'TsExtensionManager.cc').read_text().replace('ProtonElectronBirthNtuple','PhotonResponseNtuple').replace('protonelectronbirthntuple','photonresponsentuple');(out/'TsExtensionManager.cc').write_text(manager)
    commands=[]
    for n in ['PhotonResponseNtuple','TsExtensionManager']:
        cmd=[v.replace(str(old),str(out)).replace('ProtonElectronBirthNtuple',n) for v in spec['commands'][0]]
        commands.append(cmd);subprocess.run(cmd,cwd=spec['base_build'],check=True)
    cmd=[v.replace(str(old),str(out)).replace('ProtonElectronBirthNtuple','PhotonResponseNtuple') for v in spec['commands'][-1]]
    commands.append(cmd);subprocess.run(cmd,cwd=spec['base_build'],check=True)
    record=dict(source_files=['PhotonResponseNtuple.cc','PhotonResponseNtuple.hh','ElectronExtractionRange.hh','TsExtensionManager.cc'],base_build=spec['base_build'],commands=commands,native_inputs=spec['native_inputs'],physics_changes='none; read-only scorer and optional native cross-section queries, no GS override')
    (out/'build_manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(out/'topas')
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Build a fixed hard-event-energy native ion-birth diagnostic, not a scorer."""
import argparse,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    old=ROOT/'out/electron_standalone_20261003/photon_native';spec=json.loads((old/'build_manifest.json').read_text())
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);name='IonBirthConditionalDump'
    for n in [name+'.cc',name+'.hh','ElectronExtractionRange.hh']:
        shutil.copy2(ROOT/'benchmark/electron_response_model/native'/n,out/n)
    (out/'TsExtensionManager.cc').write_text((old/'TsExtensionManager.cc').read_text().replace('PhotonResponseNtuple',name).replace('photonresponsentuple',name.lower()))
    commands=[]
    for n in [name,'TsExtensionManager']:
        cmd=[v.replace(str(old),str(out)).replace('PhotonResponseNtuple',n) for v in spec['commands'][0]]
        commands.append(cmd);subprocess.run(cmd,cwd=spec['base_build'],check=True)
    cmd=[v.replace(str(old),str(out)).replace('PhotonResponseNtuple',name) for v in spec['commands'][-1]]
    commands.append(cmd);subprocess.run(cmd,cwd=spec['base_build'],check=True)
    (out/'build_manifest.json').write_text(json.dumps(dict(commands=commands,base_build=spec['base_build'],scope='Native GPIL/PostStepDoIt at fixed hard-event energy; no analog dose replacement'),indent=2)+'\n')
    print(out/'topas')
if __name__=='__main__':main()

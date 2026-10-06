#!/usr/bin/env python3
"""Build offline model-level rates/outcomes at explicit physical cuts."""
import argparse,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    old=ROOT/'out/electron_standalone_20261003/photon_native';spec=json.loads((old/'build_manifest.json').read_text())
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);name='ElectronCompositionProbe'
    for suffix in ['.cc','.hh']:shutil.copy2(ROOT/'benchmark/electron_response_model/native'/(name+suffix),out/(name+suffix))
    (out/'TsExtensionManager.cc').write_text((old/'TsExtensionManager.cc').read_text().replace('PhotonResponseNtuple',name).replace('photonresponsentuple',name.lower()))
    commands=[]
    for source in [name,'TsExtensionManager']:
        cmd=[v.replace(str(old),str(out)).replace('PhotonResponseNtuple',source) for v in spec['commands'][0]]
        commands.append(cmd);subprocess.run(cmd,cwd=spec['base_build'],check=True)
    cmd=[v.replace(str(old),str(out)).replace('PhotonResponseNtuple',name) for v in spec['commands'][-1]]
    commands.append(cmd);subprocess.run(cmd,cwd=spec['base_build'],check=True)
    (out/'build_manifest.json').write_text(json.dumps(dict(commands=commands,base_build=spec['base_build'],scope='Offline direct native models; explicit cuts; full correlated event branches; not a dose scorer or qualified package'),indent=2)+'\n')
    print(out/'topas')
if __name__=='__main__':main()

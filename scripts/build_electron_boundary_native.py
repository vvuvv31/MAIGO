#!/usr/bin/env python3
"""Isolated native charged-boundary parameter probe; reference unchanged."""
import argparse,json,subprocess,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'out/electron_standalone_20261003/electron_boundary_native_v3');a=p.parse_args()
 old=ROOT/'out/electron_standalone_20261003/photon_native';spec=json.loads((old/'build_manifest.json').read_text());out=a.out.resolve();out.mkdir(exist_ok=False)
 for n in ['ElectronBoundaryProbe.cc','ElectronBoundaryProbe.hh']:shutil.copy2(ROOT/'benchmark/electron_response_model/native'/n,out/n)
 manager=(old/'TsExtensionManager.cc').read_text().replace('PhotonResponseNtuple','ElectronBoundaryProbe').replace('photonresponsentuple','electronboundaryprobe');(out/'TsExtensionManager.cc').write_text(manager)
 commands=[]
 for n in ['ElectronBoundaryProbe','TsExtensionManager']:
  cmd=[v.replace(str(old),str(out)).replace('PhotonResponseNtuple',n) for v in spec['commands'][0]];commands.append(cmd);subprocess.run(cmd,cwd=spec['base_build'],check=True)
 cmd=[v.replace(str(old),str(out)).replace('PhotonResponseNtuple','ElectronBoundaryProbe') for v in spec['commands'][-1]];commands.append(cmd);subprocess.run(cmd,cwd=spec['base_build'],check=True)
 record=dict(binary_sha256=sha(out/'topas'),source_sha256={n:sha(out/n) for n in ['ElectronBoundaryProbe.cc','ElectronBoundaryProbe.hh','TsExtensionManager.cc']},base_build=spec['base_build'],commands=commands,native_inputs=spec['native_inputs'],scope='Dedicated offline native GS and energy-loss parameter probe. No GPU transport or dose acceptance; no model replacement.')
 (out/'build_manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(record['binary_sha256'])
if __name__=='__main__':main()

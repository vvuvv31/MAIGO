#!/usr/bin/env python3
"""Isolated native elastic kernel export and paired sampler check; reference unchanged."""
import argparse,json,subprocess,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'out/electron_standalone_20261003/electron_elastic_native_v2');a=p.parse_args()
 old=ROOT/'out/electron_standalone_20261003/photon_native';spec=json.loads((old/'build_manifest.json').read_text());out=a.out.resolve();out.mkdir(exist_ok=False)
 for n in ['ElectronElasticDump.cc','ElectronElasticDump.hh']:shutil.copy2(ROOT/'benchmark/electron_response_model/native'/n,out/n)
 shutil.copy2(ROOT/'include/carbon/electron_elastic.hpp',out/'electron_elastic.hpp')
 manager=(old/'TsExtensionManager.cc').read_text().replace('PhotonResponseNtuple','ElectronElasticDump').replace('photonresponsentuple','electronelasticdump');(out/'TsExtensionManager.cc').write_text(manager)
 commands=[]
 for n in ['ElectronElasticDump','TsExtensionManager']:
  cmd=[v.replace(str(old),str(out)).replace('PhotonResponseNtuple',n) for v in spec['commands'][0]];cmd.insert(1,'-I'+str(ROOT/'include'));commands.append(cmd);subprocess.run(cmd,cwd=spec['base_build'],check=True)
 cmd=[v.replace(str(old),str(out)).replace('PhotonResponseNtuple','ElectronElasticDump') for v in spec['commands'][-1]];cmd.insert(1,'-I'+str(ROOT/'include'));commands.append(cmd);subprocess.run(cmd,cwd=spec['base_build'],check=True)
 record=dict(binary_sha256=sha(out/'topas'),source_sha256={n:sha(out/n) for n in ['ElectronElasticDump.cc','ElectronElasticDump.hh','TsExtensionManager.cc','electron_elastic.hpp']},base_build=spec['base_build'],commands=commands,native_inputs=spec['native_inputs'],scope='Native GS/Mott compact coefficient export and identical-random-stream comparison. Isolated surrogate RNG restored after each sample; no complete charged transport/dose claim.')
 (out/'build_manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(record['binary_sha256'])
if __name__=='__main__':main()

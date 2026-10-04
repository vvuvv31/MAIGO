#!/usr/bin/env python3
"""Isolated native C12 birth scorer; original TOPAS binary and physics unchanged."""
from pathlib import Path
import json,subprocess,shutil,hashlib
root=Path(__file__).resolve().parents[1];old=root/'out/electron_standalone_20261003/native_build';out=root/'out/broadbeam_shared_electron_20261004/native';out.mkdir(exist_ok=False)
spec=json.loads((old/'build_manifest.json').read_text())
for name in ['ProtonElectronBirthNtuple.cc','ProtonElectronBirthNtuple.hh','TsExtensionManager.cc']:
 s=(old/name).read_text()
 if name=='ProtonElectronBirthNtuple.hh':s=s.replace('std::array<Bin,50>','std::array<Bin,51>')
 if name=='ProtonElectronBirthNtuple.cc':
  for a,b in [('GetPDGEncoding()==2212','GetPDGEncoding()==1000060120'),('primary must be proton','primary must be C12'),('parentE_=pre->GetKineticEnergy()/MeV','parentE_=pre->GetKineticEnergy()/(12*MeV)'),('last_=post->GetKineticEnergy()/MeV','last_=post->GetKineticEnergy()/(12*MeV)'),('250.000001','255.000001'),('std::min(49,int(parentE_/5))','std::min(50,int(parentE_/5))'),('loss=parentE_-last_','loss=12*(parentE_-last_)'),('bin_<50','bin_<51'),('total-initial_','total-12*initial_')]:
   assert a in s,a;s=s.replace(a,b)
 (out/name).write_text(s)
commands=[]
for cmd in spec['commands']:
 cmd=[v.replace(str(old),str(out)) for v in cmd];commands.append(cmd)
 subprocess.run(cmd,cwd=spec['base_build'],check=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
(out/'build_manifest.json').write_text(json.dumps(dict(commands=commands,base_build=spec['base_build'],binary_sha256=sha(out/'topas'),source_sha256={n:sha(out/n) for n in ['ProtonElectronBirthNtuple.cc','ProtonElectronBirthNtuple.hh','TsExtensionManager.cc']},physics_changes='none; C12 birth recording only, no RNG/track killing/process changes'),indent=2)+'\n')

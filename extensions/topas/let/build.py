#!/usr/bin/env python3
"""Add the LET scorer to an existing TOPAS build without modifying that build."""
from pathlib import Path
import argparse,json,re,shlex,shutil,subprocess
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--topas-build',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--with-recoil-extractor',action='store_true',
               help='Update the existing AllIonElasticDump extension for electronic recoil table extraction')
a=p.parse_args();base=a.topas_build.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
source=Path(__file__).resolve().parent;common=source.parent/'common'
flags={}
for line in (base/'extensions/CMakeFiles/extensions.dir/flags.make').read_text().splitlines():
 if line.startswith('CXX_'):
  key,value=line.split('=',1);flags[key.strip()]=shlex.split(value)
cache=(base/'CMakeCache.txt').read_text();compiler=re.search(r'^CMAKE_CXX_COMPILER:[^=]+=([^\n]+)',cache,re.M)[1]
opts=flags['CXX_DEFINES']+['-I'+str(out),'-I'+str(source),'-I'+str(common)]+flags['CXX_INCLUDES']+flags['CXX_FLAGS']
manager=(base/'extensions/TsExtensionManager.cc').read_text();manager='#include "ElectronicLETMoment.hh"\n'+manager
manager=manager.replace('// Insertion point for Scorers','''// Insertion point for Scorers
    if (quantityNameLower == "electronicletmoment")
        return new ElectronicLETMoment(pM,mM,gM,scM,this,currentScorerName,quantityName,outFileName,isSubScorer);''')
files=[out/'TsExtensionManager.cc',source/'ElectronicLETMoment.cc']
if 'physicsModuleName=="ionletstepphysics"' not in manager.replace(' ', ''):
 manager='#include "IonLETStepPhysics.hh"\n'+manager
 manager=manager.replace('// Insertion point for Physics Modules','''// Insertion point for Physics Modules
    if (physicsModuleName == "ionletstepphysics")
        return new CreatorWithPm<IonLETStepPhysics>(pM);''')
files.append(source/'IonLETStepPhysics.cc')
if a.with_recoil_extractor:
 if 'AllIonElasticDump' not in manager:
  manager='#include "AllIonElasticDump.hh"\n'+manager
  manager=manager.replace('// Insertion point for Scorers','''// Insertion point for Scorers
    if (quantityNameLower == "allionelasticdump")
        return new AllIonElasticDump(pM,mM,gM,scM,this,currentScorerName,quantityName,outFileName,isSubScorer);''')
 recoil=source.parent/'all_ion_elastic'
 opts.insert(0,'-I'+str(recoil));files.append(recoil/'AllIonElasticDump.cc')
if 'physicsModuleName=="allchargedurbanphysics"'not in manager.replace(' ',''):
 manager='#include "AllChargedUrbanPhysics.hh"\n'+manager
 manager=manager.replace('// Insertion point for Physics Modules','''// Insertion point for Physics Modules
    if (physicsModuleName == "allchargedurbanphysics")
        return new CreatorWithPm<AllChargedUrbanPhysics>(pM);''')
 files.append(common/'AllChargedUrbanPhysics.cc')
(out/'TsExtensionManager.cc').write_text(manager)
archive=out/'libextensions.a';shutil.copy2(base/'extensions/libextensions.a',archive)
for f in files:
 obj=out/(f.name+'.o');subprocess.run([compiler,*opts,'-c',str(f),'-o',str(obj)],check=True)
 subprocess.run(['ar','r',str(archive),str(obj)],check=True)
cmd=shlex.split((base/'CMakeFiles/topas.dir/link.txt').read_text());cmd=[str(archive)if x=='extensions/libextensions.a'else x for x in cmd];cmd[cmd.index('-o')+1]=str(out/'topas')
subprocess.run(cmd,cwd=base,check=True)
(out/'build.json').write_text(json.dumps(dict(base=str(base),compile_options=opts,compiler=compiler,link_command=cmd),indent=2))
print('BUILT',out/'topas',flush=True)

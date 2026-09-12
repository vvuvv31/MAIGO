from pathlib import Path
import shlex,re,shutil,subprocess,json,hashlib
root=Path(__file__).resolve().parent
old=Path('/LustreData6/home/wuwei/Applications/TOPAS/OpenTOPAS-build')
flags=(old/'extensions/CMakeFiles/extensions.dir/flags.make').read_text()
opts=['-I'+str(root)]
for name in ['CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS']:
 opts+=shlex.split(re.search('^'+name+r' = (.*)$',flags,re.M).group(1))
link=shlex.split((old/'CMakeFiles/topas.dir/link.txt').read_text());compiler=link[0]
assert Path(compiler).exists(),compiler
s=(old/'extensions/TsExtensionManager.cc').read_text()
s='#include "CarbonIonElasticPhysics.hh"\n#include "AllIonElasticDump.hh"\n'+s
s=s.replace('// Insertion point for Physics Modules','// Insertion point for Physics Modules\n if(physicsModuleName=="carbonionelasticphysics") return new CreatorWithPm<CarbonIonElasticPhysics>(pM);')
s=s.replace('// Insertion point for Scorers','// Insertion point for Scorers\n if(quantityNameLower=="allionelasticdump") return new AllIonElasticDump(pM,mM,gM,scM,this,currentScorerName,quantityName,outFileName,isSubScorer);')
(root/'TsExtensionManager.cc').write_text(s)
shutil.copyfile(old/'extensions/libextensions.a',root/'libextensions.a')
for name in ['CarbonIonElasticPhysics','AllIonElasticDump','TsExtensionManager']:
 subprocess.run([compiler,*opts,'-c',str(root/(name+'.cc')),'-o',str(root/(name+'.cc.o'))],check=True)
 subprocess.run(['ar','r',str(root/'libextensions.a'),str(root/(name+'.cc.o'))],check=True)
link=[str(root/'libextensions.a') if x=='extensions/libextensions.a' else x for x in link];link[link.index('-o')+1]=str(root/'topas')
subprocess.run(link,cwd=str(old),check=True)
(root/'build_manifest.json').write_text(json.dumps({'compiler':compiler,'options':opts,'link':link,'topas_sha256':hashlib.sha256((root/'topas').read_bytes()).hexdigest()},indent=2)+'\n')

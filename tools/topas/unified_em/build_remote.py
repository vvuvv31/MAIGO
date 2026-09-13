from pathlib import Path
import shlex,re,shutil,subprocess,json,hashlib
root=Path(__file__).resolve().parent
old=Path('/home/v/software/topas/OpenTOPAS-build-v4.2.3-carbon')
flags=(old/'extensions/CMakeFiles/extensions.dir/flags.make').read_text();opts=['-I'+str(root)]
for name in ['CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS']:opts+=shlex.split(re.search('^'+name+r' = (.*)$',flags,re.M).group(1))
link=shlex.split((old/'CMakeFiles/topas.dir/link.txt').read_text());compiler=link[0]
shutil.copyfile(old/'extensions/libextensions.a',root/'libextensions.a')
name='CarbonStoppingPowerNtuple'
subprocess.run([compiler,*opts,'-c',str(root/(name+'.cc')),'-o',str(root/(name+'.cc.o'))],check=True)
subprocess.run(['ar','r',str(root/'libextensions.a'),str(root/(name+'.cc.o'))],check=True)
link=[str(root/'libextensions.a') if x=='extensions/libextensions.a' else x for x in link];link[link.index('-o')+1]=str(root/'topas')
subprocess.run(link,cwd=old,check=True)
(root/'manifest.json').write_text(json.dumps(dict(compiler=compiler,options=opts,link=link,sha256=hashlib.sha256((root/'topas').read_bytes()).hexdigest()),indent=2))

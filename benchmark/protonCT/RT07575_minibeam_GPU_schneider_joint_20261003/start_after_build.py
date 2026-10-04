"""Freeze the successful FP32 build; screen one matched shard, then full plan."""
import json,os,re,shutil,subprocess,time
import numpy as np
from common import ROOT,REPO,sha,save_json,read_mhd
from run_gpu import validate

def main():
    build=REPO/'out/schneider_joint_density_20261003/build.log'
    deadline=time.time()+1800
    while '[100%] Built target carbon_mc' not in build.read_text():
        if time.time()>deadline:raise RuntimeError('Build timeout; no frozen executable/run')
        time.sleep(5)
    shutil.copy2(build,ROOT/'build.log')
    cache=(REPO/'build/CMakeCache.txt').read_text()
    assert 'CARBON_DOSE_FP32:BOOL=ON' in cache and 'CARBON_DOSE_FP64:BOOL=OFF' in cache
    binary=ROOT/'inputs/carbon_mc';shutil.copy2(REPO/'build/carbon_mc',binary)
    shutil.copy2(REPO/'build/CMakeCache.txt',ROOT/'inputs/CMakeCache.txt')
    files=['include/carbon/electron_joint_pilot.hpp','include/carbon/electron_joint_pilot_device.hpp','include/carbon/electron_joint_sampling.hpp','src/transport_sycl.cpp','src/config.cpp','src/run_quality.cpp']
    assert all((REPO/f).stat().st_mtime < binary.stat().st_mtime for f in files)
    save_json(ROOT/'build.json',dict(binary_sha256=sha(binary),FP32_dose=True,FP64_dose=False,source_sha256={f:sha(REPO/f) for f in files}))
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,h in manifest['input_sha256'].items():assert sha(ROOT/'inputs'/name)==h,name
    assert sha(manifest['joint_pilot']['file'])==manifest['joint_pilot']['sha256']
    (ROOT/'smoke.yaml').write_text((ROOT/'batch0000.yaml').read_text())
    (ROOT/'smoke.manifest').write_text('smoke.yaml\n')
    env=dict(os.environ,LD_LIBRARY_PATH='/home/wuwei/sycl_workspace/llvm/build/install/lib:'+os.environ.get('LD_LIBRARY_PATH',''),CARBON_RUNTIME_BREAKDOWN='1')
    start=time.perf_counter()
    with (ROOT/'smoke.log').open('x') as log:r=subprocess.run([str(binary),'--plan-manifest',str(ROOT/'smoke.manifest')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
    elapsed=time.perf_counter()-start;assert r.returncode==0,('smoke',r.returncode)
    text=(ROOT/'smoke.log').read_text();quality=validate(dict(name='smoke',histories=manifest['jobs'][0]['histories']),text)
    work=dict((k,int(v)) for k,v in re.findall(r'(\w+)=(\d+)',next(l for l in text.splitlines() if l.startswith('[electron-joint-pilot-work]'))))
    assert work['started']>0 and work['invalid']==0
    packets=int(re.search(r'\[electron-queue-time\].*?tasks=(\d+)',text)[1])
    old=__import__('pathlib').Path(manifest['previous_campaign'])
    x,_=read_mhd(old/'out/batch0000/dose.mhd');y,_=read_mhd(ROOT/'out/smoke/dose.mhd');d=abs(y.astype(float)-x)
    save_json(ROOT/'smoke_result.json',dict(quality=quality,work=work,packets=packets,started_fraction=work['started']/packets,wall_seconds=elapsed,whole_grid_relative_L1=float(d.sum()/x.sum()),max_absolute_raw_Gy=float(d.max())))
    print('SMOKE',json.dumps(json.loads((ROOT/'smoke_result.json').read_text())),flush=True)
    start=time.perf_counter()
    with (ROOT/'campaign.log').open('x') as log:r=subprocess.run([os.sys.executable,'-u',str(ROOT/'run_gpu.py'),'--group','119'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    save_json(ROOT/'completion.json',dict(returncode=r.returncode,wall_seconds=time.perf_counter()-start))
    assert r.returncode==0
    with (ROOT/'comparison_joint.log').open('x') as log:subprocess.run([os.sys.executable,str(ROOT/'compare_joint.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':main()

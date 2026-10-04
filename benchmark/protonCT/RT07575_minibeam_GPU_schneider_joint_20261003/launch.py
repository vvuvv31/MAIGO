"""Freeze a successfully built binary, then run the full joint pilot campaign."""
import json,os,shutil,subprocess,time
from pathlib import Path
from common import ROOT,REPO,sha,save_json

def main():
    log=(ROOT/'build.log').read_text();assert '[100%] Built target carbon_mc' in log
    cmake=(REPO/'build/CMakeCache.txt').read_text();assert 'CARBON_DOSE_FP32:BOOL=ON' in cmake and 'CARBON_DOSE_FP64:BOOL=OFF' in cmake
    binary=ROOT/'inputs/carbon_mc'
    if binary.exists():assert sha(binary)==sha(REPO/'build/carbon_mc')
    else:shutil.copy2(REPO/'build/carbon_mc',binary)
    shutil.copy2(REPO/'build/CMakeCache.txt',ROOT/'inputs/CMakeCache.txt')
    files=['include/carbon/electron_joint_pilot.hpp','include/carbon/electron_joint_pilot_device.hpp','include/carbon/electron_joint_sampling.hpp','src/transport_sycl.cpp','src/config.cpp','src/run_quality.cpp']
    save_json(ROOT/'build.json',dict(binary_sha256=sha(binary),FP32_dose=True,FP64_dose=False,source_sha256={f:sha(REPO/f) for f in files}))
    m=json.loads((ROOT/'manifest.json').read_text())
    for name,h in m['input_sha256'].items():assert sha(ROOT/'inputs'/name)==h,name
    assert sha(m['joint_pilot']['file'])==m['joint_pilot']['sha256']
    command=[os.sys.executable,'-u',str(ROOT/'run_gpu.py'),'--group','119']
    save_json(ROOT/'launch.json',dict(command=command,pid=os.getpid(),started_unix=time.time()))
    start=time.perf_counter()
    with (ROOT/'campaign.log').open('x') as log:r=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    save_json(ROOT/'completion.json',dict(returncode=r.returncode,wall_seconds=time.perf_counter()-start))
    if r.returncode:raise SystemExit(r.returncode)
    with (ROOT/'comparison_joint.log').open('x') as log:
        subprocess.run([os.sys.executable,str(ROOT/'compare_joint.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
if __name__=='__main__':main()

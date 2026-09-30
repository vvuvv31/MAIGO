#!/usr/bin/env python3
"""Sequential species share the allocation and inherited CPU affinity."""
import json
import os
import platform
import subprocess
import time
from pathlib import Path
HERE=Path(__file__).resolve().parent
manifest=json.loads((HERE/'manifest.json').read_text())
RAW=Path(manifest['species']['carbon']['directory']).parent
allowed=sorted(os.sched_getaffinity(0))
cores={}
for cpu in allowed:
    topology=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
    key=(int((topology/'physical_package_id').read_text()),int((topology/'core_id').read_text()))
    cores.setdefault(key,cpu)
assert len(cores)==100,dict(allowed=allowed,physical_cores=len(cores))
# Slurm's core binding includes SMT siblings. Keep one logical CPU per core.
os.sched_setaffinity(0,set(cores.values()))
result=dict(slurm_job=os.environ.get('SLURM_JOB_ID'),hostname=platform.node(),
            original_slurm_affinity=allowed,physical_cores=len(cores),affinity=sorted(os.sched_getaffinity(0)),cpu_model=subprocess.check_output(['lscpu'],text=True),runs=[])
assert len(result['affinity'])==100,result['affinity']
(RAW/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
for particle in ['carbon','proton']:
    directory=Path(manifest['species'][particle]['directory'])
    with (directory/'topas.log').open('w') as f:
        start=time.perf_counter()
        process=subprocess.run(['/usr/bin/time','-v','-o','process_time.txt',manifest['topas'],'run.txt'],cwd=directory,stdout=f,stderr=subprocess.STDOUT)
        wall=time.perf_counter()-start
    row=dict(particle=particle,external_wall_seconds=wall,exit_code=process.returncode,directory=str(directory))
    result['runs'].append(row)
    (RAW/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(row),flush=True)
    if process.returncode:raise RuntimeError(f'{particle} TOPAS failed; see {directory}/topas.log')

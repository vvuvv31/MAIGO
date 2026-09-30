#!/usr/bin/env python3
"""Read measured native TOPAS times and verify complete matched smoke outputs."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
m=json.loads((HERE/'manifest.json').read_text())
RAW=Path(m['species']['carbon']['directory']).parent
execution=json.loads((RAW/'execution.json').read_text())
assert len(execution['runs'])==2 and execution['physical_cores']==100 and len(execution['affinity'])==100
result=dict(slurm_job=execution['slurm_job'],host=execution['hostname'],threads=100,
            physical_cores=100,histories_per_species=m['histories_per_species'],
            timing_scope='TOPAS Real time for execution/all4phases; external process wall also recorded, no queue time',
            scope='single200k smoke per species, complete weighted plan; no throughput stability or particle-only causality claim',
            cpu_model=execution['cpu_model'],species={})
for run in execution['runs']:
    particle=run['particle'];directory=Path(run['directory'])
    assert run['exit_code']==0,run
    log=(directory/'topas.log').read_text()
    assert 'setting number of threads to: 100' in log
    assert 'Max kinetic energy for tables                      10 GeV' in log
    assert 'DijMode:' not in log
    label='CarbonPBS' if particle=='carbon' else 'ProtonPBS'
    match=re.search(r'Particle source '+label+r': Total number of histories: (\d+)',log)
    assert match and int(match[1])==m['histories_per_species']
    phase={name:float(re.search(r'^\s*'+name+r'\s*:.*?Real=([\d.eE+-]+)s',log,re.M)[1]) for name in ['Parameter Reading','Initialization','Execution','Finalization']}
    header=(directory/'dose_plan.binheader').read_text()
    assert '# DoseToMedium ( Gy ) : Sum' in header
    for axis,bins in [('X',417),('Y',505),('Z',35)]:assert f'# {axis} in {bins} bins' in header
    dose=np.fromfile(directory/'dose_plan.bin','<f8')
    # Select native CPU output width only from the exact grid length.
    if dose.size!=417*505*35:dose=np.fromfile(directory/'dose_plan.bin','<f4')
    assert dose.size==417*505*35 and np.isfinite(dose).all() and np.all(dose>=0) and dose.sum()>0
    t=(directory/'process_time.txt').read_text()
    maxrss=int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',t)[1])
    user=float(re.search(r'User time \(seconds\): ([\d.]+)',t)[1]);system=float(re.search(r'System time \(seconds\): ([\d.]+)',t)[1])
    omitted=re.search(r'Total amount of energy not scored for this reason: ([\d.eE+-]+) MeV',header)
    omitted_hits=re.search(r'Total number of steps not scored for this reason: (\d+)',header)
    row=dict(histories=int(match[1]),phase_real_seconds=phase,
             execution_histories_per_second=int(match[1])/phase['Execution'],
             program_histories_per_second=int(match[1])/sum(phase.values()),
             external_histories_per_second=int(match[1])/run['external_wall_seconds'],
             external_wall_seconds=run['external_wall_seconds'],allphase_real_seconds=sum(phase.values()),
             maxrss_KiB=maxrss,maxrss_GiB=maxrss/1024**2,
             process_user_seconds=user,process_system_seconds=system,
             dose_dtype=str(dose.dtype),dose_sum_Gy=float(dose.sum()),dose_max_Gy=float(dose.max()),
             unscored_energy_MeV=float(omitted[1]) if omitted else 0,
             unscored_hits=int(omitted_hits[1]) if omitted_hits else 0,
             binary_sha256=m['topas_sha256'],source=m['species'][particle])
    phase_user=float(re.search(r'^\s*Execution\s*:.*?User=([\d.eE+-]+)s',log,re.M)[1])
    phase_sys=float(re.search(r'^\s*Execution\s*:.*?Sys=([\d.eE+-]+)s',log,re.M)[1])
    row['execution_effective_busy_cores']=(phase_user+phase_sys)/phase['Execution']
    result['species'][particle]=row
c,p=result['species']['carbon'],result['species']['proton']
result['proton_over_carbon']={k:p[k]/c[k] for k in ['execution_histories_per_second','program_histories_per_second','external_histories_per_second']}
(HERE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)

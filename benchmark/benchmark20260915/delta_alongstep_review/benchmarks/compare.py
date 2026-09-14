"""Analyze fresh, quality-accepted GPU results against unchanged TOPAS."""
from pathlib import Path
import json,csv,subprocess,time
from analyze import analyze
R=Path(__file__).resolve().parent;rows=[]
for name,*_ in json.loads((R/'cases.json').read_text()):
 d=R/name
 deadline=time.monotonic()+3600
 while not (d/'gpu_status.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('GPU did not finish '+name)
  time.sleep(5)
 assert json.loads((d/'gpu_status.json').read_text())['complete'],name
 row=analyze(d);status=json.loads((d/'gpu_status.json').read_text());model=status['model'];row['primary_em_model']='legacy' if model=='g4_material_joint_v1' else model;row['em_model']=model if model=='g4_material_joint_v1' else 'legacy'
 import numpy as np
 z,t,g=np.loadtxt(d/'idd.csv',delimiter=',').T
 row['peak_error_percent']=float(100*(g.max()/t.max()-1));row['high_dose_mare_percent']=float(np.mean(abs(g[t>.1*t.max()]/t[t>.1*t.max()]-1))*100)
 (d/'comparison.json').write_text(json.dumps(row,indent=2));rows.append(row)
 with (R/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(name,'analyzed',row['peak_error_percent'],flush=True)
subprocess.run(['python3',str(R/'plot_benchmarks.py')],check=True)

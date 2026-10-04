import json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
with (root/'gpu_memory.csv').open('x') as f:
 f.write('unix_time,memory_used_MiB,gpu_utilization_percent\n')
 while not (root/'completion.json').exists():
  output=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
  f.write(str(time.time())+','+output+'\n');f.flush();time.sleep(2)

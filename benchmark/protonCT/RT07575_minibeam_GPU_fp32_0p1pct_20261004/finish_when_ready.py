from pathlib import Path
import json,subprocess,sys,time
b=Path(__file__).resolve().parent
while True:
 status=json.loads((b/'campaign_status.json').read_text())
 if status['state']=='failed':raise RuntimeError(status)
 if status['state']=='complete':break
 time.sleep(10)
with (b/'postprocess.log').open('w') as log:r=subprocess.run([sys.executable,str(b/'postprocess.py')],stdout=log,stderr=subprocess.STDOUT,cwd=b)
(b/'completion.json').write_text(json.dumps({'transport_state':'complete','postprocess_exit_code':r.returncode,'finished_unix':time.time()},indent=2)+'\n')
raise SystemExit(r.returncode)

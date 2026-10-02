"""Read complete shard outputs without treating in-progress files as accepted dose."""
import json,time
from common import ROOT
m=json.loads((ROOT/'manifest.json').read_text());done=[]
for j in m['jobs']:
 p=ROOT/'out'/j['name'];q=p/'quality_report.json';r=p/'dose.raw'
 if not(q.exists() and r.exists() and r.stat().st_size==417*35*505*4):continue
 try:quality=json.loads(q.read_text())
 except json.JSONDecodeError:continue
 if quality.get('accepted') and quality.get('queue_overflow_count')==0:done.append(j)
h=sum(j['histories'] for j in done);start=json.loads((ROOT/'launch.json').read_text())['start_unix'];elapsed=time.time()-start
print(json.dumps(dict(completed_shards=len(done),total_shards=m['parts'],completed_histories=h,percent=100*h/m['histories'],elapsed_seconds=elapsed,extrapolated_remaining_seconds=(m['histories']-h)/(h/elapsed) if h else None)))

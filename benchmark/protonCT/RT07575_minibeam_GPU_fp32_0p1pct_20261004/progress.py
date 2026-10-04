from pathlib import Path
import json
b=Path(__file__).resolve().parent;s=json.loads((b/'campaign_status.json').read_text());c=json.loads((b/'batch_checks.json').read_text()) if (b/'batch_checks.json').exists() else []
r=dict(state=s['state'],batches=s.get('completed_parts',0),primaries=s.get('completed_histories',0),elapsed_min=round(s.get('wall_seconds',0)/60,2),remaining_min=round(s.get('remaining_seconds',0)/60,2))
if len(c)>1:r['resident_primary_per_second']=round(sum(x['histories'] for x in c[1:])/sum(x['batch_seconds'] for x in c[1:]))
if c:r['max_energy_residual']=max(abs(x['energy_residual']) for x in c)
if (b/'completion.json').exists():r['postprocessing']=json.loads((b/'completion.json').read_text())
print(json.dumps(r))

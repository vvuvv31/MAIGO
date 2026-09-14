from pathlib import Path
import json,re,numpy as np
root=Path(__file__).resolve().parent;repo=root.parents[1];r=repo/'scratch/unified_em_perf_20260913'
pairs=[('runtime_current_1m_baseline','runtime_mean_guard_1m_r1'),('runtime_mean_base_1m_r2','runtime_mean_guard_1m_r2'),('runtime_mean_audit_base','runtime_mean_audit_guard')]
rows=[]
for ba,ca in pairs:
 a=r/ba/'RT07575';b=r/ca/'RT07575'
 if not (a/'status.json').exists() or not (b/'status.json').exists():continue
 sa=json.loads((a/'status.json').read_text());sb=json.loads((b/'status.json').read_text());assert sa['complete'] and sb['complete'] and sa['config_sha256']==sb['config_sha256'] and sa['audit']==sb['audit'] and sa['steps']==sb['steps']
 x=np.fromfile(a/'dose_gpu.raw','<f4').astype(float);y=np.fromfile(b/'dose_gpu.raw','<f4').astype(float);diff=100*abs(y-x).max()/x.max();assert diff<.001
 la=json.loads((a/'out/gpu/energy_ledger.json').read_text());lb=json.loads((b/'out/gpu/energy_ledger.json').read_text());events={k:[v,lb[k]] for k,v in la.items() if isinstance(v,int) and ('interactions' in k or 'overflow' in k)};assert all(x==y for x,y in events.values())
 audit=re.findall(r'^PRIMARY_LOSS_QUERY,.*$',(a/'gpu.log').read_text(),re.M);other=re.findall(r'^PRIMARY_LOSS_QUERY,.*$',(b/'gpu.log').read_text(),re.M);assert audit==other
 if 'audit' in ba:assert len(audit)==14
 row=dict(baseline_label=ba,candidate_label=ca,baseline=sa,candidate=sb,throughput_gain_percent=100*(sb['throughput']/sa['throughput']-1),primary_time_reduction_percent=100*(1-sb['primary_s']/sa['primary_s']),max_dose_difference_percent_peak=diff,event_counts=events,primary_loss_query_audit_equal=True)
 rows.append(row);print(ba,'->',ca,'throughput',sa['throughput'],sb['throughput'],'gain%',row['throughput_gain_percent'],'primary',sa['primary_s'],sb['primary_s'],'dose%peak',diff)
(root/'mean_guard_results.json').write_text(json.dumps(rows,indent=2)+'\n')

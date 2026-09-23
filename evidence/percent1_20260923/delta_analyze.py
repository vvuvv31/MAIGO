from pathlib import Path
import numpy as np,json
r=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923');o=r/'evidence/percent1_20260923';j=json.loads((o/'comparison.json').read_text());assert j['status']=='COMPLETE'
q=json.loads((o/'delta_fixed_quality.json').read_text());assert len(q['runs'])==7
x=-49.95+np.arange(1000)*.1;z=.125+np.arange(1000)*.25;vm=abs(abs(x)-1.8)<.45;pm=abs(x)<.25
old=np.stack([np.fromfile(r/f'out/valley_cu0051m_s{i}/dose.raw',dtype='<f4').reshape(1000,1000).astype(float)/1e6 for i in [1,2,3]])
new=np.stack([np.fromfile(r/f'out/percent1_delta_fixed_s{i}/dose.raw',dtype='<f4').reshape(1000,1000).astype(float)/1e6 for i in [1,2,3]])
result={'scope':'Research scoring-only diagnostic; three paired 1M source seeds. Table 32768 near deposits/channel plus all original endpoints outside 0.5 mm, original loss fractions and ancestor paths retained. Not production or 1% validation. Baseline_15M_plus_shift is a variance-reduction point estimate only, not a 15M corrected run.','rows':[]}
for row in j['rows']:
 dep=row['depth_mm'];width=row['window_width_mm'];mask=np.arange(1000)==int(np.argmin(abs(z-dep))) if width==.25 else abs(z-dep)<width/2
 rr={'depth_mm':dep,'window_width_mm':width}
 for name,m in [('valley',vm),('peak',pm),('row_mean',np.ones(1000,bool))]:
  a=old[:,mask][:,:,m].mean((1,2));b=new[:,mask][:,:,m].mean((1,2));ref=row[name]['topas']['mean'];change=(b-a)/ref
  rr[name]={'paired_change_reference_fraction':float(change.mean()),'paired_change_SE':float(change.std(ddof=1)/np.sqrt(3)),'paired_changes':change.tolist(),'direct_3M_relative_difference':float(b.mean()/ref-1),'baseline_15M_plus_shift_relative_difference':float(row[name]['relative_difference']+change.mean())}
 result['rows'].append(rr)
 print(dep,width,'valley shift pp',100*rr['valley']['paired_change_reference_fraction'],'paired SE pp',100*rr['valley']['paired_change_SE'],'peak shift pp',100*rr['peak']['paired_change_reference_fraction'],flush=True)
result['energy_change_Gy_voxels_per_source']=(new.sum((1,2))-old.sum((1,2))).tolist()
(o/'delta_comparison.json').write_text(json.dumps(result,indent=2)+'\n')

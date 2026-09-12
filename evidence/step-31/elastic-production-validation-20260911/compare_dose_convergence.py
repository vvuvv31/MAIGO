from pathlib import Path
import sys,json,hashlib
import numpy as np
REPO=Path(__file__).resolve().parents[3];sys.path.insert(0,str(REPO/'tools'))
from evaluate_topas10x_gpu_gamma import pass_mask
base=Path('/mnt/sda/wuwei/ct_previous_full20_20260909/RT07575');run=Path('/mnt/sda/wuwei/all_ion_elastic_validation_20260911/RT07575_final_build_shard');old=Path('/mnt/sda/wuwei/final_stopping_20260911/RT07575/shard_01')
m=json.loads((base/'manifest.json').read_text());import yaml
c=yaml.safe_load((run/'config.yaml').read_text());n=c['number_of_histories']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(base/'topas_sum.raw')==m['reference_sum_sha256']
assert json.loads((run/'out/config/quality_report.json').read_text())['accepted']
r=np.fromfile(base/'topas_sum.raw','<f4').reshape(m['topas_shape_zyx'])*(n/m['histories'])
mask=np.load(REPO/'evidence/step-31/rt07575-gamma-spatial-20260911/body/body_mask_zyx.npy')&(r>=.1*r.max());pts=np.argwhere(mask)
report={'case':'RT07575','histories':n,'scope':'One original shard, BODY reference >=10%; original TOPAS total scaled only by history count; no fitted dose normalization; 0.125mm search','limitation':'Existing TOPAS reference may not include the added CarbonIonElasticPhysics module; this comparison is diagnostic, not matched-physics validation. One-shard noise is larger than full20.','results':{}}
for name,p in [('baseline_512_grid20',run/'dose.raw'),('samples2048',Path('/mnt/sda/wuwei/elastic_production_validation_20260911/gpu_samples2048/out/config/dose.raw')),('grid40',Path('/mnt/sda/wuwei/elastic_production_validation_20260911/gpu_grid40/out/config/dose.raw'))]:
 g=np.flip(np.fromfile(p,'<f4').reshape(m['gpu_shape_zyx']).transpose(1,2,0),2);result={'sha256':sha(p),'gamma':{}}
 for dd,dta in [(3,3),(2,2),(1,1),(3,0)]:
  for local in [False,True]:
   passed=np.zeros(len(pts),bool)
   for step in [.5,.25,.125]:
    ix=np.flatnonzero(~passed)
    if len(ix):passed[ix]=pass_mask(g,r,pts[ix],np.array(m['spacing_zyx']),dd,dta,local,step)
    if dta==0:break
   result['gamma'][f"{'local' if local else 'global'}_{dd}{dta}"]=100*float(passed.mean())
 result['body_mean_local_error_pct']=float(np.mean((g[mask]-r[mask])/r[mask])*100);report['results'][name]=result
 print(name,result,flush=True)
(Path(__file__).parent/'dose_convergence_gamma.json').write_text(json.dumps(report,indent=2)+'\n')

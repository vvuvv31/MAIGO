import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

root = Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923')
base = root / 'evidence/urban_d51e599_20260923'
out = root / 'evidence/review_fix_20260923'
def module(name):
    spec = importlib.util.spec_from_file_location(name, base / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
r = module('replay_gate')
d = module('dose_gate')
def pair(theta, energy, weight=1):
    return np.array([[0,0,40,0,0,1,energy,weight],
                     [0.1,0,60,np.sin(theta),0,np.cos(theta),energy,weight]],float)
rows = [pair(.01,1), pair(.1,1000)]
got = r._moments(rows, np.array([0,2]), 2)
assert abs(got['theta2']-5050) < 1e-8 and got['weight_sum'] == 2, got
weighted = r._moments([pair(.01,3000,3),pair(.1,30,1)],np.array([3,72]),2)
assert abs(weighted['theta2']-2575) < 1e-8 and weighted['weight_sum'] == 4
surv = r._survival({0:rows[0],2:rows[1]}, {2:rows[1]},
                   {0:(None,3),2:(None,1)}, 2)
assert surv['survival_fraction'] == .25
assert d._ratio_gate([0,0,0],[1,1,1],.03)['status'] == 'FAIL'
assert d._fwhm_1d(np.array([.7,.8,1,.8,.7]),np.arange(-2,3)) is None

x=np.linspace(-10,10,201); z=np.linspace(0,150,601)
ref=(1+3*np.exp(-((z-100)/10)**2))[:,None]*(np.exp(-.5*(x/.4)**2)+.001)[None,:]
gpu=ref.copy()
geom=dict(nx=len(x),nz=len(z),dx_mm=.1,dz_mm=.25,dy_mm=1,
          x0_center_mm=-10,z0_center_mm=0,density_g_cm3=1,voxel_mass_kg=2.5e-8)
d.load_manifest=lambda path,role: {'runs':[{'random_seed':n} for n in range(3)]}
d.load_gpu_run=lambda run,expected_count:(gpu.copy(),dict(geom,seed=run['random_seed']))
d.load_topas_run=lambda run,shape,geometry:(ref.copy(),dict(geom,seed=run['random_seed']))
d.sha256=lambda path:'synthetic_fixture'
args=SimpleNamespace(accept=str(root/'evidence/urban_after_0f2c0ca_20260922/acceptance.yaml'),
                     gpu_manifest='synthetic_gpu',topas_manifest='synthetic_topas')
baseline,code=d.run(args)
assert code==0 and baseline['state']=='PASS',baseline
gpu[120,x>5]+=ref.sum()*.10/np.count_nonzero(x>5)
bad,code=d.run(args)
assert code==1 and bad['state']=='FAIL',bad
assert bad['whole_scored_volume_energy_J_per_primary']['status']=='FAIL'
assert all(row['status']=='PASS' for row in bad['depth_rows'])
results={'sparse_ids':'PASS','statistical_weights':'PASS','weighted_survival':'PASS',
         'zero_gpu_ratio':'PASS','undefined_fwhm':'PASS','identical_dose':'PASS',
         'full_volume_only_failure':{'state':bad['state'],'exit_code':code,'ratio':bad['whole_scored_volume_energy_J_per_primary']['ratio']}}
(out/'gate_regressions.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))

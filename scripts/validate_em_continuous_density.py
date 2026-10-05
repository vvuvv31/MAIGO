"""Explicit continuous26-composition comparisons; defaults are never overwritten.

FP32 dose only. RT07575 full-plan geometry is head, not the lung presets.
Packages are explicit, so default promotion cannot silently turn comparisons
into A/A runs. Archived controls use a frozen compatible GPU binary or a
validated prior campaign after package/config/statistics identity checks.
"""
from pathlib import Path
import argparse,json
import yaml
import validate_em_density_factorization as v

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['smoke','broadbeam','head-screen','head-0p1'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--candidate-root',type=Path,required=True);p.add_argument('--reference-root',type=Path,default=v.ROOT/'data/em');p.add_argument('--binary',type=Path,default=v.ROOT/'build/carbon_mc');p.add_argument('--reuse-head-reference',type=Path,help='Validated expanded-model0.1% campaign; verify source/configs before reuse');p.add_argument('--reuse-broadbeam-reference',type=Path,help='Reuse validated broadbeam reference arms after package/config/statistics checks');p.add_argument('--reference-binary',type=Path,help='Frozen legacy GPU binary for archived density-state reference; latest GPU requires26composition data');a=p.parse_args()
 v.OUT=a.output.resolve();v.OUT.mkdir(parents=True,exist_ok=True)
 roots={'reference':a.reference_root.resolve(),'continuous':a.candidate_root.resolve()}
 assert roots['reference']!=roots['continuous']
 def run(case,arm,tag,n,h,seed):
  files=tuple(roots[arm]/f for f in ['unified_em_v1.bin','unified_em_delta_moments_v2.bin'])
  binary=a.reference_binary if arm=='reference' and a.reference_binary else a.binary
  return v.run(case,arm,tag,n,h,seed,binary.resolve(),package_pair=files)
 if a.phase=='broadbeam' and a.reuse_broadbeam_reference:
  import subprocess
  control_root=a.reuse_broadbeam_reference.resolve()
  for case,n,h in [('proton_broadbeam',10,1253150),('carbon_broadbeam',7,925987)]:
   for tag,seed in [('c1',1001904502),('c2',1001904602)]:
    control=control_root/'runs'/f'{case}_reference_{tag}'
    old=yaml.safe_load((control/'case.yaml').read_text());new=v.config(case);new['random_seed']=seed
    ignored={'em_package_file','em_delta_moments_file'}
    differences={k:(old.get(k),new.get(k)) for k in old.keys()|new.keys() if k not in ignored and old.get(k)!=new.get(k)}
    if differences:raise RuntimeError(f'Reused control physics differs:{differences}')
    for key,name in [('em_package_file','unified_em_v1.bin'),('em_delta_moments_file','unified_em_delta_moments_v2.bin')]:
     subprocess.run(['cmp',old[key],str(roots['reference']/name)],check=True)
    rec=json.loads((control/'execution.json').read_text())
    assert rec['runtime_pass'] and (rec['n'],rec['h'],rec['seed'])==(n,h,seed)
    v.save(v.OUT/f'reused_control_{case}_{tag}.json',dict(control=str(control),histories=rec['histories'],same_source_packages=True,same_physics_config=True))
    run(case,'continuous',tag,n,h,seed)
   candidate_out=v.OUT;v.OUT=control_root
   base,ref,mask,spacing=v.analyze(case,'reference',['c1','c2'],report_dir=candidate_out)
   v.OUT=candidate_out;cand,ev,_,_=v.analyze(case,'continuous',['c1','c2'])
   from validate_em_analytic_density import compare_results
   compare_results(case,base,cand,ref,ev,mask,spacing,v.OUT)
  return
 if a.phase=='head-0p1' and a.reuse_head_reference:
  # Reuse the equal-history expanded-model control only after file/config
  # identity checks. No new low-statistics arm is substituted for0.1%.
  import subprocess,numpy as np
  control_root=a.reuse_head_reference.resolve();case='proton_head_minibeam';tags=['s1','s2','h01']
  for tag,n,seed in [('s1',12,1001904502),('s2',12,1001904602),('h01',95,1001904702)]:
   control=control_root/'runs'/f'{case}_analytic_{tag}'
   old=yaml.safe_load((control/'case.yaml').read_text());new=v.config(case);new['random_seed']=seed
   ignored={'em_package_file','em_delta_moments_file','spot_weights_file'}
   differences={k:(old.get(k),new.get(k)) for k in old.keys()|new.keys() if k not in ignored and old.get(k)!=new.get(k)}
   if differences:raise RuntimeError(f'Reused control physics differs:{differences}')
   assert np.array_equal(np.loadtxt(old['spot_weights_file']),np.loadtxt(new['spot_weights_file']))
   for key,name in [('em_package_file','unified_em_v1.bin'),('em_delta_moments_file','unified_em_delta_moments_v2.bin')]:
    subprocess.run(['cmp',old[key],str(roots['reference']/name)],check=True)
   rec=json.loads((control/'execution.json').read_text());assert rec['runtime_pass'] and (rec['n'],rec['h'],rec['seed'])==(n,1983920,seed)
   v.save(v.OUT/f'reused_control_{tag}.json',dict(control=str(control),histories=rec['histories'],same_source_packages=True,same_physics_config=True,same_weights=True))
   run(case,'continuous',tag,n,1983920,seed)
  candidate_out=v.OUT;v.OUT=control_root
  base,ref,mask,spacing=v.analyze(case,'analytic',tags,report_dir=candidate_out)
  v.OUT=candidate_out;cand,ev,_,_=v.analyze(case,'continuous',tags)
  from validate_em_analytic_density import compare_results
  compare_results(case,base,cand,ref,ev,mask,spacing,v.OUT);return
 if a.phase=='smoke':
  records=[];original=v.config
  for preset in sorted((v.ROOT/'config').glob('*.yaml')):
   v.config=lambda case,preset=preset:v.absolute_paths(yaml.safe_load(preset.read_text()))
   records.append(run(preset.stem,'continuous','smoke',2,50000,10579111)[1])
  v.config=original
  records.append(run('proton_head_minibeam','continuous','smoke',2,100000,10579111)[1])
  v.save(v.OUT/'smoke.json',records);return
 cases=[('proton_broadbeam',10,1253150),('carbon_broadbeam',7,925987)] if a.phase=='broadbeam' else [('proton_head_minibeam',119 if a.phase=='head-0p1' else 6,1983920)]
 for case,n,h in cases:
  for tag,seed,arms in [('c1',1001904502,['reference','continuous']),('c2',1001904602,['continuous','reference'])] if a.phase=='broadbeam' else [('smoke' if a.phase=='head-screen' else 'c01',1001904502,['reference','continuous'])]:
   for arm in arms:run(case,arm,tag,n,h,seed)
  tags=['c1','c2'] if a.phase=='broadbeam' else ['smoke' if a.phase=='head-screen' else 'c01']
  base,ref,mask,spacing=v.analyze(case,'reference',tags);cand,ev,_,_=v.analyze(case,'continuous',tags)
  from validate_em_analytic_density import compare_results
  compare_results(case,base,cand,ref,ev,mask,spacing,v.OUT)
if __name__=='__main__':main()

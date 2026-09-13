"""Explicit integration guard test; takes a built carbon_mc and a valid water YAML."""
import os,re,subprocess,sys,tempfile
from pathlib import Path
exe,config=map(Path,sys.argv[1:3]);base=config.read_text()
def replace(text,key,value):
    return re.sub(r'^'+re.escape(key)+r':.*$',key+': '+value,text,flags=re.M) if re.search(r'^'+re.escape(key)+':',text,re.M) else text+'\n'+key+': '+value+'\n'
cases=[('primary_em_model','unknown','Unknown primary_em_model'),('primary_joint_em_data_directory',None,'requires primary_joint_em_data_directory'),('run_mode','production','validation candidate'),('enable_ct_grid','true','CT grid path does not exist'),('water_density_g_per_cm3','1.1','requires research homogeneous'),('primary_atomic_number','2','requires research homogeneous'),('beam_energy_spread','0.01','requires research homogeneous'),('straggling_scale','0.9','requires unscaled native fluctuations'),('material_electron_response_index_file','forbidden.json','electron response stacking is forbidden'),('primary_em_model','legacy','requires g4_joint_water_v1')]
with tempfile.TemporaryDirectory() as tmp:
 for key,value,message in cases:
  p=Path(tmp)/'guard.yaml';p.write_text(re.sub(r'^'+re.escape(key)+r':.*\n?', '', base, flags=re.M) if value is None else replace(base,key,value))
  run=subprocess.run([str(exe.resolve()),'--config',str(p),'--device','cuda'],capture_output=True,text=True,timeout=60)
  assert run.returncode == 1 and 'carbon_mc:' in run.stderr,(key,run.stdout,run.stderr)
  if key in ('primary_em_model', 'primary_joint_em_data_directory', 'straggling_scale'):
   assert message in run.stdout+run.stderr,(key,run.stdout,run.stderr)
  print('PASS',key,value)

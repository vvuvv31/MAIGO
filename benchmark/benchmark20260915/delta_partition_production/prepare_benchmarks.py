from pathlib import Path
import json,shutil,hashlib
R=Path(__file__).resolve().parent;repo=R.parents[2];B=R/'benchmarks';old=repo/'benchmark/benchmark20260914/production_continuation';plot_source=repo/'benchmark/benchmark20260914/delta_off_review/benchmarks/plot_benchmarks.py';assert not B.exists();B.mkdir()
cases=json.loads((old/'cases.json').read_text())
for n,e,s in cases:
 d=B/n;d.mkdir();src=old/n
 for p in src.iterdir():
  if p.name in ['topas.txt','topas.log','topas_status.json','geometry.json','phantom.cctg'] or (p.name.startswith('dose') and p.suffix in ['.bin','.binheader']):(d/p.name).symlink_to(p.resolve())
 shutil.copyfile(src/'gpu.yaml',d/'gpu.yaml');(d/'data').symlink_to(repo/'data')
for n in ['analyze.py','sigma_fit.py','sigma_model.py','compare.py','run_gpu.py','cases.json']:shutil.copyfile(old/n,B/n)
p=B/'run_gpu.py';p.write_text(p.read_text().replace('repo=R.parents[2]','repo=R.parents[3]'))
s=plot_source.read_text().replace('GPU delta OFF','GPU Poisson partition');(B/'plot_benchmarks.py').write_text(s)
exe=B/'carbon_mc';shutil.copyfile(repo/'build/oneapi-nvidia-release/carbon_mc',exe);exe.chmod(0o755)
m=json.loads((old/'input_manifest.json').read_text());m['binary_sha256']=hashlib.sha256(exe.read_bytes()).hexdigest();m['candidate']='delta_partition_production';m['config_changes']={};m['baseline_directory']=str(old);(B/'input_manifest.json').write_text(json.dumps(m,indent=2))

"""Restore this experimental source without changing production. Requires patch and git."""
from pathlib import Path
import subprocess
R=Path(__file__).resolve().parent;repo=R.parents[2]
root=repo/'scratch'/'delta_alongstep_current_20260915';src=root/'restored_source';src.mkdir(parents=True,exist_ok=False)
a=subprocess.check_output(['git','archive','a7c382de2daf1274a66979005d9530af360c5b63'],cwd=repo)
subprocess.run(['tar','-xf','-','-C',str(src)],input=a,check=True)
subprocess.run(['patch','--batch','-p1','-i',str(R/'candidate_from_a7c382d.patch')],cwd=src,check=True)
p=src/'src/transport_sycl.cpp';p.write_text(p.read_text().replace('/mnt/sdb/wuwei/MAIGO/',str(repo)+'/'))
print(src)

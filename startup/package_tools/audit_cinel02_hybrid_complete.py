#!/usr/bin/env python3
import argparse,csv,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit_cinel02_coverage as audit
def main():
 p=argparse.ArgumentParser(); p.add_argument('--summary',type=Path,required=True); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--csv-output',type=Path); a=p.parse_args(); paths=[]
 for r in csv.DictReader(a.summary.open()):
  name='z{}a{}_e{}_h{}_{}'.format(r['projectile_z'],r['projectile_a'],r['source_energy_MeV_per_u'],r['histories'],r['version'])
  paths.extend(sorted((a.root/name).glob('raw/*/worker_*.cinel02')))
 hashes={str(path):'pilot-unbound' for path in paths}
 report=audit.audit(paths,campaign_uuid='00000000-0000-4000-8000-000000999999',raw_hashes=hashes)
 audit.write_outputs(report,a.output,a.csv_output); print(report['summary'])
if __name__=='__main__': main()

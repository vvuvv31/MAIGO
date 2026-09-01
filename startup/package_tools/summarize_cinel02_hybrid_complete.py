#!/usr/bin/env python3
"""Summarize the valid v3 p/d/t and v4 He--C CINEL02 pilot campaigns."""

import argparse, csv, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cinel02

PATTERN = re.compile(r"z(\d+)a(\d+)_e(\d+)_h(\d+)_(v3|v4)$")
LIGHT = {(1, 1), (1, 2), (1, 3)}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    rows=[]
    for run in sorted(a.root.iterdir()):
        m=PATTERN.fullmatch(run.name)
        if not m: continue
        z,aa,e,h=map(int,m.groups()[:4]); version=m.group(5)
        if ((z,aa) in LIGHT) != (version=='v3'): continue
        interactions=products=raw_bytes=workers=0
        for path in run.glob('raw/*/worker_*.cinel02'):
            with path.open('rb') as f: header=f.read(cinel02.RAW_HEADER_SIZE)
            values=cinel02.RAW_HEADER.unpack(header)
            if values[0]!=cinel02.RAW_MAGIC or values[1]!=cinel02.RAW_VERSION: raise ValueError(path)
            interactions+=int(values[5]); products+=int(values[6]); raw_bytes+=path.stat().st_size; workers+=1
        log=run/'topas.log'; completed=log.is_file() and 'TOPAS run sequence complete' in log.read_text(errors='replace')
        rows.append({'projectile_z':z,'projectile_a':aa,'source_energy_MeV_per_u':e,'histories':h,'version':version,'completed':int(completed),'raw_workers':workers,'interactions':interactions,'products':products,'raw_bytes':raw_bytes,'interactions_per_history':interactions/h})
    rows.sort(key=lambda r:(r['projectile_z'],r['projectile_a'],r['source_energy_MeV_per_u']))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print({'points':len(rows),'completed':sum(r['completed'] for r in rows),'interactions':sum(r['interactions'] for r in rows),'products':sum(r['products'] for r in rows),'raw_bytes':sum(r['raw_bytes'] for r in rows),'zero_capture':sum(r['interactions']==0 for r in rows)})
if __name__=='__main__': main()

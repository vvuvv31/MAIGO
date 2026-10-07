#!/usr/bin/env python3
"""Pack26 native shell compositions; no density-state or empirical birth CDF."""
import argparse,json,struct
from pathlib import Path
import numpy as np
from formats import shells
ROOT=Path(__file__).resolve().parents[3]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);spec=json.loads((a.native/'report.json').read_text());rows=[];bounds={s:[] for s in range(26)}
    for line in (a.calibration/'charged/index.txt').read_text().splitlines()[1:]:
        s,rho,file=line.split();bounds[int(s)].append(float(rho))
    with (a.out/'birth26.bin').open('xb') as f:
        f.write(struct.pack('<8sII',b'MAIBCA26',1,26))
        for section in range(26):
            members=[s for s in spec['results'] if s['section']==section]
            if not members:raise ValueError('incomplete26')
            tables=[shells(Path(s['directory'])/'atomic_shells.csv') for s in members];table=tables[0]
            if any(not np.array_equal(t['binding'],table['binding']) or not np.array_equal(t['z'],table['z']) for t in tables):raise ValueError('composition shells differ')
            # Integer CDF conversions can differ by tiny host rounding between
            # densities. Require the independently extracted probabilities
            # themselves to agree at floating-point precision.
            for t in tables:
                if np.max(abs(t['cdf'].astype(np.longdouble)-table['cdf'].astype(np.longdouble)))/np.longdouble(2)**64>1e-12:
                    raise ValueError('shell law density dependence')
            low,high=np.float32(min(bounds[section])),np.float32(max(bounds[section]))
            f.write(struct.pack('<IffI',section,low,high,len(table)));f.write(table.tobytes())
            rows.append(dict(section=section,shells=len(table),density_bounds=[float(low),float(high)]))
    report=dict(schema='MAIBCA26',version=1,status='EXPERIMENTAL_BIRTH_ONLY_NOT_FULL_ELECTRON26_OR_DOSE_ACCEPTANCE',compositions=26,density_states=0,projectiles=['proton','carbon'],bytes=(a.out/'birth26.bin').stat().st_size,physical_law='energy-weighted Bragg/Bethe/LS delta sampling + electron-number-weighted G4DeltaAngle shell motion',birth_yield='continuous26 ion EM at same hard-event energy and cut',rows=rows)
    (a.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

"""Report independent multi-energy water checks, never grant production acceptance."""
import argparse,json
from pathlib import Path
from run_topas10x_gpu_benchmark import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--case',nargs=2,action='append',metavar=('ENERGY','COMPARISON'),required=True)
    a=p.parse_args();rows=[];pins={}
    for energy,name in a.case:
        path=Path(name).resolve();r=json.loads(path.read_text());m=r['metrics']
        if r['status']!='DIAGNOSTIC_ONLY_NO_PRODUCTION_ACCEPTANCE':raise ValueError('Unexpected result status')
        for f,h in r['dose_pins'].items():
            if sha(Path(f))!=h:raise ValueError('Dose changed: '+f)
        pins[str(path)]=sha(path)
        rows.append(dict(energy_MeVu=float(energy),job=r['job'],
            energy_bias_percent=100*(m['gpu_over_topas']-1),
            peak_delta_mm=m['gpu']['peak_mm']-m['topas']['peak_mm'],
            r80_delta_mm=m['gpu']['r80_mm']-m['topas']['r80_mm'],
            depth_corr=m['depth_corr'],depth_rmse_percent_ref_peak=m['depth_rmse_percent_ref_peak'],
            tail_energy_bias_percent=100*(r['windows']['fragment_tail']['gpu']['MeV_per_primary']/r['windows']['fragment_tail']['topas']['MeV_per_primary']-1)))
    if len({r['energy_MeVu'] for r in rows})!=len(rows):raise ValueError('Duplicate energy')
    report=dict(status='MULTI_ENERGY_DIAGNOSTIC_NOT_PRODUCTION',rows=sorted(rows,key=lambda r:r['energy_MeVu']),pins=pins,
        limitations=['Single seed 50k per energy; no independent confidence intervals.',
                     'Not patient Gamma; no default migration or electron-response validation.',
                     'Candidate stopping grid ends at400.01MeV/u; nuclear v2.1 unchanged.'])
    with a.out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

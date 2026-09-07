"""Compare actual first-generation queued GPU spectra to conditional TOPAS births."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from analyze_water_helium_birth import rows,key
from verify_water_species_tally_fix import pair,check_pins
from run_topas10x_gpu_benchmark import sha

def clamped_hist(values,low,width,count):
    values=np.asarray(values,dtype=float)
    if not np.isfinite(values).all():raise ValueError('Nonfinite birth spectrum')
    indices=np.clip(np.floor((values-low)/width),0,count-1).astype(int)
    return np.bincount(indices,minlength=count)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    dest=p.parse_args().out;root=Path('/mnt/sda/wuwei')
    gpu=root/'unified_water300_actual_birth_20260907/run';topas=root/'unified_water300_birth_roi_20260907'
    comparison=pair(root/'unified_water300_roi_replay_20260907',gpu)
    check_pins(json.loads((topas/'analysis.json').read_text())['pins'])
    interactions={}
    for r in rows(topas/'cascade.phsp'):
        if r[0]=='interaction':interactions[key(r)]=int(r[7])==0 and (int(r[10]),int(r[11]))==(6,12)
    samples={3:[],4:[]}
    for r in rows(topas/'cascade.phsp'):
        if r[0]=='product' and int(r[10])==2 and int(r[11]) in samples and interactions.get(key(r)) and float(r[13])>.1:
            samples[int(r[11])].append([float(r[13]),float(r[24]),float(r[21])])
    with (gpu/'birth_summary.csv').open() as f:summaries=list(csv.DictReader(f))
    result={}
    for mass,name in [(3,'he3'),(4,'he4')]:
        data=np.array(samples[mass]);s=[r for r in summaries if r['species']==name and r['generation']=='0']
        if len(s)!=1 or len(data)==0:raise ValueError('Missing birth summary')
        s=s[0];total=int(s['count']);spectra={}
        for suffix,column,low,width,n,values in [
            ('mevu','mevu_bin_low',0.,2.,200,data[:,0]/mass),
            ('costheta','cos_bin_low',-1.,.1,20,data[:,1]),
            ('depth','depth_mm',0.,.5,800,data[:,2])]:
            t=clamped_hist(values,low,width,n);g=np.zeros(n,dtype=np.int64)
            with (gpu/f'birth_{suffix}.csv').open() as f:
                for row in csv.DictReader(f):
                    if row['species']!=name or row['generation']!='0':continue
                    coordinate=float(row[column])-(.25 if suffix=='depth' else 0.)
                    index=int(round((coordinate-low)/width))
                    if not 0<=index<n:raise ValueError('Invalid bin')
                    g[index]+=int(row['count'])
            if g.sum()!=total or t.sum()!=len(data):raise ValueError('Histogram count mismatch')
            centers=low+(np.arange(n)+.5)*width
            spectra[suffix]=dict(cdf_max_distance=float(np.max(np.abs(np.cumsum(t)/t.sum()-np.cumsum(g)/g.sum()))),
                topas_binned_mean=float(np.dot(centers,t)/t.sum()),gpu_binned_mean=float(np.dot(centers,g)/g.sum()),
                topas_counts=t.tolist(),gpu_counts=g.tolist())
        result[name]=dict(topas_count=len(data),gpu_count=total,topas_mean_KE_MeV=float(data[:,0].mean()),
            gpu_mean_KE_MeV=float(s['mean_kinetic_energy_MeV']),spectra=spectra)
    files=[gpu/f'birth_{suffix}.csv' for suffix in ['summary','mevu','costheta','depth']]
    files += [topas/'cascade.phsp',Path(__file__).resolve(),Path('tools/probe_water_birth_spectrum.py').resolve()]
    output=dict(status='CONDITIONAL_BIRTH_DIAGNOSTIC_NOT_PRODUCTION',dose_regression=comparison,results=result,
        pins={str(f):sha(f) for f in files},limitations=[
            'TOPAS ROI-only first-tracked records vs GPU full-water queued births; both use KE > 0.1 MeV.',
            'Only GPU generation 0 vs TOPAS direct primary C12 products; no mixed-generation energy sums.',
            'Parent-energy, parent-Z and parent-product joint CSV columns are NOT actual parent states in this GPU path; unused.',
            'Cosine bins are width 0.1; agreement cannot exclude finer angular differences.',
            'One run per engine; CDF distance is descriptive, not a confidence level or fitted correction.'])
    with dest.open('x') as f:json.dump(output,f,indent=2,allow_nan=False)
    print(json.dumps({s:{k:v for k,v in r.items() if k!='spectra'}|{'cdf_distances':{k:v['cdf_max_distance'] for k,v in r['spectra'].items()}} for s,r in result.items()},indent=2))

if __name__=='__main__':main()

"""Fixed geometric peak/valley bands, pooled only within BODY and source field."""
import json
from pathlib import Path
import numpy as np
from common import ROOT,read_mhd,packed_to_patient,save_json
from metrics import roi_indices

def analyze_peak_valley():
    m=json.loads((ROOT/'manifest.json').read_text());ref,_=read_mhd(ROOT/'inputs/physical_dose.mhd');ev,_=read_mhd(ROOT/'results/dose_4Gy.mhd')
    # Work in the verified canonical packed grid to define physical slit phase.
    pack=lambda a:np.flip(a,axis=2).transpose(2,0,1)
    ref=pack(ref);ev=pack(ev)
    roi=roi_indices();mask=np.zeros((35,417,505),bool);mask.reshape(-1)[np.union1d(roi['BODY'],roi['GTV'])]=True;mask=pack(mask.transpose(0,2,1))
    x=-169.1015+(np.arange(505)+.5)*.5;y=-33.6383+(np.arange(35)+.5)*2.;z=(np.arange(417)+.5)*.5
    phase=(x+1.8)%3.6-1.8
    domain=mask&(abs(x)[None,None,:]<=14.4)&(abs(y)[None,:,None]<=10)
    masks={};result={}
    for lo,hi in [(0,20),(20,40),(40,80),(80,120),(120,160)]:
        band=(z>=lo)&(z<hi)
        for name,select in [('peak',abs(phase)<=.25),('valley',abs(abs(phase)-1.8)<=.25)]:
            k=f'{lo}-{hi}mm_{name}';masks[k]=domain&band[:,None,None]&select[None,None,:]
            r=ref[masks[k]];g=ev[masks[k]]
            if not len(r) or r.sum()==0:continue
            result[k]=dict(voxels=len(r),reference_mean_Gy=float(r.mean()),gpu_mean_Gy=float(g.mean()),relative_difference_percent=float((g.sum()/r.sum()-1)*100))
    sums={k:[] for k in result};paired={k:[] for k in result};legacy_means={k:[] for k in result}
    for job in m['jobs']:
        d,_=read_mhd(ROOT/'out'/job['name']/'dose.mhd');d=d*(m['protons_4Gy']/job['histories'])
        old,_=read_mhd(Path(m['previous_campaign'])/'out'/job['name']/'dose.mhd');old=old*(m['protons_4Gy']/job['histories'])
        for k in result:
            current=float(d[masks[k]].mean());baseline=float(old[masks[k]].mean())
            sums[k].append(current);paired[k].append(current-baseline);legacy_means[k].append(baseline)
    for k in result:
        se=float(np.std(sums[k],ddof=1)/np.sqrt(len(sums[k])));result[k]['gpu_mean_standard_error_Gy']=se;result[k]['difference_standard_error_percentage_points']=100*se/result[k]['reference_mean_Gy']
        from scipy.stats import t
        base=float(np.mean(legacy_means[k]));mean=float(np.mean(paired[k]));half=float(t.ppf(.975,len(paired[k])-1)*np.std(paired[k],ddof=1)/np.sqrt(len(paired[k])))
        result[k]['paired_vs_legacy_percent']=100*mean/base
        result[k]['paired_vs_legacy_95CI_percent']=[100*(mean-half)/base,100*(mean+half)/base]
    save_json(ROOT/'results/peak_valley.json',dict(definition='BODY union GTV, |world X|<=14.4 mm, |world Z|<=10 mm. Peak centers at fixed Cu slit centers n*3.6 mm, half-width0.25 mm; valley midpoints half-width0.25 mm. Depth from CT box entrance. Same geometric masks for both doses, no profile fitting.',limitations='0.5 mm transverse voxel width equals peak/valley band width; CT entrance may contain air before patient surface; only BODY voxels included. At larger depths scattering blends peaks and valleys. Standard errors exclude reference and response-bank noise.',bands=result))
if __name__=='__main__':analyze_peak_valley()

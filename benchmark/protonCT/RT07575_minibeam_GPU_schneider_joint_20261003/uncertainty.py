"""Between-independent-shard Monte Carlo uncertainty; not model/bank uncertainty."""
import json
import numpy as np
from common import ROOT,read_mhd,packed_to_patient,save_json
from metrics import roi_indices

def analyze_uncertainty():
    manifest=json.loads((ROOT/'manifest.json').read_text());ref,_=read_mhd(ROOT/'inputs/physical_dose.mhd');roi=roi_indices();ids=roi['GTV'];rows=[];means=[]
    for job in manifest['jobs']:
        dose,_=read_mhd(ROOT/'out'/job['name']/'dose.mhd');dose=packed_to_patient(dose)*(manifest['protons_4Gy']/job['histories']);vals=dose.transpose(0,2,1).reshape(-1)[ids];rows.append(vals);means.append(float(vals.mean()))
    rows=np.array(rows,dtype=np.float64);mean=rows.mean(axis=0);se=rows.std(axis=0,ddof=1)/np.sqrt(len(rows));r=ref.transpose(0,2,1).reshape(-1)[ids]
    summary=dict(independent_shards=len(rows),GTV_mean_per_shard_Gy=means,GTV_mean_Gy=float(np.mean(means)),GTV_mean_standard_error_Gy=float(np.std(means,ddof=1)/np.sqrt(len(means))),GTV_voxel_relative_standard_error_median_percent=float(np.median(se/mean)*100),GTV_voxel_relative_standard_error_p90_percent=float(np.percentile(se/mean,90)*100),GTV_relative_RMS_difference_percent=float(np.sqrt(np.mean(((mean-r)/r)**2))*100),GTV_relative_RMS_MC_standard_error_percent=float(np.sqrt(np.mean((se/r)**2))*100),scope='Independent primary histories only; excludes finite TOPAS DIJ noise, response-bank uncertainty and model bias. Nearly equal shard counts; integer allocation differs slightly per shard. No noise-subtracted gamma claim.')
    save_json(ROOT/'results/uncertainty.json',summary)
    print(json.dumps(summary),flush=True)
if __name__=='__main__':analyze_uncertainty()

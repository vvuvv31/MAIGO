#!/usr/bin/env python3
"""Compare independent native full-family geometry controls and frozen models."""
import argparse
import json
from pathlib import Path
from statistics import NormalDist
import numpy as np
from prototype import save, sha
from voxel_profile import CENTRES


def pool(results,field):
    items=[r['summaries'][field] for r in results];n=np.array([s['histories'] for s in items]);total=int(n.sum())
    out={'histories':total}
    for mean_key,sem_key in [('mean','sem'),('roi_mean','roi_sem')]:
        m=np.array([s[mean_key] for s in items]);se=np.array([s[sem_key] for s in items])
        mean=(n[:,None]*m).sum(axis=0)/total
        ss=((n-1)[:,None]*n[:,None]*se**2+n[:,None]*(m-mean)**2).sum(axis=0)
        out[mean_key]=mean;out[sem_key]=np.sqrt(ss/(total-1)/total)
    return out


def compare(a,b):
    av=np.array(a['roi_mean']);bv=np.array(b['roi_mean'])
    diff=(av/bv-1)*100
    half=2.44*np.hypot(a['roi_sem'],b['roi_sem'])/bv*100
    local=(np.array(a['mean'])/np.array(b['mean'])-1)*100
    local_half=NormalDist().inv_cdf(1-.05/(2*36))*np.hypot(a['sem'],b['sem'])/np.array(b['mean'])*100
    return dict(roi_difference_pct=diff.tolist(),conditional_roi_halfwidth_pp=half.tolist(),
                max_local_difference_pct=float(abs(local).max()),local_difference_pct=local.tolist(),
                conditional_local_bonferroni_halfwidth_pp=local_half.tolist(),
                peak_valley_points_within_1pct=bool(np.all(abs(diff)<1)),
                local_points_within_1pct=bool(np.all(abs(local)<1)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--new-model',type=Path);a=p.parse_args();r=a.work
    native={};reports={}
    for label,folder in [('voxel','voxel200k_v3'),('box025','box200k'),('box005','box_step005')]:
        path=r/folder/'report.json';d=json.loads(path.read_text());reports[label]=sha(path)
        native[label]={f:pool(d['results'],f) for f in ['charged','photon','full']}
    m=json.loads((r/'model_lab/report.json').read_text())
    comparisons={'model025_vs_box025_charged':compare(m,native['box025']['charged']),
                 'model025_vs_voxel_charged':compare(m,native['voxel']['charged']),
                 'native_box025_vs_voxel_full':compare(native['box025']['full'],native['voxel']['full']),
                 'native_box005_vs_voxel_full':compare(native['box005']['full'],native['voxel']['full'])}
    if a.new_model:
        new=json.loads((a.new_model/'report.json').read_text());reports['new_model']=sha(a.new_model/'report.json')
        comparisons['model005_vs_box005_charged']=compare(new,native['box005']['charged'])
        comparisons['model005_vs_voxel_charged']=compare(new,native['voxel']['charged'])
    fractions={k:dict(roi_photon_fraction_pct=(v['photon']['roi_mean']/v['full']['roi_mean']*100).tolist(),
                     max_local_photon_fraction_pct=float((v['photon']['mean']/v['full']['mean']*100).max())) for k,v in native.items()}
    report=dict(status='FULL_FAMILY_GEOMETRY_AND_MODEL_SCREEN',ct_acceptance=False,production_promoted=False,
                comparisons=comparisons,photon_contribution=fractions,reference_hashes=reports,
                uncertainty='Native train/test pools are independent of model training; model intervals exclude training/representation uncertainty. Local intervals use 36-bin Bonferroni normal bounds.',
                model_scope='Charged-only; full-family native reference does not restore the model photon branch',
                script_sha256=sha(Path(__file__)))
    save(r/('comparison_final.json' if a.new_model else 'comparison_initial.json'),report)
    for k,v in comparisons.items():print(k,'ROI',v['roi_difference_pct'],'local',v['max_local_difference_pct'])
    print('photon',fractions)
    if a.new_model:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,1,figsize=(8,7),sharex=True)
        axes[0].plot(CENTRES,native['voxel']['full']['mean'],label='Native voxel, full family',color='black')
        axes[0].plot(CENTRES,m['mean'],label='0.25 mm model, charged',ls='--')
        axes[0].plot(CENTRES,new['mean'],label='0.05 mm + entry model, charged')
        axes[0].set_ylabel('Energy fraction / 0.1 mm bin');axes[0].legend()
        for label,key in [('0.25 mm','model025_vs_voxel_charged'),('0.05 mm + entry','model005_vs_voxel_charged')]:
            axes[1].plot(CENTRES,comparisons[key]['local_difference_pct'],label=label)
        axes[1].axhspan(-1,1,alpha=.15,color='grey');axes[1].axhline(0,color='black',lw=.5)
        axes[1].set_ylabel('Difference vs native charged (%)');axes[1].set_xlabel('Folded laboratory x (mm)');axes[1].legend()
        fig.tight_layout();fig.savefig(r/'profile_comparison.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()

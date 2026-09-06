"""Diagnostic only: frozen windows, step/seed sensitivity and 3-D radial moments.

No fit, rescaling, smoothing or production promotion. Raw dose stays immutable.
"""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_air_tissue_interface_pilot import analyze,gpu
from analyze_longitudinal_holdout import sha

def radial_metrics(image,radius2):
    image=np.asarray(image,dtype=float)
    if image.shape!=radius2.shape or not np.isfinite(image).all() or np.any(image<0) or image.sum()<=0:
        raise ValueError('Invalid radial dose image')
    return dict(radial_rms_mm=float(np.sqrt((image*radius2).sum()/image.sum())),
                outer_10mm_fraction=float(image[radius2>100].sum()/image.sum()))

def audit(root,labels):
    manifest=json.loads((root/'manifest.json').read_text())
    reference=analyze(root)
    result={};reference_seeds={}
    xy=(np.arange(100)+.5)*2-100
    radius2=xy[:,None]**2+xy[None,:]**2
    for name,spec in manifest['cases'].items():
        case=root/name; nz=2*sum(s['length_mm'] for s in spec['segments'])
        z=(np.arange(nz)+.5)*.5-spec['interface_mm']
        refs=[]
        for seed in ('s1','s2'):
            layers=[]
            for i,segment in enumerate(spec['segments']):
                rows=np.loadtxt(case/seed/f'dose{i}.csv',delimiter=',',comments='#')
                xyz=rows[:,:3].astype(int)
                layer=np.zeros((segment['length_mm']*2,100,100))
                layer[xyz[:,2],xyz[:,1],xyz[:,0]]=rows[:,3]
                layers.append(layer)
            refs.append(np.concatenate(layers)/manifest['histories'])
        truth=(refs[0]+refs[1])/2
        reference_seeds[name]=[]
        for w in reference['cases'][name]['windows']:
            low,high=w['relative_depth_mm'];take=(z>=low)&(z<high)
            reference_seeds[name].append(dict(relative_depth_mm=[low,high],
                seeds=[radial_metrics(v[take].sum(axis=0),radius2) for v in refs]))
        variants={}
        for label in labels:
            folder=case/label
            profile=gpu(folder/'dose.mhd',nz)/manifest['histories']
            actual=np.fromfile(folder/'dose.raw',dtype='<f4').reshape(nz,100,100)/manifest['histories']
            windows=[]
            for w in reference['cases'][name]['windows']:
                low,high=w['relative_depth_mm'];take=(z>=low)&(z<high)
                ref=truth[take].sum(axis=0);got=actual[take].sum(axis=0,dtype=float)
                windows.append(dict(relative_depth_mm=[low,high],
                    dose_relative_error=float(profile[take].sum()/w['topas_Gy_per_primary']-1),
                    radial_rms_mm=dict(topas=float(np.sqrt((ref*radius2).sum()/ref.sum())),
                                       gpu=float(np.sqrt((got*radius2).sum()/got.sum()))),
                    outer_10mm_fraction=dict(topas=float(ref[radius2>100].sum()/ref.sum()),
                                            gpu=float(got[radius2>100].sum()/got.sum()))))
            q=json.loads((folder/'out/run/quality_report.json').read_text())
            ledger=json.loads((folder/'out/run/energy_ledger.json').read_text())
            roi=(z>=-5)&(z<5)
            maximum=float(np.max(np.abs(profile[roi]/truth[roi].sum(axis=(1,2))-1)))
            kernel=ledger['electron_joint_response']
            numerical=([f['code'] for f in q['failures']]==['unvalidated_electron_joint_response']
                and kernel['domain_misses']==0 and kernel['invalid_marches']==0)
            variants[label]=dict(windows=windows,max_abs_bin_error=maximum,
                narrow_depth_gate=bool(numerical and maximum<=.02 and all(abs(w['dose_relative_error'])<=.01 for w in windows)),
                kernel=kernel,quality=q,
                artifacts={str(p):sha(p) for p in (folder/'dose.raw',folder/'run.yaml',folder/'out/run/quality_report.json')})
        result[name]=variants
    return dict(scope='175 MeV/u axis-aligned HU -1000/100 EM-only; radial metrics diagnostic, not an acceptance gate',
                patient_gamma_improvement_proven=False,cases=result,
                reference_seed_windows=reference_seeds,
                uncertainty_note='Two seeds describe observed spread, not a confidence interval; GPU seeds share a fixed response table and do not measure bulk-response uncertainty.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--labels',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=audit(a.root,a.labels)
    with a.output.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')

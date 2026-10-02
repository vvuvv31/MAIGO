#!/usr/bin/env python3
"""Full 3D gamma pass rates on BODY with DTA/10 isotropic search spacing.

The search computes a pass/fail decision rather than the minimum gamma value.
Each passing voxel has at least one sampled evaluation point with gamma <= 1.
Unresolved voxels exhaust the DTA sphere. Rates use all eligible voxels.
"""
import argparse
import csv
import json
import time
import numpy as np
from scipy.ndimage import map_coordinates
from common import ROOT, sha, read_mhd, write_mhd, save_json
from metrics import roi_indices


def pass_mask(evaluation, reference, points, spacing_zyx, dose_percent, dta, local,
              on_progress=None):
    """Same definition as the historical pass_mask, explicitly at DTA/10."""
    ref = reference[tuple(points.T)].astype(np.float64)
    tolerance = dose_percent*.01*(ref if local else float(reference.max()))
    initial = evaluation[tuple(points.T)].astype(np.float64)
    passed = np.square((initial-ref)/tolerance) <= 1
    if dta == 0:
        return passed
    step = dta/10
    offsets = []
    for z in range(-10,11):
        for y in range(-10,11):
            for x in range(-10,11):
                squared = x*x+y*y+z*z
                if 0 < squared <= 100:
                    distance = step*np.sqrt(squared)
                    offsets.append((distance,np.array([z,y,x])*step/spacing_zyx))
    pending = np.flatnonzero(~passed)
    for index,(distance,offset) in enumerate(sorted(offsets,key=lambda entry:entry[0])):
        if not len(pending):
            break
        coordinates = points[pending].T.astype(float)+offset[:,None]
        valid = np.all((coordinates >= 0) & (coordinates <= np.array(evaluation.shape)[:,None]-1),axis=0)
        values = map_coordinates(evaluation,coordinates,order=1,mode='constant',
                                 cval=np.nan,prefilter=False).astype(np.float64)
        denominator = tolerance[pending] if local else tolerance
        hit = valid & ((distance/dta)**2+np.square((values-ref[pending])/denominator) <= 1)
        passed[pending[hit]] = True
        pending = pending[~hit]
        if on_progress and index%200 == 0:
            on_progress(index,len(offsets),len(pending))
    return passed


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cutoff-percent',type=float,default=10.)
    parser.add_argument('--pilot', choices=['100k','2m'])
    args=parser.parse_args()
    assert 0 < args.cutoff_percent < 100
    results=ROOT/('pilot_'+args.pilot+'_results' if args.pilot else 'results')
    directory=results/f'gamma_cutoff{args.cutoff_percent:g}pct'
    directory.mkdir(exist_ok=True)
    summary=json.loads((results/'summary.json').read_text())
    assert args.pilot or (summary['state']=='complete' and summary['histories']==json.loads((ROOT/'manifest.json').read_text())['histories'])
    reference,fields=read_mhd(ROOT/'inputs/physical_dose.mhd')
    evaluation,evaluation_fields=read_mhd(results/'dose_4Gy.mhd')
    for key in ['TransformMatrix','Offset','ElementSpacing','DimSize']:
        assert fields[key]==evaluation_fields[key],key
    reference=reference.astype(np.float64)
    evaluation=evaluation.astype(np.float64)
    rois=roi_indices()
    nz,ny,nx=reference.shape
    body=np.zeros((nz,nx,ny),dtype=bool)
    body.reshape(-1)[np.union1d(rois['BODY'],rois['GTV'])]=True
    body=body.transpose(0,2,1)
    target=np.zeros((nz,nx,ny),dtype=bool)
    target.reshape(-1)[rois['GTV']]=True
    target=target.transpose(0,2,1)
    maximum=float(reference.max())
    cutoff=args.cutoff_percent*.01*maximum
    mask=body & (reference>=cutoff)
    points=np.argwhere(mask)
    assert len(points)>0
    spacing=np.array(list(map(float,fields['ElementSpacing'].split())))[::-1]
    target_points=target[tuple(points.T)]
    write_mhd(directory/'evaluation_mask.mhd',mask,fields)
    report=dict(state='running',gpu_histories=summary['histories'],
                reference='optimized TOPAS DIJ weighted physical dose',
                reference_histories_per_spot=1000000,
                reference_dose_sha256=sha(ROOT/'inputs/physical_dose.raw'),
                gpu_dose_sha256=sha(results/'dose_4Gy.raw'),
                script_sha256=sha(__file__),
                method=dict(dimension=3,reference_mask='BODY (including GTV) and reference >= cutoff',
                cutoff_percent=args.cutoff_percent,cutoff_Gy=cutoff,reference_maximum_Gy=maximum,
                global_normalization='reference maximum',local_normalization='reference dose at query voxel',
                search_step='DTA / 10 in physical mm, independent isotropic spherical lattice',
                interpolation='trilinear',distance_spacing_zyx_mm=spacing.tolist(),
                candidate_domain='whole evaluation CT grid; no outside-volume extrapolation',
                evaluated_voxels=len(points),gtv_evaluated_voxels=int(target_points.sum()),
                pass_criterion='min of sampled (distance/DTA)^2 + ((evaluation-reference)/dose_tolerance)^2 <= 1',
                zero_dta='same-voxel dose-only comparison',dose_fit=False,spatial_registration=False,
                limitation='finite search lattice; pass/fail only, not exact continuous minimum gamma'),gamma={})
    save_json(directory/'gamma.json',report)
    started=time.perf_counter()
    for dose_percent,dta in ([(3,0),(3,3)] if args.pilot else [(3,0),(3,3),(2,2),(1,.2),(2,.2),(3,.2)]):
        for local in [True,False]:
            key=f'{"local" if local else "global"}_{dose_percent}pct_{dta}mm'
            part_started=time.perf_counter()
            print('START',key,'voxels=',len(points),'step=',dta/10 if dta else 0,flush=True)
            def progress(index,total,pending):
                save_json(directory/'progress.json',dict(criterion=key,offset_index=index,
                          total_offsets=total,pending_voxels=pending,wall_seconds=time.perf_counter()-part_started))
            passed=pass_mask(evaluation,reference,points,spacing,dose_percent,dta,local,progress)
            item=dict(percent=100*float(passed.mean()),passed=int(passed.sum()),
                      failed=int(np.count_nonzero(~passed)),evaluated=len(passed),
                      gtv_percent=100*float(passed[target_points].mean()),
                      gtv_passed=int(passed[target_points].sum()),gtv_evaluated=int(target_points.sum()),
                      dose_percent=dose_percent,dta_mm=dta,search_step_mm=dta/10 if dta else 0,
                      wall_seconds=time.perf_counter()-part_started)
            report['gamma'][key]=item
            np.save(directory/f'{key}_passed.npy',passed)
            if key=='local_1pct_1mm':
                volume=np.zeros(reference.shape,dtype=np.float32)
                volume[tuple(points.T)]=passed
                write_mhd(directory/'local_1pct_1mm_pass.mhd',volume,fields)
            save_json(directory/'gamma.json',report)
            print(key,json.dumps(item),flush=True)
    report.update(state='complete',wall_seconds=time.perf_counter()-started)
    save_json(directory/'gamma.json',report)
    save_json(directory/'progress.json',dict(state='complete',wall_seconds=report['wall_seconds']))
    with (directory/'gamma_rates.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['criterion','pass_percent','passed','evaluated','gtv_pass_percent','search_step_mm'])
        for key,item in report['gamma'].items():
            writer.writerow([key,item['percent'],item['passed'],item['evaluated'],item['gtv_percent'],item['search_step_mm']])
    summary['gamma_computed']=True
    summary['gamma_result_path']=str(directory/'gamma.json')
    save_json(results/'summary.json',summary)
    from plots import make_plots
    make_plots(results)
    if not args.pilot:
        from uncertainty import analyze_uncertainty
        analyze_uncertainty()
        from peak_valley import analyze_peak_valley
        analyze_peak_valley()
        from convergence_report import main as convergence_report
        convergence_report()
    print('COMPLETE',directory/'gamma.json',flush=True)


if __name__=='__main__':
    main()

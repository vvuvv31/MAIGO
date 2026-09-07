"""Absolute, unfitted comparison of the prepared 3D pure-water reference."""
import argparse
import json
import re
import subprocess
from pathlib import Path
import numpy as np
from run_topas10x_gpu_benchmark import sha


def depth_metrics(profile, spacing=0.5):
    peak = int(np.argmax(profile))
    threshold = 0.8 * profile[peak]
    crossings = np.flatnonzero((profile[peak:-1] >= threshold) &
                              (profile[peak+1:] < threshold))
    if not len(crossings):
        raise ValueError('No distal R80 crossing within scorer')
    k = peak + int(crossings[0])
    r80 = (k + 0.5 + (profile[k]-threshold)/(profile[k]-profile[k+1])) * spacing
    return dict(peak_mm=(peak+0.5)*spacing, r80_mm=float(r80))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--job', type=int, required=True)
    parser.add_argument('--gpu', type=Path, help='Frozen alternative GPU run; verify its manifest')
    args = parser.parse_args()
    root = args.reference.resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    status = subprocess.check_output(['sacct', '-j', str(args.job),
        '--format=JobIDRaw,State,ExitCode', '-n', '-P'], text=True)
    if f'{args.job}|COMPLETED|0:0' not in status.splitlines():
        raise ValueError('Reference job is not COMPLETED 0:0: '+status)
    if 'Finalization:' not in (root/f'job_{args.job}.log').read_text():
        raise ValueError('TOPAS finalization missing')
    for p, h in manifest['pins'].items():
        if sha(Path(p)) != h:
            raise ValueError('Frozen input changed: '+p)
    gpu_root = args.gpu.resolve() if args.gpu else Path(manifest['gpu_directory'])
    if args.gpu:
        gpu_manifest=json.loads((gpu_root/'manifest.json').read_text())
        if gpu_manifest['histories']!=manifest['histories']:
            raise ValueError('Different histories require explicit normalization support')
        for p,h in gpu_manifest['pins'].items():
            if sha(Path(p))!=h:raise ValueError('GPU input changed: '+p)
        if sha(gpu_root/'dose.raw')!=gpu_manifest['dose_sha256']:
            raise ValueError('GPU output changed')
    header = (root/'dose_topas.binheader').read_text()
    nz=int(manifest.get('nz',400))
    if [int(re.search(rf'# {a} in (\d+) bins', header)[1]) for a in 'XYZ'] != [64,64,nz]:
        raise ValueError('TOPAS grid mismatch')
    if 'DoseToMedium ( Gy ) : Sum' not in header:
        raise ValueError('Expected exactly summed Gy column: '+header)
    mhd = (gpu_root/'dose.mhd').read_text()
    for field in (f'DimSize = 64 64 {nz}', 'ElementSpacing = 2 2 0.5',
                  'Offset = -63 -63 0.25', 'ElementType = MET_FLOAT'):
        if field not in mhd:
            raise ValueError('GPU geometry/type mismatch: '+field)
    # TOPAS PrintBinary loops k,j,i and appends, hence x-fastest like MHD.
    ref = np.fromfile(root/'dose_topas.bin', dtype='<f8').reshape(nz,64,64)
    gpu = np.fromfile(gpu_root/'dose.raw', dtype='<f4').reshape(nz,64,64).astype(float)
    for data in (ref,gpu):
        if not np.isfinite(data).all() or data.min()<0 or data.max()<=0:
            raise ValueError('Invalid dose payload')
    # 2*2*0.5 mm3, rho=1 g/cm3 -> 2e-6 kg per voxel; Gy=J/kg.
    factor = 2e-6 * 6.241509074e12 / manifest['histories']
    rd = ref.sum(axis=(1,2))*factor
    gd = gpu.sum(axis=(1,2))*factor
    metrics = dict(topas_MeV_per_primary=float(rd.sum()), gpu_MeV_per_primary=float(gd.sum()),
        gpu_over_topas=float(gd.sum()/rd.sum()), topas=depth_metrics(rd), gpu=depth_metrics(gd),
        depth_corr=float(np.corrcoef(rd,gd)[0,1]),
        depth_rmse_percent_ref_peak=float(100*np.sqrt(np.mean((gd-rd)**2))/rd.max()))
    x = np.arange(64)*2-63
    yy,xx = np.meshgrid(x,x,indexing='ij')
    windows = {}
    peak = metrics['topas']['peak_mm']
    for label,lo,hi in [('entrance',0,20),('plateau',20,peak-5),
                        ('peak',peak-2,peak+2),('fragment_tail',peak+5,nz*.5)]:
        z = np.arange(nz)*.5+.25
        mask = (z>=lo)&(z<hi)
        entry = {}
        for name,data in [('topas',ref),('gpu',gpu)]:
            plane = data[mask].sum(axis=0)
            total = plane.sum()
            entry[name] = dict(MeV_per_primary=float(total*factor),
                radial_rms_mm=float(np.sqrt((plane*(xx*xx+yy*yy)).sum()/total)) if total>0 else None,
                outside_10mm_fraction=float(plane[xx*xx+yy*yy>100].sum()/total) if total>0 else None)
        windows[label]=entry
    limitations=list(manifest['limitations'])
    if manifest.get('em_only',False):
        limitations=[v for v in limitations if not v.startswith('TOPAS full physics')]
        limitations.append('EM-only on both sides: opt4+decay TOPAS, no nuclear interactions; GPU has no explicit water electrons.')
    result = dict(status='DIAGNOSTIC_ONLY_NO_PRODUCTION_ACCEPTANCE',job=args.job,
        em_only=manifest.get('em_only',False),metrics=metrics,windows=windows,limitations=limitations,
        dose_pins={str(p):sha(p) for p in (root/'dose_topas.bin',gpu_root/'dose.raw')})
    output_root = gpu_root if args.gpu else root
    with (output_root/'comparison.json').open('x') as f:
        json.dump(result,f,indent=2)
    np.savetxt(output_root/'depth_profiles.csv',np.column_stack((np.arange(nz)*.5+.25,rd,gd)),
        delimiter=',',header='depth_mm,topas_MeV_per_primary_per_bin,gpu_MeV_per_primary_per_bin')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()

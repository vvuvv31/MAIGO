"""Read-only dose inputs; material/edge/noise diagnosis without fitted scaling."""
import json
import argparse
import struct
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('/mnt/sda/wuwei/electron_ct_half10_20260906/20022516'))
    args=parser.parse_args()
    old=Path('/mnt/sda/wuwei/topas10x_threecase_20260905_r3/20022516')
    new=args.run.resolve()
    state=json.loads((new/'execution.json').read_text())
    if state['status']!='complete_experiment' or state['overflow_excluded']:
        raise ValueError('Require completed zero-overflow half-plan')
    shards=state['completed']
    histories=sum(s['histories'] for s in shards)
    if len(shards)!=10 or histories!=88586520:
        raise ValueError('Require the matched 10-shard half-plan')
    m=json.loads((old/'manifest.json').read_text())
    shape=m['gpu_shape_zyx']
    assert m['mapping']=='native'
    ref=np.fromfile(old/'topas_sum.raw','<f4').reshape(shape)
    gpu=np.fromfile(new/'gpu_sum.raw','<f4').reshape(shape)*2
    with (old.parent/'20022516_origin0.bin').open('rb') as f:
        header=struct.unpack('<5I6f',f.read(44))
        assert header[0]==0x47544343 and tuple(header[2:5][::-1])==tuple(shape)
        rho=np.fromfile(f,'<f4',count=ref.size).reshape(shape)
        mat=np.fromfile(f,'u1',count=ref.size).reshape(shape)
    peak=float(ref.max());mask=ref>=.1*peak
    err=(gpu-ref)/peak
    failed=mask & (np.abs(err)>.03)
    boundary=np.zeros(shape,bool)
    for axis in range(3):
        a=[slice(None)]*3;b=a.copy();a[axis]=slice(1,None);b[axis]=slice(None,-1)
        diff=mat[tuple(a)]!=mat[tuple(b)]
        boundary[tuple(a)] |= diff; boundary[tuple(b)] |= diff
    boundary=binary_dilation(boundary,iterations=1)
    groups=[]
    for parity in range(2):
        a=np.zeros(shape,np.float64);n=0
        for t in shards[parity::2]:
            a+=np.fromfile(Path(t['directory'])/'dose.raw','<f4').reshape(shape);n+=t['histories']
        groups.append(a*(m['histories']/n))
    noise=(groups[0]-groups[1])/(2*peak)
    def stats(sel):
        n=int(sel.sum())
        if not n:return {'voxels':0}
        e=err[sel];v=noise[sel]
        return dict(voxels=n,failures=int(failed[sel].sum()),
            hot_failures=int((e>.03).sum()),cold_failures=int((e<-.03).sum()),
            mean_error_pct_peak=float(e.mean()*100),rms_error_pct_peak=float(np.sqrt(np.mean(e*e))*100),
            noise_rms_pct_peak=float(np.sqrt(np.mean(v*v))*100))
    result=dict(histories=88586520,normalization=2,all=stats(mask),
        material={str(s):stats(mask & (mat==s)) for s in range(25)},
        interface=stats(mask & boundary),interior=stats(mask & ~boundary),
        bands={str(lo):stats(mask & (ref>=lo*peak) & (ref<hi*peak)) for lo,hi in ((.1,.2),(.2,.5),(.5,1.01))})
    result['adjacent_error_correlation']={}
    for axis in range(3):
        a=[slice(None)]*3;b=a.copy();a[axis]=slice(1,None);b[axis]=slice(None,-1)
        a=tuple(a);b=tuple(b);sel=mask[a]&mask[b]&(mat[a]==1)&(mat[b]==1)
        result['adjacent_error_correlation'][str(axis)]=float(np.corrcoef(err[a][sel],err[b][sel])[0,1])
    out=new/'residual_diagnosis.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

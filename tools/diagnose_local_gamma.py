"""Frozen full-mask Local Gamma attribution; no new transport or fitted dose."""
import json
import struct
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation
from evaluate_topas10x_gpu_gamma import pass_mask
from run_topas10x_gpu_benchmark import sha


def main():
    root=Path('/mnt/sda/wuwei/electron_ct_half10_step025_20260906/20022516')
    reference=Path('/mnt/sda/wuwei/topas10x_threecase_20260905_r3/20022516')
    target=root/'local_gamma_diagnosis_with_reference_noise.json'
    if target.exists():raise FileExistsError(target)
    state=json.loads((root/'execution.json').read_text())
    m=json.loads((reference/'manifest.json').read_text())
    if state['status']!='complete_experiment' or state['overflow_excluded']:
        raise ValueError('Incomplete/overflow run')
    shards=state['completed']
    if len(shards)!=10 or sum(s['histories'] for s in shards)!=88586520:
        raise ValueError('Unexpected histories')
    if sha(root/'gpu_sum.raw')!=state['aggregate_sha256'] or sha(reference/'topas_sum.raw')!=m['reference_sum_sha256']:
        raise ValueError('Dose pin mismatch')
    if m['mapping']!='native':raise ValueError('Mapping changed')
    shape=m['gpu_shape_zyx']
    ref=np.fromfile(reference/'topas_sum.raw','<f4').reshape(shape)
    dose=np.fromfile(root/'gpu_sum.raw','<f4').reshape(shape).astype(float)*2
    if not np.isfinite(dose).all() or np.any(dose<0):raise ValueError('Invalid dose')
    peak=float(ref.max()); mask=ref>=.1*peak; pts=np.argwhere(mask); index=tuple(pts.T)
    rr=ref[index].astype(float); err=(dose[index]-rr)/rr
    ct=reference.parent/'20022516_origin0.bin'
    with ct.open('rb') as f:
        h=struct.unpack('<5I6f',f.read(44))
        if h[:2]!=(0x47544343,3) or tuple(h[2:5][::-1])!=tuple(shape):raise ValueError('CT header')
        f.seek(44+ref.size*4)
        mat=np.fromfile(f,'u1',count=ref.size).reshape(shape)
    boundary=np.zeros(shape,bool)
    for axis in range(3):
        a=[slice(None)]*3;b=a.copy();a[axis]=slice(1,None);b[axis]=slice(None,-1)
        a=tuple(a);b=tuple(b);different=mat[a]!=mat[b]
        boundary[a]|=different;boundary[b]|=different
    edge=binary_dilation(boundary,iterations=1)[index];section=mat[index]
    fail={}
    for dd,dta in ((1,1),(3,0)):
        fail[f'local_{dd}{dta}']=~pass_mask(dose,ref,pts,np.array(m['spacing_zyx']),dd,dta,True)
    groups=[]
    counts=[]
    for parity in range(2):
        total=np.zeros(len(pts),float);n=0
        for s in shards[parity::2]:
            path=Path(s['directory'])/'dose.raw'
            if sha(path)!=s['dose_sha256']:raise ValueError('Shard pin mismatch')
            total+=np.memmap(path,dtype='<f4',mode='r',shape=tuple(shape))[index]
            n+=s['histories']
        groups.append(total*m['histories']/n);counts.append(n)
    # Equal-statistics 5+5 split; essentially equal totals verified explicitly.
    if abs(counts[0]/counts[1]-1)>1e-4:raise ValueError('Unbalanced split')
    noise=(groups[0]-groups[1])/(2*rr)
    replicas=[]
    for r in m['replicas']:
        if r['histories']!=m['histories']//4:raise ValueError('Reference split unbalanced')
        p=Path(r['path'])/'OSMK_Dtotal_full_plan.bin'
        if sha(p)!=m['input_sha256'][str(p)]:raise ValueError('Reference replica pin')
        replicas.append(np.memmap(p,dtype='<f8',mode='r',shape=tuple(shape))[index])
    if len(replicas)!=4:raise ValueError('Require four reference replicas')
    if not np.array_equal(sum(replicas).astype('<f4'),ref[index]):
        raise ValueError('Replica reconstruction differs from frozen reference')
    reference_noise=(replicas[0]+replicas[2]-replicas[1]-replicas[3])/rr
    def stats(sel):
        n=int(sel.sum())
        if not n:return {'voxels':0}
        e=err[sel];v=noise[sel];t=reference_noise[sel]
        result=dict(voxels=n,mean_relative_error_pct=float(e.mean()*100),
            rms_relative_error_pct=float(np.sqrt(np.mean(e*e))*100),
            gpu_split_noise_rms_relative_pct=float(np.sqrt(np.mean(v*v))*100),
            topas_split_noise_rms_relative_pct=float(np.sqrt(np.mean(t*t))*100),
            combined_noise_rms_relative_pct=float(np.sqrt(np.mean(v*v+t*t))*100),
            noise_exceeds_local_3pct=int((np.abs(v)>.03).sum()))
        for key,f in fail.items():
            result[key]=dict(failures=int(f[sel].sum()),pass_pct=float(100*(1-f[sel].mean())),
                center_hot_failures=int((f[sel] & (e>0)).sum()),
                center_cold_failures=int((f[sel] & (e<0)).sum()))
        return result
    bands={'10-20':(rr>=.1*peak)&(rr<.2*peak),'20-50':(rr>=.2*peak)&(rr<.5*peak),'>=50':rr>=.5*peak}
    result=dict(histories=88586520,reference_sha256=m['reference_sum_sha256'],
        aggregate_sha256=state['aggregate_sha256'],ct_sha256=sha(ct),
        all=stats(np.ones(len(pts),bool)),bands={k:stats(s) for k,s in bands.items()},
        material={str(k):stats(section==k) for k in range(25)},
        interface=stats(edge),interior=stats(~edge),
        band_material_interface={f'{b}/sec{k}/{label}':stats(s & (section==k) & geom)
            for b,s in bands.items() for k in range(25) for label,geom in [('edge',edge),('interior',~edge)]},
        limitations=['Center-dose sign is not the minimizing Gamma location sign.',
            'GPU 5+5 and TOPAS 2+2 split differences estimate aggregate statistical noise, not systematic bias or Gamma uncertainty.',
            'Section boundary plus one-voxel dilation is not every continuous-density boundary.',
            'No noise-subtracted pass rate or causal attribution is asserted.'])
    expected=json.loads((root/'gamma.json').read_text())['results']['new_half10']
    for key in fail:
        if abs(result['all'][key]['pass_pct']-expected[key])>1e-8:raise ValueError('Frozen Gamma reproduction failed')
    with target.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('all','bands','interface','interior')},indent=2),flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Freeze the TOPAS full-plan reference and allocate 0.01% of 4 Gy incident protons."""
import csv, hashlib, json, math, shutil, struct, sys
from pathlib import Path
import numpy as np
import yaml
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
CASE = REPO/'benchmark/protonCT/RT07575_minibeam'
ORIGINAL = Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923/benchmark/protonCT/RT07575_minibeam')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, obj): Path(p).write_text(json.dumps(obj, indent=2)+'\n')
def allocate(weights, total):
    exact=weights/weights.sum()*total
    counts=np.floor(exact).astype(np.int64)
    counts[np.argsort(-(exact-counts),kind='stable')[:total-int(counts.sum())]]+=1
    assert counts.sum()==total
    return counts

def main():
    if (ROOT/'manifest.json').exists(): raise SystemExit('Already prepared; refusing overwrite')
    inputs=ROOT/'inputs'; inputs.mkdir(exist_ok=True)
    for n in ['spots_optimized.csv','plan_export.json','physical_dose.mhd','physical_dose.raw']:
        shutil.copy2(CASE/'code/output_minibeam'/n, inputs/n)
    shutil.copy2(CASE/'code/RT07575_cropXY_bodyZgtv20mm.mat',inputs/'patient.mat')
    for n in ['beam_model.csv','HUtoMaterialSchneider.txt','pbs_dij_template.txt','collimator.txt']:
        shutil.copy2(ORIGINAL/n,inputs/n)
    shutil.copytree(ORIGINAL/'extension',inputs/'topas_extension',dirs_exist_ok=True)
    # Independent voxel-by-voxel check against the original TOPAS DICOM.
    import importlib.util
    legacy=ORIGINAL.parent/'RT07575_proton_GPU'
    sys.path.insert(0,str(legacy))
    spec=importlib.util.spec_from_file_location('ct_verify',legacy/'prepare.py')
    check=importlib.util.module_from_spec(spec);spec.loader.exec_module(check);check.CASE=ORIGINAL
    shutil.copy2(legacy/'prepare.py',inputs/'original_ct_verification.py')
    ct_check=check.check_ct(REPO/'inputs/proton_rt07575/patient_packed.bin',inputs/'patient.mat',inputs/'HUtoMaterialSchneider.txt')
    payload=bytearray((REPO/'inputs/proton_rt07575/patient_packed.bin').read_bytes())
    header=list(struct.unpack('<5I6f',payload[:44]));header[5]+=-42.8515;header[6]+=1.3617
    payload[:44]=struct.pack('<5I6f',*header)
    (inputs/'patient_world_packed.bin').write_bytes(payload)
    ct_check['canonical_origin_xyz_mm']=header[5:8]
    ct_check['world_to_gpu']='(X,Z,Y+117.0136); patient local -> world = Rz(-90)*local + (-42.8515,-12.7636,1.3617)'
    rows=list(csv.DictReader((inputs/'spots_optimized.csv').open()))
    weights=np.array([float(r['weight']) for r in rows]);total=round(weights.sum()*.0001)
    assert len(rows)==870 and (weights>=0).all() and total==23608643
    counts=allocate(weights,total);parts=math.ceil(total/2000000)
    split=np.repeat((counts//parts)[None,:],parts,axis=0);cursor=0
    for j,rem in enumerate(counts%parts):
        idx=(np.arange(rem)+cursor)%parts;split[idx,j]+=1;cursor=(cursor+int(rem))%parts
    assert np.array_equal(split.sum(axis=0),counts) and split.sum(axis=1).max()<=2000000
    with (ROOT/'spot_history_allocation.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['plan_index','spot_id','protons_4Gy','histories_0p01pct']);w.writerows((i,r['spot_id'],weights[i],counts[i]) for i,r in enumerate(rows))
    optics=np.genfromtxt(inputs/'beam_model.csv',delimiter=',',names=True)
    energies=np.array([float(r['energy_MeV']) for r in rows]);x=np.array([float(r['x_mm']) for r in rows]);y=np.array([float(r['y_mm']) for r in rows])
    tx=x/6227.8;ty=y/7008.6
    rx=np.arctan2(np.ones_like(ty),ty);ry=np.arctan2(-tx,np.sqrt(1+ty*ty))
    ux=-np.sin(ry);uy=np.cos(rx)*np.cos(ry);uz=np.sin(rx)*np.cos(ry)
    assert np.max(abs(np.arctan2(ux,uz)-np.arctan(tx)))<1e-14
    assert np.max(abs(np.arctan2(uy,uz)-np.arctan(ty)))<1e-14
    # Verify FP32 kernel angular reconstruction against reference on all spots,
    # including correlated emittance samples (angles, not slopes).
    rng=np.random.default_rng(100207575);xp=rng.normal(0,.006,(870,1024));yp=rng.normal(0,.006,(870,1024))
    ax=np.arctan2(ux.astype('f4'),uz.astype('f4'))[:,None]+xp.astype('f4')
    ay=np.arctan2(uy.astype('f4'),uz.astype('f4'))[:,None]+yp.astype('f4')
    slopes=np.stack([np.tan(ax),np.tan(ay),np.ones_like(ax)],axis=-1);slopes/=np.linalg.norm(slopes,axis=-1,keepdims=True)
    ref=np.stack([np.tan(np.arctan(tx)[:,None]+xp),np.tan(np.arctan(ty)[:,None]+yp),np.ones_like(xp)],axis=-1);ref/=np.linalg.norm(ref,axis=-1,keepdims=True)
    maxerr=float(abs(slopes-ref).max());assert maxerr<3e-7
    channels={0:np.array([int(r['spot_id']) for r in rows]),2:energies,5:x*(1-450/6227.8),6:y*(1-450/7008.6),7:np.rad2deg(rx),8:np.rad2deg(ry)}
    for k,name in {3:'energy_spread_percent',9:'sigma_x_mm',10:'sigma_xp_rad',11:'corr_x',12:'sigma_y_mm',13:'sigma_yp_rad',14:'corr_y'}.items():
        channels[k]=np.interp(energies,optics['energy_MeV'],optics[name])
    base=yaml.safe_load((REPO/'config/proton_ct_minibeam.yaml').read_text())
    base.update(enable_flat_source=False,enable_emittance_source=True,spots_geometry_mode='minibeam_pbs_y',spots_sad_mm=450.0,
        ct_grid_file=str(inputs/'patient_world_packed.bin'),ct_schneider_file=str(inputs/'HUtoMaterialSchneider.txt'),
        initial_energy_MeV=float(energies[weights>0][0]),source_origin_z_mm=-332.9864,
        minibeam_water_entrance_world_y_mm=-117.0136,minibeam_collimator_center_to_isocenter_mm=167.0136,
        voxel_bins_x=505,voxel_bins_y=35,voxel_bins_z=417,voxel_size_x_mm=.5,voxel_size_y_mm=2.,voxel_size_z_mm=.5,
        phantom_length_mm=208.5,depth_bin_width_mm=.5,scorer_area_mm2=17675.,dose_output_scale=1.)
    for k,v in list(base.items()):
        if k.endswith('_file') and isinstance(v,str) and v and not Path(v).is_absolute():base[k]=str(REPO/v)
    for name in ['data','inputs_shared']:
        target=REPO/('inputs' if name=='inputs_shared' else name)
        if not (ROOT/name).exists():(ROOT/name).symlink_to(target,target_is_directory=True)
    def make(name,ns,seed):
        src=ROOT/'sources'/f'{name}.txt';src.parent.mkdir(exist_ok=True)
        ch=dict(channels);ch[4]=ns
        src.write_text('\n'.join(f'dv:Tf/Scatterer1/L{k}/Values = {len(a)} '+' '.join(format(float(v),'.17g') for v in a) for k,a in sorted(ch.items()))+'\n')
        cfg=dict(base,topas_spots_file=str(src),number_of_histories=int(ns.sum()),random_seed=seed)
        path=ROOT/f'{name}.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        return dict(name=name,histories=int(ns.sum()),seed=seed,config_sha256=sha(path),source_sha256=sha(src))
    pilots=[make('pilot_100k',allocate(weights,100000),1002075751),make('pilot_2m',allocate(weights,2000000),1002075752)]
    jobs=[make(f'batch{i:04d}',ns,2000000001+i*10000019) for i,ns in enumerate(split)]
    save(ROOT/'manifest.json',dict(histories=total,parts=parts,protons_4Gy=float(weights.sum()),scale_4Gy=float(weights.sum()/total),jobs=jobs,pilots=pilots,ct_check=ct_check,source_direction_check_max_abs=maxerr,active_spots=int((weights>0).sum()),input_sha256={str(p.relative_to(inputs)):sha(p) for p in inputs.rglob('*') if p.is_file()},dose_mapping='np.flip(packed_zyx.transpose(1,2,0),axis=2)',scope='FP32, group32, material electron replay, full Cu/air, no CT neutron transport'))
    print(json.dumps(dict(histories=total,parts=parts,active_spots=int((weights>0).sum()),source_error=maxerr)))
if __name__=='__main__':main()

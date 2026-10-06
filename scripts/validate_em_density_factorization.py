"""Matched full-plan validation of factorized ion EM data; FP32 dose only.

RT07575 is a head case. Its proton minibeam geometry/optimized weights are
independent of the lung preset. Complete plans may use reduced history counts;
report those counts and Monte Carlo limitations explicitly.
"""
from pathlib import Path
import argparse, csv, importlib.util, json, os, re, struct, subprocess, sys, time
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'out/em_density_factorization_20261005'

def save(p,v):
    p.write_text(json.dumps(v,indent=2)+'\n')

def absolute_paths(cfg):
    for k,v in list(cfg.items()):
        if isinstance(v,str) and v.startswith(('data/','inputs/')):cfg[k]=str(ROOT/v)
        elif isinstance(v,list):cfg[k]=[str(ROOT/x) if isinstance(x,str) and x.startswith(('data/','inputs/')) else x for x in v]
    return cfg

def config(case):
    if case=='proton_head_minibeam':
        cfg=yaml.safe_load((ROOT/'config/proton_ct_minibeam.yaml').read_text())
        old=yaml.safe_load((ROOT/'out/electron_rate_certificate_20261003/screen/reference_b/fresh0000.yaml').read_text())
        keys=['initial_energy_MeV','beam_energy_spread','enable_flat_source','enable_emittance_source','source_origin_z_mm',
              'minibeam_collimator_center_to_isocenter_mm','minibeam_water_entrance_world_y_mm','ct_grid_file',
              'voxel_bins_x','voxel_bins_y','voxel_bins_z','voxel_size_x_mm','voxel_size_y_mm','voxel_size_z_mm',
              'phantom_length_mm','depth_bin_width_mm','scorer_area_mm2','spots_geometry_mode','spots_sad_mm','topas_spots_file']
        cfg.update({k:old[k] for k in keys})
        rows=list(csv.DictReader((ROOT/'benchmark/protonCT/RT07575_minibeam_GPU/inputs/spots_optimized.csv').open()))
        assert len(rows)==870 and sum(float(r['weight'])>0 for r in rows)==487
        weights=OUT/'head_optimized_weights.txt'
        source=Path(cfg['topas_spots_file']).read_text()
        l4=next(l for l in source.splitlines() if '/L4/Values' in l).split('=',1)[1].split()
        assert int(l4[0])==len(rows)
        assert all((float(v)>0)==(float(r['weight'])>0) for v,r in zip(l4[1:],rows))
        weights.write_text(''.join(r['weight']+'\n' for r in rows if float(r['weight'])>0))
        cfg['spot_weights_file']=str(weights)
    else:
        species,beam=case.split('_');cfg=yaml.safe_load((ROOT/f'config/{species}_ct_{beam}.yaml').read_text())
    return absolute_paths(cfg)

def case_identity(case):
    if 'water' in case:return 'Water_75eV phantom'
    if case in ['proton_head_minibeam','proton_broadbeam','carbon_broadbeam','proton_ct_broadbeam','carbon_ct_broadbeam']:return 'RT07575 head'
    return 'lung CT; accuracy deferred'


def run(case,arm,tag,n,h,seed,binary,package_pair=None,config_updates=None,runtime_env=None):
    directory=OUT/'runs'/f'{case}_{arm}_{tag}';directory.mkdir(parents=True,exist_ok=False)
    cfg=config(case);cfg['random_seed']=seed
    package_root=ROOT/({'native':'data/archive/ion_em_native_20261005', 'factorized':'data/archive/ion_em_density_residual_20261005'}.get(arm,'data/em'))
    em_file,delta_file=package_pair if package_pair is not None else (package_root/'unified_em_v1.bin',package_root/'unified_em_delta_moments_v2.bin')
    cfg['em_package_file']=str(Path(em_file).resolve())
    cfg['em_delta_moments_file']=str(Path(delta_file).resolve())
    if config_updates:cfg.update(config_updates)
    file=directory/'case.yaml';file.write_text(yaml.safe_dump(cfg,sort_keys=False))
    command=[str(binary),'--config',str(file),'--batch-count',str(n),'--histories-per-batch',str(h),'--batch-output',str(directory/'batches')]
    save(directory/'command.json',command)
    env={k:v for k,v in os.environ.items() if not k.startswith('CARBON_')}
    env.update(LD_LIBRARY_PATH='/home/wuwei/sycl_workspace/llvm/build/install/lib',CARBON_RUNTIME_BREAKDOWN='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    if runtime_env:env.update(runtime_env)
    start=time.monotonic()
    with (directory/'gpu.log').open('w') as log:
        p=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    text=(directory/'gpu.log').read_text()
    if p.returncode:raise RuntimeError(f'{directory.name} exit {p.returncode}: {text[-3500:]}')
    quality=[]
    for i in range(n):
        folder=directory/'batches'/f'batch{i:06d}';q=json.loads((folder/'quality_report.json').read_text())
        assert q['accepted'] and not q['failures'] and q['queue_overflow_count']==0 and q['physical_relative_energy_residual']<1e-4,q
        dose=np.fromfile(folder/'dose.raw','<f4');assert np.isfinite(dose).all() and (dose>=0).all()
        quality.append(q['physical_relative_energy_residual'])
    plan_summary=None
    if case=='proton_head_minibeam':
        weights=np.loadtxt(cfg['spot_weights_file'],dtype=np.float64).astype(np.longdouble)
        allocated=np.zeros(len(weights),dtype=np.int64)
        for i in range(n):
            rows=list(csv.DictReader((directory/'batches'/f'batch{i:06d}'/'primary_batches.csv').open()))
            assert sum(int(r['history_end'])-int(r['history_begin']) for r in rows)==h
            for r in rows:
                assert (int(r['seed'])-seed)%1000003==0
                ordinal=(int(r['seed'])-seed)//1000003-i*len(weights)
                assert 0<=ordinal<len(weights)
                allocated[ordinal]+=int(r['history_end'])-int(r['history_begin'])
        exact=weights/weights.sum()*(n*h);expected=np.floor(exact).astype(np.int64)
        order=np.argsort(-(exact-expected),kind='stable')
        expected[order[:n*h-int(expected.sum())]]+=1
        assert np.array_equal(allocated,expected)
        plan_summary=dict(active_spots=len(weights),sampled_spots=int((allocated>0).sum()),histories=int(allocated.sum()),allocation='whole-campaign largest remainder; verified per-spot seeds and counts')
        if tag not in ['smoke','closeout']:assert plan_summary['sampled_spots']==487
    completions=re.findall(r'\[independent-fp32-completion\] expected=(\d+) completed=(\d+)',text)
    empty_water='water_broadbeam' in case and text.count('effective=empty_uniform_water_selection')==n
    if empty_water:
        assert not completions and len(re.findall(r'Kernel time: primary=[\d.eE+-]+ s electron=0 s',text))==n
    else:
        assert len(completions)==n and all(a==b for a,b in completions)
    for line in text.splitlines():
        if line.startswith(('[independent-charged-work]','[independent-photon-work]','[electron-joint-pilot-work]')):
            assert all(int(v)==0 for v in re.findall(r'(?:coverage_missing|invalid|photon_gap)=(\d+)',line)),line
    audits=re.findall(r'^\[unified-em-audit\] (.*)$',text,re.M);assert len(audits)==n and all(int(x.split()[0])==0 for x in audits)
    times=[float(x) for x in re.findall(r'\[batch-wall\] index=\d+ seconds=([\d.eE+-]+)',text)];assert len(times)==n
    rec=dict(case=case,arm=arm,tag=tag,binary=str(binary),seed=seed,n=n,h=h,histories=n*h,wall_seconds=time.monotonic()-start,batch_wall_seconds=times,resident_primary_s=(n-1)*h/sum(times[1:]) if n>1 else None,max_energy_residual=max(quality),runtime_pass=True,ion_maximum_step_mm=cfg['maximum_step_mm'],case_identity=case_identity(case))
    rec['spot_allocation']=plan_summary
    rec['config_updates']=config_updates or {};rec['runtime_env']=runtime_env or {}
    rec['em_package_file']=cfg['em_package_file'];rec['em_delta_moments_file']=cfg['em_delta_moments_file']
    save(directory/'execution.json',rec);print(json.dumps(rec),flush=True);return directory,rec

def summed(directory):
    rec=json.loads((directory/'execution.json').read_text());dose=None
    for i in range(rec['n']):
        a=np.fromfile(directory/'batches'/f'batch{i:06d}'/'dose.raw','<f4').astype(float)
        dose=a if dose is None else dose+a
    return dose,rec

def evaluator():
    path=ROOT/'out/broadbeam_shared_electron_20261004/carbon_analysis5/gamma_lattice.py'
    spec=importlib.util.spec_from_file_location('density_gamma_lattice',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.pass_mask

def head_reference():
    base=ROOT/'benchmark/protonCT/RT07575_minibeam_GPU/inputs'
    fields=dict(line.strip().split(' = ',1) for line in (base/'physical_dose.mhd').read_text().splitlines() if ' = ' in line)
    shape=tuple(map(int,fields['DimSize'].split()))[::-1];reference=np.fromfile(base/fields['ElementDataFile'],'<f4').reshape(shape)
    import h5py
    rois={}
    with h5py.File(base/'patient.mat','r') as patient:
        cells=patient['cst'][()]
        for col in range(cells.shape[1]):
            name=''.join(chr(int(v)) for v in patient[cells[1,col]][()].reshape(-1))
            if name not in ['BODY','GTV']:continue
            container=patient[cells[3,col]];data=patient[container[()].reshape(-1)[0]]
            if not data.attrs.get('MATLAB_empty',False):rois[name]=np.unique(data[()].astype(np.int64).reshape(-1))-1
    nx,ny,nz=tuple(map(int,fields['DimSize'].split()))
    masks={}
    for name,ids in rois.items():
        mask=np.zeros((nz,nx,ny),bool);mask.reshape(-1)[ids]=True;masks[name]=mask.transpose(0,2,1)
    weights=list(csv.DictReader((base/'spots_optimized.csv').open()));scale=sum(float(row['weight']) for row in weights)
    return reference,(masks['BODY']|masks['GTV']),masks['GTV'],np.array(list(map(float,fields['ElementSpacing'].split())))[::-1],scale,str(base/'physical_dose.mhd')

def analyze(case,arm,tags,report_dir=None):
    total=None;histories=0;members=[]
    for tag in tags:
        directory=OUT/'runs'/f'{case}_{arm}_{tag}';dose,rec=summed(directory)
        total=dose if total is None else total+dose;histories+=rec['histories'];members.append(rec)
    if case=='proton_head_minibeam':
        ref,body,gtv,spacing,scale,refpath=head_reference();shape=(417,35,505)
        ev=np.flip((total/histories).reshape(shape).transpose(1,2,0),axis=2)*scale
        mask=body & (ref>=ref.max()*.1);percent,dta=3,.2
    else:
        prior=ROOT/'out/broadbeam_shared_electron_20261004'
        refpath=str(prior/('proton_shared5_arrays.npz' if case=='proton_broadbeam' else 'carbon_analysis5/plot_arrays.npz'))
        a=np.load(refpath);ref=a['reference'];spacing=a['spacing'];mask=ref>=ref.max()*.1;gtv=np.zeros_like(mask)
        if case=='proton_broadbeam':shape=(417,35,505);scale=25062649959.598873
        else:
            with (ROOT/'inputs/carbon_rt07575/rt07575_packed.bin').open('rb') as f:header=struct.unpack('<5I6f',f.read(44))
            shape=tuple(reversed(header[2:5]));scale=1.
        ev=np.flip((total/histories).reshape(shape).transpose(1,2,0),axis=2)*scale;percent,dta=1,1
    start=time.monotonic();pts=np.argwhere(mask);hit=evaluator()(ev,ref,pts,spacing,percent,dta,True,step=dta*.1)
    rec=dict(case=case,arm=arm,case_identity='RT07575 head',dimension=3,complete_spot_plan=all(m.get('spot_allocation',{}).get('sampled_spots')==487 for m in members) if case=='proton_head_minibeam' else True,histories=histories,reference=refpath,mask='BODY including GTV, reference >=10% reference max' if case=='proton_head_minibeam' else 'full grid reference >=10% reference max',points=len(pts),dose_percent=percent,dta_mm=dta,search_step_mm=dta*.1,pass_percent=float(hit.mean()*100),passed=bool(hit.mean()>.99),dose_sum_bias_percent=float(100*(ev[mask].sum()/ref[mask].sum()-1)),normalization='absolute plan weights/per-history; no fitting or registration',gamma_seconds=time.monotonic()-start)
    if gtv.any():
        select=gtv[tuple(pts.T)];rec['GTV_pass_percent']=float(hit[select].mean()*100);rec['prescription_fraction_sampled']=histories/scale
    save((Path(report_dir) if report_dir is not None else OUT)/f'{case}_{arm}_gamma.json',rec)
    print(json.dumps(rec),flush=True)
    return rec,ev,mask,spacing

def compare(case,tags):
    baseline,ref,mask,spacing=analyze(case,'native',tags);candidate,ev,_,_=analyze(case,'factorized',tags)
    hit=evaluator()(ev,ref,np.argwhere(mask),spacing,baseline['dose_percent'],baseline['dta_mm'],True,step=baseline['search_step_mm'])
    delta=ev[mask]-ref[mask];local=100*abs(delta)/ref[mask]
    rec=dict(case=case,native_TOPAS=baseline,factorized_TOPAS=candidate,against_native_pass_percent=float(hit.mean()*100),relative_L1=float(abs(delta).sum()/ref[mask].sum()),dose_sum_change_percent=float(100*delta.sum()/ref[mask].sum()),local_percent_quantiles={str(p):float(np.percentile(local,p)) for p in [50,95,99,99.9,100]},accuracy_accepted=bool(candidate['passed']))
    save(OUT/f'{case}_comparison.json',rec);return rec

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['smoke','full','head-0p1','analyze'],required=True);p.add_argument('--binary',type=Path,default=ROOT/'build/carbon_mc');a=p.parse_args();binary=a.binary.resolve()
    if a.phase=='head-0p1':
        #24 existing batches plus95 supplements =119*1,983,920 histories.
        # This rounds0.1% of the optimized plan by only48 incident protons.
        case='proton_head_minibeam'
        for arm in ['factorized','native']:
            for tag in ['s1','s2']:
                prior=json.loads((OUT/'runs'/f'{case}_{arm}_{tag}'/'execution.json').read_text())
                assert prior['runtime_pass'] and prior['histories']==23807040
            run(case,arm,'h01',95,1983920,1001904702,binary)
        compare(case,['s1','s2','h01'])
        return
    if a.phase=='smoke':
        for case in ['proton_broadbeam','carbon_broadbeam','proton_head_minibeam','carbon_minibeam']:
            for arm in ['native','factorized']:run(case,arm,'smoke',2,50000 if case!='carbon_minibeam' else 20000,1001904402,binary)
    else:
        results=[]
        for case,n,h in [('proton_broadbeam',10,1253150),('carbon_broadbeam',7,925987),('proton_head_minibeam',12,1983920)]:
            if a.phase=='full':
                for tag,seed,arms in [('s1',1001904502,['native','factorized']),('s2',1001904602,['factorized','native'])]:
                    for arm in arms:run(case,arm,tag,n,h,seed,binary)
            tags=['s1','s2']
            if case=='proton_head_minibeam' and all((OUT/'runs'/f'{case}_{arm}_h01'/'execution.json').exists() for arm in ['native','factorized']):tags.append('h01')
            results.append(compare(case,tags));save(OUT/'full_plan_report.json',results)

if __name__=='__main__':main()

from pathlib import Path
import numpy as np, json, hashlib, importlib.util, subprocess, csv, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923')
out=root/'evidence/boundary_fix_20260923'
old=root/'evidence/review_fix_20260923'
ref=Path('/mnt/sda/wuwei/minibeam_single_center_em_only_water75ev_e250_10m/topas_7378')
gate_path=root/'evidence/urban_d51e599_20260923/dose_gate.py'
spec=importlib.util.spec_from_file_location('gate',gate_path)
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
x=-49.95+np.arange(1000)*.1
mass=2.5e-6;mevj=1.602176634e-13
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(a):
    a=np.asarray(a,float)
    return {'mean':float(a.mean()),'se':float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None,'values':a.tolist()}
def ratio(a,b):
    ga,ta=stats(a),stats(b);v=ga['mean']/ta['mean']-1 if ta['mean'] else None
    se=float(np.hypot(ga['se']/ta['mean'],ga['mean']*ta['se']/ta['mean']**2)) if ta['mean'] and ta['se'] is not None and ga['se'] is not None else None
    return {'gpu':ga,'topas':ta,'relative_difference':v,'ratio_se':se}
def gpu(stem,nz,n):
    return np.stack([np.fromfile(root/f'out/{stem}_s{i}/dose.raw',dtype='<f4').reshape(nz,1000).astype(float)/n for i in [1,2,3]])
def topas(folder,stem):
    return np.stack([np.fromfile(folder/f'{stem}_s{i}/dose.bin',dtype='<f8').reshape(600,1000)/20000 for i in [1,2,3]])
def metrics(g,t):
    z=.125+np.arange(g.shape[1])*.25
    r={'energy_MeV_per_primary':ratio(g.sum((1,2))*mass/mevj,t.sum((1,2))*mass/mevj),'rows':[]}
    for dep in [20,40,60,80,100,120]:
        iz=np.argmin(abs(z-dep));pm=abs(x)<.25;vm=abs(abs(x)-1.8)<.45
        wg=[gate._fwhm_1d(a[iz],x) for a in g];wt=[gate._fwhm_1d(a[iz],x) for a in t]
        pg,pt=g[:,iz,pm].mean(1),t[:,iz,pm].mean(1)
        vg,vt=g[:,iz,vm].mean(1),t[:,iz,vm].mean(1)
        r['rows'].append({'depth_mm':float(z[iz]),'row_energy':ratio(g[:,iz].sum(1),t[:,iz].sum(1)),
            'peak':ratio(pg,pt),'valley':ratio(vg,vt),'pvdr':ratio(pg/vg,pt/vt) if np.all(vg>0) and np.all(vt>0) else None,
            'fwhm_mm':ratio(wg,wt) if None not in wg+wt else None})
    rg=[gate._r80(a.sum(1),z) for a in g];rt=[gate._r80(a.sum(1),z) for a in t]
    r['integrated_depth_R80_mm']={'gpu':stats(rg),'topas':stats(rt),'difference_mm':float(np.mean(rg)-np.mean(rt))}
    return r
def manifest(stem,tfolder,tstem,label):
    for role in ['gpu','topas']:
        m=json.loads((old/f'{role}_manifest.json').read_text())
        for i,r in enumerate(m['runs'],1):
            if role=='gpu':
                d=root/f'out/{stem}_s{i}';c=root/f'config/{stem}_s{i}.yaml';dose=d/'dose.mhd'
                r.update(mhd=str(dose),mhd_sha256=sha(dose),raw_sha256=sha(d/'dose.raw'))
            else:
                d=tfolder/f'{tstem}_s{i}';c=d/'input.txt';dose=d/'dose.bin'
                r.update(dose_bin=str(dose),dose_header=str(d/'dose.binheader'),header_sha256=sha(d/'dose.binheader'))
            r.update(effective_config=str(c),config_sha256=sha(c),dose_sha256=sha(dose))
        (out/f'{label}_{role}_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    run=subprocess.run([sys.executable,str(gate_path),'--gpu-manifest',str(out/f'{label}_gpu_manifest.json'),
        '--topas-manifest',str(out/f'{label}_topas_manifest.json'),'--accept',str(root/'evidence/urban_after_0f2c0ca_20260922/acceptance.yaml'),
        '--out',str(out/f'{label}_dose_gate.json')],capture_output=True,text=True)
    (out/f'{label}_dose_gate.log').write_text(run.stdout+run.stderr+f'\nexit_code={run.returncode}\n')
    assert run.returncode in [0,1,2],run.stderr
    return {'exit_code':run.returncode,'state':json.loads((out/f'{label}_dose_gate.json').read_text())['state']}

data={
 'water050':(gpu('boundary_final_water',600,20000),topas(old,'topas_water'),gpu('review_water_fixed',600,20000)),
 'water025':(gpu('boundary_final_water_step025',600,20000),topas(out,'topas_step025'),gpu('review_water_step025',600,20000)),
 'fullchain':(gpu('boundary_final_fullchain',1000,250000),np.fromfile(ref/'dose.bin',dtype='<f8').reshape(1,1000,1000)/10000000,gpu('review_fullchain',1000,250000))}
results={'scope':'C12 250 MeV/u, Water_75eV, EM only, physical voxel navigation opt-in; no fitted dose normalization or scattering coefficients',
         'water_histories_per_seed':20000,'gpu_seeds':3,'fullchain_gpu_histories_per_seed':250000,
         'fullchain_status':'EXPLORATORY_TOPAS_REFERENCE_SE_MISSING','cases':{}}
for label,(g,t,before) in data.items():
    r=metrics(g,t);r['before_boundary_fix']=metrics(before,t)
    if label!='fullchain':
        r['gate']=manifest('boundary_final_water' if label=='water050' else 'boundary_final_water_step025',old if label=='water050' else out,'topas_water' if label=='water050' else 'topas_step025',label)
    results['cases'][label]=r
    with (out/f'{label}_comparison.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['depth_mm','peak_rel_percent','valley_rel_percent','PVDR_rel_percent','GPU_FWHM_mm','TOPAS_FWHM_mm','FWHM_rel_percent','FWHM_ratio_SE_pp','before_FWHM_rel_percent'])
        for row,b in zip(r['rows'],r['before_boundary_fix']['rows']):
            fw=row['fwhm_mm']
            w.writerow([row['depth_mm']]+[100*row[k]['relative_difference'] if row[k] and row[k]['relative_difference'] is not None else '' for k in ['peak','valley','pvdr']]+
                [fw['gpu']['mean'],fw['topas']['mean'],100*fw['relative_difference'],100*fw['ratio_se'] if fw['ratio_se'] is not None else '',100*b['fwhm_mm']['relative_difference']])
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    z=.125+np.arange(g.shape[1])*.25
    for a,name,c in [(t,'TOPAS','#111827'),(before,'GPU before boundary fix','#94a3b8'),(g,'GPU physical voxels','#dc2626')]:
        axes[0,0].plot(z,a.sum(2).mean(0)*mass/mevj/.25,label=name,c=c,lw=1.3)
    axes[0,0].set(xlabel='Depth (mm)',ylabel='Energy deposit (MeV/mm/source primary)',xlim=(0,140));axes[0,0].legend(fontsize=8)
    for mr,c,name in [(r['before_boundary_fix'],'#94a3b8','Before'),(r,'#dc2626','Physical voxels')]:
        axes[0,1].errorbar([a['depth_mm'] for a in mr['rows']],[100*a['fwhm_mm']['relative_difference'] for a in mr['rows']],
            yerr=[200*(a['fwhm_mm']['ratio_se'] or 0) for a in mr['rows']],fmt='o-',c=c,label=name,capsize=3)
    axes[0,1].axhline(0,c='#111',lw=.7);axes[0,1].set(xlabel='Depth (mm)',ylabel='FWHM GPU/TOPAS - 1 (%)');axes[0,1].legend(fontsize=8)
    for role,name,c in [('gpu','GPU physical voxels','#dc2626'),('topas','TOPAS','#111827')]:
        axes[0,2].errorbar([a['depth_mm'] for a in r['rows']],[a['fwhm_mm'][role]['mean'] for a in r['rows']],
            yerr=[2*(a['fwhm_mm'][role]['se'] or 0) for a in r['rows']],fmt='o-',c=c,label=name,capsize=3)
    axes[0,2].set(xlabel='Depth (mm)',ylabel='Central beamlet FWHM (mm)');axes[0,2].legend(fontsize=8)
    for ax,dep in zip(axes[1],[20,80,120]):
        iz=np.argmin(abs(z-dep))
        for a,name,c in [(t,'TOPAS','#111827'),(before,'GPU before','#94a3b8'),(g,'GPU physical voxels','#dc2626')]:ax.plot(x,a[:,iz].mean(0),c=c,label=name)
        ax.set(xlabel='Transverse x (mm)',ylabel='Dose (Gy/source primary)',xlim=(-8,8) if label=='fullchain' else (-4,4),title=f'Depth {z[iz]:.3f} mm')
    subtitle='TOPAS 10M x 1; GPU 250k x 3. TOPAS SE unavailable; FWHM ratio error bars omitted.' if label=='fullchain' else f'20k x 3 seeds per engine; step {0.05 if label=="water050" else 0.025} mm. Error bars: +/-2 run SE.'
    fig.suptitle(f'C12 250 MeV/u | {label} | Water_75eV | No fitted normalization\n{subtitle}',fontsize=12)
    fig.savefig(out/f'{label}_comparison.png',dpi=160);plt.close(fig)
    print(label,'energy error %',100*r['energy_MeV_per_primary']['relative_difference'],'R80 delta mm',r['integrated_depth_R80_mm']['difference_mm'],'gate',r.get('gate'),flush=True)
    print((out/f'{label}_comparison.csv').read_text(),flush=True)

response={}
for label,idx in [('TOPAS',1),('GPU before',2),('GPU physical voxels',0)]:
    a=data['water050'][idx];b=data['water025'][idx];z=.125+np.arange(600)*.25
    rows=[]
    for dep in [20,40,60,80,100,120]:
        iz=np.argmin(abs(z-dep));wa=np.array([gate._fwhm_1d(v[iz],x) for v in a]);wb=np.array([gate._fwhm_1d(v[iz],x) for v in b])
        rows.append({'depth_mm':float(z[iz]),'relative_change':float(wb.mean()/wa.mean()-1),
                     'paired_run_change':stats(wb/wa-1),'fwhm050':stats(wa),'fwhm025':stats(wb)})
    response[label]=rows
results['step050_to_025']=response
(out/'comparison.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
for (label,rows),c in zip(response.items(),['#111827','#94a3b8','#dc2626']):
    ax.errorbar([r['depth_mm'] for r in rows],[100*r['relative_change'] for r in rows],
        yerr=[200*r['paired_run_change']['se'] for r in rows],fmt='o-',label=label,c=c,capsize=3)
ax.axhline(0,c='#888',lw=.7);ax.set(xlabel='Depth (mm)',ylabel='FWHM change at 0.025 vs 0.05 mm step (%)',title='Step response | 20,000 histories x 3 seeds\nError bars: +/-2 SE of paired per-seed relative changes')
ax.legend();fig.savefig(out/'step_response.png',dpi=160);plt.close(fig)
print('STEP_RESPONSE',json.dumps({k:[round(100*r['relative_change'],4) for r in v] for k,v in response.items()}),flush=True)

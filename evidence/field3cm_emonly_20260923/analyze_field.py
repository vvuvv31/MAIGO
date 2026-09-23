from pathlib import Path
import csv,hashlib,json,math
import numpy as np
from scipy.stats import t as student
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

R=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923')
O=R/'evidence/field3cm_emonly_20260923';D=O/'results'
M=json.loads((O/'manifest.json').read_text());P=json.loads((O/'preflight.json').read_text())
assert P['accepted']
n=M['histories_per_batch'];g=[];t=[];quality=[]
for i in range(1,5):
    td=Path(M['cases'][f'topas_b{i}']);gd=Path(M['cases'][f'gpu_b{i}']).parent
    tq=json.loads((td/'quality.json').read_text());gq=json.loads((gd/'quality.json').read_text())
    assert tq['accepted'] and gq['accepted'] and tq['histories']==gq['histories']==n
    assert tq['topas_binary_sha256']==P['topas_binary_sha256']
    assert gq['gpu_binary_sha256']==M['gpu_binary_sha256']
    t.append(np.fromfile(td/'dose.bin',dtype='<f8').reshape(1000,1000)/n)
    g.append(np.fromfile(gq['dose_path'],dtype='<f4').reshape(1000,1000).astype(float)/n)
    quality.extend([tq,gq])
g=np.stack(g);t=np.stack(t);assert np.isfinite(g).all() and np.isfinite(t).all()
gm=g.mean(0);tm=t.mean(0);x=-49.95+np.arange(1000)*.1;z=.125+np.arange(1000)*.25
peak=abs(x)<.25;valley=abs(abs(x)-1.8)<.45
field_peak=np.any(abs(x[None,:]-np.arange(-3,4)[:,None]*3.6)<.25,axis=0)
field_valley=np.any(abs(x[None,:]-(np.arange(-3,3)+.5)[:,None]*3.6)<.45,axis=0)

def ratio(a,b):
    a=np.asarray(a);b=np.asarray(b);am=a.mean(0);bm=b.mean(0)
    ase=a.std(0,ddof=1)/2;bse=b.std(0,ddof=1)/2
    r=np.divide(am,bm,out=np.full_like(am,np.nan),where=bm>0)
    av=np.divide(ase,bm,out=np.zeros_like(am),where=bm>0)**2
    bv=np.divide(r*bse,bm,out=np.zeros_like(am),where=bm>0)**2
    var=av+bv;den=av**2/3+bv**2/3
    df=np.divide(var**2,den,out=np.full_like(am,np.nan),where=den>0)
    half=student.ppf(.975,df)*np.sqrt(var)
    return dict(gpu=am,topas=bm,gpu_SE=ase,topas_SE=bse,difference_percent=(r-1)*100,
                CI95_low_percent=(r-1-half)*100,CI95_high_percent=(r-1+half)*100)

curves={label:ratio(g[:,:,mask].mean(2),t[:,:,mask].mean(2)) for label,mask in
        [('peak',peak),('valley',valley),('inner_field_peaks',field_peak),('inner_field_valleys',field_valley)]}
rows=[]
for dep in [20,40,60,80,100,120]:
    k=int(np.argmin(abs(z-dep)));row=dict(depth_mm=dep,actual_depth_mm=float(z[k]))
    for label,c in curves.items():row[label]={key:float(v[k]) for key,v in c.items()}
    rows.append(row)
summary=dict(status='COMPLETE',extension_commit=M['extension_commit'],histories_per_engine=12800000,
             batches_per_engine=4,source_audit_passed=True,rows=rows,quality=quality,
             total_deposited_energy_difference_percent=float((gm.sum()/tm.sum()-1)*100),
             method='Absolute dose per original source. Four independent equal-size batches per engine. Ratios use delta-method approximate Welch-t 95% intervals. No fitted normalization or depth smoothing.',
             scope='Original 16x16 scan grid. Homogeneous Water_75eV volume, no TOPAS-only same-material Box2 daughter. Scoring averages over the original 100 mm vertical bin.',
             implementation_limits=M['implementation_limits'])
(D/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
with (D/'depth_curves.csv').open('w',newline='') as f:
    writer=csv.writer(f);keys=list(curves['peak']);writer.writerow(['depth_mm','ROI']+keys)
    for label,c in curves.items():
        for k,depth in enumerate(z):writer.writerow([depth,label]+[float(c[key][k]) for key in keys])

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})
fig,axs=plt.subplots(2,2,figsize=(14,8),constrained_layout=True)
for col,label in enumerate(['peak','valley']):
    c=curves[label];ax=axs[0,col]
    ax.plot(z,c['topas']*1e9,label='TOPAS',color='#2563eb',lw=1.5)
    ax.plot(z,c['gpu']*1e9,label='GPU',color='#ea7b24',ls='--',lw=1.3)
    ax.set(title='Central '+label,xlim=(0,140),ylim=(0,None),xlabel='Water depth (mm)',ylabel='Mean dose (nGy / source)');ax.legend()
    ax=axs[1,col];valid=(z<=125)&(c['topas']>c['topas'].max()*.01)
    ax.axhspan(-1,1,color='#d7efda');ax.axhline(0,color='#64748b',lw=.6)
    ax.plot(z[valid],c['difference_percent'][valid],color='#007d83',lw=.85)
    ax.fill_between(z[valid],c['CI95_low_percent'][valid],c['CI95_high_percent'][valid],color='#007d83',alpha=.15)
    ax.set(xlim=(0,125),xlabel='Water depth (mm)',ylabel='(GPU/TOPAS - 1) (%)',title='Relative difference; approx. 95% CI')
fig.suptitle('3 cm x 3 cm field | 256 spots | 250 MeV/u C12 | EM-only\nGPU / TOPAS: 4 x 3.2M each; 0.25 mm depth bins; green target +/-1%')
fig.savefig(D/'peak_valley_depth.png',dpi=180);plt.close(fig)

xs=abs(x)<=25;zs=z<=140;ta=tm[zs][:,xs]*1e9;ga=gm[zs][:,xs]*1e9
vmax=max(ta.max(),ga.max());norm=LogNorm(vmin=vmax*.001,vmax=vmax)
extent=[x[xs][0]-.05,x[xs][-1]+.05,z[zs][-1]+.125,0]
fig,axs=plt.subplots(1,3,figsize=(15,6),constrained_layout=True)
for ax,v,label in zip(axs[:2],[ta,ga],['TOPAS','GPU']):
    im=ax.imshow(v,extent=extent,aspect='auto',norm=norm,cmap='viridis');ax.set(title=label,xlabel='Transverse x (mm)',ylabel='Water depth (mm)');ax.grid(False)
fig.colorbar(im,ax=list(axs[:2]),label='Mean dose (nGy / source)')
rel=np.divide(ga,ta,out=np.full_like(ga,np.nan),where=ta>=.002*ta.max())*100-100
im=axs[2].imshow(rel,extent=extent,aspect='auto',vmin=-10,vmax=10,cmap='RdBu_r')
axs[2].set(title='Relative difference (color clipped at +/-10%)',xlabel='Transverse x (mm)',ylabel='Water depth (mm)');axs[2].grid(False)
fig.colorbar(im,ax=axs[2],label='(GPU/TOPAS - 1) (%)',extend='both')
fig.suptitle('3 cm x 3 cm field | Same absolute source normalization\nOriginal 100 mm vertical scoring bin; ratios omitted below 0.2% of maximum TOPAS dose')
fig.savefig(D/'field_dose_maps.png',dpi=180);plt.close(fig)

fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
for ax,dep in zip(axs.flat,[20,60,100,124]):
    k=int(np.argmin(abs(z-dep)))
    ax.plot(x,tm[k]*1e9,label='TOPAS',color='#2563eb',lw=1.4)
    ax.plot(x,gm[k]*1e9,label='GPU',color='#ea7b24',lw=1.2,ls='--')
    ax.set(xlim=(-25,25),ylim=(0,None),title=f'Depth {z[k]:.3f} mm',xlabel='Transverse x (mm)',ylabel='Mean dose (nGy / source)');ax.legend()
fig.suptitle('3 cm x 3 cm field: transverse minibeam profiles | 0.25 mm depth layer')
fig.savefig(D/'transverse_profiles.png',dpi=180);plt.close(fig)

report=['# 3 cm × 3 cm field：EM-only TOPAS/GPU 对照','',
        f'扩展commit：`{M["extension_commit"]}`；TOPAS 4.2.3 / Geant4 11.3.2。',
        '原计划为16×16个spot，两个方向−15至+15 mm、间距2 mm；C12 250 MeV/u。两端各4个独立批次，每批320万历史，共1280万，每spot总计50000历史。',
        '256个spot的非均匀权重、多线程Event ID和零发射度几何检查通过。正式运行使用原始非零发射度、1.2%能散，未使用DijMode。',
        'Cu/水最大步长均0.05 mm，Water_75eV，EM-only。水箱统一为100×100×250 mm且无内部Box2子体积，剂量网格沿用0.1×100×0.25 mm。',
        '','| 深度 mm | 中央峰区差 | 邻近谷区差 | 谷区近似95%区间 |','|---:|---:|---:|---:|']
for a in rows:
    p=a['peak'];v=a['valley'];report.append(f'| {a["depth_mm"]} | {p["difference_percent"]:+.2f}% | {v["difference_percent"]:+.2f}% | [{v["CI95_low_percent"]:+.2f}%, {v["CI95_high_percent"]:+.2f}%] |')
report += ['',f'全体积沉积能量差：{summary["total_deposited_energy_difference_percent"]:+.3f}%。',
           '中央峰区为 |x|<0.25 mm；邻近谷区为 ||x|−1.8|<0.45 mm，和此前单spot的横向ROI相同。整场内部多个峰/谷的合并指标另存comparison.json和depth_curves.csv。',
           '所有剂量以原始源粒子数归一化，没有以峰值或总剂量拟合归一化。每个ROI先在批次内部合并，再估计独立批次误差。4批的置信区间仍属近似，不据此自动宣布1%精度。',
           '', '模型实现仍有已知差别：GPU主比较保持电子local沉积、空气平均能损近似，TOPAS显式输运；参数对齐不表示近似输运算法完全相同。',
           '', '图：field_dose_maps.png、transverse_profiles.png、peak_valley_depth.png。运行配置、Slurm脚本、哈希及验证结果位于父目录。']
(D/'RESULTS.md').write_text('\n'.join(report)+'\n')
print(json.dumps({k:summary[k] for k in ['status','histories_per_engine','extension_commit','total_deposited_energy_difference_percent']},indent=2),flush=True)

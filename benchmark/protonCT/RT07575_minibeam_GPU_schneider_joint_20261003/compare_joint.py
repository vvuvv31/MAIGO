"""Compare the opt-in joint full plan with the unchanged legacy and TOPAS doses."""
import json,re,sys
from pathlib import Path
import numpy as np
from common import ROOT,REPO,read_mhd,save_json
from metrics import roi_indices
sys.path.insert(0,str(REPO/'scripts'))
from compare_minibeam import AUDITS,integer_fields
OLD=Path(json.loads((ROOT/'manifest.json').read_text())['previous_campaign'])
def read(p):return json.loads(p.read_text())
def records(root):
    chunks={};logs=[]
    for p in sorted(root.glob('group[0-9][0-9][0-9].log')):
        t=p.read_text();logs.append(t)
        for part in t.split('[plan-shard] ')[1:]:
            name=re.search(r'config="[^"]*/(batch\d+)\.yaml"',part).group(1);assert '[plan-shard-done]' in part
            chunks[name]=part
    return chunks,'\n'.join(logs)
def delta(x,y):
    x=x.astype(float);y=y.astype(float);d=abs(y-x);m=x>0
    return dict(relative_L1=float(d.sum()/x.sum()),max_absolute_Gy=float(d.max()),max_local_percent=float((d[m]/x[m]).max()*100))
def main():
    manifest=read(ROOT/'manifest.json');summary=read(ROOT/'results/summary.json');oldsummary=read(OLD/'results/summary.json')
    assert summary['state']=='complete' and summary['histories']==oldsummary['histories']==236086432
    previous,plog=records(OLD);current,clog=records(ROOT);checks=[];work={};packets=0
    for job in manifest['jobs']:
        name=job['name'];a=previous[name];b=current[name];q=read(ROOT/f'out/{name}/quality_report.json')
        aa=[l for l in a.splitlines() if l.startswith(AUDITS)];bb=[l for l in b.splitlines() if l.startswith(AUDITS)]
        ia=integer_fields(read(OLD/f'out/{name}/energy_ledger.json'));ib=integer_fields(read(ROOT/f'out/{name}/energy_ledger.json'))
        w=dict((k,int(v)) for k,v in re.findall(r'(\w+)=(\d+)',next(l for l in b.splitlines() if l.startswith('[electron-joint-pilot-work]'))))
        for k,v in w.items():work[k]=work.get(k,0)+v
        packets+=int(re.search(r'\[electron-queue-time\].*?tasks=(\d+)',b)[1])
        checks.append(dict(name=name,primary_audits_equal=aa==bb,integer_mismatches=[k for k in ia.keys()|ib.keys() if ia.get(k)!=ib.get(k)],quality=q,joint_work=w))
        assert q['accepted'] and not q['failures'] and q['queue_overflow_count']==0 and w['started']>0 and w['invalid']==0
    save_json(ROOT/'results/batch_checks.json',checks)
    x,_=read_mhd(OLD/'results/dose_4Gy.mhd');y,_=read_mhd(ROOT/'results/dose_4Gy.mhd');rois=roi_indices();gtv=rois['GTV']
    gx=x.transpose(0,2,1).reshape(-1)[gtv];gy=y.transpose(0,2,1).reshape(-1)[gtv]
    oldgamma=read(OLD/'results/gamma_cutoff10pct/gamma.json');gamma=read(ROOT/'results/gamma_cutoff10pct/gamma.json')
    assert oldgamma['method']==gamma['method'] and oldgamma['reference_dose_sha256']==gamma['reference_dose_sha256']
    gammas={k:dict(legacy_percent=v['percent'],joint_percent=gamma['gamma'][k]['percent'],change_percentage_points=gamma['gamma'][k]['percent']-v['percent']) for k,v in oldgamma['gamma'].items()}
    opv=read(OLD/'results/peak_valley.json')['bands'];npv=read(ROOT/'results/peak_valley.json')['bands']
    pv={k:dict(legacy_vs_TOPAS_pct=v['relative_difference_percent'],joint_vs_TOPAS_pct=npv[k]['relative_difference_percent'],joint_vs_legacy_pct=100*(npv[k]['gpu_mean_Gy']/v['gpu_mean_Gy']-1),paired_95CI_percent=npv[k]['paired_vs_legacy_95CI_percent']) for k,v in opv.items()}
    def timing(log):
        stages=lambda name:sum(map(float,re.findall(r'\[runtime-stage\] '+name+r' seconds=([\d.eE+-]+)',log)))
        return dict(transport_seconds=stages('transport_loop_including_scoring_and_queue_transfers'),setup_seconds=stages('transport_setup_including_upload'),electron_seconds=sum(map(float,re.findall(r'\[electron-queue-time\] seconds=([\d.eE+-]+)',log))))
    result=dict(status='EXPERIMENTAL_RT07575_HYBRID_NOT_FULL_MODEL_ACCEPTANCE',histories=summary['histories'],work=work,total_packets=packets,started_fraction=work['started']/packets,
      primary_audits_equal=all(c['primary_audits_equal'] and not c['integer_mismatches'] for c in checks),
      accepted_batches=len(checks),max_energy_residual=max(c['quality']['relative_energy_residual'] for c in checks),dose_change=dict(whole_grid=delta(x,y),GTV=delta(gx,gy)),gamma=gammas,peak_valley=pv,
      GTV=dict(TOPAS=summary['reference_metrics']['GTV'],legacy=oldsummary['gpu_metrics']['GTV'],joint=summary['gpu_metrics']['GTV']),
      timing=dict(legacy=timing(plog),joint=timing(clog),legacy_wall_seconds=oldsummary['wall_seconds'],joint_wall_seconds=summary['wall_seconds']),limitations=['25 Schneider models, spatial density scaling to82 selected legacy nodes; experimental10k training histories per composition; legacy bank retained.','Historical TOPAS DIJ and historical GPU control reused; no new TOPAS transport.','Single timing pair at different dates/cache states, not a speed certification.','0.5x0.5x2mm reference; 0.2mm gamma depends on interpolation and DTA/10 finite search.'])
    ref,_=read_mhd(ROOT/'inputs/physical_dose.mhd')
    domain=np.union1d(rois['BODY'],rois['GTV'])
    vx=x.transpose(0,2,1).reshape(-1)[domain];vy=y.transpose(0,2,1).reshape(-1)[domain];vr=ref.transpose(0,2,1).reshape(-1)[domain]
    cut=(vr>=ref.max()*.1)&(vx>0);local=100*(vy[cut].astype(float)-vx[cut])/vx[cut]
    result['dose_change']['BODY_reference_cut10pct']=dict(voxels=int(cut.sum()),mean_signed_percent=float(local.mean()),absolute_local_percentiles={str(p):float(np.percentile(abs(local),p)) for p in [50,90,95,99,100]},within_1pct_percent=float(100*np.mean(abs(local)<=1)))
    save_json(ROOT/'results/joint_comparison.json',result)
    assert result['primary_audits_equal'],'Review changed primary/EM audits'
    lines=['# RT07575：Schneider全成分、82密度节点缩放模型首轮接入','',f"完成 {summary['histories']:,} primary，4 Gy full plan 的0.1%，487 spots，119批单进程，FP32 dose、group32。未归一化拟合剂量，未修改TOPAS参考。",'',
      f"新模型进入 {work['started']:,}/{packets:,} 个电子包（{100*result['started_fraction']:.4f}%）；推进 {work['advances']:,} 步，切回旧路径 {work['fallback']:,} 个；无效包 {work['invalid']}，显式光子续接缺口 {work['photon_gap']}。全包仍保留，此结果不能代表全面替换后的精度/内存。",'',
      f"119批质量检查通过，零队列溢出，主输运/EM离散审计一致，最大相对能量残差 {result['max_energy_residual']:.3e}。状态仍是 research/non_production，未完成新模型正式患者验收。",'',
      '| Local gamma vs TOPAS | 旧GPU | 新混合GPU |','|---|---:|---:|']
    for pct,dta in [(1,.2),(2,.2),(3,.2),(3,0)]:
        g=gammas[f'local_{pct}pct_{dta}mm'];lines.append(f"| {pct}% / {dta}mm | {g['legacy_percent']:.4f}% | {g['joint_percent']:.4f}% |")
    lines+=['','BODY∪GTV，TOPAS最大剂量10%阈值；185559体素；既有TOPAS加权DIJ。0.2mm gamma使用三线性插值与DTA/10球形离散搜索。','',
      '| GTV Gy | TOPAS | 旧GPU | 新混合GPU |','|---|---:|---:|---:|']
    for label,k in [('Mean','mean_Gy'),('D95','D95_Gy'),('D98','D98_Gy'),('D50','D50_Gy'),('D2','D2_Gy')]:
        v=result['GTV'];lines.append(f"| {label} | {v['TOPAS'][k]:.6f} | {v['legacy'][k]:.6f} | {v['joint'][k]:.6f} |")
    lines+=['','| 深度 | 峰区新旧变化 | 谷区新旧变化 |','|---|---:|---:|']
    for band in ['20-40','40-80','80-120','120-160']:lines.append(f"| {band}mm | {pv[band+'mm_peak']['joint_vs_legacy_pct']:+.5f}% | {pv[band+'mm_valley']['joint_vs_legacy_pct']:+.5f}% |")
    lines+=['',f"进程墙钟 {summary['wall_seconds']:.2f}s，吞吐 {summary['histories_per_second']:.0f} primary/s；电子队列合计 {result['timing']['joint']['electron_seconds']:.2f}s。历史旧GPU墙钟 {oldsummary['wall_seconds']:.2f}s；不同日期/缓存状态，不能据单次结果认证加速。",'',
      '[完整结果](results/joint_comparison.json) · [剂量图](results/dose_comparison.png) · [DVH与剖面](results/dvh_profile.png) · [新旧差异图](results/joint_vs_legacy.png) · [4Gy剂量](results/dose_4Gy.mhd)','']
    (ROOT/'REPORT_zh.md').write_text('\n'.join(lines))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    mask=np.zeros((35,417,505),bool);mask.reshape(-1)[gtv]=True;mask=mask.transpose(0,2,1);iz=int(np.argmax(abs(y.astype(float)-x).sum(axis=(1,2))))
    fig,axes=plt.subplots(1,3,figsize=(15,6),layout='constrained')
    for ax,v,title in zip(axes,[x[iz],y[iz],y[iz]-x[iz]],['Legacy GPU','Joint hybrid GPU','Joint - legacy (Gy)']):
        diff=title.startswith('Joint -');im=ax.imshow(v,origin='lower',cmap='RdBu_r' if diff else 'inferno',vmin=-.01 if diff else 0,vmax=.01 if diff else max(x.max(),y.max()));ax.contour(mask[iz],levels=[.5],colors=['cyan'],linewidths=.6);ax.set_title(title);fig.colorbar(im,ax=ax)
    fig.suptitle(f'Patient slice index {iz}: largest summed absolute joint/legacy difference')
    fig.savefig(ROOT/'results/joint_vs_legacy.png',dpi=150)
    print(json.dumps({k:result[k] for k in ['started_fraction','work','GTV','gamma','timing']},indent=2),flush=True)
if __name__=='__main__':main()

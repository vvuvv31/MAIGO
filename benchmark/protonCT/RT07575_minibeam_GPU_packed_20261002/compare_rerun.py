"""Compare the latest same-source/seed rerun with the completed historical 0.1% plan."""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

from common import ROOT, REPO, read_mhd, save_json
from metrics import roi_indices

sys.path.insert(0, str(REPO / 'scripts'))
from compare_minibeam import AUDITS, integer_fields

OLD = ROOT.parent / 'RT07575_minibeam_GPU_0p1pct'


def read(root, name):
    return json.loads((root / name).read_text())


def log_records(root):
    records = {}
    logs = []
    for path in sorted(root.glob('group[0-9][0-9][0-9].log')):
        text = path.read_text()
        logs.append(text)
        for chunk in text.split('[plan-shard] ')[1:]:
            name = re.search(r'config="[^"]*/(batch\d+)\.yaml"', chunk).group(1)
            assert '[plan-shard-done]' in chunk, name
            records[name] = chunk
    return records, '\n'.join(logs)


def timing(text):
    stage = lambda name: [float(s) for s in re.findall(r'\[runtime-stage\] '+name+r' seconds=([\d.eE+-]+)', text)]
    setups = stage('transport_setup_including_upload')
    return dict(transport_seconds=sum(stage('transport_loop_including_scoring_and_queue_transfers')),
                setup_seconds=sum(setups), first_setup_seconds=setups[0],
                electron_seconds=sum(float(s) for s in re.findall(r'\[electron-queue-time\] seconds=([\d.eE+-]+)', text)),
                batch_wall_seconds=[float(s) for s in re.findall(r'\[batch-wall\] index=\d+ seconds=([\d.eE+-]+)', text)],
                bank_cache_hits=text.count('[electron-bank-cache] hit'),
                bank_cache_misses=text.count('[electron-bank-cache] miss'),
                prepared_tables_loaded=text.count('[material-response-prepared]'))


def dose_delta(old, new):
    old, new = np.asarray(old, dtype=float), np.asarray(new, dtype=float)
    delta = np.abs(new-old)
    nonzero = old > 0
    return dict(relative_L1=float(delta.sum()/old.sum()),
                max_absolute_difference_Gy=float(delta.max()),
                max_difference_over_peak=float(delta.max()/old.max()),
                max_voxel_relative_difference=float(np.max(delta[nonzero]/old[nonzero])),
                voxels_above_1pct_change=int(np.sum(delta[nonzero] > .01*old[nonzero])),
                new_nonzero_voxels=int(np.sum((old==0)&(new>0))),
                lost_nonzero_voxels=int(np.sum((old>0)&(new==0))))


def main():
    manifest=read(ROOT,'manifest.json')
    old_summary=read(OLD,'results/summary.json'); summary=read(ROOT,'results/summary.json')
    assert old_summary['histories']==summary['histories']==manifest['histories']==236086432
    assert summary['state']=='complete'
    old_records,old_log=log_records(OLD);new_records,new_log=log_records(ROOT)
    assert len(old_records)==len(new_records)==119
    checks=[]
    for job in manifest['jobs']:
        name=job['name'];a,b=old_records[name],new_records[name]
        audits=lambda text:[line for line in text.splitlines() if line.startswith(AUDITS)]
        work=lambda text:[re.findall(r'\b(tasks|peak_tasks|advances|source_restarts|boundary_restarts)=(\d+)',line) for line in text.splitlines() if line.startswith('[electron-queue-time]')]
        ia=integer_fields(read(OLD,f'out/{name}/energy_ledger.json'))
        ib=integer_fields(read(ROOT,f'out/{name}/energy_ledger.json'))
        q=read(ROOT,f'out/{name}/quality_report.json')
        x=np.fromfile(OLD/f'out/{name}/dose.raw',dtype='<f4')
        y=np.fromfile(ROOT/f'out/{name}/dose.raw',dtype='<f4')
        check=dict(name=name,audits_equal=audits(a)==audits(b),electron_work_equal=work(a)==work(b),
                   integer_mismatches=[(k,ia.get(k),ib.get(k)) for k in sorted(ia.keys()|ib.keys()) if ia.get(k)!=ib.get(k)],
                   quality_accepted=q['accepted'],queue_overflow_count=q['queue_overflow_count'],
                   relative_energy_residual=q['relative_energy_residual'],dose=dose_delta(x,y))
        assert q['accepted'] and not q['failures'] and not q['queue_overflow_count'],name
        checks.append(check)
    save_json(ROOT/'results/batch_parity.json',checks)
    a,_=read_mhd(OLD/'results/dose_4Gy.mhd');b,_=read_mhd(ROOT/'results/dose_4Gy.mhd')
    rois=roi_indices();af=a.transpose(0,2,1).reshape(-1);bf=b.transpose(0,2,1).reshape(-1)
    delta=dict(whole_grid=dose_delta(a,b),GTV=dose_delta(af[rois['GTV']],bf[rois['GTV']]))
    old_gamma=read(OLD,'results/gamma_cutoff10pct/gamma.json');gamma=read(ROOT,'results/gamma_cutoff10pct/gamma.json')
    assert gamma['state']=='complete' and gamma['reference_dose_sha256']==old_gamma['reference_dose_sha256']
    assert gamma['method']==old_gamma['method']
    gammas={k:dict(previous=v['percent'],current=gamma['gamma'][k]['percent'],
                  change_percentage_points=gamma['gamma'][k]['percent']-v['percent']) for k,v in old_gamma['gamma'].items()}
    opv=read(OLD,'results/peak_valley.json');npv=read(ROOT,'results/peak_valley.json')
    pv={k:dict(previous_vs_TOPAS_percent=v['relative_difference_percent'],
               current_vs_TOPAS_percent=npv['bands'][k]['relative_difference_percent'],
               new_vs_old_percent=100*(npv['bands'][k]['gpu_mean_Gy']/v['gpu_mean_Gy']-1)) for k,v in opv['bands'].items()}
    ot,nt=timing(old_log),timing(new_log)
    telemetry=list(csv.DictReader((ROOT/'gpu_telemetry.csv').open()))
    result=dict(histories=manifest['histories'],same_seeds_sources_and_physics_configs=True,
                dose=delta,batches_exact_discrete_audits=all(c['audits_equal'] and c['electron_work_equal'] and not c['integer_mismatches'] for c in checks),
                quality=dict(accepted_batches=len(checks),queue_overflow_count=0,
                             max_relative_energy_residual=max(c['relative_energy_residual'] for c in checks)),
                gamma=gammas,peak_valley=pv,
                timing=dict(previous=ot,current=nt,previous_wall_seconds=old_summary['wall_seconds'],
                            current_wall_seconds=summary['wall_seconds'],
                            wall_reduction_percent=100*(1-summary['wall_seconds']/old_summary['wall_seconds']),
                            throughput_gain_percent=100*(old_summary['wall_seconds']/summary['wall_seconds']-1),
                            current_primary_per_second=summary['histories_per_second']),
                max_sampled_gpu_memory_MiB=max(float(r['memory_MiB']) for r in telemetry),
                artifact_preparation=read(ROOT,'prepare.json'),artifact_validation=read(ROOT,'artifact_check.json'),
                limitations=['Historical baseline was 8 processes of up to 16 batches; new run is one process of 119 batches. This is not an isolated electron-artifact A/B timing.',
                             'No OS page-cache flush; warm filesystem caching may apply.',
                             'Same finite TOPAS weighted DIJ reference; no new TOPAS simulation.',
                             'Research non_production quality status and existing physics approximations retained.',
                             'Reference grid is 0.5 x 0.5 x 2 mm; 0.2 mm gamma uses interpolation and a finite DTA/10 search.'])
    save_json(ROOT/'results/rerun_comparison.json',result)
    assert result['batches_exact_discrete_audits'],'Review discrete regression before acceptance'
    lines=['# RT07575 proton minibeam full plan：最新代码复跑','',
           f"使用 RTX 2080 Ti、FP32 dose、group32，完成 {manifest['histories']:,} 个入射质子（4 Gy full plan 的 0.1%）、487 个有效 spot、119 批。与上一轮保持逐批粒子分配、源文件、随机种子和物理配置；唯一 YAML 增量是材料电子紧凑包路径。所有批次在同一进程运行。",'',
           '| 耗时 / 吞吐 | 上次 | 本次 |','|---|---:|---:|',
           f"| 蒙卡进程墙钟（含初始化、逐批输出、退出） | {old_summary['wall_seconds']:.2f} s | {summary['wall_seconds']:.2f} s |",
           f"| 全程吞吐 | {old_summary['histories_per_second']:,.0f} primary/s | {summary['histories_per_second']:,.0f} primary/s |",
           f"| 输运循环合计 | {ot['transport_seconds']:.2f} s | {nt['transport_seconds']:.2f} s |",
           f"| 电子队列内核合计 | {ot['electron_seconds']:.2f} s | {nt['electron_seconds']:.2f} s |",
           f"| 全批次 setup 合计 | {ot['setup_seconds']:.2f} s | {nt['setup_seconds']:.2f} s |",
           f"| 首批 setup | {ot['first_setup_seconds']:.2f} s | {nt['first_setup_seconds']:.2f} s |",'',
           f"时间减少 {result['timing']['wall_reduction_percent']:.2f}%，吞吐增加 {result['timing']['throughput_gain_percent']:.2f}%。上次为 8 个进程、最多 16 批/进程；本次单进程加载 82 张紧凑表一次、{nt['bank_cache_hits']} 次缓存命中。因此总收益包含少启动 7 次、共享输入缓存和紧凑包/电子执行改善，不能全部归因于电子包。未清理文件系统页缓存，这不是严格冷盘 I/O 基准。",'',
           f"包准备耗时 {result['artifact_preparation']['wall_seconds']:.3f} s，复用已验证的 63 表，仅生成另外 19 表；额外表与原始记录逐字段验证通过。此时间未计入蒙卡墙钟，也不是从零生成整个包的时间。采样显存峰值 {result['max_sampled_gpu_memory_MiB']:.0f} MiB。",'',
           f"119/119 批质量通过，queue overflow=0；所有输运/EM 离散审计和电子任务工作量与旧版完全一致。最大相对能量残差 {result['quality']['max_relative_energy_residual']:.3e}。累加剂量新旧相对 L1={delta['whole_grid']['relative_L1']:.3e}，全网格非零体素最大相对变化={100*delta['whole_grid']['max_voxel_relative_difference']:.6f}%；变化超过 1% 的体素数为 {delta['whole_grid']['voxels_above_1pct_change']}。",'',
           '| GTV 指标 | TOPAS Gy | 上次 GPU Gy | 本次 GPU Gy |','|---|---:|---:|---:|']
    for label,key in [('平均','mean_Gy'),('D95','D95_Gy'),('D98','D98_Gy'),('D50','D50_Gy'),('D2','D2_Gy'),('Dmax','max_Gy')]:
        lines.append(f"| {label} | {summary['reference_metrics']['GTV'][key]:.6f} | {old_summary['gpu_metrics']['GTV'][key]:.6f} | {summary['gpu_metrics']['GTV'][key]:.6f} |")
    lines+=['','| Local gamma vs TOPAS | 上次 | 本次 |','|---|---:|---:|']
    for pct,dta in [(1,.2),(2,.2),(3,.2),(3,0),(2,2),(3,3)]:
        item=gammas[f'local_{pct}pct_{dta}mm'];lines.append(f"| {pct}% / {dta} mm | {item['previous']:.4f}% | {item['current']:.4f}% |")
    lines+=['','与上次相同的 BODY∪GTV、参考最大剂量 10% 阈值和 185,559 个体素。只按原计划入射粒子数换算 4 Gy，无剂量拟合或配准；TOPAS 是既有加权 DIJ，没有重新运行。参考网格 0.5×0.5×2 mm，0.2 mm gamma 依赖三线性插值和 DTA/10 离散搜索。','',
            '| CT 入口深度 | 本次峰区相对 TOPAS | 本次谷区相对 TOPAS | 峰区新旧变化 | 谷区新旧变化 |','|---|---:|---:|---:|---:|']
    for band in ['20-40','40-80','80-120','120-160']:
        p,v=pv[band+'mm_peak'],pv[band+'mm_valley'];lines.append(f"| {band} mm | {p['current_vs_TOPAS_percent']:+.3f}% | {v['current_vs_TOPAS_percent']:+.3f}% | {p['new_vs_old_percent']:+.6f}% | {v['new_vs_old_percent']:+.6f}% |")
    lines+=['','峰谷采用上一轮固定几何掩膜，低剂量谷区单独检查。原有 research/non_production 状态不变：有限响应库、主粒子电子空间响应、带电核次级电子损失局部处理、CT 中子剂量缺失等限制仍存在。本次验证的是当前实现保持上一轮结果。','',
            '[完整机器可读对比](results/rerun_comparison.json) · [逐批审计](results/batch_parity.json) · [Gamma](results/gamma_cutoff10pct/gamma.json) · [剂量图](results/dose_comparison.png) · [DVH/剖面](results/dvh_profile.png) · [4 Gy 剂量](results/dose_4Gy.mhd)','']
    (ROOT/'REPORT_zh.md').write_text('\n'.join(lines))
    print(json.dumps(dict(timing=result['timing'],dose=delta,quality=result['quality'],gamma=gammas),indent=2))


if __name__=='__main__':
    main()

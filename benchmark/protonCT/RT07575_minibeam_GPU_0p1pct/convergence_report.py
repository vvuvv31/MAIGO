#!/usr/bin/env python3
"""Compare completed independent 0.01% and 0.1% runs without dose fitting."""
import json,re
from common import ROOT,save_json

def main():
    read=lambda root,n:json.loads((root/n).read_text())
    manifest=read(ROOT,'manifest.json');previous=ROOT.parent/'RT07575_minibeam_GPU'
    a=read(previous,'results/summary.json');b=read(ROOT,'results/summary.json')
    ga=read(previous,'results/gamma_cutoff10pct/gamma.json');gb=read(ROOT,'results/gamma_cutoff10pct/gamma.json')
    ua=read(previous,'results/uncertainty.json');ub=read(ROOT,'results/uncertainty.json');pv=read(ROOT,'results/peak_valley.json')
    assert b['state']=='complete' and b['histories']==manifest['histories']==236086432 and gb['state']=='complete'
    quality=[read(ROOT,'out/'+j['name']+'/quality_report.json') for j in manifest['jobs']]
    assert all(q['accepted'] and q['queue_overflow_count']==0 and not q['failures'] for q in quality)
    loops=sum(float(x) for log in ROOT.glob('group[0-9][0-9][0-9].log') for x in re.findall(r'transport_loop_including_scoring_and_queue_transfers seconds=([\d.eE+-]+)',log.read_text()))
    means={}
    for key in ['mean_Gy','D95_Gy','D98_Gy','D50_Gy','D2_Gy','max_Gy']:
        r=b['reference_metrics']['GTV'][key];v=b['gpu_metrics']['GTV'][key]
        means[key]=dict(reference=r,previous=a['gpu_metrics']['GTV'][key],current=v,relative_change_percent=100*(v/r-1))
    gamma={k:dict(previous=ga['gamma'][k]['percent'],current=item['percent'],change_percentage_points=item['percent']-ga['gamma'][k]['percent'],GTV_current=item['gtv_percent']) for k,item in gb['gamma'].items()}
    result=dict(histories=b['histories'],previous_histories=a['histories'],independent_runs=True,GTV=means,gamma=gamma,noise=dict(previous=ua,current=ub),quality=dict(accepted_shards=len(quality),overflow_count=0,max_relative_energy_residual=max(q['relative_energy_residual'] for q in quality)),wall_seconds=b['wall_seconds'],transport_loop_seconds=loops,histories_per_second=b['histories_per_second'],peak_valley=pv)
    save_json(ROOT/'results/convergence.json',result)
    lines=['# RT07575 minibeam full plan：0.1% / group32', '',f'已完成{b["histories"]:,}个准直器前入射质子（4Gy计划总量的0.1%，汇总取整），487个有效spot、119批。使用RTX2080Ti、FP32 dose ON / FP64 dose OFF、group32，以及与0.01%比较相同的冻结GPU二进制、源束、CT、材料电子响应库及物理参数；两次模拟使用独立随机种子，旧剂量没有复用。', '',f'正式分批进程墙钟{b["wall_seconds"]:.2f}s（{b["wall_seconds"]/60:.2f}min），平均{b["histories_per_second"]:,.0f}primary/s；输运循环合计{loops:.2f}s。16批/进程以复用已加载响应库，每批不超过200万；这改变进程间重复初始化次数，不改变采样或输运。所有批次审计通过、无队列溢出，最大相对能量残差{result["quality"]["max_relative_energy_residual"]:.3e}。状态仍为研究non_production，原有近似声明保持。', '', '| GTV指标 | TOPAS Gy | 0.01% GPU Gy | 0.1% GPU Gy | 0.1%相对TOPAS差异 |','|---|---:|---:|---:|---:|']
    for label,key in [('平均','mean_Gy'),('D95','D95_Gy'),('D98','D98_Gy'),('D50','D50_Gy'),('D2','D2_Gy'),('Dmax','max_Gy')]:
        item=means[key];lines.append(f'| {label} | {item["reference"]:.6f} | {item["previous"]:.6f} | {item["current"]:.6f} | {item["relative_change_percent"]:+.3f}% |')
    lines+=['','| Gamma | 0.01% Local | 0.1% Local | 0.1% Global | 0.1% GTV Local |','|---|---:|---:|---:|---:|']
    for pct,dta in [(1,.2),(2,.2),(3,.2),(3,0),(2,2),(3,3)]:
        local=gamma[f'local_{pct}pct_{dta}mm'];global_=gamma[f'global_{pct}pct_{dta}mm'];lines.append(f'| {pct}% / {dta}mm | {local["previous"]:.4f}% | {local["current"]:.4f}% | {global_["current"]:.4f}% | {local["GTV_current"]:.4f}% |')
    lines+=['',f'相同gamma定义：完整3D，BODY∪GTV，参考剂量≥全参考最大值10%，{gb["method"]["evaluated_voxels"]:,}体素；Global按参考最大值{gb["method"]["reference_maximum_Gy"]:.6f}Gy归一化，Local按每个参考体素剂量归一化。三线性插值、DTA/10球形搜索网格，离散通过率不等于连续精确最小gamma。0.5×0.5×2mm的参考网格限制了亚毫米结论；0.2mm结果依赖插值。只按原始入射质子数换算4Gy，未拟合剂量或配准。', '',f'GTV单体素相对标准误中位数由{ua["GTV_voxel_relative_standard_error_median_percent"]:.3f}%降到{ub["GTV_voxel_relative_standard_error_median_percent"]:.3f}%，P90={ub["GTV_voxel_relative_standard_error_p90_percent"]:.3f}%；平均剂量标准误为{ub["GTV_mean_standard_error_Gy"]:.6f}Gy。GTV逐体素相对RMS差异={ub["GTV_relative_RMS_difference_percent"]:.3f}%，GPU蒙卡RMS标准误={ub["GTV_relative_RMS_MC_standard_error_percent"]:.3f}%。这些误差估计只包括独立primary统计，不包括有限TOPAS DIJ参考和有限电子响应库的误差。没有去噪gamma或无限粒子数上限声明。', '', '| CT入口深度 | 峰区平均差异 | 谷区平均差异 |','|---|---:|---:|']
    for band in ['20-40','40-80','80-120','120-160']:
        peak=pv['bands'][band+'mm_peak'];valley=pv['bands'][band+'mm_valley'];lines.append(f'| {band}mm | {peak["relative_difference_percent"]:+.3f}% ± {peak["difference_standard_error_percentage_points"]:.3f}pp | {valley["relative_difference_percent"]:+.3f}% ± {valley["difference_standard_error_percentage_points"]:.3f}pp |')
    lines+=['','峰谷使用与上一轮完全相同的固定几何掩膜，±为GPU批间1标准误；低剂量谷区独立检查，不依赖gamma阈值。横向体素宽与掩膜宽均为0.5mm，深部峰谷逐渐混合。原TOPAS参考为每spot各100万入射质子的DIJ加权结果，优化器原始nonconverged状态和权重均未更改。GPU CT中子输运关闭，主粒子材料电子响应仍是有限库研究近似，带电次级电子损失仍局部处理；增加primary统计量不消除这些差异。', '', '[剂量图](results/dose_comparison.png) · [DVH/剖面](results/dvh_profile.png) · [4Gy剂量](results/dose_4Gy.mhd) · [收敛对比JSON](results/convergence.json) · [Gamma完整数据](results/gamma_cutoff10pct/gamma.json)', '']
    (ROOT/'REPORT_zh.md').write_text('\n'.join(lines))
    print('Comparison report:',ROOT/'REPORT_zh.md',flush=True)
if __name__=='__main__':main()

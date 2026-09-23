from pathlib import Path
import json,hashlib,subprocess
r=Path('/mnt/sdb/wuwei/MAIGO_review_fix_20260923');o=r/'evidence/percent1_20260923'
p=json.loads((o/'entrance_comparison.json').read_text());d=json.loads((o/'delta_comparison.json').read_text());q=json.loads((o/'delta_fixed_quality.json').read_text())
a=p['regions']['all'];v=p['regions']['valley'];reg=q['regression_checks'];sm=q['runs']['percent1_delta_fixed_smoke']['quality']
lines=['| 深度 mm | 电子迁移造成的谷区变化 pp | 配对1SE pp | 峰区变化 pp |','|---:|---:|---:|---:|']
for row in d['rows']:
 if row['window_width_mm']!=.25:continue
 lines.append(f"| {row['depth_mm']} | {100*row['valley']['paired_change_reference_fraction']:+.3f} | {100*row['valley']['paired_change_SE']:.3f} | {100*row['peak']['paired_change_reference_fraction']:+.3f} |")
s=f'''# 过程诊断与电子迁移评分修复

## 水入口相空间

新TOPAS入口运行200万历史，GPU为同一Cu0.05 mm上下文100万历史（独立种子），全C12总透过率分别为 {100*a['TOPAS']['fluence_per_source']:.5f}% 和 {100*a['GPU']['fluence_per_source']:.5f}%。固定谷区角度RMS分别 {v['TOPAS']['theta_x_mrad_RMS']:.5f} 与 {v['GPU']['theta_x_mrad_RMS']:.5f} mrad；谷区平均能量差 GPU−TOPAS = {v['GPU_vs_TOPAS']['mean_energy_difference_MeVu']:+.3f} ± {v['GPU_vs_TOPAS']['mean_energy_difference_SE']:.3f} MeV/u（1SE）。这些结果没有显示需要大幅调整散射幅度，但样本不足以排除百分之一级的谷区通量偏差。

TOPAS在Box/YMinusSurface记录后终止C12，入口面为世界Y=60 mm，没有添加新材料薄膜。此运行的剂量不是完整水体剂量，未纳入主比较。没有将TOPAS多线程EventID与GPU source_history连接。

## 本轮实际代码修复

`src/transport_sycl.cpp` 修复了实验性 `water_response_v1` 分支的积分问题：

1. 水体网格只用 `phantom_length_mm/depth_bin_width_mm` 定义时，原始 `voxel_bins_z` 和 `voxel_size_z_mm` 均为0。分支直接使用这两个原始值，错误地把电子迁移端点判断为网格外。现在与网格分配/文件头一样，使用 `number_of_bins()` 和 `scorer_spacing_z_mm()` 的解析结果。
2. 物理逃逸能量不再同时记成网格外沉积，避免网格能量被重复扣减；实际逃逸加入每历史逃逸能量。
3. 迁移到目标体素的能量同时写入已启用的charged-origin/minibeam component评分器，保持分量记账一致。

旧分支20k运行：物理能量残差0.78757%，3D/网格内能量比1.085197，运行拒绝。修复后相同输入：残差 {100*sm['relative_energy_residual']:.6f}%，3D/网格内比 {sm['voxel_to_ingrid_ratio']:.10f}，通过全部运行检查。

构建及 urban_localize 通过；默认local模式修复前后最大体素差/峰值 {reg['baseline_before_after_max_difference_over_peak']:.3g}，隐式/显式Z网格最大体素差/峰值 {reg['legacy_vs_explicit_z_max_difference_over_peak']:.3g}。三个百万历史迁移对照均通过能量、队列及Urban fatal/cap/guard/subulp检查。GPU剂量仍为FP32。旧二进制和源文件已保留，主统计比较来自旧二进制，以上回归支持其对默认local路径继续适用。

## 电子迁移是否解释100 mm谷区偏高

以下为3组百万历史、相同source/seed的GPU配对差，水/Cu输运参数不变。变化用主比较的TOPAS绝对剂量作为分母；没有归一化拟合。

{chr(10).join(lines)}

100 mm谷区增加约0.066个百分点，与消除约1.84%的正偏差方向相反。因此电子当地沉积近似不是这一剩余偏差的主要解释。它对20 mm峰区影响较明显，但不能借此宣称整模型已达1%。主比较保留local模式，修复后的迁移模式保持显式研究开关。

## 响应表与限制

原Water_75eV、0.05 mm生产cut响应表包含37,501,208段与43,244,518路径节点，超过现有诊断加载器容量。最初简单CDF压缩丢失少量大半径事件，使空间二阶矩最大相对误差超过400%，在GPU运行前拒绝。

用于对照的版本精确保留所有任一端点距源超过0.5 mm的原始段及其权重，只对近距离部分做每能量通道32768个分层样本，保留完整祖先链、原始能量分配/逃逸比例。共3,675,695段、9,646,814节点，各空间二阶矩最大相对误差0.09745%。这是敏感性诊断表，矩一致并不证明所有局部尾剂量误差小于1%，不能代替高统计独立空间分布验证，也没有修改原表。

水Urban本身0.05→0.025 mm步长响应与TOPAS的旧差异仍未宣告解决；入口角分布一致不等于水内步长响应已通过。目前正在检查Cu0.05→0.025 mm的独立TOPAS和配对GPU收敛，结果存于 `cu_step025/`。
'''
(o/'PROCESS_DIAGNOSIS.md').write_text(s)
for file,marker,body in [
 ('urban.md','## 2026-09-23 Independent batches and delta scoring integration fix', 'Main Cu0.05 comparison now pools GPU15M and TOPAS16M. Fixed 100 mm valley is +1.838% with approximate 95% CI [0.671%,3.004%]; 4/6 point estimates, no full 95% interval, are within +/-1%. Experimental delta response now resolves legacy z geometry, credits physical escape once, and updates enabled component scores. Seven runtime checks and urban_localize pass; default local dose regression <2e-7 of peak. Delta relocation shifts 100 mm valley +0.066 pp, so it is not a correction for that positive residual. See evidence/percent1_20260923/RESULTS.md and PROCESS_DIAGNOSIS.md. No default physics promotion.'),
 ('failed.md','### 2026-09-23 Simple inverse-CDF water-response compression loses rare tails','32768 equally weighted samples per energy channel without retaining far deposits changes spatial second moments by up to 414%; rejected before GPU use. Retry used exact retention of every original segment beyond 0.5 mm plus stratified near samples and exact ancestor closure; maximum moment error 0.09745%, diagnostic only. See evidence/percent1_20260923/delta_diagnostic_table and delta_tail_diagnostic_table.'),
 ('failed.md','### 2026-09-23 Treating all residual valley error as noise or electron-local deposition','GPU15M / TOPAS16M independent batches: 100 mm valley +1.838%, approximate 95% CI [0.671%,3.004%]. Pure zero-bias noise is not supported at that point, although a <1% bound remains unresolved. Three paired 1M delta-relocation runs shift that valley +0.066 pp, not downward. Do not normalize/scale MCS or promote the delta diagnostic as the valley fix. Investigate matched Cu step convergence and transport/loss distributions; no coefficient fitting.')]:
 path=r/file;old=path.read_text()
 if marker not in old:path.write_text(old+'\n\n'+marker+'\n\n'+body+'\n')
subprocess.run(['git','diff','--check'],cwd=r,check=True)
print('DIAGNOSIS_WRITTEN',flush=True)

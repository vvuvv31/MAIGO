# 全计划蒙卡统计误差与剂量差异分布

本报告的原始实验目录为工作区 `tmp/full_plan_summary_20261017/uncertainty/`。下述原始文件清单相对于该实验目录；Git 中保存汇总 JSON 和 PNG/PDF 图。

本次工作只整理统计与绘图，不改变现有 TOPAS/GPU 参考剂量、归一化、gamma 或生产源码。

## 评价区域与统计口径

- Broadbeam：BODY 内，原始 TOPAS 剂量大于全局 Dmax 的 10%。
- Minibeam：BODY 内，原始 TOPAS 剂量大于 0.4 Gy。
- 直接读取原 gamma 分析保存的评价 mask，不用新低统计量剂量重新筛选体素。
- 每个体素的相对标准误差为 `100 × σ / D_TOPAS`。汇总值是这些数值的 RMS，属于 1σ 统计量；不是 ROI 平均剂量的误差，也不意味着区域内每个体素都达到该误差。
- GPU 使用原全计划独立批次估计的方差；有原生方差的 TOPAS 使用其全计划逐 history 统计。两者独立，合并方差为二者之和。
- 统计误差不包括物理模型、几何、步长或源模型的系统偏差。

## 三个碳宽束的补充估算

历史完整 TOPAS 结果没有保存原生逐体素方差。本次每个病例用本地 TOPAS 重新运行 300,000 个粒子，仅用于估计方差，不替换已有完整剂量。

保留原始 CT/DICOM、材料映射、患者平移旋转、束流模型、spot 位置/能量/相对权重和八模块物理列表（含 CarbonIonElasticPhysics）；每个病例 64 线程，三例同时运行。原始历史日志和新日志均为 Geant4 11.3.2；保留历史 EM 表上限 600 MeV。

源粒子按整个原计划的合并计数，用最大余数法分配到 300,000 粒子。低权重 spot 的整数舍入会留下很小的源分布差异，各病例总变差记录在 manifest/statistics.json 中，约 0.036%–0.053%。不使用 crop DIJ 或拟合剂量修正来推算统计误差。

TOPAS 原生 `Variance` 是每 history 剂量贡献的样本方差 `s²`。原历史参考是 N_full 个粒子的累计剂量，因此其方差估算为：

`Var(D_full) ≈ N_full × s²_pilot`

等价地，把 pilot 剂量缩放到原全计划归一化后，其标准误差再乘 `sqrt(N_pilot/N_full)`。估算分母使用原始完整 TOPAS 剂量，避免低统计 pilot 自身的剂量波动污染分母。混合 spot 的 pooled 方差没有扣除 spot 均值之间的差异，对固定各 spot 粒子数的分层源属于保守估计。

额外从原始 4–5 份独立 TOPAS 复本估计复本间方差，作为交叉检查。每个病例均验证复本剂量求和、转换 FP32 后精确等于归档参考。复本数少，其逐体素方差分布较粗，适合交叉检查区域 RMS。

## 本次外推与独立复本交叉检查

|病例|30万粒子外推的 TOPAS SE RMS|原独立复本估算|
|---|---:|---:|
|RT07575|0.594%|0.596%|
|RT06423|0.610%|0.608%|
|20022516|0.816%|0.816%|

以上均对应原始完整计划粒子数，复本与原生方差外推的区域 RMS 相对差异均小于 0.5%。它不保证低统计 pilot 的逐体素方差图已完全收敛。

## 文件

- `RESULTS.md`、`statistics.json`：所有已完成结果的统计误差、分位数、pilot 来源和复本交叉检查。
- `broadbeam_statistical_uncertainty.*`、`minibeam_statistical_uncertainty.*`：统计误差直方图，蓝色 GPU、橙色 TOPAS、绿色合并；虚线表示碳宽束的低统计 pilot 外推值。
- `all_cases_statistical_uncertainty.*`：全部病例与模态的汇总图。
- `*_dose_difference.*`：有符号相对剂量差 `(GPU−TOPAS)/TOPAS` 的直方图。它包含随机噪声和系统差异，不能作为纯蒙卡统计误差。
- 每图提供 PNG、PDF、SVG。纵轴是每个 bin 占全部评价体素的百分比；显示范围外尾部比例写入 JSON，不在截取后重新归一化。
- `histogram_bins.csv`：所有直方图 bin 的体素数与比例。
- `*_voxel_errors.npz`：评价区域内的逐体素相对标准误差与剂量差，便于重新绘图。
- `carbon_broadbeam_*/TOPAS_estimated_variance_Gy2.*`：单独存放的外推方差，不写入原始 TOPAS 归档目录。

20022516 Carbon minibeam 尚无完成且匹配的 GPU/TOPAS 全计划结果，图中留空，不从别的病例推测其误差。

## 重现

已完成 pilot 后运行：

```bash
python3 /mnt/sdb/wuwei/MAIGO_clean/tmp/full_plan_summary_20261017/plot_uncertainty.py
```

绘图脚本会读取原始报告和完成审计标记，核对已有统计误差，生成全部图与 CSV，并更新上一级 `RESULTS.md` / `RESULTS.json`。它不会重新运行 GPU/TOPAS 或修改 gamma。

Git 内的图表和全部结果见 [汇总表](README.md)。

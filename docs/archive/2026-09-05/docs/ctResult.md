# CT 患者剂量与 LET 验证结果

本文汇总截至 **2026-08-09** 有完整证据链的患者 CT 蒙特卡罗比较。常规宽束采用独立 TOPAS
full-plan scorer；RT07575 minibeam 采用 TOPAS 逐 spot sparse-Dij 与 matRad
优化权重的乘积。两类参考的统计含义不同，不能直接互称为 full-plan TOPAS。
几何与输入生成方法见 [ctplan.md](ctplan.md)。

## 1. 统一比较口径

当前 rotation-fixed 常规束 GPU 与 TOPAS 使用相同的正权重 spot 整数粒子分配，剂量采用绝对标度
`scale=1`，不拟合归一化。比较前将参考和评价剂量在 RTSTRUCT BODY 外同时置零；
评价集合为 `BODY ∩ raw TOPAS dose >= 10% BODY Dmax`。

当前常规束权威汇总只报告全部选择体素上的 3%/0 mm identical-voxel 通过率；
`G/L` 分别表示以参考峰值和参考局部值定义剂量容差的 global/local 通过率，
`E/R` 表示选择集内 evaluation/reference。机器可读结果见
[rotation-fixed summary](../out/ct/generic_rotation_fix_equal_history/final_summary.json)。

Minibeam 也采用 BODY 和参考剂量 10% 阈值，但非零 DTA 只在固定 seed-0 的至多
50,000 个选择体素上计算；0.3/0.5 mm DTA 的搜索步长为 0.1 mm，1/2/3 mm 为
0.5 mm，0 mm 使用全部选择体素。

## 2. 病例证据状态

| Case | 解剖与几何 | 当前可报告状态 | 处理 |
|---|---|---|---|
| RT06423 | head，TPS 90° | rotation-fixed 严格同粒子常规束 dose | 纳入 |
| RT07575 | head，TPS 90° | rotation-fixed 严格同粒子常规束 dose；另有 minibeam | 纳入并分节报告 |
| 20022516 | lung，TPS 0° | rotation-fixed 严格同粒子常规束 dose | 纳入 |
| RT06541 | head，TPS 270° | 已有 TOPAS full-plan 数据被确认无效 | 排除 |

RT06541 的旧 fitted-scale Dij shape 回归不能作为独立 TOPAS 验证。20022516 的
5-spot 子集和旧 10M plan-shape 回归也不能替代这里的完整 full-plan 比较。

## 3. 常规束 full-plan

### 3.1 Rotation-fixed equal-history dose

| Case | GPU / TOPAS histories | GPU seed | selected voxels | NRMSE | selected E/R | G/L 3%/0mm |
|---|---:|---:|---:|---:|---:|---:|
| RT07575 | 12,963,817 / 12,963,817 | 20260801 | 318,711 | 1.000210% | 0.997578 | 98.4610 / 81.4493% |
| RT06423 | 15,108,664 / 15,108,664 | 20260730 | 335,807 | 1.090661% | 0.997912 | 97.8491 / 80.7348% |
| 20022516 | 17,717,177 / 17,717,177 | 20260802 | 755,302 | 1.171132% | 0.992402 | 97.6631 / 68.9863% |

TOPAS 剂量直接读取原始 scorer；MHD 副本只用于转换完整性检查。三个 case 均采用
入口面无 shear/旋转修补，并启用相同的通用 residual-heat 和
upstream-air 设置。详细配置、history 四向核对和输入路径见
[rotation-fixed summary](../out/ct/generic_rotation_fix_equal_history/final_summary.md)。

### 3.2 Runtime 与当前 LET 边界

| Case | GPU transport / wall | throughput | memory estimate | overflow S/C |
|---|---:|---:|---:|---:|
| RT07575 | 271.44 / 272.89 s | 47.76k s⁻¹ | 9,399 MiB | 0/0 |
| RT06423 | 306.38 / 307.95 s | 49.31k s⁻¹ | 9,507 MiB | 0/0 |
| 20022516 | 349.33 / 351.56 s | 50.72k s⁻¹ | 10,366 MiB | 0/0 |

这些 run 写出了 primary-C12 和 all-hadron LET_d，但当前 rotation-fixed 汇总尚未对
它们执行统一 TOPAS-dose mask 的 LET 比较。旧 `out/fullplan_result` LET 表和旧
RT07575 1B fast/best profile 均早于本次通用 RotX/RotY inverse-order 修复，不能
作为 current-code LET 或 dose 结论；当前也没有可报告的 medium 结果。

## 4. RT07575 rotation-fixed minibeam

### 4.1 参考与运行口径

本节使用 commit `56b5343214f3970a9077247171060e589c628f14` 的新构建结果。计划含
1,943 spots，其中 1,617 个权重为正；四个 subfield 组成两个 opposed angles。
每个 GPU seed 共 129,496,548 histories，angle01/angle02 分别为
79,737,018 / 49,759,530，相当于旧 divisor-100 GPU 预算的 3×。两个角映射回患者
坐标后取 3× 平均并乘固定 baseline `×100`，即 `(angle01 + angle02) × 100/3`；
不拟合归一化。

参考剂量由 TOPAS 每 spot 100k-history sparse Dij 乘 matRad 优化权重得到，源为
`RBE_dose_result_c.mat/resultGUI.physicalDose`。它不是一次独立的 TOPAS whole-plan
scorer。1,943 个 spot 各输运 100,000 histories，原始逐 spot 预算合计 194.3M；
这个数不是按优化权重定义的 full-plan 独立粒子数。Dij builder 使用过
`2e-6 Gy/spot/voxel` 稀疏阈值并删除 87.281% nnz；
本地没有原始 per-spot scorer，无法重建无阈值参考。因此以下结果可验证旋转修复、
计划映射和当前 GPU/TOPAS-Dij 一致性，但残差同时包含 sparse-Dij 阈值和
参考端逐 spot 有限统计，不能解释为纯粹的 GPU–TOPAS full-plan 物理差异。本轮
没有权威 minibeam LET 结果。

### 4.2 GPU–TOPAS-Dij dose

| Comparison | NRMSE | selected E/R | BODY E/R | G/L 3%/3mm | G/L 2%/2mm | G/L 1%/1mm |
|---|---:|---:|---:|---:|---:|---:|
| GPU A / TOPAS-Dij | 1.628% | 0.984766 | 1.001640 | 99.504 / 98.534% | 97.720 / 92.988% | 85.040 / 57.948% |
| GPU B / TOPAS-Dij | 1.639% | 0.984648 | 1.001640 | 99.536 / 98.508% | 97.706 / 93.014% | 85.178 / 57.902% |
| A/B ensemble / TOPAS-Dij | 1.586% | 0.984707 | 1.001640 | 99.444 / 98.252% | 97.494 / 92.162% | 84.894 / 57.272% |

| Comparison | G/L 3%/0mm | G/L 3%/0.3mm | G/L 3%/0.5mm |
|---|---:|---:|---:|
| GPU A / TOPAS-Dij | 92.052 / 56.840% | 95.512 / 82.376% | 96.888 / 88.836% |
| GPU B / TOPAS-Dij | 91.966 / 56.650% | 95.366 / 82.894% | 96.716 / 88.946% |
| A/B ensemble / TOPAS-Dij | 92.502 / 58.470% | 95.622 / 83.414% | 96.838 / 89.212% |

这里 0 mm 使用全部 317,867 个选择体素，其余 gamma 使用确定性 50,000 点。
权威数值见 [comparison.md](../out/benchmark/ct/RT07575/conventional/minibeam_plan/rotation_fix_56b5343/three_x_129m_two_seed/comparison.md)
和 [comparison.json](../out/benchmark/ct/RT07575/conventional/minibeam_plan/rotation_fix_56b5343/three_x_129m_two_seed/comparison.json)。

### 4.3 GPU–GPU 重复性

以 GPU A 为 reference、GPU B 为 evaluation，选择集为
`BODY ∩ GPU A >= 10% GPU A BODY Dmax`，共 320,030 voxels；固定绝对标度：

| NRMSE | selected E/R | BODY E/R | G/L 3%/3mm | G/L 2%/2mm | G/L 1%/1mm | G/L 3%/0mm | G/L 3%/0.3mm | G/L 3%/0.5mm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.777% | 0.999809 | 0.999999 | 100.000 / 99.910% | 99.992 / 98.610% | 97.414 / 75.120% | 99.693 / 82.136% | 99.952 / 96.276% | 99.976 / 98.036% |

GPU–GPU 明显优于 GPU–TOPAS-Dij，说明 minibeam 当前差异不能只归因于 GPU seed
噪声；但受上述参考限制，也不能据此把全部残差归因于输运模型。

### 4.4 Runtime、显存与可复现性

| Seed | Angle | Histories | random seed | elapsed | throughput | memory estimate | overflow S/C/N |
|---|---|---:|---:|---:|---:|---:|---:|
| A | angle01 | 79,737,018 | 20260811 | 2,264.12 s | 35.22k s⁻¹ | 20,063 MiB | 0/0/0 |
| A | angle02 | 49,759,530 | 20260812 | 1,308.31 s | 38.03k s⁻¹ | 18,133 MiB | 0/0/0 |
| B | angle01 | 79,737,018 | 20260821 | 2,260.37 s | 35.28k s⁻¹ | 20,063 MiB | 0/0/0 |
| B | angle02 | 49,759,530 | 20260822 | 1,313.28 s | 37.89k s⁻¹ | 18,133 MiB | 0/0/0 |

每个 seed 两角串行约 59.5 min。Manifest 记录的是设备内存估算而非采样到的峰值；
预算为 20,151 MiB，secondary/neutral queue 容量为 58M/70M。构建二进制 SHA-256
为 `0e1f0fc08312700a2253f85add6eb1350739c558a2ad2ef8daf102fef8f3609e`；运行时
工作树为 dirty，但 manifest 保存了输入 hash、命令和 dirty 状态。完整证据见
[manifest.md](../out/benchmark/ct/RT07575/conventional/minibeam_plan/rotation_fix_56b5343/three_x_129m_two_seed/manifest.md)
和 [manifest.json](../out/benchmark/ct/RT07575/conventional/minibeam_plan/rotation_fix_56b5343/three_x_129m_two_seed/manifest.json)。

## 5. 结论边界

- 常规束当前有三例 rotation-fixed 严格同粒子 full-plan dose：RT06423、RT07575
  与 20022516；当前统一汇总只支持 3%/0 mm，不能把旧 3/3、2/2 或 LET 数值当作
  current-code 结果。当前没有 medium 的权威结果。
- RT07575 minibeam rotation fix 后两 GPU seed 高度一致；与 TOPAS-Dij 的差异必须
  连同 sparse threshold 和参考统计共同解释。获得真正的 GPU–TOPAS full-plan
  结论仍需独立、无阈值的 TOPAS whole-plan scorer。
- RT06541 在 TOPAS reference 重算和审计完成前不得恢复到有效病例表。

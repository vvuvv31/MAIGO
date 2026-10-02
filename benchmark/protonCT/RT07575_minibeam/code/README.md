# RT07575 proton minibeam ADMM

由 `RT07575_proton/code` 的物理剂量 ADMM 改编，读取本 minibeam 病例已经完成的
`../merged/dij.mat`，不依赖 broadbeam dij 或其优化结果。

## MATLAB 运行

```matlab
cd('/mnt/sdb/wuwei/MAIGO_review_fix_20260923/benchmark/protonCT/RT07575_minibeam/code');
build_mex;
resultFile = run_admm_minibeam;
```

需要支持 `readtable(...,'VariableNamingRule','preserve')` 和
`jsonencode(...,PrettyPrint=true)` 的 MATLAB（建议 R2021a 或更新版本），以及已配置的
C MEX 编译器。若尚未配置，先运行 `mex -setup C`。
只编译实际用到的 `CG033114`、`JtJ_id` 两个源文件，使用原来的 `-R2017b` API。
没有携带旧平台的 MEX 二进制，也不需要 matRad 工具箱。

默认 ADMM **10 次**，每次 CG 上限 20 次；保留原 GTV/BODY/OAR 目标、非负权重约束、
每次 **4 Gy**、共 **16 次**的物理剂量处方。默认最后按 GTV D95 归一化到 4 Gy。
这不是额外 ADMM 迭代；归一化前后指标均保存。10 次上限不表示已经收敛，
`solverInfo.converged` 和残差如实保存。优化目标是原来的 GTV 覆盖目标，未添加峰谷剂量目标。

输入矩阵为 MATLAB v7.3 稀疏矩阵，7,370,475 行 × 870 列，约 8.35 GiB；
不会转换成全尺寸稠密矩阵。运行需要超过矩阵大小的可用内存，建议至少 24–32 GiB。
CT/cst 从本目录 `RT07575_cropXY_bodyZgtv20mm.mat` 加载，spot 信息来自 `../spots.csv`。
代码检查完成状态、尺寸、spacing、每列 histories、spot 顺序、位置及能量。

## 最终结果

全部保存在本目录下 `output_minibeam/`，不写迭代 checkpoint：

| 文件 | 内容 |
|---|---|
| `minibeam_dose_result.mat` | 最终每次/全疗程 dose、权重、grid、指标、求解状态、迭代记录及 dij 元数据 |
| `physical_dose.mhd/.raw` | 每次物理剂量 Gy，FP32 二进制 |
| `physical_dose_total.mhd/.raw` | 16 次总物理剂量 Gy |
| `spots_optimized.csv`, `spots_optimized_total.csv` | 原 spot 表加优化后的每次/全疗程 `weight` |
| `spot_weights.csv`, `spot_weights_total.csv` | 无表头单列每次/全疗程权重，原始 spot 顺序 |
| `spot_weights_details.csv` | `plan_index`、`spot_id` 和两种权重 |
| `plan_export.json` | 单位、归一化、剂量指标及求解状态 |

权重是**铜准直器前入射质子数**，不是 MU，也不是透过准直器后的质子数。
每列 dij 已累计 **1,000,000** 个入射质子的 DoseToMedium（Gy）：

```matlab
s = load('../merged/dij.mat', 'Dij', 'histories_per_spot');
r = load(resultFile, 'weights', 'physicalDose', 'grid');
dose = reshape(s.Dij * (r.weights / s.histories_per_spot), r.grid.dimensions_YXZ);
```

`weights` 按 dij 列序，CSV 按原始 `spots.csv` 顺序，`planIndex`（0 起始）记录对应关系。
MAT dose 顺序是 `[Y,X,Z]=[505,417,35]`；MHD/raw 为 X-fastest，
`[X,Y,Z]=[417,505,35]`。不旋转或翻转物理坐标。

## 精简同步副本

独立可运行的副本位于：

```
/mnt/sdb/wuwei/MAIGO_minibeam/benchmark/protonCT/RT07575_minibeam/
  code/                  本代码、必要 utils、CT/cst，以及运行后最终输出
  merged/dij.mat         最终合并矩阵
  merged/dij.json        列映射与剂量元数据
  merged/COMPLETE.json   完成状态
  spots.csv              原始 spot 参数
```

没有复制 `part_*`、单 spot dose、日志、build、临时矩阵或重复的 `.npz`。
运行结束后代码自动把上述最终输出同步到该副本的 `code/output_minibeam/`；
不会重复复制大矩阵。也可以直接在副本 `code` 目录执行相同 MATLAB 命令，结果就地保存。
如果需要在另一台机器运行并关闭固定路径同步：

```matlab
resultFile = run_admm_minibeam(struct('sync_code_dir', ''));
```

同步失败时本地结果保留，无需重新优化，运行 `sync_final_results` 可重试。
如需修改迭代次数，显式传入 `struct('N_iter',10)`；默认已是 10 次。

此版本已检查输入数据布局和映射；按照要求，尚未执行 MATLAB 编译或优化，
优化权重和最终 dose 将在手动运行后生成。

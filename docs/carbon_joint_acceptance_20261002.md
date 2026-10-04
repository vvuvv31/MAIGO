# Carbon CT broadbeam joint response：RT07575 剂量验收

2026-10-02，当前 `carbon_ct_plan_fe_joint` 在 RT07575 对照中通过用户指定的 **local gamma 1%/1 mm >98%** 标准。结论限定于本配置、病例、束流谱和评估阈值；未将通用电子响应包标记为已验证，未更改物理模型、参数或质量门禁。

## 配置与参考

- 当前源码提交：`13b7a7d860c4c78bfd84b10ee88a7005b9738d28`。
- GPU 二进制 SHA256：`a29f19e23795c77d709b4c4593f16b516d14f6503faefcc4d4335bc2646a8cb0`。
- RTX 2080 Ti；`CARBON_DOSE_FP32=ON`、`CARBON_DOSE_FP64=OFF`。
- GPU：三个独立种子 `1001800701/702/703`，每个 6,481,909 primary，合计 **19,445,727**。保持原预设，只有种子与隔离输出目录不同；queued joint response、采样概率下限 0.05、FE 内部分段 1 mm、平均损失上限 5%。
- TOPAS：既有 RT07575 五个独立 replica，每个 25,927,634 primary，合计 **129,638,170**。从原始 DoseToMedium Sum 重新读取并逐文件验证历史 SHA256；原始双精度 TOPAS 输出仅用于读取和分析，GPU 剂量评分始终为 FP32。
- CT 与 beam model 和历史参考的 SHA256 相同。853 个非零 spot 的 ID、位置、能量一致；TOPAS 表额外含 64 个零权重行。GPU 预设是按 spot 取整的计划分片，归一化源权重 L1 差异 `3.19016e-5`；不是精确复现 TOPAS 整数 spot 配额的全量重跑。
- 只除以各自实际 primary 数，使用 Gy/primary；不拟合剂量比例，不做剂量配准。固定几何映射为 `flip(gpu.transpose(1,2,0), axis=2)`，来自原 CT 打包坐标约定。
- 3D 全体素 gamma：TOPAS 剂量达到其最大值 10% 的 **459,684 个体素**，无随机体素抽样。网格间距 0.5 × 0.5 × 2 mm；global 分母为 TOPAS 最大剂量，local 分母为查询点 TOPAS 剂量。

## 合并剂量结果

| 标准 | Global | Local |
|---|---:|---:|
| 3% / 3 mm | 100.0000% | 100.0000% |
| 2% / 2 mm | 100.0000% | 99.9956% |
| 1% / 1 mm | 99.9458% | **99.8294%** |
| 3% / 0 mm | 99.9078% | **92.7572%** |

采用三线性插值、DTA/10 的球内笛卡尔搜索格点；这是有限格点近似，不是连续最小化。0 mm 为精确同体素剂量差测试。local 1%/1 mm 将搜索步长从 0.1 mm 减至 0.05 mm 后，通过率为 **99.8375%**，仅增加 37 个通过体素，不影响验收。

独立 PyMedPhys 0.41.0 全评估区复核：3D local 1%/1 mm、`interp_fraction=10`、无体素抽样，通过率 **99.8138%**。不同搜索方向离散造成少量临界点分类差异，两种实现均通过。

| GPU 种子 | Local 1% / 1 mm |
|---|---:|
| 1001800701 | 99.9149% |
| 1001800702 | 99.8777% |
| 1001800703 | 99.8105% |

三个独立种子均通过。增加粒子数并不保证 gamma 单调升高；零距离差异与空间容差结果应分别解释。

## 绝对剂量与局部差异

- 评估区剂量总和偏差 **−0.4874%**；按 GPU 三批和 TOPAS 五批计算的批间 SEM 为 **0.0034 个百分点**，仅表示本次蒙卡批间统计误差，不涵盖模型或数据包误差。这是体素剂量总和，不能解释为质量加权的总沉积能量。
- 全 CT 网格剂量总和偏差 −0.7360%；评估区平均局部偏差 −0.4956%，平均绝对局部差异 1.2554%。
- 评估区绝对局部差异 P95=3.4387%、P99=5.6185%、最大值=16.9756%。因此，本次 gamma 通过不等于每个体素剂量都在 1% 或 3% 内。
- 评估区涉及 Schneider section 0–23；各材料分区 local 1%/1 mm 为 99.5168%–100%。section 23 仅 5 个评估体素，section 24 没有评估体素，不能视为覆盖验证。
- 与不同 section 的体素面相邻的体素：188,038 个，local 1%/1 mm=99.8559%，剂量总和偏差 −0.5119%。其余 271,646 个体素为 99.8112%、−0.4709%。此分区只识别材料 section 改变，不覆盖同一 section 内的密度界面。

图及原始差异数组：`out/carbon_joint_acceptance_20261002/comparison.png`、`plot_arrays.npz`。

## 运行与包审计

- 三批电子响应查询约 16.08 亿次/批，均 `domain_misses=0`、`invalid_marches=0`；空间响应确实启用。
- Unified EM 审计零错误，队列溢出为零；能量相对残差 `3.49e-6`–`3.55e-6`。剂量均有限、非负。
- 包的 CSV、metadata、ordered path SHA 均通过。25 个材料 × 100 个能量 bin 共 2,500 个通道齐全，C12 能量上限 500 MeV/u；通道能量分区最大相对残差 `1.87e-11`。
- 三批运行总 wall time 109.74 s，不含 gamma 分析；未新增 TOPAS 蒙卡。
- 原始 TOPAS 五批 scorer header 含并行世界未计分警告，未计分能量合计 0.19048 MeV；本报告保留原始参考及警告，没有修补参考剂量。

程序三批均以原有 `unvalidated_electron_joint_response` 返回 quality fail/exit 1，其他运行审计通过。本报告的 **scoped dose acceptance** 是本病例的外部验收结论，不声称程序 quality 已变为 accepted。

包仍保留的局限包括：跨界后沿用源材料的电子散射、有限提取盒内的光子逃逸能量保留在局部、压缩轨迹库需要独立样本/种子收敛验证。当前 broadbeam 路径只重分配主粒子电离电子能损；带电次级电子仍局部沉积，核反应中性粒子能量未输运。当前结果不能直接授权 carbon CT minibeam 或其他病例的剂量精度。

## 复现与留档

本地完整运行目录：`out/carbon_joint_acceptance_20261002/`。

- `run.py`、`harness.py`：三种子运行，拒绝覆盖已有运行目录。
- `analyze.py`、`gamma_lattice.py`：参考来源和源参数核查、全部 gamma、剂量偏差。
- `supplement.py`：独立 PyMedPhys、材料分区、包审计、图。
- `reference_audit.json`、`runtime_audits.json`、`analysis.json`、`independent_gamma.json`、`material_regions.json`、`package_audit.json`：详细证据。
- `carbon_broadbeam_r701/702/703/`：实际配置、日志、原始剂量和能量账。
- 机器可读摘要：`docs/carbon_joint_acceptance_20261002.json`。

分析命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 out/carbon_joint_acceptance_20261002/analyze.py
PYTHONPATH=/mnt/sda/wuwei/maigo_gamma_python_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 out/carbon_joint_acceptance_20261002/supplement.py
```

首次分析脚本的两个前置断言失败日志已保留：先对参考按 histories 除乘后要求旧 FP32 汇总逐位相同，以及把 TOPAS 零权重 spot 也计入行数比较。分别改为原始剂量先求和再核查，以及只比较正权重源；原始剂量、参考、GPU 运行、阈值和验收标准均未修改。

# FRED 非弹性模型修复执行总表

本目录把仓库根目录 [`plan.md`](../plan.md) 的审查结论转换成可执行、可验收、按依赖排序的工作包。本文是唯一的进度入口；执行时只推进当前阶段，不跨阶段调参。

## 总目标

修复 GPU 非弹性过程中的截面查表、守恒、碎片 channel、运动学、remnant 和次级 MCS，使 100–300 MeV/u 水模体结果达到论文级验证要求，并确认 400 MeV/u 外推稳定。GPU 作业只在本地 WSL 的 RTX 2080Ti（sm_75）运行；TOPAS 作业只在 `wuwei@127.0.0.1` 通过 `sbatch` 运行。剂量比较使用 3D scorer 后对横向求和，不使用 1D dose scorer。

## 执行规则

1. 严格按编号执行；前一阶段的硬门槛全部通过后，才能进入下一阶段。
2. 每个物理修改独立提交，禁止同时用 normalization、yield 或 halo scale 掩盖偏差。
3. 每次运行记录配置、commit、随机种子、history 数、耗时、诊断摘要和输出路径。
4. GPU Monte Carlo 必须本地运行，不得提交到远程主机或集群。
5. TOPAS 数据放在远端 `/mnt/sda/wuwei`；总资源不超过 192 CPU 和 128G 内存，并用 `sbatch`。
6. 开发阶段依次使用 100k、1M、10M histories；100M 只用于最终论文级验证。

## 进度状态

状态只使用：`未开始`、`进行中`、`阻塞`、`已完成`。更新状态时，同时填写证据链接或输出路径。

| 顺序 | 工作包 | 状态 | 完成证据 |
| ---: | --- | --- | --- |
| 00 | [冻结基线与诊断框架](steps/00-baseline-and-diagnostics.md) | 已完成 | 基线诊断核查；128-history GPU smoke |
| 01 | [修复 GPU 截面与 H/O 查表网格](steps/01-xs-grid.md) | 已完成 | host transport-grid 重采样；carbon_tests |
| 02 | [建立完整事件能量守恒](steps/02-energy-ledger.md) | 已完成 | 无 clip/scale；GPU balance error 8.1e-08 |
| 03 | [关闭固定 neutron vertex kerma](steps/03-neutron-kerma.md) | 已完成 | vertex kerma 计分路径已删除 |
| 04 | [恢复论文 Table-1 固定产额](steps/04-paper-yields.md) | 已完成 | 95–400 MeV/u 固定权重测试 |
| 05 | [替换 nearest-remnant channel sampler](steps/05-fragment-channels.md) | 已完成 | constrained sampler；projectile A/Z open 0 |
| 06 | [重实现 Eq. 12–21 运动学](steps/06-paper-kinematics.md) | 进行中 | 固定reference域、截断权重、全projectile correlation |
| 07 | [显式 remnant、Q 与 excitation](steps/07-remnant-q-excitation.md) | 已完成 | paper-minimal remnant；经验8 MeV deposit已删除 |
| 08 | [标定次级碎片 MCS](steps/08-secondary-mcs.md) | 进行中 | secondary scale接线完成；LUT标定待办 |
| 09 | [分层水模体回归与最终验收](steps/09-layered-validation.md) | 进行中 | 100/400 MeV/u 128-history GPU smoke |

## 当前应执行

继续完成 [步骤 06](steps/06-paper-kinematics.md) 的薄靶分布核对，再执行步骤 08 的独立离子 MCS 标定；不使用 halo scale 补偿 event generator。

## 全局硬门槛

```text
GPU/CPU XS interpolation relative error       < 1e-5
GPU/CPU target-H probability relative error   < 1e-5
A closure failures                            = 0
Z closure failures                            = 0
negative energy residual count                = 0
|numerical residual| / incident energy        < 1e-5
product queue overflow                        = 0
secondary step-limit count                    = 0
NaN/invalid stopping power or range           = 0
```

## 建议提交顺序

每份步骤文档给出建议提交。实际提交前先检查工作树，保留用户已有修改；不要把无关文件混入提交。

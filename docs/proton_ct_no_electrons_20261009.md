# Proton CT broadbeam 不加载电子包性能消融（2026-10-09）

本次按用户要求关闭电子包，仅实验配置变更，发布八预设与源码不变。使用最新proton-primary同一步视图复用的同一selective binary，RTX2080Ti sm75，CUDA_DISABLE_PTX_JIT=1。原RT07575 head CT、TPS源、plan权重/种子、0.5mm离子步长、continuous26 ion EM、涨落、核反应与带电碎片保持原配置。

实验配置存于 out/proton_ct_no_electrons_20261009/no_electrons/config/proton_ct_broadbeam.yaml：broadbeam_electron_tracking=false；删除birth/joint/charged/photon四个电子包路径；关闭电子bank retain、出生排序，并将电子roulette survival设1。现有broadbeam_fast解析路径支持此配置，无新增诊断构建模式或缺包回退。电子及其EM光子不执行独立输运，对应primary能损沿现有离子计分路径局部沉积；核反应中性产物依旧是未追踪能量汇，没有改成局部剂量。

这不是“只跳过最后一个电子kernel”的实验：出生/分组/入队、队列与包初始化也移除，主粒子dispatch从shared-electron stage13变为现有nonion stage7。不能把全部收益归为electron worker时间，不能视为相同空间剂量模型。

## 实测

ABBA每进程4×1M primaries，每臂8M，共16M；每进程首批剔除，性能统计每臂6M。连续批次复用GPU context/packages。

| 模式 | 暖批primary/s | 秒/百万primary |
|---|---:|---:|
| 完整电子/光子模型 | 152,659 | 6.5505 |
| 不加载电子包、局部计分 | 298,837 | 3.3463 |

合并吞吐比 1.95754，wall减少 48.92%。两对吞吐比 1.93618/1.97909。GPU未锁频，运行中温度31→约70°C，1875/1860MHz等频率变化；此处给出这轮本机观测，不作为跨硬件常数。没有做reverse独立种子复测，宽范围消融差异约两倍，不用于精确百分比收益认证。

所有批次quality accepted，energy/coverage/completion/zero-overflow检查通过。8个逐批对照中source CSV、核反应、核弹性、live ion terminal计数与transport ledger匹配。日志effective=disabled，未出现electron queue计时或电子包加载；保留的electron-geometry-cache标签是CT几何缓存，不是电子输运包。未进行剂量等价/gamma验收；本实验允许空间剂量改变，不推广为默认配置。

证据：out/proton_ct_no_electrons_20261009/{benchmark.py,summary.json,performance/}。脚本、配置和同binary SHA均保留。

后续已完成原始RT07575完整优化计划的16M/臂剂量验证：TOPAS local1%/1mm在BODY∪GTV内开启99.951991%、关闭99.948873%；全网格≥10%peak开启99.909317%、关闭99.339310%，均通过>99%。该后续采用原始完整浮点权重，非本页性能实验的part_01整数分块权重；参考为冻结的加权TOPAS物理DIJ。详见[双域必要性报告](rt07575_broadbeam_electron_gate_20261009.md)。

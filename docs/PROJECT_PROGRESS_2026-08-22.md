# MAIGO 多离子输运项目进展（2026-08-22）

## 1. 想实现什么

目标是把 MAIGO 从针对碳离子的专用程序，改造成由配置和外部物理数据驱动的
通用带电离子蒙特卡洛输运程序。用户应当只修改 YAML，并替换与粒子、材料和
物理模型对应的 package/data，就能运行 proton、helium、carbon、oxygen、neon
等常见离子，而不需要修改 GPU 输运算法。

希望固定在代码中的通用算法包括：

- voxel traversal、step control 和 RNG；
- 连续能损、能损涨落和多重库仑散射框架；
- 弹性/非弹性反应抽样和相关末态装载；
- secondary/cascade/neutral/electron queue；
- dose 和 LET scorer。

随离子、材料或用户选择的物理过程变化的内容应放在 YAML 和数据文件中：

- primary 的 Z、A、静质量和电荷相关数据；
- stopping-power、能损涨落、核反应截面和弹性截面；
- primary/cascade/elastic/neutral reaction package；
- 碎片 stopping power、有效电荷、衰变等扩展数据。

Package 不应被写死为 INCL++。只要离线数据满足运行时约定，用户可以使用任意
Geant4/TOPAS inelastic 或 elastic 过程产生 package。当前碳离子基准固定使用：

- `data/packages/topas_water_inclxx_1M_stitch7_primary_3d.bin`
- `data/packages/topas_400MeVu_water_inclxx_1M_cascade_3d.bin`

## 2. 已完成的工作

### 2.1 Primary ion 通用化

- YAML 显式配置 `primary_atomic_number`、`primary_mass_number` 和可选静质量。
- primary stopping power、inelastic cross section、primary package 和 cascade
  package 改为显式输入，不再由 C-12 身份隐式决定。
- CPU/SYCL 主粒子路径使用配置的 Z/A、质量、有效电荷、MCS 和 straggling 参数。
- package 工具保留 physics model 和版本作为 provenance，不把 INCL++ 当作格式限制。
- 修复 aggregate secondary depth scorer：总 IDD 始终累计全部 secondary 剂量，按
  species 分类的 scorer 仍由独立开关控制，避免关闭分类 scorer 时丢失总 IDD。

### 2.2 弹性过程

- 增加可配置 elastic cross-section table 和 correlated elastic package。
- 实现 elastic 抽样、GPU 存储、队列适配、方向更新和 overflow/accounting。
- 增加 proton-water 直接弹性截面、package 编译和自适应 TOPAS campaign 工具。
- 弹性、非弹性和连续电磁过程保持独立输入，输运核按竞争过程推进。

### 2.3 连续能损与能损涨落

- 增加可选 residual-range CSDA 能损，降低有限步长造成的射程偏移。
- primary straggling 使用固定 block 的随机变量，避免单纯细分步长时重复产生独立
  Gaussian 而人为改变总方差。
- 增加 Gaussian clamp audit，确认旧的 `2 * mean loss` 上限会在高能区截断方差。
- 增加 TOPAS/Geant4 能损涨落提取 extension、campaign、编译器和严格 CSV loader，
  允许以 `packaged_fluctuation` 替代经验性的 energy-wise scale。

### 2.4 Neutral 与电子运输

- neutral package 可按 `off`、局部沉积或 `full` 模式处理 gamma/neutron 末态。
- 增加 e-/e+ stopping-power table 的严格装载和 YAML 校验。目前 v1 仅接受均匀
  `G4_WATER`，并要求 neutral full transport。
- 增加 `ElectronParticle3D`、electron queue、generation limit、step/cutoff 设置和
  能量账本。
- 实现电子/正电子 condensed-history：碰撞能损、MCS、brems reservoir、逃逸与
  cutoff 局部沉积。
- brems gamma 会返回 neutral queue；停止的正电子产生两个背对背 0.51099895 MeV
  光子。gamma/neutron 的后续代不再自动并入 residual energy。
- 增加 Geant4 11.3.2 / TOPAS 4.2.p3、`g4em-standard_opt4` 的电子 stopping-power
  提取 extension，分别记录 e-/e+ 的 collision、radiation 和 total stopping power。

### 2.5 工程与验证基础

- 增加 clean-build 所需 CT API 依赖、配置校验、package identity/provenance 和多项
  cross-language package 测试。
- 增加多能量绝对剂量比较、scale-free straggling audit 和 TOPAS campaign 脚本。
- GPU 作业固定在本地 WSL 执行；TOPAS CPU campaign 可在限定的集群 CPU 节点执行。
- 本次发布未上传 LFS `.bin`。运行者必须在本地准备 YAML 引用的 package 文件。

## 3. 当前结果

### 3.1 构建和测试

- 2026-08-22 的 CPU rebuild 成功。
- CTest 结果为 8/8 通过，包括核心、elastic sampling/storage/application/queue、
  cross-language package、配置预加载和 Python campaign 测试。
- 电子耦合实现已在本地 WSL 的 NVIDIA TITAN RTX CUDA SYCL 后端通过
  `carbon_tests`。真实 neutral package smoke 观察到电子入队、电子剂量、brems
  gamma 回灌和能量账本闭合，electron/gamma queue overflow 均为 0。

这些结果证明工程链路可运行，不等同于生产级电子剂量已经与 TOPAS 一致。

### 3.2 碳离子 full-physics 绝对 IDD

使用 1M GPU histories、TOPAS 五个独立 100k seed 的均值乘以 10，仅做 histories
等价换算，不做 profile normalization；straggling scale 为 1.0，且使用上述规定的
carbon primary/cascade package。

| Energy | Peak height | Peak depth | Peak +/-2 mm | Pre-Bragg | Tail | Total |
|---:|---:|---:|---:|---:|---:|---:|
| 100 MeV/u | -1.478% | 0.000 mm | -0.888% | +0.991% | -24.237% | -0.357% |
| 200 MeV/u | -1.244% | 0.000 mm | -0.886% | +0.680% | -10.545% | -0.616% |
| 300 MeV/u | -1.070% | 0.000 mm | -1.237% | -1.012% | -8.126% | -2.020% |
| 400 MeV/u | -1.444% | 0.000 mm | -1.434% | -3.516% | -10.768% | -4.246% |

因此碳离子 Bragg peak 的位置和宽度已经接近 TOPAS，100/200 MeV/u 的峰区积分
约在 1% 内；但 300/400 MeV/u 的 pre-Bragg/总剂量以及所有能量的 distal tail
仍未达到生产级一致性。详细数据由
`scripts/audit_carbon_scale_free_idd_multienergy.py` 生成。

### 3.3 Proton 电磁与 full-physics

不含核过程的 1M proton water 绝对剂量，在 `packaged_fluctuation`、scale=1.0 下：

- 70--250 MeV 的峰高误差为 -0.574% 到 +0.621%；
- 峰深误差为 0--0.5 mm；
- ROI MAE 为 0.068%--0.478%；
- 总积分误差绝对值不超过 0.0027%。

这说明不依赖经验 scale 的 package-driven 电磁涨落路径是可行的。

Proton full-physics 目前仍不是无校准结果。已有 70--250 MeV 对比通过区域积分
门限时使用了 1.015--1.700 的 energy-wise straggling scale；去掉该 scale 后，
70--200 MeV 的峰高仍低 3.746%--13.964%。低能量 nuclear package 还是 pilot
数据，因此这组“通过”结果不能作为通用离子模型已经完成的证据。

## 4. 当前问题

### 4.1 Package 的能量与过程覆盖仍是主要限制

- 当前 carbon primary 是 stitch7 多能量 package，而 cascade 是 400 MeV/u
  package。对低能 cascade projectile 使用单一高能来源，可能导致碎片产额、方向和
  distal tail 偏差，需要按 projectile species 和能量做覆盖审计或生成更密能量网格。
- 50 MeV/u 间隔对快速变化的低能反应末态可能过粗。运行时插值只能改善连续表，
  不能恢复离散 package 中缺失的相关末态分布。
- carbon 尚无生产级 elastic package。忽略 elastic 会影响横向输运和少量深度剂量；
  但目前明显的 distal-tail/总剂量差异不能在没有独立诊断的情况下全部归因于 elastic。
- neutral package 不是所有计算都必需；只做 charged-particle 近似时可关闭。但要闭合
  gamma/neutron 能量并启用电子运输时，neutral full package 是必要输入。

### 4.2 电子运输尚未形成物理验收结果

- TOPAS electron extension 已编译到中途，但真实 0.001--500 MeV water stopping-power
  CSV 尚未生成；当前 smoke 使用合成临时表。
- 尚未完成相同 histories、相同 scoring definition 下 electron on/off 的生产级
  400 MeV/u carbon 绝对 IDD，也没有回答电子运输对 Bragg peak、entrance 和 tail
  分别改变多少。
- 尚缺 electron maximum-step、relative-loss 和 cutoff 的收敛性测试。
- 当前电子 v1 只支持均匀水，不支持 CT、异质 slab/insert、minibeam 或通用材料表。
- 电子统计尚未完整写入所有控制台输出和 energy-ledger JSON 字段。

### 4.3 通用离子目标尚未完全实现

- proton 已打通工程链路，但 helium/oxygen/neon 尚无真实 stopping/XS/package 的
  end-to-end TOPAS 对比。
- fragment stopping data 仍可能回退到有效电荷缩放；这不是任意离子的高精度替代品。
- 衰变、精确同位素质量、材料相关有效电荷数据和任意 Z/A 动态 species scorer
  尚未实现。
- package identity 主要依赖 YAML 和 sidecar provenance；还需要在运行前更严格地
  检查 projectile、material、process 和能量覆盖，避免“文件能加载但物理不匹配”。
- 目前不能宣称只替换 package 就能得到生产级新离子结果。除了反应 package，用户
  还必须同时提供匹配的 stopping power、截面、涨落/弹性数据以及完整 provenance。

## 5. 下一步验收顺序

1. 完成 TOPAS electron build，生成并审计真实 water electron stopping-power CSV。
2. 在本地 WSL CUDA 上完成固定配置的 electron off/on 1M carbon 绝对 IDD，对电子
   step 和 cutoff 做收敛性测试，并检查完整能量账本。
3. 为 carbon 生成 energy-resolved elastic package 和更密的 cascade package，逐项
   分离 elastic、neutral、cascade 对 pre-Bragg 和 distal tail 的贡献。
4. 冻结无经验 scale 的 carbon/proton 多能量验收协议，先解决 package coverage，
   再判断是否仍需修改输运算法。
5. 选择 helium 或 oxygen 作为第二个真实重离子案例，只通过 YAML + data/package
   完成端到端验证；通过后再扩展到 neon。

生产级完成标准应至少包括：绝对剂量不归一化、1M 或更高统计量、峰深和射程门限、
Bragg peak 前区域积分约 1% 精度、queue overflow 为 0、能量账本闭合，以及跨能量、
跨粒子验证中不使用按能量人工拟合的 scale。

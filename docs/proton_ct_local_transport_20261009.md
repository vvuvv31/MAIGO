# Proton CT broadbeam：显式关闭电子后专用 primary 优化（2026-10-09）

按用户要求，当前 `config/proton_ct_broadbeam.yaml` 暂时关闭电子输运：`broadbeam_electron_tracking: false`，移除 birth/joint/charged/photon 四个包路径并关闭 retain。电子能量沿已有 broadbeam_fast 路径局部计分；不是删除电子的能量贡献，也不是缺包回退。其它七个配置保持本轮之前的设置。离子 continuous26、涨落、0.5 mm 步长、散射、核反应、带电碎片和中性核产物能量汇保持不变。同步更新 `data/manifest.json` 与 `docs/preset_dependencies.json`，物理资源文件未删除。

本次默认修改是用户授权的暂时近似。剂量适用证据为 RT07575 **head 患者内部**，不推广到肺部或任意低密度几何。已有原始完整计划电子关闭的 TOPAS local1%/1mm（search0.1mm）BODY∪GTV 结果为 **99.948873%**，见 [原始双域验收](rt07575_broadbeam_electron_gate_20261009.md)。这项已有验收与本轮的匹配实现检查分开报告。

## 实现

关闭电子时此前使用 stage7（固定 non-ion 质子模型、其它设置保留运行时分支）。新增 proton CT 专用 stage15：主机先证明 CT、计分、散射、核反应、质子 Z/A 和包 non-ion 安全条件，以及关闭全部电子/队列条件，才在设备端折叠这些固定值。prepare 与 loss 复用同一步 continuous26 材料视图，沿用当前原始均值、涨落、delta 与 RNG。视图不跨步或进入 continuation state。其它自定义设置沿用原有 dispatch。

开启电子仍为 stage13；水宽束沿用原路径；碳和 minibeam 没有增加 stage15 dispatch。共享 fragment worker 保留通用带电物种。

## 性能

RTX2080Ti，本地 CUDA AOT，`CUDA_DISABLE_PTX_JIT=1`，仅编译 proton broadbeam 管线。ABBA 与独立种子 BABA，各进程连续4×1M；每进程首批剔除，context/packages在连续批次间复用。每臂共16M，计时每臂12M。**两臂均关闭电子**，因此此处是关闭电子后的额外实现收益。

| 实现 | 暖批 primary/s | 秒/百万 primary |
|---|---:|---:|
| 原无电子路径 | 298,550 | 3.3495 |
| 专用无电子路径 | 310,155 | 3.2242 |

合并吞吐 **+3.89%**，耗时减少 **3.74%**。ABBA比值1.042614（配对1.037680/1.047506），BABA比值1.035216（配对1.038638/1.031834）；四对方向均为正，保留候选。GPU未锁频，约1905→1830MHz、31→75°C；这是本轮本机观测，不是固定跨硬件比例。

历史开电子观测约152,659/s，相比当前约310,155/s接近两倍，但该跨实验比较不是本轮同时间交错测量。关闭电子本身的收益与本轮+3.89%不混算。

原生 primary 资源 stage7 为222寄存器/2224B stack，stage15为240/2368B。资源反而增加，不能把收益解释为降低寄存器或 spill。`cuobjdump`需要先从SYCL ELF提取device images；直接对ELF查询无device code不作为资源结果。

## 验证

两轮32个完整性能批次，source/种子、核反应/弹性、live-ion terminal与能量账目及EM离散审计匹配，剂量相对L1≤5.19e-8。参考≥10%峰值的体素，同位置 local1%通过100%，给出local1%/1mm通过率100%的下界。

另外原始RT07575完整浮点权重计划每臂2M，source/审计匹配，剂量L1≤5.15e-8；proton water和重新开启电子的CT各每臂40k，匹配通过。这两个小规模检查不用于吞吐宣称。全部44批quality accepted、energy/coverage/completion审计通过、零溢出，最大物理相对能量残差4.206391e-07。安装manifest校验通过。

本轮没有重新运行TOPAS，也没有把局部沉积认定为任意患者通用的物理等价模型。

## 产物与恢复

当前已构建专用可执行文件：`build-proton-broadbeam/carbon_mc`，SHA256 `1c7ea1f007279d651c27d8259bc1a94f0570618524b1e4a81768f3b395b0c14f`。完整五管线 `build/carbon_mc` 本轮未重新构建；其读取新YAML也会关闭电子，但没有本轮stage15优化。

原始证据位于 `out/proton_ct_local_perf_20261009/`：冻结基线/候选源码与binary、ABBA/BABA、逐批dose/审计、original-full-plan/water/electron-enabled匹配检查、device images/resources与summary。开启电子原配置保留为 `electrons_on.yaml`；恢复生产配置时恢复以下四个包路径并设 tracking/retain为true，再重建依赖清单：

```yaml
material_electron_birth_file: data/em/electrons/shared/birth26.bin
material_electron_joint_pilot_file: data/em/electrons/shared/joint.bin
material_photon_response_index_file: data/em/electrons/shared/photon.index
material_electron_charged_index_file: data/em/electrons/shared/charged.index
material_electron_retain_bank: true
broadbeam_electron_tracking: true
```

后续已保留同一步CT/scorer cell复用（stage16），因此当前专用程序已经更新；本页stage15 SHA与吞吐是初轮历史记录。当前实现与额外收益见[后续报告](proton_ct_shared_cell_20261009.md)。

# futureStep — 准备做 / 以后做

更新日期：2026-07-27

本文件只列**尚未完成**或**明确推迟**的工作。  
已实现能力见 [`README.md`](README.md)。  
**当前主线（明确下一步）见 [`LET.md`](LET.md)**，不要在这里重复展开 LET 细节。

分支约定：[`BRANCH_WORKFLOW.md`](BRANCH_WORKFLOW.md)（默认 `master`）。

---

## 0. 当前主线（不在本文件展开）

按 [`LET.md`](LET.md) 执行，优先级从高到低：

1. 定位 **300–400 MeV/u** all-hadron fragment tail：GPU vs TOPAS 产物能谱 / 代数 / 方向  
2. 若确认采样偏硬/偏软 → **能量条件化 cascade package**（保留事件内相关性）  
3. 独立 case 泛化（新 SOBP、单能、至少一个 CT plan）的 LET + dose  
4. 高统计量 3D LET 参考（GPU 1M；TOPAS 300k–500k，≤40 线程）

完成后再回到下面的中性/CT/性能项。

---

## 1. 近期（LET 主线之后）

### 1.1 中性粒子正式闭合

| 状态 | 说明 |
|------|------|
| 已有 | 方案 D / full 模式、neutral package 加载、interim `neutral_local_kerma_fraction` |
| 待做 | 用正式 neutron/gamma package 替换 interim local kerma |
| 待做 | 水中与 TOPAS neutral-origin 剂量闭合；再评估 CT 上是否可开 `enable_neutral_transport` |
| 注意 | CT 上 11.1.3 water-derived neutral package 曾恶化 IDD correlation；患者材料需单独验证 |

### 1.2 高能峰高残差（300/400 MeV/u）

- 带电 IDD 峰高仍可能有约 −3–4% 量级残差  
- 在 cascade 能谱修复（LET 主线）后复测，避免在谱错误时对 SP/straggling 经验调参  
- 检查限制能损语义与碎片 cutoff 是否仍主导峰区展宽

### 1.3 材料相关 reaction / cascade final-state

| 状态 | 说明 |
|------|------|
| 已有 | 水包 + 材料相关 XS / mass-SP；骨/肺 **诊断** package（G4 11.3.2）已导出 |
| 待做 | 与 production 参考 **同 Geant4 版本**（目标 11.1.3）的骨/肺/组织 final-state 包 |
| 待做 | CT 按 material_id 选择 package；Q 值 / 重残核 / 局部沉积闭合 |
| 禁止 | 用 11.3.2 诊断库静默替换 11.1.3 production 而不做对波 |

### 1.4 CT / TOPAS 对照补强

- 等权层、多 spot 在**统一旋转修复**后复算（历史 prelim 层需刷新）  
- 提高 TOPAS prelim 统计（≥2e5–5e5）再评 local γ 噪声底  
- 全 plan TOPAS MC 参考（在中性/材料包满意后）  
- CT binary 0.25 mm origin 元数据债务：下次从 DICOM 完整重建时统一，禁止只改单边配置  
- 跨病例盲测：固定 MU↔ions 响应因子后测新计划，避免单病例过拟合

---

## 2. 中期（计划接口与泛化）

### 2.1 多野 / 任意角度计划

| 已有 | 待做 |
|------|------|
| TPS 90° 轴置换 + TOPAS spots 批处理 | 通用 `beam_dir` / gantry 循环脚本叠加多野剂量 |
| opt-in `tpsSource`（碳离子 + PBS CSV） | 多粒子种、更完整 couch/collimator/患者方位（HFS/HFP…） |
| 单计划 batch primary | 外层脚本多角度 seed 流与绝对 MeV/primary（或 fluence）叠加 |

最小增量曾规划（仍可参考）：

1. `TransportConfig` 显式 `beam_dir_*` + `isocenter_mm`（与现有 tps_source 协调，勿双轨）  
2. 先支持与体轴共线 ±z smoke  
3. `run_ct_plan_angles.py`：两角度 vs TOPAS 两野  

### 2.2 验证与论文资产（暂缓优先）

- 论文级总图与消融表（用户曾要求暂缓）  
- 固定 seed 套件的一键回归报告（水 + hetero + CT smoke + LET）  
- 跨能量 / 跨几何 acceptance 清单写入 CI 或脚本门禁

---

## 3. 性能与工程（以后做）

以下**不改变**已锁定物理约定时再做；每步独立 A/B，对照当前 full-physics 基线。

### 3.1 次级输运

| 项 | 说明 |
|----|------|
| **P4 step-level wavefront** | 每 worker 固定 quantum steps 后 compact；缓解 secondary 长尾 SIMD 空等 |
| secondary 调度参数 | `secondary_batch_size`、local_size（CUDA 128 / L0 256）统一 A/B |
| CUDA secondary 非进度尾 | 历史 TITAN RTX oneAPI-CUDA 路径 secondary 极慢；需同 commit、同 FP64、同 seed 严格 A/B（见 archive 中 cudaVSlevel0 笔记） |

### 3.2 Kernel / tally

| 项 | 说明 |
|----|------|
| **P5** 编译期 kernel 特化 | CT on、neutral off 等恒定分支消掉；防寄存器爆炸 |
| **P6** local/tiled tally | 热点 voxel 本地合并再 flush FP64；患者 CT 需测 hit rate |
| **P7** 可选 FP32 batch tally + FP64 reduction | 仅 fast mode；必须过 gamma / 能量闭合 |
| FP32 vs FP64 dose atomic 统一 A/B | Level Zero 生产多为 FP64；历史 CUDA 曾用 FP32，不可混比性能 |

### 3.3 分级物理 / 快速剂量（仅 preview）

生产默认保持 `energy_cutoff_MeV=0.1`、`maximum_relative_energy_loss=0.005`、full cascade。

| 档位 | 用途 |
|------|------|
| `preview-primary` | 坐标 / 射程 smoke |
| `direct-secondary` | 直接带电次级、无 cascade |
| `full-cascade` | 最终报告默认 |

另测 `energy_cutoff_MeV=1.0` 等 **fast-dose** 配置：必须过 3D γ、R80、积分、峰位、能量闭合后才能标为 preview，**不得**静默替代最终剂量。  
每个输出记录 backend / physics-tier 与关闭的能量通道。

### 3.4 架构债（低优先级）

- 拆分 `transport_sycl.cpp` 巨型单体（几何 / primary / secondary / neutral / scorer）  
- 收窄 `TransportConfig` 职责或分组  
- Serial 与 SYCL 能力不对齐：完整路径依赖 SYCL + TOPAS，无全功能 CPU oracle  

细节见 [`structure.md`](structure.md) §14。

---

## 4. 明确不做 / 已排除的方向

- 为贴 TOPAS 或 `physical_dose` 做**按能量/按病例**的 SP、straggling、MCS 经验 scale  
- 把 ridge/能层拟合权重当作跨病例通用物理（入口面 bug 修复后已废弃该路径作 production）  
- 仅为性能用 2 mm 下采样 CT 替换细网格生产（实测端到端加速可忽略，次级仍主导）  
- 在 LET 主线未完成前对 SOBP LET 曲线做 case-specific 乘子校准  

---

## 5. 建议验收门槛（任意新改动）

- 构建 + `carbon_tests` 通过  
- queue overflow = 0  
- energy-balance 不劣于当前统计可解释范围  
- 固定 seed 对照：IDD / R80 / 积分 / 3D dose / 峰位  
- 报告 2%/2 mm 或 3%/3 mm γ（视场景）  
- 同时报 wall time、kernel time、histories/s、steps/history，避免把“少算物理”写成“GPU 加速”  

---

## 6. 历史文档位置

下列工作笔记内容已并入 README / 本文件 / LET.md，原文移至 `docs/archive/` 备查：

- `codex.md` — TPS 90° CT 性能 P0–P8  
- `grok.md` — CT gamma、入口面修复、full-plan 标定  
- `furtherStep.md` / `nextStep.md` — 中性、emittance、异质、CT 执行记录  
- `compileAndRun.md` — 构建命令（已写入 README）  
- `cudaVSlevel0.md` — CUDA vs Level Zero 性能分析  
- `GPU_MC_TPS_Source_Module_Design.md` — TPS source 设计（实现契约已写入 README）  
- `intel_oneapi_carbon_ion_gpu_monte_carlo_guide.md` — 长篇从零开发指南与早期状态  

# furtherStep — 按序执行记录

源清单：`D:\OneDrive\DoctorDocuments\myproject\carbonGPU\furtherStep.md`  
工作区：`carbonGPU_grok`

---

## 1. 中性 mode D — 已完成（见前文）

## 2. 峰高诊断 — 已完成（见前文）

---

## 3. 数据表与版本对齐 — **已完成**

### 截面 XS

- 远程 TOPAS 4.1.p1 / **G4 11.1.p3** 导出 `CarbonCrossSectionNtuple`
- 产出：`data/c12_inelastic_cross_sections_water_geant4_11_1_3.csv`
- 与旧 11.3.2 表在 1–400 MeV/u 网格上 **宏观水截面相对差 = 0**（数值一致）

### 停止本领 SP

- 新 scorer：`CarbonStoppingPowerNtuple`（G4EmCalculator electronic dE/dx）
- 产出：`data/stopping_power_water_geant4_11_1_3.csv`
- vs 开发 Bethe–Bloch 表（200 MeV/u）：16.127 vs 16.098 MeV/mm（**+0.18%**）

### Package 策略（文档）

见 `data/README_physics_tables.md`：≤200 用 200 包，>200 用 400 包；中性用 200 development package。

### 接线

- 所有 `beam_*MeVu_multi_energy*.yaml` 已指向 `*_geant4_11_1_3.csv`
- 抽检 mode D + 新表 100k：
  - 200 MeV：积分 ≈ −0.2% 量级（与换表前一致）
  - 400 MeV：积分 ≈ −3.9% 量级（换表未改变主偏差结构）

### 结论

- **XS 版本混用不是主因**（两版表相同）
- **SP 对齐后略有差异（~0.2%）**，不解释 300/400 峰高 −4%
- 高能剩余偏差仍主要来自：mode D residual 中性 + 峰区展宽（条目 1/2）

---

## 4. Emittance 多能量 σ(z) — **已完成**

### 设置

- 源：TOPAS BiGaussian emittance  
  σx=σy=0.2 mm，σ′=0.032，ρx=−0.9411，ρy=+0.9411  
- 能量：150 / 200 / 250 / 350 / 400 MeV/u  
- TOPAS development：50k histories/E（远程 v@192.168.31.5，顺序跑）  
- GPU Arc B580：100k histories/E，`beam_*MeVu_emittance_sigma.yaml`  
- 观测量：剂量加权横向 σ_rms(z)（1 mm 深度 bin，0.5 mm 侧向体素）

### 工具与产出

| 角色 | 路径 |
|------|------|
| TOPAS 配置 | `validation/topas/carbon_*MeVu_water_emittance*.txt` |
| 远程 runner | `validation/scripts/_remote_emittance_multi.py` |
| GPU 套件 | `out/emittance/e{150,200,250,350,400}_voxels.csv` |
| 分析 | `validation/scripts/analyze_emittance_multi_sigma.py` |
| 汇总 JSON | `validation/results/emittance_multi_sigma_summary.metrics.json` |
| 总览图 | `validation/results/emittance_multi_sigma_overview.png` |
| 分能曲线 | `validation/results/emittance_{E}_sigma/{gpu,topas}_sigma_vs_depth.csv` |

### 指标（GPU − TOPAS）/ TOPAS

| E (MeV/u) | mean rel σ | peak-region rel | entrance σ GPU / TOPAS (mm) | source σ |
|-----------|------------|-----------------|------------------------------|----------|
| 150 | **−2.20%** | −1.33% | 0.303 / 0.354 | 0.2 |
| 200 | **−2.04%** | −1.24% | 0.369 / 0.435 | 0.2 |
| 250 | **−1.88%** | −1.22% | 0.460 / 0.432 | 0.2 |
| 350 | **−1.85%** | −1.08% | 0.456 / 0.498 | 0.2 |
| 400 | **−2.06%** | −0.95% | 0.489 / 0.568 | 0.2 |

### 结论

- 全能量 σ(z) 平均相对偏差稳定在 **约 −2%**（GPU 略窄），峰区约 **−1%**，与先前单独 200 MeV 结果一致。
- **入口 dose-weighted σ 明显大于源几何 σ=0.2 mm**（两侧均如此）：0.5 mm 体素 + σ′ 发散 + 首 bin 内 MCS/straggling；不宜把入口 bin 直接等同束流 σ。
- 入口处 GPU 多数能量仍略窄于 TOPAS（250 入口 bin 统计有起伏）；**无能量相关发散恶化**，不支持按能量再调 MCS/straggling 去贴 TOPAS。
- 本项验证 emittance 采样与横向展宽链在 150–400 MeV/u 上可用；剩余 IDD 峰高/积分问题仍归条目 1（中性 residual）与 2（峰区展宽诊断），而非 emittance 源。

---

## 5. RNG 键 → GPU bit-reproducible — **已完成**

### 问题

次级/级联/中性输运把 **原子队列下标 `particle_index`** 当作 Philox 的 `history_id`。  
同 seed 两次运行时队列分配顺序不同 → 同一物理粒子走不同随机流 → IDD **非 bit-identical**（旧 1e6：NRMSE~3e-5）。

### 改动

| 项 | 说明 |
|----|------|
| `SecondaryParticle3D` / `NeutralParticle3D` | 增加 `uint64_t rng_stream`（布局 48 B） |
| `rng::child_stream(parent, branch_tag)` | 由父 stream + 角色/产物序号派生，**与 queue slot 无关** |
| 角色 tag | primary charged/neutral、cascade charged/neutral、neutral→charged、continuation |
| 输运 | MCS / 级联抽样 / 中性 free-path 一律用 `particle.rng_stream` |

初级 history 仍用 `history` 作 stream；其直接次级：`child_stream(history, tag(role, secondary_index))`；级联/中性子代：`child_stream(parent.rng_stream, tag(...))`。

### 验证

- `carbon_tests`：Philox + child_stream；cascade 同 seed 两次 **IDD 逐 bin 相等**（CPU SYCL）
- Arc B580 GPU：`config/beam_200MeVu_rng_repro.yaml` 10k cascade 双跑  
  - `identical True`，`max_abs=0`，步数/反应数/次级能量计数完全一致  
  - 输出：`out/rng_repro/run_{a,b}_idd.csv`

### 说明

剂量仍用 `double` atomic 累加；在本机 B580 上轨迹一致后 IDD 已位级一致。若他卡上仍见 ulp 级差异，再考虑定点记分；当前 **不必**。

### 未改

- 物理模型、SP/XS、禁止按能量调 straggling/MCS  
- 论文图（条目 6 暂缓）

---

## 6. 论文图 — **暂缓**（用户：暂不急论文）

---

## 7. 异质体 — 进行中

### 7a-1. 轴向密度分层 slab（水当量）— **已完成（含 TOPAS 对比）**

**物理**

- 成分仍为水表；层内 **ρ 缩放** dE/dx、宏观 XS、straggling、MCS
- 界面截步；几何：0–50 mm ρ=1 / 50–70 mm ρ=1.85 / 70–400 mm ρ=1

**TOPAS vs GPU（各 50k，B580 / 远程 56 线程）**

| 量 | TOPAS | GPU | GPU−TOPAS |
|----|-------|-----|-----------|
| R80 (mm) | 69.86 | 69.89 | **+0.03 mm** |
| peak z (mm) | 69.75 | 69.75 | 0 |
| 积分相对 | — | — | **−0.94%** |
| 密层均值 | — | — | −2.8% |
| NRMSE | — | — | 0.010 |

产物：`validation/results/slab_density/compare.metrics.json`、`compare.png`  
工具：`_remote_slab_density.py`、`stitch_slab_density_idd.py`、`compare_slab_density.py`

### 7a-2. 真实 bone/lung 表 + 多材料 slab — **已完成（含 TOPAS 对波）**

**表（远程 G4 11.1.3 / TOPAS 导出）**

| 材料 | SP @200 MeV/u (MeV/mm) | 宏观 XS @200 (1/mm) | 文件 |
|------|------------------------|---------------------|------|
| water | ~16.13 | ~0.0045 | `*_water_geant4_11_1_3.csv` |
| bone (G4_BONE_COMPACT_ICRU) | **27.81** | **0.00765** | `*_bone_geant4_11_1_3.csv` |
| lung (G4_LUNG_ICRP) | **16.66** | **0.00490** | `*_lung_geant4_11_1_3.csv` |

**GPU 多材料**

```yaml
enable_layered_phantom: true
slab_stopping_power_files: water.csv,bone.csv,water.csv
slab_cross_section_files: ...
# 有 material 表时：SP/XS 用绝对表（不再 ρ 缩放）；ρ 仅 MCS/straggling
```

- 配置：`config/beam_200MeVu_slab_real_bone.yaml`
- Backend：`+layered-material`

**TOPAS 修复说明**

先前异常（剂量≈0、Real~8s）来自参数/组件命名与密度 slab 不一致、以及并发覆盖污染输出。  
已改为与密度 slab **同几何组件名**（`LayerDense`）+ 完整 BeamPosition 旋转；smoke 与 50k development 正常（Real~218s）。

**TOPAS vs GPU（各 50k，真实 G4_BONE_COMPACT_ICRU）**

| 量 | TOPAS | GPU | GPU−TOPAS |
|----|-------|-----|-----------|
| R80 (mm) | 72.45 | 72.52 | **+0.07 mm** |
| peak z (mm) | 72.25 | 72.25 | 0 |
| 积分相对 | — | — | **−0.81%** |
| 骨层均值 | — | — | −1.11% |
| NRMSE | — | — | **0.0047** |

产物：`validation/results/slab_density/compare_real_bone.metrics.json`、`compare_real_bone.png`

相对密度 slab（ρ=1.85 水当量 R80≈69.9 mm），真实骨 R80≈72.5 mm → **组成差异可见**（骨非纯密度缩放水）。

**局限（已知）**

- 级联 reaction package 仍为水中 TOPAS 包；骨层主要校验 **SP/初级 XS + 射程**
- MCS 仍用 water radiation length + 局部密度

### 7b. 横向/3D 异质插入体 — **已完成（含 TOPAS 对波）**

**模型**

- 水背景 + 有限 AABB 插入体（`enable_hetero_insert`）
- 几何：`x,y ∈ [-10,10] mm`，`z ∈ [50,70] mm`，材料 `G4_BONE_COMPACT_ICRU`（绝对 SP/XS 表）
- 3D 材料判定 + 射线–AABB 界面限步（避免卡死）；与轴向 slab **互斥**

**配置 / 工具**

| 角色 | 路径 |
|------|------|
| GPU | `config/beam_200MeVu_hetero_bone_insert.yaml`（及 `_idd` 无 voxel 版） |
| TOPAS | `validation/topas/carbon_200MeVu_water_hetero_bone_insert*.txt` |
| 远程 | `validation/scripts/_remote_hetero_bone.py` |
| 对比 | `validation/results/hetero/compare_bone_insert.*` |

**GPU 自洽（50k）**

| 案 | R80 (mm) | 插入区均值剂量 |
|----|----------|----------------|
| 均匀水 | 86.93 | 11.3 |
| 骨插入 | **72.52** | **23.5** |

（轴上铅笔束穿过插入体 → 与全宽骨 slab 射程效应一致。）

**TOPAS vs GPU（50k）**

| 量 | TOPAS | GPU | GPU−TOPAS |
|----|-------|-----|-----------|
| R80 (mm) | 72.45 | 72.52 | **+0.07 mm** |
| peak z | 72.25 | 72.25 | 0 |
| 积分 | — | — | **−0.64%** |
| 插入区均值 | — | — | −0.93% |
| NRMSE | — | — | **0.0046** |

### 7b 扩展. 偏轴骨条 + 空气腔 — **已完成（GPU + 空气腔 TOPAS）**

**空气腔（轴上，水 SP×ρ_air）**

| 角色 | 路径 |
|------|------|
| GPU | `config/beam_200MeVu_hetero_air_cavity.yaml` |
| TOPAS | `validation/topas/carbon_200MeVu_water_hetero_air_cavity*.txt` |
| 远程 | `validation/scripts/_remote_hetero_extended.py air …` |
| 对比 | `validation/results/hetero/compare_air_cavity.metrics.json` |

| 案 | R80 (mm) |
|----|----------|
| 均匀水 | 86.93 |
| 空气腔 GPU | **106.92** |
| 空气腔 TOPAS | **106.89** |
| GPU−TOPAS | **+0.03 mm** |

NRMSE ≈ 0.007；积分相对差 ≈ −1.0%。远端射程拉长与低密度腔定性一致。

**偏轴骨条（x∈[5,25] mm，弱 emittance σ=3 mm）**

| 角色 | 路径 |
|------|------|
| GPU | `config/beam_200MeVu_hetero_bone_offset.yaml` |
| TOPAS | `validation/topas/carbon_200MeVu_water_hetero_bone_offset*.txt` |
| 远程 | `validation/scripts/_remote_hetero_extended.py offset …` |

| 案 | R80 (mm) |
|----|----------|
| 均匀水 | 86.93 |
| 偏轴骨 GPU | **86.93** |
| 偏轴骨 TOPAS | **86.90** |
| GPU−TOPAS | **+0.03 mm** |

NRMSE ≈ 0.007；积分相对差 ≈ −0.8%。相对 on-axis 骨插入（R80→72.5），偏轴条对中央 IDD 几乎不变（仅旁瓣穿过骨），符合预期。TOPAS 需 `BeamAngularCutoffX/Y`（已写入参数文件）。产物：`validation/results/hetero/compare_bone_offset.metrics.json`。

### 7c. CT 体素网格 — **已完成（合成水立方 + 患者 CT smoke）**

**模型**

- 二进制网格 `CCTG`：`nx,ny,nz` + origin/spacing + density + material_id
- HU→密度/材料类：`prepare_ct_grid.py` 读 **TOPAS `HUtoMaterialSchneider.txt`**（默认 `ct/HUtoMaterialSchneider.txt`）
  - 密度：Schneider 公式（与 TsDicomPatient 同表）
  - 材料类：Schneider 组织段折叠为 air/lung/water/bone（供 GPU 4 表 SP/XS）
- GPU 运行时只读 `ct/grid/*.bin` 的密度+material_id（不再现场做 HU 分段）
- 水当量模式：水 SP/XS × 局部密度；多材料模式：绝对表 × ρ/ρ_ref
- 患者 DICOM/网格 **不入库**：`.gitignore` 含 `ct/`（含用户复制的 Schneider 表）

**工具 / 配置**

| 角色 | 路径 |
|------|------|
| 准备 | `validation/scripts/prepare_ct_grid.py`（`ct/dicom` → `ct/grid/patient_ct.bin`） |
| 头文件 | `include/carbon/ct_grid.hpp`（`ct_sample`、`clamp_step_to_ct_faces`） |
| 合成水 | `config/beam_150MeVu_ct_water_cube.yaml` + `ct/grid/water_cube.bin` |
| 患者 | `config/beam_200MeVu_ct_patient.yaml` + `ct/grid/patient_ct.bin` |
| 指标 | `validation/results/ct/ct_smoke.metrics.json` |

**GPU smoke**

| 案 | 入口 MeV/primary | peak z (mm) | R80 (mm) |
|----|------------------|-------------|----------|
| 合成水立方 150 MeV/u | ~9.77 | 52.75 | 52.85 |
| 患者 CT 150 MeV/u（20k） | ~18.1（轴上 ρ≈1.85） | 37.25 | 37.5 |

患者入口 ≈ 水 × 轴上密度，Bragg 前移，能量守恒正常。

**已修 bug**

- CT 体素面限步在面上产生 `step≈0` → 粒子 `break` 后把剩余动能整笔倒入当前 bin（入口 ~1349 MeV/primary）。
- 修复：`distance_to_next_ct_face_1d` 跳过零长度面 + 面上 nudge；患者 IDD 恢复物理量级。

**多材料 CT（lung/water/bone 绝对表）— 已接通**

- 配置：`config/beam_200MeVu_ct_patient_multimat.yaml`
- 规则：`SP/XS = 表_mat(E) × (ρ / ρ_ref)`；ρ_ref：air/lung/water=1，bone=1.85
- Backend：`+ct-grid-material`
- GPU 20k 对比（同患者网格，primary-only + MCS）

| 模式 | 入口 MeV/p | peak z | R80 |
|------|------------|--------|-----|
| 水当量 | 18.13 | 37.25 | 37.52 |
| 多材料 | 16.88 | 39.25 | **39.54** |

入口比 ≈ 0.93（骨绝对 SP/水×ρ ≈ 1.72/1.85），射程略拉长，与 7a 真实骨 vs 密度 slab 趋势一致。  
产物：`validation/results/ct/ct_multimat_vs_we.metrics.json`、`gpu_e150_patient_multimat_idd.csv`

**TOPAS 患者 CT 对波 — 已完成（20k development）**

| 角色 | 路径 |
|------|------|
| TOPAS | `validation/topas/carbon_150MeVu_ct_patient*.txt` + `HUtoMaterialSchneider.txt` |
| 远程 | `validation/scripts/_remote_ct_patient.py` |
| 对比 | `validation/scripts/compare_ct_patient.py` → `validation/results/ct/compare_patient.metrics.json` |

**几何约定（重要）**

- TOPAS `TsDicomPatient` 把 CT **isocenter 放在组件原点**（Trans=0 → 体中心在世界原点）
- 本 CT：nz=35×2 mm → 半深 35 mm；入口（首片）z=−35 mm；束流 (0,0,−35.1) 沿 +z
- GPU：`prepare_ct_grid` 将首片映射为 z=0、xy 居中；对比时用**入口相对深度**
- 平行世界 IDD 管（无 material）：0.5 mm × 240 bins，覆盖 CT + 出口水

**CT 验证一键入口**

```text
python validation/scripts/run_ct_baseline.py
# 或仅汇总已有 CSV：
python validation/scripts/run_ct_baseline.py --skip-gpu
```

产出：primary-only / multimat+secondary / TOPAS 三份绝对 MeV/primary IDD +  
`validation/results/ct/ct_baseline_summary.metrics.json`（含 `delta_R80_mm`、`integral_rel_diff`、`nrmse`）。

**SP 模型（CCTG v3；v2 可读兼容）**

`SP = SP_water(E) × f_E(za_rel, I, E) × ρ`  
- `za_rel = (Z/A)_section / (Z/A)_water`（高能极限）  
- `I`：Bragg 平均激发能（eV），由 Schneider 元素份额  
- `f_E`：简化 Bethe 阻止数比 `za_rel × L(I,E)/L(I_w,E)`（I_w=75 eV）  
- CCTG v2 网格：仅存常数 `mass_sp_factor`（读入时 I≡75 → f_E = za）  
Backend：`+ct-grid-mass-sp-e`。初级与次级均采样 CT 密度/段 (za,I)。

**GPU mass-SP-e + secondary vs TOPAS（各 20k，150 MeV/u；CCTG v3）**

| 量 | GPU primary | GPU **+secondary mass-SP-e** | TOPAS |
|----|-------------|------------------------------|-------|
| R80 (mm) | 43.49 | **43.50** | 43.51 |
| 入口 (MeV/p /0.5mm) | 14.16 | **14.22** | 14.16 |
| 积分 (MeV/p) | 1483.6 | **1721.1** | 1721.0 |
| 积分相对差 | ~−13.8% | **≈+0.008%** | — |
| NRMSE | 0.020 | **0.011** | — |
| ΔR80 | −0.02 mm | **−0.01 mm** | — |
| Wall (s, B580) | 3.25 | **9.35** | — |

Backend：`+ct-grid-mass-sp-e`。配置：`config/beam_200MeVu_ct_patient_multimat(_secondary).yaml`  
指标：`validation/results/ct/compare_patient_secondary.metrics.json` / `compare_patient_mass_sp.metrics.json`

**性能（同 mass-SP + secondary，20k，隔离 A/B）**

- 开关：`ct_skip_homogeneous_face_clamp`（默认 true）
- full clamp：Steps 5.883e9，**10.31 s**
- homog skip：Steps 5.836e9（**−0.8%**），**8.02 s**（**−22%**）
- 另：步长 < 0.2×min voxel spacing 时跳过 face 计算（Bragg 峰 thrash）
- 能量平衡均 ≪1e-3；对照配置：`…_secondary_full_clamp.yaml`

**局限 / 后续**

- f_E 为简化 Bethe 比（非完整 25 材料 G4 表 / shell / density-effect 全项）
- 可再做更激进的 DDA / 同质合并

**Future plan：允许物理近似的性能档位（当前生产配置不启用）**

1. 低能次级局部沉积 A/B：测试 `energy_cutoff_MeV: 1.0` 与
   `maximum_relative_energy_loss: 0.01`（必要时再测 0.02），将 cutoff 以下的
   剩余能量沉积到当前 voxel。必须以当前 0.1 MeV / 0.005 结果为基准，比较
   3D gamma、R80、D95/D2、积分剂量、峰值位置和能量闭合；通过阈值后才能作为
   fast-dose 配置，不能直接替代最终剂量配置。
2. 分级物理模式：提供 `preview-primary`（仅 primary）、`direct-secondary`
   （直接带电次级、无 fragment cascade）和 `full-cascade`（当前完整模型）三档。
   优化迭代和坐标 smoke 可使用前两档，最终报告默认使用 `full-cascade`；每档输出
   必须记录 backend 标签和被关闭的能量通道，避免把预览剂量误当最终剂量。

### 多角度 / 真实计划钩子（设计，未实现）

当前 GPU CT 约定：**束流固定 +z，CT 首片 z=0、xy 居中**。真实计划需要：

| 层级 | 内容 | 建议实现顺序 |
|------|------|----------------|
| 1. 单野旋转 | 配置 `beam_direction` 或 `gantry_angle_deg` + isocenter；粒子在等中心坐标系采样，CT 查询用旋转矩阵 `R^T (x - iso)` | 先只转束、不转网格 |
| 2. 多野叠加 | 外层脚本循环角度/权重，累计绝对 MeV/primary（或 fluence 权重）IDD/3D dose | 不改核心核，脚本叠 dose |
| 3. 计划接口 | DICOM RT Plan / spot list → 每 spot 能量、权重、角度；输出与 TOPAS 同 scorer | 对齐 TOPAS 坐标后再扩 |
| 4. 性能 | 每野独立 seed 流；同质 face-skip / short-step 保持；可选预旋转 CT 副本（内存换查询） | 先 1 再测 |

**坐标对齐要点**

- TOPAS `TsDicomPatient`：isocenter 在组件原点；Gantry 转束不转 CT
- GPU 现状：入口帧（首片 z=0）；对比时用入口相对深度
- 多角度对比 TOPAS：统一到 **isocenter 帧** 再叠剂量；IDD 沿束轴重投影

**最小下一步（未做）**

1. `TransportConfig` 增加 `beam_dir_x/y/z`（默认 0,0,1）与 `isocenter_mm`
2. 源抽样与初始方向用该方向；CT sample 仍用世界坐标（先只支持与体轴共线 ±z 的 180° 翻转作 smoke）
3. 脚本 `run_ct_plan_angles.py`：两角度叠加 vs TOPAS 两野

在精度/性能基线锁定（A–C）后再动 1–2。

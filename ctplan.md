# CT Plan Workflow（TOPAS DIJ / matRad / GPU 蒙卡）

本文记录 **患者 CT 碳离子计划** 在 TOPAS 与 GPU 两侧的几何约定，避免以后混淆
「旋转 CT」与「旋转束流」、以及扫描磁铁带来的微倾角。

相关代码入口：

| 位置 | 作用 |
|------|------|
| `ct/<case>/scripts_c/run_c_*.txt` | TOPAS DIJ / dose 几何与 spots include |
| `ct/<case>/scripts_c/spots_c_*.txt` | 逐 spot 能量、位置、Rx/Ry、histories |
| `ct/<case>/code/physical_dose_result.mat` | 优化权重 `x` 与计划剂量 `d3d` |
| `ct/<case>/code/physical_dose.mhd` | 物理剂量参考（通常为 Dij·x） |
| `spots_geometry_mode: tps_90` | patient X 为深度的 TPS 路径（既有 head cases） |
| `spots_geometry_mode: tps_gantry_y` | patient +Y 为深度的 TPS 0° 路径（20022516 lung） |
| `validation/scripts/prepare_full_plan_weights.py` | 从 `x` 导出权重与 histories 分配 |
| `validation/scripts/match_gpu_to_physical_dose.py` | GPU 剂量对齐参考并算 gamma |

---

## 1. 核心思想（务必先记住）

在 TOPAS 里 **转机架（转束流方向）不方便**，因此采用等价做法：

```text
真实 TPS：固定患者，机架角 θ 转束流
TOPAS 实现：固定「TPS 0° 束流基准」，用 Patient/RotZ = θ 转 CT
```

再配合 BeamPosition 上的 **Rx / Ry**，把 TOPAS 默认「沿局部 +Z 出射」的源改成 **TPS 0° 方向**，并叠加扫描磁铁引起的 **非整数小角度**。

```text
                    ┌─────────────────────────────┐
                    │  TPS 机架角 θ                │
                    │  （真实世界转束流）           │
                    └──────────────┬──────────────┘
                                   │  等价实现
                                   ▼
        Patient/RotZ = θ           旋转 CT（患者体位）
              +
        BeamPosition RotX/RotY     把源从默认 +Z 改到 TPS 0°
              +
        每 spot 的微小 ΔRx/ΔRy     扫描磁铁 → 虚拟源到等中心不平行
```

---

## 2. TOPAS 几何分解

以 lung case `ct/20022516` 的 `scripts_c/run_c_01.txt` 为例。

### 2.1 患者：用 RotZ 模拟 TPS 角度

```text
d:Ge/Patient/TransX = -69.6605 mm
d:Ge/Patient/TransY =  9.3887 mm
d:Ge/Patient/TransZ =  0.0819 mm
d:Ge/Patient/RotZ   =  0.0000 deg    ← 本 case 的 TPS 计划角 = 0°
```

| 参数 | 含义 |
|------|------|
| `TransX/Y/Z` | 把 DICOM 患者放到 world（等中心/摆位相关） |
| **`RotZ`** | **TPS 计划角 θ**。0° / 90° / … 用转 CT 代替转束流 |

- **θ = 0°**（本 lung case）：`RotZ = 0`，束流在「TPS 0° 基准」下入射患者。
- **θ = 90°**（如既有 RT06423 等 case）：`RotZ = 90`，CT 绕 Z 转 90°，等价机架转 90°。

> TOPAS 组件旋转是 **passive** 约定；GPU 侧若做 world↔patient，逆变换符号要与
> `transform_tps_90_pose_to_ct` 注释一致（见 `src/topas_spots.cpp`）。

### 2.2 束流基准：从默认 +Z 改到 TPS 0°

```text
d:Ge/BeamPosition2/TransX = Tf/Scatterer1/L5/Value mm   # 扫描位置 X
d:Ge/BeamPosition2/TransY = -0.45 m                    # SAD：源在 -Y
d:Ge/BeamPosition2/TransZ = Tf/Scatterer1/L6/Value mm   # 扫描位置 Z
d:Ge/BeamPosition2/RotX   = Tf/Scatterer1/L7/Value deg  # ~90° + 微扰
d:Ge/BeamPosition2/RotY   = Tf/Scatterer1/L8/Value deg  # ~0°  + 微扰
```

| 量 | 作用 |
|----|------|
| `TransY = -SAD`（此处 450 mm） | 虚拟源到等中心距离 |
| **`RotX ≈ 90°`** | 把源局部 **+Z 出射** 拧成 **TPS 0°** 主方向（本约定下为 world **+Y**） |
| **`RotY ≈ 0°`** | 同平面内微调 |
| `L5/L6`（TransX/TransZ） | 扫描磁铁在源平面上的横向位移（spot 位置） |

源粒子挂在 `BeamPosition2` 上（`So/.../Component = "BeamPosition2"`），默认沿组件局部 +Z。
**没有 Rx≈90° 时束流仍是 TOPAS 默认轴向；有了 Rx/Ry 才是「TPS 0° 束流」。**

与 `RotZ` 合在一起：

```text
TPS 计划角 θ 的完整语义
  =  固定「TPS 0° 束流」（BeamPosition Rx/Ry ≈ 90°/0°）
  +  患者 RotZ = θ
```

### 2.3 扫描磁铁：非整数微角度

spots 文件里 **L7/L8 不是整 90 / 0**，例如 20022516（1549 spots）：

| 通道 | 含义 | 典型范围（本 case） |
|------|------|---------------------|
| L7 RotX | 俯仰类 | 约 **89.81° ~ 90.19°**（中心 ≈ 90.00°） |
| L8 RotY | 偏航类 | 约 **−0.17° ~ +0.20°**（中心 ≈ 0°） |
| L5 TransX | 源平面扫描 X | 约 ±19.5 mm |
| L6 TransZ | 源平面扫描 Z | 约 −20 ~ +22 mm |

含义：

- 主值 **90° / 0°**：TPS 0° 束流基准；
- **相对 90°/0° 的毫度～0.2° 级偏差**：从虚拟源到等中心，扫描磁铁偏转后射线 **并不严格平行**，每 spot 有独立小倾角；
- 与 L5/L6 的横向位移一起，复现 PBS 扫描几何，而不是理想平行束栅格。

GPU 目标几何也应 **保留每 spot 的 Rx/Ry（及由此推出的方向）**，不能只取整数 90°。

### 2.4 Spot 通道速查（Scatterer1）

| 层 | 典型含义 |
|----|----------|
| L0 | spot 序号 |
| L2 | 总动能 MeV（C-12） |
| L3 | 能量散度（% 量级，按 TOPAS 定义） |
| L4 | 该 spot histories（DIJ 常为常数，如 1e5） |
| L5 / L6 | 源平面 TransX / TransZ |
| L7 / L8 | RotX / RotY（0° 基准 + 扫描倾角） |
| L9–L14 | BiGaussian emittance |

多野/分文件：`spots_c_01.txt` + `spots_c_02.txt` 按 **优化/Dij 列顺序** 拼接（ID 连续 1…N）。

---

## 3. 从 TOPAS DIJ 到物理剂量计划

```text
TOPAS (scripts_c)
  run_c_01 / run_c_02  +  spots  +  DICOM
        │
        ▼
  每 spot 剂量核（DIJ 列）  ──►  dij_physical_sparse_*.mat
        │
        ▼
  matRad / ADMM 等优化 (code/run_admm.m)
        │
        ▼
  physical_dose_result.mat
        ├── x          每 spot 权重（长度 = 总 spot 数）
        └── d3d        计划三维剂量（常与 physical_dose.mhd 一致）
        │
        ▼
  physical_dose.mhd/.raw   参考物理剂量立方（患者/计划网格）
```

**剂量关系（概念上）**

```text
physical_dose ≈ Dij · x
```

- `x`：优化变量（非负）；
- DIJ 列通常对应 **固定每 spot 粒子数**（如 L4=1e5），GPU 复现时用
  `N_i ∝ x_i`（Hamilton 分配），再用 **全局 scale** 吸收 MU↔离子数与历史总预算。

权重导出：

```bash
python3 validation/scripts/prepare_full_plan_weights.py \
  --mat ct/<case>/code/physical_dose_result.mat \
  --spot-files ct/<case>/scripts_c/spots_c_01.txt ct/<case>/scripts_c/spots_c_02.txt \
  --output-weights ct/<case>/plan/full_plan_weights_exact.csv \
  --output-allocation ct/<case>/plan/full_plan_allocation.csv \
  --summary ct/<case>/plan/full_plan_weights_summary.json \
  --pilot-histories 10000000 \
  --actual-histories-per-spot 100000
```

---

## 4. 当前 GPU 路径（X-depth 与 Y-depth 均已实现）

历史/生产配置多走 **`spots_geometry_mode: tps_90`**：

1. CT 用 `reorient_ct_grid_tps_90.py` 把 **深度轴排到 GPU +Z**
   （束流沿 patient ±X 时的 packing）；
2. `tps_zero_beam_pose_for_spot`：用 **−Rx/−Ry** 得到与 TOPAS 一致的 world 方向；
3. `transform_tps_90_pose_to_ct`：world → patient（含 `RotZ`），再映到 reoriented CT；
4. 入口面投影到 GPU z=0，emittance 沿空气隙传播。

```text
90° case：Patient RotZ=90 + reorient(X 为深度)  →  tps_90 闭环已验证
 0° case：Patient RotZ=0  + 束流沿 +Y          →  tps_gantry_y 闭环已验证
```

`tps_gantry_y` 使用 `patient_ct_beam_y.bin`，其轴映射为
`GPU(x,y,z) = patient(x,z,+y)`；source/方向先按 TOPAS passive `RotZ`
从 world 变换到 patient，再映到 GPU，并保留逐 spot L7/L8 与 emittance。
因此目前已支持本 lung case 的 +Y 入射闭环。下一节的“固定 CT、直接转束流”
仍是进一步统一任意机架角的长期方向，而不是本次 0° 验证的前置条件。

---

## 5. 任意机架角的统一 GPU 几何（已实现为 opt-in TPS source）

### 5.1 原则

| TOPAS 做法 | 目标 GPU 做法 |
|------------|----------------|
| 转 CT（`Patient/RotZ`）模拟机架角 | **直接指定 TPS 角度 θ**，CT 保持患者坐标系（或一次固定摆位） |
| 转 CT 很重 | **等中心放在原点**，束流方向 = 机架角，不再为每个角度重排体素 |
| Rx/Ry 含扫描微倾角 | **保留每 spot 的方向**（由 Rx/Ry + θ 合成），不要压成理想平行束 |

### 5.2 推荐坐标系（概念）

```text
患者/计划系（DICOM 或 matRad 网格）
  · 等中心 = (0,0,0)     ← 显式约定，Trans 吸收到摆位
  · CT 密度网格：固定，不随 θ 旋转
  · 第 k 个 spot：
        源点  s_k（虚拟源球面/平面上，含扫描位移）
        方向  u_k（由 TPS 0° 基准 + 机架角 θ + 扫描微倾角）
        粒子从 s_k 沿 u_k 入射，穿 CT
```

机架角 θ 作用在 **束流方向** 上，例如绕患者 Z（或 TPS 定义轴）：

```text
u_k = R_gantry(θ) · u_0°(spot_k 的 Rx/Ry 扫描倾角)
s_k = R_gantry(θ) · s_0°(spot_k 的横向扫描位置)   # 若源随机架转
```

使用 `tps_angle_convention: topas_patient_rot_z` 时符号已经与本文
TOPAS passive `Patient/RotZ` 约定钉死：

```text
w(theta) = (-sin(theta), cos(theta), 0)
theta=0°   → patient +Y
theta=90°  → patient -X
theta=270° → patient +X
```

角度为连续浮点数；37°、37.5°、225° 不走基数角特例。

### 5.3 与 TOPAS 参数的对应

| TOPAS | GPU 目标模型 |
|-------|----------------|
| `Patient/RotZ = θ` | 配置项 **`tps_angle_deg = θ`**（转束流，不转 CT） |
| `Patient/Trans*` | 并入 **isocenter 在原点** 的摆位，或显式 isocenter 偏移 |
| `BeamPosition TransY = −SAD` | `spots_sad_mm` / 虚拟源距离 |
| L5/L6 横向扫描 | spot 在源平面的 (x′, y′) |
| L7/L8 ≈ 90°+ε, 0°+δ | **TPS 0° 基准 + 扫描磁铁微倾角** → 每 spot `u_k` |
| L4 · x | histories 分配；绝对 Gy 用全局 scale 对齐 `physical_dose` |

### 5.4 当前配置

```yaml
tpsSource: true
tps_angle_convention: topas_patient_rot_z
tps_gantry_angle_deg: 37.0
tps_couch_angle_deg: 0
tps_collimator_angle_deg: 0
tps_isocenter_mm: [0, 0, 40]
tps_sad_mm: 450
```

固定患者 CT 直接使用未做轴置换的 `patient_ct.bin`。scorer 的
`voxel_bins_x/y`、`voxel_size_x/y`、`depth_bin_width_mm` 和
`phantom_length_mm` 必须与 CCTG 网格完全一致。patient CT 的非零 Z
低边界只在输运内部重基准；输出 MHD 仍使用原患者坐标。

样例：

- `config/beam_tps_source_arbitrary_angle.yaml`：37° 水箱；
- `config/beam_tps_source_ct_20022516_angle37_smoke.yaml`：37° 固定 lung CT。

默认 `tpsSource: false`，所以既有 `tps_90`、`tps_gantry_y` 的 RNG 和
source 路径不变。

### 5.5 为何这样更好

1. **同一 CT 网格** 可算多机架角，不必为每个 θ 生成 `patient_ct_tps_θ_*.bin`。
2. **等中心在原点** 后，与 matRad / 物理剂量网格对齐更直观。
3. **自然保留扫描磁铁非平行性**（每 spot 独立方向），与 TOPAS L7/L8 一致。
4. 90° 现有路径可视为「θ=90 + 曾用转 CT 实现」的特例；长期应收敛到同一「转束流」模型。

### 5.6 使用时注意

- TOPAS **passive RotZ** 与 GPU **主动转束流** 的符号已由
  0°/37°/90°/180°/270° 几何测试锁定；新 patient position 或 couch
  组合仍应做单 spot 入口验证。
- 剂量 scorer 网格应与 `physical_dose.mhd` 同轴或可严格重采样（block-average）。
- 全局 dose scale：吸收 history 预算与 MU→离子数；gamma 在 scale 之后算。
- 扫描微角虽小（~0.2°），在长 SAD 下入口位移可达毫米级，**匹配入口与 penumbra 时不要丢掉**。

---

## 6. 推荐工作流清单（以后直接照做）

### A. 读懂一个 TOPAS case

1. 打开 `run_c_*.txt`：记下 `Patient/RotZ`（= TPS 角 θ）、`Trans*`、`BeamPosition2/TransY`（SAD）。
2. 打开 `spots_c_*.txt`：确认 L7≈90°、L8≈0° 为 0° 基准；看 L7/L8 散布 = 扫描倾角。
3. 打开 `physical_dose_result.mat`：`x` 长度 = 总 spot 数；与 L0 顺序一致。
4. 打开 `physical_dose.mhd`：DimSize / Spacing / Offset = 参考剂量网格。

### B. 导出 GPU 权重

见 §3 `prepare_full_plan_weights.py`。

### C. 跑 GPU（当前已验证路径 / 未来任意角统一模型）

- **现状 90°**：`spots_geometry_mode: tps_90` + reoriented CT + `spots_patient_rot_z_deg: 90`。
- **现状 0°、+Y 入射**：`spots_geometry_mode: tps_gantry_y` + `patient_ct_beam_y.bin` +
  `spots_patient_rot_z_deg: 0`，每 spot 方向含 L7/L8。
- **任意角**：`tpsSource: true` +
  `tps_angle_convention: topas_patient_rot_z` + 固定 patient CT（本文 §5）。

### D. 与 physical_dose 对齐 + gamma

```bash
python3 validation/scripts/match_gpu_to_physical_dose.py \
  out/.../dose.mhd \
  ct/<case>/code/physical_dose.mhd \
  --output-dir out/ct/<case>/match \
  --patient-shape <nx ny nz>   # 全分辨率 CT 体素数，用于重采样
```

关注：全局 scale、3%/3mm、2%/2mm、1%/1mm、3%/0mm（DTA=0 即纯剂量差准则）。

---

## 7. Case 速记：20022516（lung，TPS 0°）

| 项 | 值 |
|----|-----|
| TPS 角 | **0°**（`Patient/RotZ = 0`） |
| SAD | 450 mm（`TransY = -0.45 m`） |
| Spot 数 | 1549（775 + 774） |
| DIJ L4 | 1e5 / spot |
| 权重 `x` | 1549 维，`physical_dose_result.mat` |
| 参考剂量 | `physical_dose.mhd` 240×151×42 @ 2 mm |
| TOPAS 细网格 | 960×607×42 @ 0.5/0.5/2 mm（4× 下采样 → 参考） |
| L7 / L8 | ~90°±0.2° / ~0°±0.2°（扫描倾角） |

10M GPU 验证（参考为 TOPAS Dij·优化权重得到的 `physical_dose.mhd`，10% 阈值，
GPU 先做高剂量区最小二乘全局 scale）：

| 指标 | 结果 |
|------|------|
| global 3%/3 mm | **99.778%** |
| local 3%/3 mm | **98.969%** |
| global 2%/2 mm | **98.316%** |
| local 2%/2 mm | **94.853%** |
| global 1%/1 mm | 79.117% |
| global 3%/0 mm | 79.150% |
| 高剂量 NRMSE / 3D cosine | 2.482% / 0.998470 |
| IDD 峰 | 相差 1 个 2 mm bin |
| 三维峰坐标 | 完全一致 |

完整的四病例统一结果和复现命令见 `CTPlan4CaseMatch.md`。

### 7.1 本地 TOPAS 与 GPU 同粒子验证（2026-07-29）

为避免拿 `physical_dose.mhd`（TOPAS Dij 加权和）间接比较，本次直接让
TOPAS 和 GPU 使用完全相同的 spot 参数与整数 history 数。选择
`20022516` lung case，是因为它包含低密度肺组织、骨和软组织，比既有 head
case 更能暴露 CT 材料映射、能损和散射差异。

完整计划有 1549 个 spot。TOPAS 的 time feature 会对每个 spot 建立一个
Geant4 run；本地实测其 run 切换开销约 30--40 s/spot，所以即使把总粒子数
降到约 100k，完整 spot 列表仍需十余小时。为在本地得到可重复的直接验证，
从两束中按位置、能量和权重选了 5 个代表性 spot：

| 原计划序号 | 能量 (MeV/u) | L5/L6 (mm) | GPU/TOPAS histories |
|---:|---:|---:|---:|
| 1 | 120 | -19.4827 / -8.4221 | 2920 |
| 470 | 125 | -5.5665 / -2.8074 | 30 |
| 841 | 135 | 2.7832 / -19.6515 | 40370 |
| 1200 | 130 | 8.3497 / 0 | 2770 |
| 1549 | 155 | 19.4827 / 19.6515 | 53910 |
| **合计** |  |  | **100000** |

GPU 与 TOPAS 都保留了 L7/L8 的逐 spot 偏转角、相同 SAD、患者平移、DICOM
CT 和 DoseToMedium 网格。比较采用 TOPAS 剂量峰值 10% 阈值、三线性插值及
0.5 mm gamma 搜索步长。以下是**绝对同粒子数、GPU scale 固定为 1**的结果：

| 指标 | 100k 结果 |
|---|---:|
| GPU/TOPAS 剂量积分差 | **+0.704%** |
| global 3%/3 mm | **99.834%** |
| local 3%/3 mm | **99.670%** |
| global 2%/2 mm | **98.486%** |
| local 2%/2 mm | **94.958%** |
| 高剂量区 NRMSE | 5.073% |
| IDD correlation | 0.999385 |
| IDD 峰位差 | 1 个 0.5 mm bin |

高剂量区最小二乘的诊断 scale 为 0.975394，即 GPU 高剂量响应平均约高
2.52%；应用该 scale 后 global/local 3%/3 mm 为 99.768%/99.532%，但这不是
上表的主结果。绝对积分仅高 0.704%，说明剩余差异主要是局部形状和统计涨落，
不能简单归结为统一归一化偏差。少数单平面的 local gamma 仍约 82--95%，集中
在很窄、采样点少的单 spot 高梯度区；这也是 5-spot/100k 子集不能替代完整
临床计划验证的原因。

速度（相同 100k histories）：

| 程序 | 配置 | 墙钟/输运时间 | 吞吐 |
|---|---|---:|---:|
| GPU | CUDA | 0.875 s | 114342 histories/s |
| TOPAS | 40 CPU threads | 1207.0 s / 1199.5 s | 83.4 histories/s |

本次 GPU 墙钟约快 **1380×**。TOPAS scorer 因 Geant4 parallel-world
navigation 未计入 4838 个 step，但总能量仅 `2.27e-6 MeV`，对剂量结果可忽略。

复现入口：

- GPU：`config/generated/beam_ct_20022516_topas_smoke100k.yaml`
- TOPAS：`ct/fullplan_local_ct_compare/20022516/run_smoke_100k_dose_only.txt`
- spot：`ct/fullplan_local_ct_compare/20022516/spots_smoke_100k.txt`
- TOPAS RTDOSE 转 MHD：`validation/scripts/convert_topas_rtdose_to_mhd.py`
- 数值结果：`out/ct/20022516/topas_local_compare_100k/match_absolute/match_metrics.json`
- 三解剖面图：`out/ct/20022516/topas_local_compare_100k/multiplanar_absolute/`

注意：本地 TOPAS 4.2.p3 使用 Geant4 11.3.2，而该 GPU CT 配置仍使用
Geant4 11.3.2 生成的 stopping-power、截面和 cascade 表；因此残差同时包含
GPU 近似误差和 Geant4 版本差异。下一轮严格物理归因应先把 GPU 表统一到
11.3.2，再增加 spot 数或在集群运行完整计划。

---

## 8. 一句话版

- **TOPAS**：`RotZ` 转 CT = TPS 角；`BeamPosition Rx/Ry` 把默认 +Z 源改成 **TPS 0° 束流**；L7/L8 的小数 = **扫描磁铁非平行**。
- **GPU 目标**：**等中心在原点，直接设 TPS 角转束流，CT 不转**；方向仍带每 spot 扫描微倾角。
- **当前代码**：head 的 X-depth `tps_90` 与 lung 的 Y-depth
  `tps_gantry_y` 继续用于已验证复现；opt-in `tpsSource` 已能在固定患者 CT
  上按 `topas_patient_rot_z` 连续转束流。

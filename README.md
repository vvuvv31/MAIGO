# MAIGO — carbon-oneapi-mc

面向 **Intel oneAPI / SYCL** 的碳离子 condensed-history 蒙特卡洛剂量与 LET 引擎。  
主 C-12 与带电碎片支持三维方向、voxel 边界步进、Highland 多重库仑散射、反应/级联末态包、患者 CT 与 TPS spot plan；后端可在 Intel Arc（Level Zero）与 NVIDIA（CUDA plugin）上运行。

> **研究用途**：结果可用于与 TOPAS/Geant4 对照和算法开发，**不能用于临床或治疗计划**。

## 文档索引

| 文档 | 作用 |
|------|------|
| **本 README** | 已实现能力、构建运行、验证摘要 |
| [`futureStep.md`](futureStep.md) | **当前主线与后续计划**（A 节：LET_d / cascade 能谱；B–D：中性/CT/性能等） |
| [`BRANCH_WORKFLOW.md`](BRANCH_WORKFLOW.md) | 分支约定：默认在 `master` 开发与推送 |
| [`structure.md`](structure.md) | 代码架构与模块依赖 |
| [`docs/archive/`](docs/archive/) | 历史工作笔记与长篇开发日志（已整合） |

---

## 当前能力（已实现）

### 输运与物理

- C++20 **serial CPU** CSDA 子集（连续能损 / straggling / 初级衰减；无完整 MCS/次级/级联）
- **SYCL** 全功能路径：serial 与 SYCL 共用配置与物理表
- MeV/u ↔ C-12 总动能显式转换；自适应步长与深度 bin / voxel 面限步
- 能量涨落（Bohr straggling，可选材料 `Z/A` 缩放）
- Highland / Lynch–Dahl **多重库仑散射**（主 C-12 与带电碎片；可选 CT 材料辐射长度）
- 能量相关 **C-12 非弹性截面**（TOPAS/Geant4 导出表插值）
- 事件级 **reaction package** 联合采样 + 固定容量带电次级队列
- 最多两代 **fragment cascade**（breadth-first；产物按 Z/A 重分类）
- 可选 **中性粒子**（neutron/gamma）队列：`first_interaction`（方案 D）或 `full` continuation
- 可选 **粒子种同位素停止本领表**（Geant4 提取；缺失同位素退回 C-12 有效电荷缩放）
- 同 seed 次级/级联 **RNG stream 与 queue slot 解耦**（GPU 轨迹可复现）

### LET_d

- YAML：`scorerLET: true/false`（关闭时不分配 LET buffer、无原子开销）
- primary C-12 / all-hadron 深度 LET_d；可选三维 MHD
- 元素分组（C/B/Be/Li/He/H）与可选轻同位素诊断
- 分子/分母 FP64 atomic moments，写盘前再相除
- 与 TOPAS `myHadronLET` 对齐的定义与水中 SOBP 验证（细节见 [`futureStep.md`](futureStep.md) A 节）

### 几何与材料

- 均匀水 / 轴向 **slab**（密度缩放或真实 bone/lung 绝对 SP+XS 表）
- 横向 **hetero insert**（AABB 骨条、空气腔等）
- **CT 体素网格**（CCTG v2/v3）：Schneider 密度、`(Z/A,I)` mass-SP、DDA 步进与同质 span
- CT 多材料 SP/XS 表；核残余本地沉积；可选材料 MCS

### 束流与计划

- 单能 / 多能 / SOBP 配置；BiGaussian **emittance**
- **TOPAS spots** 解析与多文件按序连接；`spotWeight.csv` 按优化权重分配 histories（largest-remainder）
- **TPS 90°** 患者 CT：轴置换网格 + `Patient/RotZ` 被动旋转约定；入口面投影（含基数角 `cos(90°)` 残差修复）
- 可选 **TPS source 模块**（`tpsSource: true`）：gantry/couch/collimator/isocenter/SAD + PBS spot CSV

### 计分与输出

- 深度剂量 CSV；稀疏 voxel CSV；**dense MHD/RAW**（总 Gy 或 per-primary 视配置）
- charged-origin 8 类 voxel 分类 + 闭合检查
- fragment species / reaction / neutral / 能量账本
- dose scorer 默认 **FP64 atomic**（可选编译期 FP32）

### 性能相关已落地

- 整份 spot plan **一次 batch launch**（`--sequential-spots` 仅作 A/B）
- secondary **track-level persistent workers** + energy-sorted batches
- 线程本地 dose pending 合并；primary near-Z CT face fast path
- CT **integer DDA** + homogeneous span；mass-SP **预计算 LUT**
- Linux 双目标编译：`spir64` + `nvptx64-nvidia-cuda`

### 构建与测试

- CMake presets：`cpu-debug`、`oneapi-release`（双目标）、`oneapi-intel-release`、`oneapi-nvidia-release` 等
- 无第三方测试依赖的 `carbon_tests`（单元 + 集成 smoke）
- 辅助脚本：`scripts/build_linux_oneapi.sh`、`scripts/run_linux_nvidia.sh`、Windows B580 系列脚本

---

## 验证状态摘要

下列为已固化对照结论（无 MCS/全局 dose 的“贴曲线”调参作为主线）。

### 水中带电 IDD（Arc B580 / TOPAS）

| 层级 | 要点 |
|------|------|
| EM 隔离 + `straggling_scale=1.2` | R80 ≈ +0.1 mm；2%/2 mm γ ≈ 97%（固定后续 100–400 MeV/u 参数） |
| 直接截面 + reaction package | serial/GPU 曲线高度一致；尾部由碎片输运闭合到约 10% 量级 |
| 两代 cascade + MCS 3D | 100k charged-origin 绝对 3D 比较：高剂量 voxel MAE ~3%、Pearson ~0.97 |
| 多能量 100–400 MeV/u | 固定模型参数；高能坪区曾用 interim `neutral_local_kerma_fraction` 改善 |
| Emittance σ(z) | 150–400 MeV/u 剂量加权 σ 平均相对差约 −2% |
| 步长 / 横向 voxel / 统计收敛 | 生产默认步长 0.5 mm；横向对比常用 5 mm |

### 异质与 CT

| 场景 | 状态 |
|------|------|
| 密度 slab / 真实骨 slab / 骨插入 / 空气腔 | TOPAS 对波：ΔR80 约 0.03–0.07 mm，积分差 ~1% 内 |
| 患者 CT mass-SP + secondary | 发展级 IDD 与 TOPAS 积分/R80 闭合良好 |
| TPS 90° prelim 单点 vs TOPAS | 旋转约定修复后 3D cosine ~0.98、IDD corr ~0.998 |
| CT full-plan vs matRad physical dose | 入口面修复后 10M：global/local 3%/3 mm 约 **99.2% / 98.1%**（固定 spot 响应 scale） |
| CUDA vs Level Zero 同 plan | 剂量高度一致；历史 wall-time 差不可直接归因于 backend API |

### LET_d

水中 SOBP all-hadron / primary C-12 已较好复现 TOPAS；低能 SP 表扩展至 0.01 MeV/u 为关键修复。  
**剩余主误差在 300–400 MeV/u fragment tail** → 见 [`futureStep.md`](futureStep.md) **A.4 第一优先级**。

中性粒子与 TOPAS 全闭合、跨病例通用多材料 final-state 包等 → 见 [`futureStep.md`](futureStep.md) B–D 节。

---

## 构建与运行

### 环境

| 平台 | 用途 |
|------|------|
| Linux + oneAPI `icpx` | 主开发：Intel SPIR-V 与/或 NVIDIA CUDA 目标 |
| Windows + VS + oneAPI | Arc B580 Level Zero 生产路径之一 |
| 远程 TOPAS 主机 | 参考模拟与物理表导出（勿在 WSL 提交新 TOPAS 作业） |

### Linux（推荐脚本）

```bash
source /opt/intel/oneapi/setvars.sh   # 按实际安装路径
scripts/build_linux_oneapi.sh        # 默认 spir64 + nvptx64-nvidia-cuda

# 或
cmake --preset oneapi-release
cmake --build --preset oneapi-release
ctest --preset oneapi-release
```

CPU 调试（无 SYCL）：

```bash
cmake --preset cpu-debug && cmake --build --preset cpu-debug
ctest --preset cpu-debug
./build/cpu-debug/carbon_mc --config config/beam_200MeVu.yaml --device serial
```

### 运行设备

| `--device` | 含义 |
|------------|------|
| `serial` | 非 SYCL CPU |
| `cpu` | SYCL CPU |
| `gpu` | 跟随 `ONEAPI_DEVICE_SELECTOR` |
| `cuda` / `nvidia` | NVIDIA CUDA plugin |
| `level_zero` / `intel` / `arc` | Intel Level Zero |
| `opencl` | OpenCL GPU |

**NVIDIA**：默认仅 SPIR-V 的二进制在 CUDA 上会找不到 kernel；需编译期 `nvptx64-nvidia-cuda`（`oneapi-release` 或 `oneapi-nvidia-release`）。

```bash
export ONEAPI_DEVICE_SELECTOR=cuda:gpu
./build/oneapi-release/carbon_mc \
  --config config/beam_200MeVu_attenuation.yaml \
  --device cuda
```

**Intel Arc B580（Linux Level Zero）**：若遇 V2 kernel-args 兼容问题：

```bash
export UR_L0_V2_DISABLE_ZE_LAUNCH_KERNEL_WITH_ARGS=1
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
./build/oneapi-intel-make/carbon_mc \
  --config config/beam_200MeVu_voxel_smoke.yaml \
  --device level_zero
```

| 变量 / preset | 含义 |
|----------------|------|
| `oneapi-release` | 双目标 `spir64,nvptx64-nvidia-cuda` |
| `oneapi-nvidia-release` | 仅 CUDA AOT |
| `oneapi-intel-release` | 仅 SPIR-V |
| `CARBON_SYCL_TARGETS` | 自定义 `-fsycl-targets` |
| `CARBON_CUDA_ARCH` | 可选 AOT，如 `sm_75` |

### Windows Arc B580

```bat
scripts\build_windows_oneapi.cmd
scripts\run_windows_b580.cmd
```

PowerShell 可先加载环境：`. .\scripts\enter_windows_oneapi.ps1`（若存在）。

### 患者 CT / TPS 90°（本地数据不入库）

需要本地 `ct/dicom/`、`ct/HUtoMaterialSchneider.txt`、`ct/grid/*.bin`、`ct/spotWeight.csv` 等（见 `.gitignore` 的 `ct/`）。

```bash
python validation/scripts/prepare_ct_grid.py --schneider-file ct/HUtoMaterialSchneider.txt
python validation/scripts/reorient_ct_grid_tps_90.py

# 仅检查计划
./build/.../carbon_mc --config config/beam_ct_optimized_tps_90.yaml --plan-only

# 输运
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./build/.../carbon_mc \
  --config config/beam_ct_optimized_tps_90.yaml --device level_zero
```

一键 CT baseline（Windows 脚本或）：

```bash
python validation/scripts/run_ct_baseline.py
python validation/scripts/run_ct_baseline.py --skip-gpu
```

### LET 示例配置

```bash
# 见 config/beam_*_letd*.yaml 与 out/letd_*
./build/.../carbon_mc --config config/beam_sobp_water_letd.yaml --device gpu
```

---

## 物理数据

| 资产 | 说明 |
|------|------|
| `data/stopping_power_water*.csv` | 水 SP；生产验证常用 Geant4 导出表 |
| `data/ion_stopping_power_water_geant4_11_3_2.csv` | 多同位素 SP（LET / 精确碎片） |
| `data/let_delta_electron_fraction_*.csv` | delta 电子能量份额 |
| `data/c12_inelastic_cross_sections_*.csv` | C-12 水/骨/肺宏观非弹性截面 |
| `data/ct_full_plan_weights_*.csv` | CT full-plan 权重辅助（病例相关） |
| `validation/results/*.bin` | reaction / cascade / neutral 二进制包 |
| `data/README_physics_tables.md` | 表版本与选用策略 |

开发用 Bethe–Bloch 近似表仅供联调；正式对照使用 TOPAS/Geant4 导出表，**禁止按能量手工贴 Bragg 曲线**。

---

## 输出格式

默认深度剂量 CSV：

```text
depth_mm,energy_deposition_MeV,dose_Gy,relative_dose
```

启用 voxel scoring 时：稀疏 `ix,iy,iz,...` 与/或 dense MHD（`DoseUnits = Gy`，dose-to-medium 用局部密度）。  
CT plan 下 `number_of_histories` 为**整份计划**的统计预算，不是每 spot 等权。临床 fraction 绝对 Gy 另需 MU↔离子数标定。

---

## 仓库布局（简）

```text
include/carbon/   公共头文件与物理工具
src/              配置、I/O、serial/SYCL 输运、spot/TPS source
config/           束流与验证 YAML
data/             物理表
tests/            carbon_tests
validation/       TOPAS 扩展、脚本、规范化结果与指标
scripts/          构建与运行辅助
out/              本地输出（git ignore）
ct/               本地 DICOM/网格（git ignore）
```

架构细节见 [`structure.md`](structure.md)。

---

## 许可与分支

- 许可见 [`LICENSE`](LICENSE)
- 自 2026-07-27 起默认在 **`master`** 上修改并推送 `origin/master`（见 [`BRANCH_WORKFLOW.md`](BRANCH_WORKFLOW.md)）
- `legacy` / `cuda` 为归档或兼容别名，不作为日常开发目标分支

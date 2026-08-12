# MAIGO 仓库架构速读

> 当前源码快照：2026-08-11。本文以 `CMakeLists.txt`、公共头文件和当前实现为准，目标是让新的对话或模型只读本文件即可开始定位和修改代码。历史实验结论仍应回到对应报告和原始结果核实。

## 0. 一页结论

MAIGO（CMake 项目名 `carbon_oneapi_mc`）是一个研究用途的碳离子 condensed-history Monte Carlo：

- C++20 核心，无外部运行时库；完整加速后端使用 Intel oneAPI SYCL。
- 输入是自定义的 `key: value` 配置、CSV 物理表、二进制末态 package、可选 CCTG CT 网格，以及 TOPAS/TPS spot 计划。
- CPU serial 后端只覆盖一维 CSDA/straggling/初级衰减子集；水箱、CT、MCS、碎片、级联、中性粒子、LET 和完整三维计分主要在 SYCL 后端。
- 普通水/CT/TPS 走 legacy SYCL kernel；Copper minibeam 是编译期开关控制的独立大 kernel，开启构建后再由运行时配置分发。
- `src/main.cpp` 同时承担 CLI、资源装载、spot 计划编排、结果累加、输出和运行摘要，是应用层总入口。
- 核心状态对象是 `TransportConfig`（所有输入策略）和 `TransportResult`（所有 tally、账本、计时及诊断）。
- 物理数据不是运行时生成的：CSV lookup table 和 `.bin` 事件 package 由 TOPAS/Geant4 离线生成，运行时只装载、插值和采样。
- 当前 CMake 只生成 `carbon_core` 与 `carbon_mc`；`tests/carbon_tests.cpp` 存在，但没有被当前 `CMakeLists.txt` 构建，也没有 CTest 注册。

整体数据流：

```text
CLI + key:value config
          │
          ▼
   TransportConfig::validate()
          │
          ├── CSV: stopping power / cross section / ion tables
          ├── BIN: reaction / cascade / neutral packages
          ├── CCTG: CT density + material grid
          └── TOPAS spots / TPS CSV → PrimarySpotBatchEntry[]
          │
          ▼
 transport_serial  或  transport_sycl
                          │
              ┌───────────┴────────────┐
              │ legacy water/CT/TPS    │
              │ optional minibeam path │
              └───────────┬────────────┘
                          ▼
                   TransportResult
                          │
                          ├── depth CSV（MeV / Gy / species / LET）
                          ├── sparse voxel CSV
                          ├── dense MHD + RAW（dose / LET）
                          └── stdout：能量账本、queue、kernel timing、诊断
```

## 1. 顶层目录与可信度

| 路径 | 角色 | 修改时的判断 |
|---|---|---|
| `include/carbon/` | 公共类型、API、几何/物理内联函数 | 接口和设备可用结构的第一事实源 |
| `src/` | 配置、输运、I/O、物理表实现 | 当前行为的第一事实源 |
| `src/detail/` | 两个 SYCL 输运 TU 共用的 include-only 实现片段 | 不能独立编译；修改会同时影响 legacy/minibeam |
| `config/` | 约百个水箱、异质体、CT、LET、SOBP、TPS、minibeam 配置 | 示例兼验证入口；并非每个都适合当前 build/profile |
| `data/` | 入库的 CSV 物理表、metadata、`packages/` reaction/cascade/neutral/soft-tissue 与 Copper package | 运行时资产；格式需和 loader 保持一致 |
| `tests/` | 单文件自建测试程序 | 当前未接入 CMake，不能把它等同于自动 CI |
| `validation/` | TOPAS 参数、Python/脚本、参考数据、报告、TPS 示例 | 离线生成、对照、gamma/绘图与回归；不承载 carbon_mc 默认 package |
| `startup/` | TOPAS database scorer extensions 与提取脚本 | 生成 GPU 物理数据库的上游工具，不参与 `carbon_mc` 构建 |
| `scripts/` | Linux/Windows 构建与运行包装 | 环境便利层，不定义核心物理 |
| `benchmark/` | A1–A12 TOPAS/GPU runner 及本地结果 | 当前工作树中的 runner/结果层；正式结论需看 manifest，目录当前未被 Git 跟踪 |
| `ct/` | 患者 CT/DICOM/计划及派生产物 | 大型病例资产；`.gitignore` 将新文件视为本地数据，即使已有历史 tracked 文件 |
| `build/`, `out/` | 编译目录与运行输出 | 被忽略的可再生产物，不要从中反推源码现状 |
| `.plot-*`, `.venv-*` | 本地 Python 绘图依赖 | 本地环境，不是项目依赖声明 |

顶层文档分工：

| 文档 | 用途 |
|---|---|
| `README.md` | 简要构建/运行介绍；其“slim core tree”叙述不覆盖当前工作树全部验证资产 |
| `TOPAS_GPU_Physics_Model.md` | TOPAS/Geant4 与 GPU 物理过程对应和限制 |
| `minibeamStructure.md` | minibeam 编译隔离、双 kernel、安全门和历史实现状态 |
| `minibeam.md` | minibeam 物理实现与逐阶段验证日志，长且带时间线 |
| `ctplan.md` | TOPAS Dij / matRad / GPU 计划与坐标变换工作流 |
| `ctResult.md` | CT dose/LET 证据状态与结论边界 |
| `futureStep.md` | LET、cascade、中性、性能的历史进展和后续计划 |
| `local30.md` | RT07575 局部 gamma 诊断日志，不是通用架构规范 |
| `MAIGO_TOPAS_GPU_Benchmark_CT_Match_Checklist.md` | 发文前 A1–A12 和 CT match 验收清单 |
| `BRANCH_WORKFLOW.md` | 默认在 `master` 开发的分支约定 |

## 2. 构建与二进制组成

### 2.1 当前目标

`CMakeLists.txt` 定义：

- `carbon_core`：静态/普通 CMake library，包含配置、数据表、几何、I/O、serial 后端，以及按开关加入的 SYCL 源。
- `carbon_mc`：唯一可执行程序，`src/main.cpp`，链接 `carbon_core`。

当前没有 `enable_testing()`、`add_executable(carbon_tests ...)` 或 `add_test(...)`。若任务涉及测试基础设施，需先明确是否要把现有测试重新接回 CMake。

### 2.2 编译开关

| 选项 | 默认 | 效果 |
|---|---:|---|
| `CARBON_ENABLE_SYCL` | OFF | 要求 IntelLLVM/icpx，定义 `CARBON_HAS_SYCL`，加入 SYCL device/transport |
| `CARBON_ENABLE_MINIBEAM` | OFF | 定义同名宏；SYCL 构建时加入 Copper kernel 与 dispatch |
| `CARBON_DOSE_FP32` | OFF | dose atomic 使用 FP32；否则 FP64 |
| `CARBON_ENABLE_TRANSPORT_PROFILE` | OFF | 定义 `CARBON_TRANSPORT_PROFILE`，启用低开销路径计数 |
| `CARBON_SYCL_TARGETS` | 空 | 传给 `-fsycl-targets`，例如 `spir64`、`nvptx64-nvidia-cuda` |
| `CARBON_CUDA_ARCH` | 空 | 可选 NVIDIA AOT 架构，例如 `sm_75` |

所有构建均编译 `src/transport_profile.cpp`；无 SYCL 时仍可保存空/禁用的 profile 结果。

### 2.3 SYCL 源矩阵

```text
CARBON_ENABLE_SYCL=OFF
  transport_cpu.cpp only

SYCL=ON, MINIBEAM=OFF
  transport_sycl_legacy.cpp 导出 transport_sycl()

SYCL=ON, MINIBEAM=ON
  transport_sycl_legacy.cpp  → transport_sycl_legacy()
  transport_sycl.cpp         → transport_sycl_minibeam()
  transport_sycl_dispatch.cpp→ transport_sycl() 按 config.enable_minibeam 分发
```

ON 构建中，`minibeam:false` 仍必须进入 legacy TU；这是避免 Copper 逻辑改变普通 CT kernel 寄存器压力和行为的核心隔离约束。

### 2.4 Presets

当前 `CMakePresets.json` 只有五个 configure/build preset：

| preset | 后端 | minibeam | dose atomic |
|---|---|---:|---|
| `cpu-debug` | serial Debug | OFF | FP64 定义未启用 |
| `oneapi-release` | SPIR-V + NVIDIA 双目标 | OFF | FP32 |
| `oneapi-nvidia-release` | NVIDIA CUDA plugin | OFF | FP32 |
| `oneapi-nvidia-minibeam` | NVIDIA CUDA plugin | ON | FP32 |
| `oneapi-intel-release` | Intel SPIR-V | OFF | FP64 |

常用命令：

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug

cmake --preset oneapi-nvidia-release
cmake --build --preset oneapi-nvidia-release

./build/oneapi-nvidia-release/carbon_mc \
  --config config/beam_200MeVu_letd.yaml --device cuda
```

## 3. 程序入口：`src/main.cpp`

### 3.1 CLI 生命周期

`main()` 的顺序不可随意重排：

1. 第一遍参数只寻找 `--config`，默认 `config/beam_200MeVu.yaml`。
2. `load_config()` 读取配置并在返回前调用一次 `validate()`。
3. 第二遍 CLI 覆盖 device、histories、seed、profile、CT、queue、spots、输出等字段。
4. 再次 `validate()`，因此 CLI 也受相同组合约束。
5. 装载主 stopping-power / cross-section 表，以及按 feature 开关装载 reaction、cascade、neutral package。
6. 按束流来源选择单次运行、TOPAS spots 或 TPS source。
7. 调用 serial/SYCL 后端。
8. 根据非空输出路径写 scorer；最后打印 backend tag、吞吐、kernel timing、能量账本和队列统计。

CLI 覆盖项包括：`--device`、`--histories`、`--random-seed`、`--physics-profile`、`--ct-grid`、CT SP scale、secondary/neutral queue capacity、可重复 `--spots`、spot weights、straggling scale、MeV/Gy/LET/MHD 输出、LET 开关、`--plan-only`、`--sequential-spots` 和 Dij 阈值诊断。

### 3.2 计划运行的三条路径

- 无 spots/TPS：直接运行一次 `TransportConfig`。
- `topas_spots_file(s)`：`TopasSpotPlan` 解析 L0–L14；可由外部权重重新分配 histories。SYCL 默认将所有 spot 压成一个 `primary_spot_batch`；serial 或 `--sequential-spots` 逐 spot 运行并用 `accumulate_transport_result()` 累加。
- `tpsSource:true`：`TpsSourcePlan` 从 CSV/单中心 spot 构造任意角源，按 MU 用 Hamilton/largest-remainder 精确分配 histories，然后强制 batched SYCL；serial 与 sequential 模式不支持。

`SyclTransportContext` 只在多 spot/TPS 路径建立，用于复用 queue 与不变的 device lookup/package allocations。普通单次运行让输运函数自行创建 queue。

`--plan-only` 只校验并打印 source bounds/方向，不启动 transport。Dij threshold 模式逐 spot 运行，在乘 optimizer weight 前按每 spot voxel Gy 阈值裁剪，只保留受支持的 dense voxel dose 流程。

## 4. 配置系统

### 4.1 解析模型

`src/config.cpp` 是手写解析器，不使用 YAML library。文件只是逐行 `key: value`：

- 支持去空白、布尔/数值、简单逗号列表和方括号列表。
- 不支持 YAML 层级、锚点、复杂对象或通用序列语义。
- `tpsSource`/`tps_source` 与 `scorerLET`/内部 snake-case 等少数公开别名由代码显式处理；未知能力不能按“YAML 应该支持”推断。
- 输出 path 的空字符串常表示禁用文件。

字段的完整默认值和注释在 `include/carbon/transport_config.hpp`；解析映射与组合校验在 `src/config.cpp`。添加配置项通常必须同时修改这两个文件，并视情况修改 CLI、kernel capture 和示例配置。

### 4.2 `TransportConfig` 的功能分区

| 分区 | 代表内容 |
|---|---|
| 基础输运 | histories、MeV/u、A、步长、相对能损、cutoff、seed |
| 几何 | 均匀水、轴向 slab、AABB insert、CCTG CT（互斥） |
| 束源 | flat、BiGaussian emittance、世界坐标 beam basis、能散 |
| 计划 | TOPAS files/weights/geometry transforms、TPS angles/isocenter/SAD/patient position |
| 带电物理 | straggling、Highland MCS、初级衰减、reaction、secondary、cascade |
| 中性物理 | first interaction/full continuation，或关闭输运时的 interim kerma proxy |
| 材料物理 | 四类/Schneider CT SP/XS、材料条件化 reaction/cascade、粒子特异 SP |
| scorer | depth、voxel、species、origin、LET moments、birth spectra、MHD |
| GPU 调度 | history chunk、secondary batch/persistent workers、queue capacity、显存比例 |
| minibeam | slit/Copper 几何、EM/MCS/straggling、核反应、Copper product tables、诊断 |
| 输出 | MeV/Gy CSV、MHD、LET、species、origin、birth-spectrum 路径及 dose scale |

### 4.3 当前 profile 事实

| 值 | 当前校验行为 |
|---|---|
| `accurate` | 内部默认；为旧配置/minibeam 兼容，具体开关仍由 YAML 决定 |
| `best` | 常规 CT 高精度公开档；强制 LET、粒子特异 SP、完整 attenuation/secondary/cascade、步长和 cutoff 上限 |
| `fast` | 常规 CT dose-only 档；强制 CT 与完整 charged chain，禁止 minibeam 和 LET |
| `medium` | **已移除，`validate()` 直接抛错**；kernel 内仍有遗留分支，不代表配置可用 |

不要照搬旧文档中“medium 可用”或“fast 可开 LET”的结论。当前源码明确拒绝二者。

### 4.4 关键组合约束

- layered phantom、hetero insert、CT grid 三选一。
- secondary transport 依赖 secondary generation；cascade 还依赖 transport 和 `maximum_cascade_generations > 0`。
- neutral transport 依赖 generation + charged transport；`full` 要至少两代。
- minibeam YAML 需要以 `CARBON_ENABLE_MINIBEAM=ON` 编译。
- charged-origin voxel 依赖 voxel scorer 和 secondary transport。
- voxel LET MHD 依赖 LET + voxel scorer。
- TPS source 当前只允许 carbon、要求 voxel scorer，并依赖 SYCL 主路径。
- queue overflow 不会自动变成“有效物理近似”；正式结果必须审计 stdout 中 secondary/cascade/neutral overflow。

## 5. 公共模块地图

| 模块 | 头文件 / 实现 | 责任 |
|---|---|---|
| Transport API | `transport.hpp`, `transport_cpu.cpp`, `transport_sycl*.cpp` | backend 入口、结果、SYCL context |
| Config | `transport_config.hpp`, `config.cpp` | 所有策略、默认值、解析、组合校验 |
| Particle/RNG | `particle.hpp`, `rng.hpp` | device-safe track structs、species/category、Philox counter RNG |
| Continuous physics | `stopping_power.*`, `straggling.hpp`, `multiple_scattering.hpp` | SP 插值、有效电荷/同位素比例、Bohr、Highland |
| Nuclear XS | `cross_section.*` | C-12、离子和 gamma/neutron lookup table |
| Event packages | `reaction_package.*`, `cascade_package.*`, `neutral_package.*` | 读取预编译相关末态和 projectile XS |
| Geometry | `slab_phantom.hpp`, `ct_grid.*`, `minibeam_collimator.hpp` | slab/insert、CCTG+DDA、slit/Copper 解析几何 |
| Plan/source | `topas_spots.*`, `spot_plan_geometry.*`, `tps_source.*` | spot 解析、权重分配、坐标变换、batch 构造 |
| I/O | `io.*` | MeV/Gy、LET、species、sparse CSV、dense MetaImage |
| Device | `device.*` | SYCL selector、backend pinning、设备描述、async handler |
| Profiling | `transport_profile.*` | CT face clamp/步数等计数与文本摘要 |

### 5.1 重要数据布局

- `PrimarySpotBatchEntry`：两个 history 边界、per-spot seed 和固定 `float[20]`。使用数组是为了绕开某些 SYCL host/device `offsetof` 不一致；不要轻易改字段布局。
- `SecondaryParticle3D`、`NeutralParticle3D` 及 summary structs 有 `static_assert(sizeof(...))`，是 host/device queue ABI。
- depth scorer 长度是 `ceil(phantom_length_mm / depth_bin_width_mm)`。
- voxel index 为 z-major、x fastest；CT 的线性 index 是 `x + nx*(y + ny*z)`。
- species/origin/LET 数组多为 category-major：`category * bins_or_voxels + index`。
- dose atomic 类型由 `CARBON_DOSE_FP32` 决定；结果回传统一为 `double` vector，不意味着设备累加是 FP64。

## 6. 几何和束流

### 6.1 Phantom 优先级

1. 均匀水是默认。
2. layered phantom 按 z 层切换密度，可选每层绝对 SP/XS 与辐射长度。
3. hetero insert 是水背景中的 AABB，可选 insert 绝对 SP/XS。
4. CT grid 是有限三维 voxel volume，使用 DDA/face clamp 处理材料边界。

### 6.2 CCTG 格式

`CtGrid` 读取 little-endian `CCTG` v1/v2/v3：

- 共同头：尺寸、origin、spacing，然后 density `float[]` 与 material id `uint8[]`。
- v1：四类 air/lung/water/bone。
- v2：附加每 section 的常数 mass stopping-power factor。
- v3：附加 `(Z/A)_section/(Z/A)_water` 与 mean excitation energy `I_eV`，运行时形成能量相关 mass-SP factor。

新写出的网格固定为 v3。`validation/scripts/prepare_ct_grid.py`、reorient/downsample 工具负责从 DICOM/中间网格准备 CCTG；transport 不读取 DICOM。

### 6.3 TOPAS 与 TPS 计划

- TOPAS spot parser 把 L0–L14 TimeFeature dump 当输入格式，但不模拟 timeline，只按文件顺序执行。
- `spots_geometry_mode` 支持 `topas`、`beam_plus_z`、`tps_90`、`tps_gantry_y`、`minibeam_topas_y`；这些模式把外部世界/患者坐标映射到 GPU 的 +z transport frame。
- TPS source 是另一条 opt-in 路径，支持每 spot 覆盖 gantry/couch/collimator，可按 `iec61217` 或仓库的 `topas_patient_rot_z` 约定生成 beam basis。
- 所有计划最终都变成相同的 `PrimarySpotBatchEntry[]`，所以 source 参数可在一个 primary kernel launch 中按 history range 查表。

## 7. 输运后端

### 7.1 Serial：`src/transport_cpu.cpp`

这是轻量一维参考子集：

- 只输运 C-12 primary；按 stopping power 选择步长。
- 可选 Bohr straggling、基于宏观 XS 的 primary attenuation。
- 维护 depth/可选 voxel 的简单能量沉积与能量账本。
- 不实现完整三维 MCS、reaction package 产物、charged secondary/cascade/neutral/TPS/minibeam。

因此 serial 与 SYCL 的用途不是完整物理等价；它适合表插值、步长、基础账本和小型 smoke。

### 7.2 Legacy SYCL：`src/transport_sycl_legacy.cpp`

这是普通水/CT/TPS 的生产主路径，也是仓库最大的核心文件之一。高层流水线：

```text
host load/convert tables
  → estimate device memory, clamp/allocate queues and scorers
  → copy immutable lookup/package data（context 可复用）
  → primary kernels（按 history chunk）
       CSDA + straggling + MCS + CT/material boundary
       nuclear attenuation + correlated reaction products
       depth/voxel/species/LET scoring
  → optional secondary bucket count/scatter
  → charged secondary + cascade kernels（按 batch/generation）
  → neutral kernels（first_interaction 或 continuation）
  → charged products born from neutrals 再输运
  → copy/reduce tallies → TransportResult
```

队列和 scorer 使用 SYCL USM。CUDA 后端还有 WSL-safe chunk/queue cap、显存预算估算和可选 clock warmup。不要只调大 queue：`max_device_memory_fraction` 会触发自动缩放，stdout 才是实际容量与 overflow 的事实源。

### 7.3 Minibeam SYCL：`src/transport_sycl.cpp`

该 TU 大体复制 legacy downstream transport，并在 primary 前加入 beamline：

```text
source
  → 圆柱 Copper collimator / 周期 slit 解析几何
  → absorbing_geometry 或 Copper EM 模式
  → 可选 Copper MCS / straggling / C-12 attenuation
  → 可选 reaction products 与 Copper neutral continuation
  → water/CT entrance survivor queues
  → 下游 charged / cascade / neutral / scorer
```

`MinibeamDiagnostics` 记录 incident/direct/touched/nuclear/survivor、按 slit 计数、物种/能谱和入口相空间 moments。关闭详细 diagnostics 仍保留全局 beamline removed energy，确保能量账本成立。

架构债：legacy 与 minibeam 两个超大 TU 含大量重复 downstream 代码，共享 `.inc` 只能降低部分漂移。修普通 CT bug 时必须检查是否也存在于 minibeam TU；修 Copper 专属逻辑则只应动 minibeam 路径。

### 7.4 `src/detail/` 的作用

- `sycl_device_math.inc`：设备数学/方向工具。
- `sycl_dose_atomic.inc`：FP32/FP64 dose atomic 抽象。
- `sycl_score_device.inc`：连续沉积、species/origin/LET/buildup/kerma 计分。
- `sycl_cascade_select.inc`：cascade/neutral 查找与能量/深度条件化选择。
- `sycl_cascade_host_lut.inc`：host 侧把 package XS 对齐到 transport energy grid。
- `sycl_transport_context_impl.inc` / `methods.inc`：queue 与 immutable device allocation 缓存。
- `sycl_profile.inc`：编译期可选路径计数。

这些文件通过 `#include` 注入两个 transport TU；不要把它们加入 CMake source list 单独编译。

## 8. 物理模型与 package

### 8.1 连续输运

- `StoppingPowerTable`：按 MeV/u 插值总离子 `dE/dx`（MeV/mm）。
- primary/fragment 步长由最大几何步长、最大相对能损、材料/CT face、scorer face 等共同 clamp。
- energy straggling 使用重粒子 condensed total-loss dispersion：在 Bohr 低速极限上加入 `Tmax/beta^2` 相对论因子，并将采样限制在 `0..min(2*meanLoss,E)`；它仍未显式输运 delta electron，也未完整复现 Geant4 的 Glandz/Gamma/Uniform regime。MCS 是 Highland projected RMS，可由材料 radiation length 和 scale 调整。
- 粒子特异模式用 `IonStoppingPowerTables` 中各 Z/A 相对 C-12 的比例及 delta-electron fraction；缺表物种回退到有效电荷缩放。

### 8.2 核反应链

```text
C-12 macro XS
  → primary attenuation
  → ReactionPackageTable：按 incident energy 采样相关一级末态
  → charged products 进入 SecondaryParticle3D queue
  → CascadePackageTable：按 projectile Z/A 与能量（可选参考深度）继续反应
  → neutron/gamma 进入 neutral queue
  → NeutralPackageTable：总截面、continuation、local deposit、带电产物
```

package 保存离线抽取的相关事件样本，不是解析核模型。关键二进制格式有 magic/version/offset/count，loader 在 `*_package.cpp` 做 bounds/version 校验。改变 struct 或编译脚本必须同步格式版本与 metadata。

当 neutral transport 关闭时，可用 `neutral_local_kerma_fraction` 等 proxy 计分部分中性能量；它与真实 neutral queue 互斥，不能把两者结果混为同一物理模式。

### 8.3 计分定义

- raw transport tally 是 deposited energy（MeV）。writer 依据几何质量换算 Gy，并应用仅输出层的 `dose_output_scale`。
- depth IDD 质量来自面积 × depth bin × 局部/配置密度；voxel Gy 使用每 voxel 局部 CT 密度。
- LET 保存 numerator/denominator 原始 moments，分别有 primary C-12、all hadron、charged species 和可选轻同位素；最终 LETd 由 writer 相除。
- `TransportResult::relative_energy_balance_error()` 用 initial、deposited、escaped、beamline removed、untracked/overflow 等账本项审计闭合。
- `backend` 字符串逐项追加实际 feature tag，是确认运行路径、dose 精度、材料模型与 scorer 的快速证据。

## 9. 数据与离线生成链

### 9.1 `data/`

主要资产：

- `stopping_power_{water,air,lung,bone,copper,...}.csv`
- `c12_inelastic_cross_sections_*.csv`
- `ion_stopping_power_*.csv`、`ion_cross_sections_copper_*.csv`
- `let_delta_electron_fraction_*.csv`
- `data/packages/*.bin`：water reaction/cascade/neutral 与 soft-tissue 运行时 package
- `data/copper_*.bin`：minibeam Copper reaction/neutral package
- 相邻 `.metadata.json` / `.compiled.json`：生成版本和编译参数

`TransportConfig` 默认 package path 指向 `data/packages/`。`validation/results/` 只保留离线生成中间产物、消融变体与对照输出，不再承载 carbon_mc 硬依赖 package。

### 9.2 `startup/`

`startup/extensions/*.cc/.hh` 是 TOPAS scorer extension：抽取材料属性、C-12/离子 stopping power、cross section、reaction/cascade/neutral 末态、dose origin 等。`run_topas_database_extraction.sh` 默认 dry-run，通过模板生成多个能量的 TOPAS 输入。

链路是：

```text
重新编译带 extensions 的 TOPAS
  → startup 脚本生成 .header/.phsp
  → validation/scripts/prepare_* / compile_* / merge_*
  → CSV lookup 或 versioned .bin package
  → config 引用 → carbon_mc runtime loader
```

## 10. I/O 契约

`src/io.cpp` 负责：

- depth deposited energy 和 Gy CSV；
- fragment species energy/Gy；
- LETd 及 raw numerator/denominator；
- selected fragment birth-spectrum 多文件前缀输出；
- sparse voxel/origin CSV；
- dense dose MHD+RAW 与 primary/all-hadron LET MHD+RAW。

writer 会创建父目录。MetaImage 的维度、spacing、offset 必须与 scorer/CT 对齐；TPS+CT 路径可能在 transport 内 rebase z origin，writer 再恢复患者坐标 metadata。修改轴序、index 或 MHD offset 时应同时检查 `CtGrid`、TPS transform、`voxel_masses_kg()` 和相关验证脚本。

## 11. 测试、验证与 benchmark

### 11.1 `tests/carbon_tests.cpp`

这是无测试框架的单一 `main()`，覆盖单位/插值、RNG、straggling/MCS、serial 能量闭合、package loader、CT/几何、source/spot/TPS、I/O，以及在 `CARBON_HAS_SYCL` 下的 CPU-SYCL、context、queue、CT、neutral smoke。

**当前它没有被 CMake 编译或 CTest 注册。** 看到历史文档中的 `carbon_tests`、`reorient_ct_grid_tps_90_tests`、isolation CTest 名称时，不要假设当前 checkout 可直接 `ctest`；先检查/恢复构建定义。

### 11.2 `validation/`

- `topas/`：约两百个 TOPAS 参数/extension 辅助文件，是 reference simulation 输入。
- `scripts/`：准备 CT/物理表/package、运行套件、转换 MHD/RTDOSE、比较 depth/3D dose/LET、gamma、绘图、消融和 isolation。
- `results/`：部分入库的 package、metadata、CSV/report；也可能与本地产物混居。
- `references/`：冻结的 minibeam isolation 参考。
- `tps/`：TPS source 说明与示例 spots CSV。

验证脚本没有统一 Python packaging/requirements；很多依赖本地 NumPy/SciPy/pydicom/matplotlib 环境和外部 TOPAS。运行前先读脚本 CLI、输入路径与 normalization，不能把相似脚本当成等价协议。

### 11.3 `benchmark/`

`A_README.md` 将 A1–A12 映射到 water IDD、MCS、energy spread、fragment、LET、heterogeneity、SOBP、spot batch、步长、统计、账本与性能。runner 通过环境变量指定 `TOPAS_BIN`、G4 data、`CARBON_MC` 和 device，并支持无副作用 `--dry-run`。当前初始策略不用 smoke 输入，单次 GPU/TOPAS 运行默认硬上限为 100k histories；只有确认统计不足后才通过 `BENCHMARK_MAX_HISTORIES` 显式提高。A7/A8 使用精确总和为 100k 的 21 层 L4 权重，禁止标量 histories override，避免误变成每层 100k。

当前整个 `benchmark/` 在工作树中显示为未跟踪，因此修改它不会自动成为项目提交的一部分；正式保存前需单独确认纳入范围。其 `formal_*` 输出、图片和 driver logs 是结果，不是 core 架构。

## 12. 常见修改应落在哪里

| 任务 | 首要文件 | 同步检查 |
|---|---|---|
| 新增 YAML key | `transport_config.hpp`, `config.cpp` | validate、CLI、示例 config、kernel capture |
| 改 CLI/spot 编排 | `main.cpp` | 两遍解析、batch/sequential 累加、输出 normalization |
| 改普通 CT/LET 物理 | `transport_sycl_legacy.cpp` | minibeam TU 是否有重复实现、共享 `.inc`、serial 是否应同步 |
| 改 Copper beamline | `transport_sycl.cpp`, `minibeam_collimator.hpp` | dispatch、MINIBEAM build gate、Copper tables/diagnostics |
| 改公共 scorer | `sycl_score_device.inc`, `transport.hpp`, `io.cpp` | 两个 SYCL TU、category layout、FP32/FP64 atomic |
| 改 reaction/cascade/neutral 格式 | 对应 `*_package.hpp/.cpp` | offline compiler、magic/version、metadata、static_assert |
| 改 CT 材料/边界 | `ct_grid.hpp`, 两个 SYCL TU | CCTG preparation、DDA、dose mass、MHD coordinates |
| 改 TOPAS plan transform | `topas_spots.*`, `main.cpp` | tps_90/y pack、upstream air、plan-only、CT axis |
| 改任意角 TPS | `tps_source.*` | patient orientation、SAD、voxel entry、batch ABI |
| 改 device 选择 | `device.cpp` | `ONEAPI_DEVICE_SELECTOR`、backend pin、async errors |
| 改 dose/LET 文件 | `io.*` | units、history normalization、local density、downstream scripts |
| 改 GPU 性能 | transport TU、`transport_profile.*` | physics equivalence、overflow、显存、kernel vs wall timing |

## 13. 必须保留的工程不变量

1. 默认配置和默认编译开关应保持 legacy 行为；新的近似必须显式 opt-in。
2. `MINIBEAM=ON + minibeam:false` 必须仍走 legacy kernel。
3. host/device queue struct 和 `PrimarySpotBatchEntry` 布局必须稳定。
4. CT material face 与必要的 scorer face clamp 不能因“加速”被无证据绕过。
5. raw MeV tally、Gy conversion 和 output-only dose scale 必须分层，避免破坏能量账本。
6. reaction/cascade/neutral package 的单位、reference density、energy conditioning 和版本必须和离线编译器一致。
7. 正式结果必须记录 commit/build preset、device、seed、histories、profile、scorer、package/table metadata、queue overflow 和 normalization。
8. FP32 atomic 的累加顺序通常不保证 bitwise 可复现；比较应使用约定的数值指标，不应默认逐字节相等。
9. `build/`、`out/`、患者数据和绘图虚拟环境不是源码；不要在清理或提交时误纳入。
10. 工作树存在用户改动时，修改架构相关文件前必须先查看 `git diff`，避免覆盖实验中的变更。

## 14. 新对话的推荐读取顺序

只需做一般代码任务时：

1. 本文件。
2. `CMakeLists.txt` 与相关 preset。
3. `include/carbon/transport_config.hpp`、`transport.hpp`。
4. 任务对应的 `.cpp`；若是 SYCL，再读其引用的相关 `src/detail/*.inc`。
5. 一个最接近任务的 `config/*.yaml`。

专项任务再追加：

- 物理一致性：`TOPAS_GPU_Physics_Model.md` + 对应 validation script/metadata。
- CT/计划：`ctplan.md`、`validation/tps/README.md`、相关 CCTG preparation script。
- minibeam：`minibeamStructure.md` 后再按需要查 `minibeam.md` 的具体阶段。
- 发文 benchmark：`MAIGO_TOPAS_GPU_Benchmark_CT_Match_Checklist.md` + `benchmark/A_README.md`。
- 历史指标：`ctResult.md` / `futureStep.md` / `local30.md`，但以当前源码和新跑结果复核。

## 15. 当前已知文档/架构差异

- 旧 `structure.md` 曾把 `medium` 列为可用；当前 `TransportConfig::validate()` 已明确移除它。
- 旧文档曾列出多项 CTest；当前 CMake 没有任何测试 target/注册。
- `README.md` 描述的是精简 core tree，但当前 checkout 同时包含大量 tracked validation/CT 资产和未跟踪 benchmark 结果。
- minibeam 与 legacy downstream kernel 仍是双大文件，不是完全抽取后的单一 water kernel 架构。
- 配置扩展名虽是 `.yaml`，解析器不是完整 YAML。

以上差异是新模型最容易误判的地方；发生冲突时，以当前头文件、`config.cpp::validate()`、CMake 和实际 backend tag 为准。

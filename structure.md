# MAIGO / carbon-oneapi-mc 代码架构

本文描述仓库**当前**源码结构、构建矩阵、输运路径与验证入口。  
更细的 minibeam 设计与安全合并条件见 [`minibeamStructure.md`](minibeamStructure.md)；  
使用与验证摘要见 [`README.md`](README.md)。

---

## 1. 架构概览

系统是 **配置驱动的 condensed-history 蒙特卡洛核心**，带 **CPU serial 子集** 与 **SYCL 全功能路径**，并用 TOPAS/Geant4 表与独立脚本做对照。

```text
CLI / YAML 配置 / TOPAS spots / TPS source
                    │
                    ▼
             TransportConfig
                    │
         ┌──────────┴──────────┐
         │  物理数据与几何加载   │
         │  SP / XS / packages │
         │  CT / slab / insert │
         └──────────┬──────────┘
                    ▼
         ┌──────────────────────┐
         │   Transport backend  │
         ├──────────────────────┤
         │  serial (CPU 子集)   │
         │  SYCL CPU / GPU      │
         │    ├─ legacy path    │  ← 水箱 / CT / LET / TPS
         │    │   ├─ accurate   │  ← 默认、已验证物理
         │    │   └─ fast       │  ← 显式启用、仅普通 CT dose
         │    └─ minibeam path  │  ← Copper 准直器 beamline（可选编译）
         └──────────┬───────────┘
                    ▼
             TransportResult
                    │
                    ▼
      depth / voxel / LET / fragment / neutral / ledger
```

### 1.1 仓库目录

```text
MAIGO/
├── CMakeLists.txt / CMakePresets.json   构建与预设
├── include/carbon/                      公共 API 与头文件
├── src/                                 核心实现
│   ├── main.cpp                         CLI 入口
│   ├── config.cpp                       配置解析与校验
│   ├── transport_cpu.cpp                serial 输运
│   ├── transport_sycl_legacy.cpp        master 兼容 SYCL 内核（水/CT/LET）
│   ├── transport_sycl.cpp               minibeam Copper 路径（可选）
│   ├── transport_sycl_dispatch.cpp      ON 时 runtime 分发
│   ├── detail/                          共享 SyclTransportContext Impl
│   ├── device.cpp / io.cpp / …          设备选择、写出、物理表
│   └── topas_spots.cpp / tps_source.cpp 束流计划
├── config/                              束流与验证 YAML
├── data/                                SP / XS / Copper / 同位素表
├── tests/                               carbon_tests
├── validation/
│   ├── scripts/                         对照、转换、isolation 回归
│   ├── references/                      冻结 isolation 参考（入库）
│   ├── results/                         本地/大文件结果（gitignore）
│   └── topas/                           TOPAS 配置与 scorer 扩展
├── scripts/                             构建/运行辅助
├── ct/                                  本地 CT/DICOM（gitignore）
├── out/                                 本地输出（gitignore）
├── structure.md                         本文
├── minibeamStructure.md                 minibeam 隔离设计
├── minibeam.md                          minibeam 物理与验证笔记
└── README.md / futureStep.md / BRANCH_WORKFLOW.md
```

---

## 2. 构建结构

### 2.1 目标

| 目标 | 作用 |
|------|------|
| `carbon_core` | 配置、物理表、serial/SYCL 输运、I/O |
| `carbon_mc` | 命令行可执行程序 |
| `carbon_tests` | 无第三方依赖的单元/集成测试 |

### 2.2 编译开关

| CMake 选项 | 默认 | 含义 |
|------------|------|------|
| `CARBON_ENABLE_SYCL` | OFF | 编译 SYCL 后端与 `device.cpp`；定义 `CARBON_HAS_SYCL` |
| `CARBON_ENABLE_MINIBEAM` | **OFF** | 编译 Copper minibeam 路径 + dispatch；定义 `CARBON_ENABLE_MINIBEAM` |
| `CARBON_DOSE_FP32` | OFF* | SYCL 剂量 scorer 用 float atomic（*NVIDIA 预设默认 ON） |
| `CARBON_ENABLE_TRANSPORT_PROFILE` | OFF | 步级 profile 计数 |
| `CARBON_SYCL_TARGETS` | 空 | 如 `nvptx64-nvidia-cuda` 或 `spir64,nvptx64-nvidia-cuda` |
| `CARBON_CUDA_ARCH` | 空 | 可选 AOT，如 `sm_75` |

\* 项目面向消费级 NVIDIA 时，预设 `oneapi-nvidia-*` / `oneapi-release` 默认打开 FP32 dose。

### 2.3 SYCL 源文件矩阵

| 构建 | 编译的输运源 | 导出入口 |
|------|----------------|----------|
| SYCL + **MINIBEAM=OFF** | `transport_sycl_legacy.cpp` | `transport_sycl` ≡ legacy |
| SYCL + **MINIBEAM=ON** | `transport_sycl_legacy.cpp` + `transport_sycl.cpp` + `transport_sycl_dispatch.cpp` | `transport_sycl` 按 YAML 分发 |

共享设备上下文：

- `src/detail/sycl_transport_context_impl.inc` — `SyclTransportContext::Impl`
- `src/detail/sycl_transport_context_methods.inc` — 构造/析构（仅 legacy TU 在 ON 时定义方法，`CARBON_DEFINE_SYCL_CONTEXT`）

```text
                 CARBON_ENABLE_MINIBEAM=OFF
                 ─────────────────────────
                 transport_sycl_legacy.cpp
                          │
                          ▼
                   transport_sycl()


                 CARBON_ENABLE_MINIBEAM=ON
                 ─────────────────────────
   transport_sycl_dispatch.cpp
            │
            ├── minibeam:false ──► transport_sycl_legacy()
            └── minibeam:true  ──► transport_sycl_minibeam()
```

### 2.4 推荐 presets（节选）

| Preset | 角色 |
|--------|------|
| `cpu-debug` | 无 SYCL，调试 |
| `oneapi-nvidia-release` | NVIDIA OFF，**FP32**，legacy only |
| `oneapi-nvidia-minibeam` | NVIDIA ON，**FP32**，双 kernel |
| `oneapi-nvidia-release-fp64` / `oneapi-nvidia-minibeam-fp64` | 可选 FP64 精度 |
| `oneapi-release` | 双目标 + FP32 |
| `oneapi-intel-release` | 仅 SPIR-V |

CMake presets schema version **3**（兼容 CMake 3.22）。

---

## 3. 程序入口与数据流

入口：`src/main.cpp`。

1. 解析 `--config` / CLI 覆盖项 → `TransportConfig`
2. `TransportConfig::validate()`（含 minibeam 编译期门控）
3. 加载 SP / XS / reaction / cascade / neutral 等表
4. 可选 CT grid、TOPAS spots、TPS source、spot batch
5. `transport_serial` 或 `transport_sycl`
6. 写出 depth / voxel MHD / LET / fragment / diagnostics

配置文件扩展名为 `.yaml`，解析器是 **自定义 key-value**（非完整 YAML 库）：`src/config.cpp`。

```text
main
  ├── load_config / CLI
  ├── validate()
  ├── load physics tables
  ├── load geometry / spots / TPS
  ├── transport_serial  OR  transport_sycl
  └── write scorers
```

---

## 4. 核心接口

### 4.1 `TransportConfig`

定义于 `include/carbon/transport_config.hpp`，集中存放：

- 束流、histories、种子、步长
- `physics_profile: accurate|fast` 输运策略（默认 `accurate`）
- slab / insert / CT 几何
- 次级、级联、中性、LET、voxel scorer
- GPU 批大小 / `secondary_persistent_workers` 等
- **minibeam** 准直器与 Copper 表路径
- 独立门控优化项（默认 legacy 关闭）：
  - `secondary_fp32_energy_residual`
  - `robust_boundary_nudge`
  - `secondary_condensed_step_mm`
  - `electronic_buildup_lateral_sigma_mm`
  - `dose_output_scale`

### 4.2 `TransportResult`

定义于 `include/carbon/transport.hpp`：

- depth / voxel / charged-origin / LET moments
- 核反应、级联、中性队列统计与能量账本
- kernel 计时字段（`primary_kernel_seconds` 等）
- 可选 `MinibeamDiagnostics`

### 4.3 模块表

| 模块 | 职责 | 主要位置 |
|------|------|----------|
| Config | 解析与校验 | `config.cpp`, `transport_config.hpp` |
| Stopping power / XS | 水与材料表 | `stopping_power.*`, `cross_section.*` |
| Reaction / cascade / neutral packages | 二进制末态包 | `*_package.*` |
| CT grid | CCTG 读写与材料 | `ct_grid.*` |
| Spots / TPS | TOPAS plan、TPS 几何 | `topas_spots.*`, `tps_source.*` |
| MCS / straggling / RNG | 物理工具 | `multiple_scattering.hpp`, `straggling.hpp`, `rng.hpp` |
| Minibeam 几何 | 狭缝/铜准直纯函数 | `minibeam_collimator.hpp` |
| I/O | CSV / MHD / 稀疏 dose | `io.*` |
| Device | SYCL queue 选择 | `device.*` |

---

## 5. 输运后端

### 5.1 Serial CPU（`transport_cpu.cpp`）

子集：CSDA、straggling、初级衰减。  
**无**完整 MCS / 次级 / 级联 / 中性 / 完整 CT。  
用途：轻量调试与能量账本子集对照。

### 5.2 SYCL legacy accurate（`transport_sycl_legacy.cpp`）

全功能 **水箱 / CT / TPS / LET** 路径（与隔离前 master 内核同源快照 + 共享 context）：

```text
Primary kernel
  → continuous loss / straggling / MCS / nuclear
  → secondary queue
Secondary / cascade kernel
  → fragment transport / cascade
Neutral kernel（可选）
  → result reduction
```

`physics_profile` 未设置时为 `accurate`，因此已有 YAML、CT case 和
minibeam isolation 的行为不变。

### 5.3 普通 CT fast profile

`physics_profile: fast` 是普通 CT 物理剂量的显式快速档，仍调用 legacy
SYCL 内核，但使用独立、可审计的策略。它不是 minibeam 的低精度模式。

```text
YAML / CLI
  → validate profile and feature compatibility
  → CUDA primary chunk 16k（减少 host submit / wait）
  → primary physics 不变
  → charged-secondary step 上限 1 mm
  → ≤2 MeV 短程 charged secondary 在当前 voxel 局部沉积
  → CT material face / dose voxel face / energy-loss limit 仍然 clamp
  → result.backend 追加 +physics-fast
```

其中次级步长取
`max(maximum_step_mm, secondary_condensed_step_mm, 1 mm)`；低能局部沉积阈值取
`max(secondary_local_deposit_cutoff_MeV, 2 MeV)`。以下安全门由
`TransportConfig::validate()` 强制执行：

| 条件 | 行为 |
|------|------|
| 未设置 profile | `accurate`，保持原行为 |
| `fast` + 普通 CT dose | 允许 |
| `fast` + minibeam | **拒绝配置** |
| `fast` + LET scorer | **拒绝配置** |
| `fast` + 非 CT 几何 | **拒绝配置** |
| 未知 profile | **拒绝配置** |

fast 不允许用缩小 queue 偷取速度。正式结果仍要求
`Secondary queue overflow: 0` 和 `Cascade queue overflow: 0`。

### 5.4 SYCL minibeam（`transport_sycl.cpp`，仅 MINIBEAM=ON）

在 legacy 水中/CT 输运之外增加 **Copper beamline**：

```text
Source
  → air / slit / Copper EM+nuclear
  → water-entrance phase space
  → charged/neutral product queues
  → 下游水/CT 输运与 scorer
```

`minibeam: false` 时 **不得**进入该路径：由 dispatch 调用 `transport_sycl_legacy`。

### 5.5 后端能力对比

```text
                        Serial   accurate   CT fast   minibeam
CSDA / straggling         ✓         ✓          ✓          ✓
MCS / secondaries         ✗         ✓          ✓          ✓
Cascade / neutral         ✗         ✓          ✓          ✓
CT / TPS dose             有限       ✓          ✓          ✓
LET                       ✗         ✓          ✗          ✓
Copper collimator         ✗         ✗          ✗          ✓
```

---

## 6. Minibeam 编译与运行矩阵

| 构建 | YAML | 路径 |
|------|------|------|
| MINIBEAM=OFF | 普通 CT/水 | legacy |
| MINIBEAM=OFF | `minibeam: true` | **配置报错** |
| MINIBEAM=ON | `minibeam: false` | **legacy**（无 Copper 寄存器压力） |
| MINIBEAM=ON | `minibeam: true` | minibeam kernel |

正式 minibeam 配置需**显式**打开 non-legacy 优化（见 `config/beam_minibeam_*.yaml`），例如：

```yaml
minibeam: true
secondary_fp32_energy_residual: true
robust_boundary_nudge: true
secondary_condensed_step_mm: 0.25   # 若使用
electronic_buildup_lateral_sigma_mm: 0.5  # 若使用
```

设计细节、验收门与历史 perf 数据：[`minibeamStructure.md`](minibeamStructure.md)。

---

## 7. 几何、束流与计分（摘要）

- **几何**：均匀水、轴向 slab、hetero insert、CT CCTG v2/v3  
- **束流**：emittance、TOPAS multi-spot batch、TPS 90°、可选 `tpsSource`  
- **计分**：depth dose、dense MHD、charged-origin、LET_d、birth spectrum、minibeam diagnostics  

配置样例见 `config/`；minibeam 样例为 `config/beam_minibeam_*.yaml`。

普通 CT 快速档可通过 YAML 或 CLI 启用：

```yaml
physics_profile: fast
```

```bash
carbon_mc --config config/beam_ct_20022516_1M.yaml \
  --physics-profile fast --no-scorer-let
```

正式 TOPAS match、LET、minibeam 和新病例验收仍使用默认的 `accurate`；
`fast` 适合优化迭代、统计预跑和已经做过 accurate 交叉验证的常规 CT dose。

---

## 8. 物理数据资产

| 类型 | 位置 |
|------|------|
| 水/骨/肺/空气/铜 SP | `data/stopping_power_*.csv` |
| C-12 / 离子 / 中性 XS | `data/*cross_sections*.csv` |
| Reaction / cascade / neutral packages | `validation/results/*.bin`（部分入库规则见 `.gitignore`） |
| Copper reaction / neutral packages | `data/copper_*.bin` |
| 表说明 | `data/README_physics_tables.md` |

---

## 9. 测试与 isolation 回归

### 9.1 默认 CTest

| 测试名 | 内容 |
|--------|------|
| `carbon_tests` | 单元 + SYCL smoke（有 GPU 时） |
| `reorient_ct_grid_tps_90_tests` | CT 重定向几何 |
| `dose_mapping_geometry_tests` | dose 映射几何 |
| `minibeam_isolation_log_parser` | isolation 脚本 log 解析（无 GPU） |

### 9.2 三门 isolation（可选）

脚本：`validation/scripts/regression_minibeam_isolation.py`

| 门 | 含义 |
|----|------|
| Gate 1 | MASTER 兼容二进制 vs MINIBEAM=OFF（同 dose 精度） |
| Gate 2 | ON + `minibeam:false` vs OFF（legacy 路径 + 性能） |
| Gate 3 | ON + `minibeam:true`：backend、非零 diagnostics、MHD、冻结 179.17 MeV/u 中心轴剂量 |

冻结参考：

```text
validation/references/minibeam_isolation/
  gate3_179p17_copper_em_10k_central_depth.csv
  gate3_179p17_copper_em_10k.metadata.json
```

启用 CTest 门（缺任一路径 configure 失败）：

```bash
cmake -S . -B build/iso \
  -DCARBON_RUN_ISOLATION_GATES=ON \
  -DCARBON_MC_MASTER=/path/master_fp32/carbon_mc \
  -DCARBON_MC_OFF=/path/off_fp32/carbon_mc \
  -DCARBON_MC_ON=/path/on_fp32/carbon_mc
ctest -L isolation --output-on-failure
```

**注意**：默认 NVIDIA 为 FP32 dose atomic，Gate1/2 以 **steps/反应数一致 + 紧 L1** 为准，不强制 CSV bitwise（FP32 atomic 顺序非确定）。FP64 配对可加 `--g1-bitwise --g2-bitwise`。

性能门使用 **`Kernel time: primary=… secondary=…` 中位数**（warmup + 重复），不用进程 wall time。

### 9.3 CT fast A/B 验证

2026-07-28 在 TITAN RTX / CUDA 12.6 上，以相同随机种子、相同 20022516
肺部 full-plan spot 权重和 1,000,000 histories 配对：

| 指标 | accurate | fast | 变化 |
|------|----------|------|------|
| wall time | 5.123 s | 2.672 s | **1.92×** |
| throughput | 195,192 h/s | 374,287 h/s | **+91.8%** |
| primary kernel | 3.768 s | 1.508 s | 2.50× |
| secondary kernel | 0.697 s | 0.573 s | 1.22× |
| secondary steps | 1.451 B | 1.112 B | −23.4% |
| secondary / cascade overflow | 0 / 0 | 0 / 0 | 无丢粒子 |

三维剂量网格为 `0.5 × 2.0 × 0.5 mm`，以 accurate 为 reference，10% dose
threshold，固定抽样 20,000 个 reference voxels：

| 剂量指标 | fast vs accurate |
|----------|------------------|
| integral ratio | 1.000010 |
| high-dose NRMSE / reference max | 0.168% |
| global gamma 1% / 1 mm | 99.950% |
| local gamma 1% / 1 mm | 96.285% |
| global gamma 2% / 2 mm | 99.995% |
| local gamma 2% / 2 mm | 99.915% |

该数据只证明 fast 对当前病例的 accurate 等价程度，不替代跨病例 TOPAS 验证。
水立方 1M、无 queue overflow 的独立 A/B 中，总速度为 1.13×、secondary
kernel 为 1.55×，1D global gamma 1%/1 mm 与 2%/2 mm 均为 100%。

---

## 10. 关键代码路径速查

| 主题 | 路径 |
|------|------|
| CLI | `src/main.cpp` |
| 配置 | `src/config.cpp`, `include/carbon/transport_config.hpp` |
| Serial | `src/transport_cpu.cpp` |
| Legacy SYCL | `src/transport_sycl_legacy.cpp` |
| Accurate / fast 策略与安全门 | `src/config.cpp`, `src/transport_sycl_legacy.cpp` |
| Minibeam SYCL | `src/transport_sycl.cpp` |
| 分发 | `src/transport_sycl_dispatch.cpp` |
| 设备 | `src/device.cpp` |
| 写出 | `src/io.cpp` |
| 构建 | `CMakeLists.txt`, `CMakePresets.json` |
| Isolation | `validation/scripts/regression_minibeam_isolation.py` |

---

## 11. 维护约定

1. **默认生产 CT/LET 构建**：`CARBON_ENABLE_MINIBEAM=OFF`，走 legacy，不编 Copper。  
2. **准确性默认值**：新增近似必须保持中性默认值；未设置 `physics_profile` 必须等价于 `accurate`。
3. **fast 边界**：仅普通 CT dose；不得绕过 material/dose boundary，也不得依赖 queue overflow。
4. **改水/CT/LET 物理**：改 `transport_sycl_legacy.cpp`（OFF 与 ON+`minibeam:false` 共用）。
5. **改 Copper beamline**：改 `transport_sycl.cpp` 与相关表/配置。
6. **legacy 与 master 快照漂移风险**：长期应将 Copper 完全拆出单一 water kernel；短期 dual 文件需同步修 bug。
7. **分支**：日常开发默认 `master`（见 `BRANCH_WORKFLOW.md`）。

---

## 12. 相关文档

| 文档 | 内容 |
|------|------|
| [`README.md`](README.md) | 能力、构建、运行、验证摘要 |
| [`minibeamStructure.md`](minibeamStructure.md) | 编译隔离设计、门控、验收标准 |
| [`minibeam.md`](minibeam.md) | minibeam 物理与 TOPAS 对照笔记 |
| [`futureStep.md`](futureStep.md) | 后续 LET/中性/性能计划 |
| [`BRANCH_WORKFLOW.md`](BRANCH_WORKFLOW.md) | 分支约定 |
| [`docs/archive/`](docs/archive/) | 历史开发日志 |

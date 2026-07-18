# CarbonGPU 代码架构

## 1. 架构概览

CarbonGPU 当前采用 **配置驱动的 Monte Carlo 输运核心 + CPU/SYCL 双后端 + TOPAS 独立验证体系**。

```text
CLI / 配置文件 / TOPAS spots
             │
             ▼
      TransportConfig
             │
      ┌──────┴─────────┐
      │ 物理数据资产加载 │
      │ SP / XS / CT   │
      │ reaction bins  │
      └──────┬─────────┘
             ▼
    ┌───────────────────┐
    │ Transport Backend │
    ├───────────────────┤
    │ Serial CPU        │
    │ SYCL CPU / GPU    │
    └─────────┬─────────┘
              ▼
       TransportResult
              │
              ▼
 Dose / LET / voxel / fragment
 neutral / reaction / energy ledger
```

主要目录职责：

```text
carbonGPU_grok/
├── CMakeLists.txt           构建配置
├── include/carbon/          公共接口、数据结构与物理工具
├── src/                     核心实现和命令行入口
├── config/                  模拟与束流配置
├── data/                    stopping power、截面等物理数据
├── tests/                   单元测试和集成测试
├── tools/                   数据生成与辅助工具
├── validation/
│   ├── scripts/             验证、转换、比较和远程运行脚本
│   ├── topas/               TOPAS 配置、自定义 scorer 和 spot plan
│   └── results/             规范化参考数据、指标与报告
└── scripts/                 构建和运行辅助脚本
```

## 2. 构建结构

CMake 构建三个主要目标：

- `carbon_core`：输运、物理模型、数据加载和输出。
- `carbon_mc`：命令行可执行程序。
- `carbon_tests`：单元测试和集成测试。

SYCL 是编译时可选功能，由 `CARBON_ENABLE_SYCL` 控制。启用后才会编译设备管理和 SYCL 输运实现，并定义 `CARBON_HAS_SYCL`。

关键位置：

- `CMakeLists.txt:1-55`

## 3. 程序入口与执行流程

主入口位于 `src/main.cpp:205-479`，总体流程如下：

1. 加载默认配置 `config/beam_200MeVu.yaml`。
2. 解析命令行参数并覆盖配置。
3. 加载 stopping-power、核截面、反应、级联和中性粒子数据。
4. 可选加载 CT grid、材料表和 TOPAS spot plan。
5. 选择 serial 或 SYCL 后端。
6. 执行一个束流或逐 spot 执行 SOBP 计划。
7. 汇总 `TransportResult`。
8. 写出 depth-dose、LET、voxel dose、fragment 和 reaction 等结果。

```text
main
  │
  ├── load_config
  ├── apply CLI overrides
  ├── load physics data
  │     ├── stopping power
  │     ├── cross sections
  │     ├── reaction package
  │     ├── cascade package
  │     ├── neutral package
  │     └── CT/material data
  │
  ├── load optional spot plan
  │
  ├── transport_serial
  │          或
  ├── transport_sycl
  │
  └── write scorers/results
```

配置文件使用 `.yaml` 后缀，但当前实现是自定义 key-value 解析器，而不是完整 YAML parser：

- `src/config.cpp:377-519`
- `src/config.cpp:199-375`

## 4. 核心接口和数据结构

### 4.1 TransportConfig

`TransportConfig` 是模拟输入的集中定义，包含：

- 束流参数。
- histories 和随机种子。
- 步长和输运限制。
- slab、insert 和 CT 几何。
- stopping-power 和核反应数据路径。
- 多重散射、straggling、级联和中性粒子开关。
- voxel、fragment、reaction 等 scorer 配置。
- CPU/GPU 执行相关参数。

位置：

- `include/carbon/transport_config.hpp:13-181`

配置解析完成后，由 `TransportConfig::validate()` 检查参数范围和模型组合是否合法：

- `src/config.cpp:199-375`

### 4.2 TransportResult

`TransportResult` 是输运后端的统一结果容器，主要包含：

- depth-dose 和 LET 数据。
- voxel scorer。
- 粒子与碎片统计。
- 反应和中性粒子统计。
- 入射、沉积、逃逸和剩余能量账本。
- 队列溢出等运行状态。

位置：

- `include/carbon/transport.hpp:16-73`

统一后端接口：

- `transport_serial(...)`
- `transport_sycl(...)`

位置：

- `include/carbon/transport.hpp:75-93`

### 4.3 基础领域模块

| 模块 | 职责 | 位置 |
|---|---|---|
| Particle | 粒子类型、谱系和统计结构 | `include/carbon/particle.hpp:8-139` |
| SlabPhantom | 分层幻影和材料区域 | `include/carbon/slab_phantom.hpp:9-178` |
| CtGrid | CT voxel 数据和材料参数 | `include/carbon/ct_grid.hpp:10-55` |
| MultipleScattering | 多重库仑散射 | `include/carbon/multiple_scattering.hpp:6-46` |
| Straggling | 能量涨落 | `include/carbon/straggling.hpp:6-40` |
| RNG | 并行可复现随机数 | `include/carbon/rng.hpp:6-85` |

## 5. 输运后端

### 5.1 Serial CPU

Serial 后端位于 `src/transport_cpu.cpp:36-198`，主要支持：

- CSDA 连续能损。
- energy straggling。
- primary attenuation。
- 基础能量统计。

Serial 后端并不是完整物理后端。例如，多重散射会被明确拒绝，也没有完整的二级带电粒子、级联和中性粒子输运链路。

其主要作用是：

- 提供轻量 CPU 执行路径。
- 验证基础连续能损和能量账本。
- 作为 SYCL 共同功能子集的参考实现。

### 5.2 SYCL CPU/GPU

SYCL 后端位于 `src/transport_sycl.cpp:1201-3210`，是当前全功能输运路径。

主要阶段：

```text
Primary particle kernel
          │
          ├── continuous energy loss
          ├── energy straggling
          ├── multiple scattering
          ├── nuclear interaction
          └── enqueue secondary particles
          │
          ▼
Charged secondary / cascade kernel
          │
          ├── fragment transport
          ├── cascade generation
          └── enqueue neutral particles
          │
          ▼
Neutral particle kernel
          │
          ▼
Result collection and scorer reduction
```

运行时通过 `--device serial|cpu|gpu|default` 选择执行路径：

- `serial`：调用 `transport_serial`。
- `cpu`：创建 SYCL CPU queue。
- `gpu`：创建 SYCL GPU queue。
- `default`：使用默认 SYCL selector。

关键位置：

- `src/main.cpp:175-201`
- `src/device.cpp:11-38`

SYCL 后端还会根据设备内存预算调整粒子队列容量：

- `src/transport_sycl.cpp:333-528`

### 5.3 后端不对称性

Serial 和 SYCL 后端共享 `TransportConfig`、`TransportResult` 及部分物理工具，但物理能力并不完全一致：

```text
                     Serial    SYCL
CSDA                   ✓         ✓
Energy straggling      ✓         ✓
Primary attenuation    ✓         ✓
Multiple scattering    ✗         ✓
Charged secondaries    ✗         ✓
Cascade                ✗         ✓
Neutral transport      ✗         ✓
完整 CT/voxel 路径      有限       ✓
```

因此 Serial 不能作为 SYCL 全功能路径的逐功能 oracle。

## 6. 几何与材料模型

当前支持三类异质模型：

1. slab 分层模型。
2. insert 局部材料模型。
3. CT voxel 模型。

配置位置：

- `include/carbon/transport_config.hpp:26-55`
- `include/carbon/transport_config.hpp:89-120`

这些模型最终在 SYCL kernel 中参与当前位置到材料属性的映射：

- `src/transport_sycl.cpp:1342-1474`
- `src/transport_sycl.cpp:2177-2299`

CT 使用自定义二进制格式，支持 v1、v2 和 v3。v3 增加 `(Z/A)_rel` 和 `I_eV`，用于 Schneider mass stopping-power 计算：

- `include/carbon/ct_grid.hpp:10-55`
- `src/ct_grid.cpp:42-157`

模型组合和优先级主要由 `TransportConfig::validate()` 约束。CT、insert、slab、绝对材料表和 mass-SP 之间存在组合限制，维护这些限制时需要同时检查配置校验和 kernel 分支。

## 7. 束流与 SOBP

束流可以直接由配置指定，也可以通过 TOPAS spot plan 输入。

`TopasSpotPlan::from_file()` 解析 `Tf`、`Scatterer1` 和 `L0-L14` 等字段：

- `include/carbon/topas_spots.hpp:11-71`
- `src/topas_spots.cpp:158-256`

`main` 逐 spot 运行输运，并将各 spot 的 scorer 累加：

- `src/main.cpp:282-330`

```text
TOPAS spot plan
  ├── spot 1 ──► transport ──► partial result
  ├── spot 2 ──► transport ──► partial result
  ├── ...
  └── spot N ──► transport ──► partial result
                                  │
                                  ▼
                         accumulated result
```

因此 SOBP 是建立在单束流输运之上的调度和累加层，而不是独立的输运引擎。

## 8. 物理数据资产

物理模型通过外部文件资产注入核心程序：

- stopping-power CSV。
- inelastic cross-section CSV。
- reaction binary package。
- cascade binary package。
- neutral binary package。
- CT binary grid。
- TOPAS spot plan。

默认 stopping-power 表由开发辅助脚本生成：

- `tools/generate_stopping_power.py:2-66`

TOPAS 截面数据通过验证脚本规范化：

- `validation/scripts/prepare_topas_cross_sections.py:72-182`

二进制 package 具有 magic、version、endian 和尺寸校验：

- `src/reaction_package.cpp:73-195`
- `src/cascade_package.cpp:65-183`
- `src/neutral_package.cpp:57-166`

这种设计使物理数据生成与输运核心解耦，但格式升级需要同步维护：

1. 数据生成脚本。
2. 二进制加载器。
3. metadata。
4. 测试和验证资产。

## 9. 输出与 scorer

输运结果通过 writer 层输出，主要包括：

- depth-dose。
- LET。
- voxel dose。
- fragment spectra/statistics。
- reaction records。
- neutral statistics。
- 能量守恒账本。

整体数据方向：

```text
Transport kernels
       │
       ▼
TransportResult
       │
       ├── depth-dose writer
       ├── LET writer
       ├── voxel scorer writer
       ├── fragment writer
       ├── reaction writer
       └── summary / energy ledger
```

集中式 `TransportResult` 便于统一输出和检查能量守恒，但也使结果结构随着 scorer 增加而持续扩大。

## 10. 验证体系

`validation/` 构成相对独立的参考验证系统：

```text
TOPAS simulation
      │
      ├── custom scorer extensions
      ▼
raw phsp / csv / log
      │
      ├── prepare / normalize scripts
      ▼
normalized CSV / binary packages
      │
      ├── comparison scripts
      ▼
metrics / plots / markdown results
```

### 10.1 TOPAS 自定义 scorer

- 反应 ntuple：`validation/topas/extensions/CarbonReactionNtuple.cc:11-125`
- 截面 ntuple：`validation/topas/extensions/CarbonCrossSectionNtuple.cc:15-103`
- 扩展构建：`validation/topas/build_extensions.sh:1-29`

### 10.2 数据规范化和比较

- 反应数据准备：`validation/scripts/prepare_topas_reactions.py:113-170`
- depth-dose 比较：`validation/scripts/compare_depth_dose.py:88-205`
- TOPAS voxel CSV 规范化：`validation/scripts/normalize_topas_csv.py:13-72`

### 10.3 验证结果

`validation/results/` 同时承担以下职责：

- 规范化参考数据。
- binary package 测试资产。
- 比较指标和图表。
- 验证报告。
- 来源和物理配置 metadata。

这使验证结果具有较好的可追溯性，但测试也会依赖部分 validation 资产。

## 11. 测试结构

主要测试集中在 `tests/carbon_tests.cpp:133-1090`，覆盖：

- 单位换算。
- RNG 和 Philox 可复现性。
- energy straggling。
- Serial 输运和能量守恒。
- primary attenuation。
- reaction、neutral 和 cascade package 加载。
- dose writer。
- TOPAS spot plan 解析。
- SYCL CPU 与 Serial 共同子集对齐。
- secondary queue 和 cascade smoke test。
- CT 输运 smoke test。

`carbon_tests` 直接链接 `carbon_core`：

- `CMakeLists.txt:47-55`

测试体系兼有单元测试、数据格式测试和端到端集成测试。

## 12. 依赖方向

当前主要依赖方向可以概括为：

```text
main / CLI
    │
    ▼
config + data loaders + spot plan
    │
    ▼
transport public API
    │
    ├── serial implementation
    └── SYCL implementation
           │
           ├── geometry/material helpers
           ├── physics helpers
           ├── RNG
           └── binary physics packages
    │
    ▼
TransportResult
    │
    ▼
I/O writers
```

验证系统从外部生成和比较数据，不应反向依赖核心模拟器实现细节；但部分规范化 binary asset 会作为核心程序和测试的运行时输入。

## 13. 架构优点

1. 配置、输运和结果边界较明确。
2. CPU 与 GPU 共享领域数据结构和上层输出接口。
3. 并行 RNG 设计有利于执行结果可复现。
4. CT、反应、级联和中性粒子数据具有版本化格式及强校验。
5. TOPAS 验证链独立于核心模拟器，参考结果来源可追踪。
6. 能量账本集中在 `TransportResult`，便于检查守恒和遗漏。
7. SOBP 复用单 spot 输运能力，没有建立第二套输运实现。

## 14. 主要耦合点与维护风险

### 14.1 `transport_sycl.cpp` 是巨型单体

`src/transport_sycl.cpp:305-3210` 同时承担：

- 设备内存预算。
- 几何和材料选择。
- primary transport。
- secondary transport。
- cascade。
- neutral transport。
- voxel scoring。
- 结果回收和归并。

这是当前最明显的高耦合维护热点。修改一个物理模型时，容易影响队列、内存布局、kernel 参数和 scorer。

### 14.2 `TransportConfig` 职责过宽

`TransportConfig` 同时包含：

- 设备配置。
- 束流配置。
- 几何配置。
- 物理开关。
- 数据文件路径。
- scorer 配置。

新增功能通常需要同时修改配置结构、配置解析、校验、CLI 和 kernel 参数。

### 14.3 模型组合依赖集中校验

slab、insert、CT、绝对材料表和 mass-SP 的兼容关系主要由 `validate()` 维护。随着模型增多，合法组合和优先级容易变得难以推断。

### 14.4 CPU/SYCL 功能不对称

Serial 只能验证共同功能子集。完整二级粒子、异质材料和中性粒子链路主要依赖 SYCL 集成测试及 TOPAS 对照，定位回归时缺少完全独立的本地参考实现。

### 14.5 二进制资产版本耦合

二进制包依赖特定版本和 little-endian 格式。格式升级时，需要同步更新 loader、生成脚本、测试资产和验证 metadata。

### 14.6 核心测试依赖验证资产

部分测试直接使用 `validation/results/*.bin` 和 TOPAS spot 文件。这样能够提高端到端覆盖，但也增加了测试环境和资产版本管理成本。

## 15. 总结

当前架构已经形成一条完整的领域链路：

```text
配置与束流
    → 物理数据加载
    → CPU/SYCL 输运
    → 二级粒子与中性粒子处理
    → scorer 和能量账本
    → TOPAS 独立验证
```

架构上最稳定的部分是数据格式、公共结果接口和验证链。当前扩展成本主要集中在两个位置：

1. `include/carbon/transport_config.hpp` 中持续扩大的全局配置结构。
2. `src/transport_sycl.cpp` 中集中实现的几何、物理、队列和 scorer 逻辑。

后续如果进行结构重整，最值得优先控制的是 SYCL 实现内部的职责边界，同时保持现有 `TransportConfig`、`TransportResult` 和外部数据格式接口稳定。

# Minibeam 编译隔离与安全合并设计

## 1. 目标

Minibeam 的 Copper 准直器、材料输运、核反应、次级粒子和诊断逻辑应与原有
水箱、CT、TPS source 和 LET GPU 蒙卡隔离。

设计需要同时满足：

1. 默认编译不包含 minibeam，使用原 `master` 输运路径和参数；
2. 未启用 minibeam 时，非 minibeam GPU 结果应尽可能逐 bit 复现原
   `master`；
3. 未启用 minibeam 时，不能因为 minibeam 代码体积、寄存器压力或运行时分支
   导致性能回退；
4. 启用 minibeam 的二进制仍应保留原 legacy kernel，使
   `minibeam: false` 的普通 CT case 不进入 minibeam kernel；
5. 当前已经验证的 100--400 MeV/u 和 179.17 MeV/u 三维 minibeam 精度必须
   继续复现。

## 2. 当前分支不能直接安全合并的原因

当前实现的绝大多数 minibeam 参数默认关闭，但 Copper/minibeam 逻辑仍被编译进
普通 primary SYCL kernel，只在运行时判断 `enable_minibeam`。

对同一 TITAN RTX、同一 seed、同一输入进行了 `master` 和 `minibeam` 双二进制
回归。

100k 非 minibeam CT + secondary case 的物理结果基本不变：

| 指标 | `master` 与 `minibeam` 的差异 |
|---|---:|
| total 积分差 | `-1.14e-6%` |
| normalized L1 | `2.10e-5%` |
| 最大 bin 差 / reference peak | `2.28e-5%` |
| 核反应数、secondary 数、transport steps | 完全一致 |

但性能已经出现回退：

| 100k CT + secondary | `master` | 当前 `minibeam` | 变化 |
|---|---:|---:|---:|
| transport elapsed | 0.290 s | 0.307 s | `+5.7%` |
| primary kernel | 0.155 s | 0.189 s | `+21.8%` |
| secondary kernel | 0.0422 s | 0.0467 s | `+10.8%` |

10k 均匀水箱、cascade、MCS 和 straggling case 也不再逐 bit 相同：

- total 积分差：`-0.0088%`；
- normalized L1：`0.831%`；
- primary nuclear interactions：`3957` 对 `3947`。

剂量差仍在低统计量蒙卡涨落范围内，但说明当前实现不能承诺 legacy
bitwise reproduction。

主要原因：

- 巨大的 Copper beamline 逻辑与普通 primary kernel 共存；
- `secondary_persistent_workers=0` 时仍使用新增的通用循环结构；
- boundary nudge 改动全局生效；
- FP32 secondary energy-loss residual 全局生效；
- electronic lateral redistribution 路径进入普通 voxel kernel，即使 sigma 为 0；
- 运行时 `if (enable_minibeam)` 不能保证编译器消除寄存器和代码体积开销。

## 3. 编译开关

新增默认关闭的 CMake 选项：

```cmake
option(
    CARBON_ENABLE_MINIBEAM
    "Build Copper minibeam transport support"
    OFF
)
```

### 3.1 `CARBON_ENABLE_MINIBEAM=OFF`

这是正式 CT、TPS source 和 LET build 的默认模式：

- 不编译 minibeam SYCL kernel；
- 不编译 Copper material/reaction/neutral table loader；
- 不分配 minibeam device buffer；
- 不把 minibeam lambda 或运行时判断放进 legacy kernel；
- 使用原 `master` primary/secondary kernel；
- 使用原 `master` 数值默认值；
- YAML 中出现 `minibeam: true` 时明确报错；
- YAML 中没有 minibeam 参数时不产生任何额外行为。

建议报错信息：

```text
minibeam=true requires a build configured with
CARBON_ENABLE_MINIBEAM=ON
```

不能静默忽略 `minibeam: true`，否则可能在没有准直器的情况下输出错误剂量。

### 3.2 `CARBON_ENABLE_MINIBEAM=ON`

启用时编译独立 minibeam 模块，同时保留 legacy kernel：

- `minibeam: false`：调用 legacy kernel；
- `minibeam: true`：调用 minibeam beamline/kernel；
- Copper 产生的 charged/neutral particles通过公共 queue 接口进入水中输运；
- minibeam diagnostics 继续由 YAML 独立控制；
- 普通 CT case 不应因为使用支持 minibeam 的二进制而明显变慢。

## 4. 编译和运行矩阵

| Build | YAML | 执行路径 | 预期 |
|---|---|---|---|
| Minibeam OFF | 普通 CT/水箱/LET | legacy kernel | 复现 `master` |
| Minibeam OFF | `minibeam: true` | 不运行 | 配置阶段明确报错 |
| Minibeam ON | `minibeam: false` | legacy kernel | 与 OFF build 一致 |
| Minibeam ON | `minibeam: true` | minibeam kernel | 复现当前 TOPAS match |

## 5. 推荐源码结构

```text
include/carbon/
  transport.hpp
  transport_config.hpp
  minibeam_collimator.hpp
  minibeam_transport.hpp
  minibeam_tables.hpp

src/
  transport_sycl.cpp
  transport_cpu.cpp
  minibeam_transport_sycl.cpp
  minibeam_tables.cpp
```

各文件职责：

- `transport_sycl.cpp`
  - 保留原水、CT、TPS source、LET primary/secondary/neutral transport；
  - 不包含 Copper 几何、Copper MCS 或 Copper nuclear reaction 代码；
  - OFF build 应尽量保持与原 `master` 相同的 kernel。

- `minibeam_transport_sycl.cpp`
  - source 到准直器入口的 air transport；
  - slit/Copper 几何判断；
  - Copper stopping power、straggling 和 MCS；
  - Copper nuclear attenuation；
  - Copper reaction package sampling；
  - Copper charged/neutral survivor；
  - water entrance phase-space；
  - minibeam diagnostics。

- `minibeam_tables.cpp`
  - Copper C-12 stopping power/cross section；
  - isotope-specific Copper stopping power/cross section；
  - Copper neutral cross section/package；
  - 表格验证和加载。

- `minibeam_collimator.hpp`
  - 只保留纯几何、小型、可单元测试的函数；
  - 不负责表格加载或设备内存。

## 6. Kernel 拆分方案

### 6.1 推荐：独立 beamline kernel + legacy water/CT kernel

Minibeam build 使用两个阶段：

```text
source histories
      |
      v
minibeam beamline kernel
  - air
  - slit/Copper
  - Copper reactions
  - water-entry phase space
      |
      +--> Copper charged secondary queue
      +--> Copper neutral queue
      |
      v
legacy water/CT transport kernel
      |
      v
common secondary/neutral transport
```

beamline kernel 为每个 history 输出：

- alive/dead 状态；
- water-entry position；
- water-entry direction；
- water-entry kinetic energy；
- history id / RNG stream；
- beamline removed energy；
- Copper reaction产生的 queue entries。

不必压缩 surviving histories。可以保留一项一 history 的 phase-space buffer，dead
history 在 water kernel 入口立即返回，从而保持 history id 和 Philox stream
稳定。

优点：

- legacy kernel 完全不含 Copper 代码；
- OFF build 不产生 minibeam 寄存器压力；
- minibeam 的材料输运可以独立 profile；
- beamline 与 water transport 的物理边界清晰；
- 更容易保存和比较 water-entry phase space。

代价：

- minibeam case 多一次 kernel launch；
- 需要一个 water-entry phase-space buffer；
- Copper 反应产生的 queue 必须与公共 secondary/neutral queue 对接。

相对 Copper 输运成本，这一次 kernel launch 的开销可以忽略。

### 6.2 次选：模板生成两套 primary kernel

也可以使用：

```cpp
template <bool EnableMinibeam>
class PrimaryTransportKernel;
```

编译为两个独立 SYCL kernel：

- `PrimaryTransportKernel<false>`；
- `PrimaryTransportKernel<true>`。

host 根据配置选择 kernel。必须确保 kernel name 和模板实例完全独立，使
`<false>` 的设备代码不包含 Copper 分支。

该方案改动较小，但 primary kernel 仍会非常大，minibeam 物理和水中物理继续
耦合，长期维护性不如独立 beamline kernel。

## 7. 非 minibeam 通用优化必须独立门控

以下改动不是 Copper minibeam 本身，不应简单绑定到
`CARBON_ENABLE_MINIBEAM`：

- persistent secondary workers；
- secondary condensed-history；
- FP32 energy-loss residual；
- electronic lateral redistribution；
- boundary nudge；
- dose output scale。

建议分别处理。

### 7.1 Persistent secondary workers

YAML：

```yaml
secondary_persistent_workers: 0
```

当值为 0 时，应提交原 legacy secondary kernel，而不是在同一个 kernel 中通过
运行时布尔值模拟旧行为。

当值大于 0 时，提交独立 persistent-worker kernel。

### 7.2 Secondary condensed-history

YAML：

```yaml
secondary_condensed_step_mm: 0
```

0 必须使用原 legacy step/scoring 实现。正值才进入独立 condensed-history
kernel 或模板实例。

当前限制应继续保留：

- CT grid 禁用；
- layered phantom 禁用；
- heterogeneity insert 禁用；
- LET scorer 禁用。

### 7.3 FP32 energy-loss residual

该改动属于正确性改进，但会改变 secondary 轨迹和随机分叉，不能在要求 legacy
bitwise reproduction 时无条件启用。

建议新增：

```yaml
secondary_fp32_energy_residual: false
```

- 默认 `false`：复现 `master`；
- minibeam 正式配置显式设为 `true`；
- 如果未来决定将其升级为全局默认，需要单独提交并重新建立所有 golden
  reference，而不是作为 minibeam merge 的隐式副作用。

### 7.4 Boundary nudge

原 `master` 使用 `nextafter`。新的固定物理距离 nudge 会改变边界附近轨迹。

建议：

```yaml
robust_boundary_nudge: false
```

- `false`：legacy `nextafter`；
- `true`：使用当前 `1e-5 mm` nudge；
- minibeam 的 0.1 mm scorer case 显式开启。

也可以仅在检测到连续 boundary non-progress 后才升级为固定 nudge，使正常 legacy
轨迹不变。

### 7.5 Electronic lateral redistribution

当：

```yaml
electronic_buildup_lateral_sigma_mm: 0
```

必须直接走原 voxel atomic scoring 路径，不调用 redistribution helper，也不生成
额外 RNG 值。

正值才提交带 lateral redistribution 的专用 kernel。

### 7.6 Dose output scale

`dose_output_scale: 1.0` 在数值上无影响，可以保留为通用输出功能，但：

- 只能作用于 Gy 输出；
- 不能改变 raw deposited-energy scorer；
- 不能改变 energy balance；
- legacy 配置缺省值必须为 1；
- 需要保留输出 scale 的日志提示。

## 8. CMake 组织建议

```cmake
option(CARBON_ENABLE_MINIBEAM
       "Build Copper minibeam transport support"
       OFF)

target_sources(carbon_core PRIVATE
    src/transport_sycl.cpp
)

if(CARBON_ENABLE_MINIBEAM)
    target_sources(carbon_core PRIVATE
        src/minibeam_transport_sycl.cpp
        src/minibeam_tables.cpp
    )
    target_compile_definitions(
        carbon_core PUBLIC CARBON_ENABLE_MINIBEAM=1
    )
endif()
```

头文件中只暴露稳定接口：

```cpp
#if defined(CARBON_ENABLE_MINIBEAM)
TransportResult transport_minibeam_sycl(
    const TransportConfig& config,
    SyclTransportContext& context);
#endif
```

普通入口：

```cpp
if (!config.enable_minibeam) {
    return transport_sycl(config);
}

#if defined(CARBON_ENABLE_MINIBEAM)
return transport_minibeam_sycl(config);
#else
throw std::runtime_error(
    "minibeam=true requires CARBON_ENABLE_MINIBEAM=ON");
#endif
```

实际实现中应复用 device tables、queue 和 scorer context，避免为两个入口重复加载
大表。

## 9. 配置兼容性

### 9.1 Legacy 配置

所有旧 YAML 不添加任何字段时：

- minibeam OFF；
- persistent worker OFF；
- condensed-history OFF；
- FP32 residual OFF；
- robust boundary nudge OFF；
- lateral redistribution OFF；
- dose output scale = 1。

这应构成严格的 legacy compatibility contract。

### 9.2 Minibeam 正式配置

当前正式配置需要显式写出所有非 legacy 优化：

```yaml
minibeam: true
minibeam_diagnostics: false

secondary_persistent_workers: 32768
secondary_condensed_step_mm: 0.25
secondary_fp32_energy_residual: true
robust_boundary_nudge: true

electronic_buildup_lateral_sigma_mm: 0.5
```

Copper 最大步长：

```text
100 MeV/u:       0.10 mm
200--400 MeV/u: 0.25 mm
```

## 10. 测试矩阵

### 10.1 Build matrix

CI 至少构建：

1. CPU build；
2. SYCL/CUDA minibeam OFF；
3. SYCL/CUDA minibeam ON。

### 10.2 Legacy compatibility

用同一设备、seed、编译器和 FP32/FP64 dose 配置比较：

- primary-only 均匀水箱；
- cascade + MCS + straggling 水箱；
- CT primary-only；
- CT + secondary + cascade；
- TPS source；
- LET scorer；
- full CT plan。

验收标准：

- OFF build 对原 `master` 尽可能逐 bit 一致；
- 若编译器导致 atomic order 差异，至少要求 history/steps/reaction count 完全一致；
- total 积分差和 L1 应接近 FP32 atomic rounding；
- 100k CT kernel 时间不得显著回退；
- 正式 full plan gamma 不得退化。

### 10.3 ON build 的 legacy 路径

`CARBON_ENABLE_MINIBEAM=ON` 且 `minibeam: false`：

- 对 OFF build 比较输出；
- 比较 kernel time；
- 验证没有分配 Copper buffer；
- 验证 backend 字符串不包含 minibeam；
- 验证没有读取 Copper data 文件。

### 10.4 Minibeam 物理回归

保持现有正式结果：

| 能量 | ΔR80 | total 积分差 | depth L1 |
|---:|---:|---:|---:|
| 100 MeV/u | +0.011 mm | +0.742% | 1.314% |
| 200 MeV/u | -0.122 mm | -0.019% | 1.775% |
| 300 MeV/u | -0.130 mm | -0.810% | 1.779% |
| 400 MeV/u | +0.025 mm | -0.726% | 1.999% |

179.17 MeV/u 三维回归：

- total 积分差约 `-0.102%`；
- depth L1 约 `1.789%`；
- ΔR80 约 `-0.186 mm`；
- lateral-x / lateral-y L1 约 `7.987% / 4.339%`。

建议容差：

- 四能量 `|ΔR80| < 0.20 mm`；
- 四能量 `|total integral difference| < 1.0%`；
- 四能量 depth L1 `< 2.1%`；
- 三维 total integral difference `< 0.5%`；
- 三维 depth L1 `< 2.0%`。

## 11. 迁移顺序

建议按以下顺序实施，每一步都单独验证：

1. 添加 `CARBON_ENABLE_MINIBEAM`，默认 OFF；
2. 建立 OFF build，对原 `master` 做数值和性能基线；
3. 将表格加载移到 `minibeam_tables.cpp`；
4. 将 Copper beamline 移到独立 kernel；
5. 让 minibeam survivor phase space 接入 legacy water kernel；
6. 接入 Copper charged/neutral queue；
7. 拆分 legacy 和 persistent secondary kernel；
8. 给 FP32 residual、boundary nudge 和 lateral redistribution 增加独立门控；
9. 跑 OFF/ON build matrix；
10. 重跑四能量 100k 和 179.17 MeV/u 三维 minibeam；
11. 重跑正式非 minibeam CT full plan gamma 和性能；
12. 通过所有验收标准后再合并到 `master`。

## 12. Safe merge 标准

只有同时满足以下条件才建议合并：

1. `CARBON_ENABLE_MINIBEAM=OFF` 是默认值；
2. OFF build 不编译或引用 Copper/minibeam kernel；
3. OFF build 的非 minibeam 数值复现原 `master`；
4. OFF build 的 CT 性能无显著回退；
5. ON build + `minibeam:false` 使用 legacy kernel；
6. ON build + `minibeam:true` 复现当前 TOPAS match；
7. 配置错误不会静默绕过准直器；
8. CPU、CUDA 和现有测试全部通过；
9. full CT plan gamma 和运行时间没有退化；
10. minibeam 四能量和三维回归全部通过。

在完成上述隔离前，当前 `minibeam` 分支在物理剂量上基本安全，但不能承诺
非 minibeam 性能和 bitwise compatibility，因此不应直接合并到 `master`。

## 13. 实现状态（2026-07-28）

已落地：

1. CMake `CARBON_ENABLE_MINIBEAM`（默认 OFF）；
2. OFF build 使用 `src/transport_sycl_legacy.cpp`（与合并前 `master` 的
   `transport_sycl.cpp` 同源），不把 Copper 代码编入 kernel；
3. ON build 使用 `src/transport_sycl.cpp`，`minibeam: false` 走 legacy 物理默认；
4. 独立门控：`secondary_fp32_energy_residual`、`robust_boundary_nudge`（默认
   false）；`electronic_buildup_lateral_sigma_mm=0` 时不进入 lateral helper；
5. `minibeam: true` 在 OFF build 上配置阶段明确报错；
6. 回归脚本：`validation/scripts/regression_minibeam_isolation.py`；
7. CMake preset：`oneapi-nvidia-release`（OFF）、`oneapi-nvidia-minibeam`（ON）。

正式 minibeam YAML 显式打开 non-legacy 开关以保持 TOPAS match。

三门验收命令见该脚本 `run` 子命令。

## 14. P1 修复（ON 双 kernel 分发 + 回归门）

1. **ON build 双 kernel**：`CARBON_ENABLE_MINIBEAM=ON` 时同时编译
   `transport_sycl_legacy.cpp`（`transport_sycl_legacy`）与
   `transport_sycl.cpp`（`transport_sycl_minibeam`），由
   `transport_sycl_dispatch.cpp` 在运行时按 `minibeam` 选择。
   `minibeam: false` 走 legacy kernel，不再把 Copper 逻辑留在执行路径中。
2. **共享 context**：`src/detail/sycl_transport_context_impl.inc` +
   methods.inc；仅 legacy TU 定义 `SyclTransportContext` 方法
   （`CARBON_DEFINE_SYCL_CONTEXT`）。
3. **Gate 3**：无条件通过已删除；强制检查 Backend 含 minibeam、diagnostics、
   新生成且非零的 MHD/raw；`--minibeam-ref-csv` 可选做深度剂量对比。
4. **性能门**：解析 `Steps:` / `Kernel time: primary=...`；缺字段失败；使用
   transport kernel 时间（warmup + 中位数 repeats），不再用进程 wall time。
5. **CTest**：`minibeam_isolation_log_parser`（默认）；
   `minibeam_isolation_gates`（`CARBON_RUN_ISOLATION_GATES=ON` + 双二进制路径）。
6. **Preset**：`oneapi-nvidia-minibeam` 默认 `CARBON_DOSE_FP32=ON`；
   `oneapi-nvidia-minibeam-fp64` 供高精度。

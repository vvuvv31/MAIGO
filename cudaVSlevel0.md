# CUDA 与 Level Zero CT Full-Plan 对比

本文记录同一 CT full-plan 输运代码在 NVIDIA CUDA oneAPI backend 与 Intel Arc B580 Level Zero backend 上的验证结果，并说明为什么当前约 `12.25x` 的运行时间差异不能直接解释为“Level Zero backend 比 CUDA backend 快 12 倍”。

## 1. 验证对象

共同的物理和计划设置：

- 917 个 TPS spots，其中 853 个权重大于零；
- 原始 TPS optimizer 权重，权重和 `25927.6347446291`；
- 10,000,000 histories；
- `random_seed=20260722`；
- TPS-90 patient CT；
- energy straggling、multiple scattering、primary attenuation；
- charged secondary transport 和两代 fragment cascade；
- energy-sorted secondary batches；
- 相同的 reaction/cascade package；
- 相同的 `ct/code/physical_dose.mhd` Dij·x/matRad physical-dose 参考。

Level Zero 结果使用：

- Git commit：`672e5efd1f5d1f40da03d593550bee1a429a5677`；
- Intel oneAPI DPC++/C++ 2026.1；
- Intel Arc B580；
- Level Zero driver `1.15.38646+4`；
- Intel-only `spir64` 构建；
- FP64 voxel-dose atomic scoring；
- `UR_L0_V2_DISABLE_ZE_LAUNCH_KERNEL_WITH_ARGS=1`。

CUDA 结果来自 NVIDIA TITAN RTX oneAPI CUDA-plugin 运行。对应日志后来在
`out/ct/full_plan_physics_fixed_10M/run.log` 找到，明确记录
`dose_atomic=fp32`、`local_size=128` 和 CUDA 12.6；运行前的构建缓存记录
`CARBON_CUDA_ARCH=sm_75`、`CARBON_DOSE_FP32=ON` 和 dual
`spir64,nvptx64-nvidia-cuda` target。CUDA executable SHA-256 为
`25379445ebe6364a2162a3ab46cfcd1a1abfb24fe8fae99166be85f8af44953e`。
因此当前性能表是 Level Zero FP64 对 CUDA FP32，并非相同 atomic 精度的严格A/B。

## 2. 10M 剂量验证结果

两次结果均使用独立 spot calibration 给出的固定 scale：

```text
126.310283132833
```

| 指标 | Arc B580 Level Zero | TITAN RTX CUDA | Level Zero − CUDA |
|---|---:|---:|---:|
| Global 2%/2 mm gamma | 97.6768% | 97.582% | +0.0948 pp |
| Local 2%/2 mm gamma | 95.2247% | 95.058% | +0.1667 pp |
| Global 3%/3 mm gamma | 99.1928% | 99.200% | −0.0072 pp |
| Local 3%/3 mm gamma | 98.1884% | 98.067% | +0.1214 pp |
| High-dose NRMSE | 2.2190% | 2.381% | −0.1620 pp |
| IDD correlation | 0.999865 | 0.999030 | +0.000835 |
| 自由拟合 scale | 126.352446 | 126.348104 | +0.00344% |
| 输运带电次级数 | 31,597,193 | 31,611,408 | −0.0450% |
| Primary/cascade queue overflow | 0 | 0 | 相同 |

Level Zero 的自由拟合 scale 相对 CUDA 仅差 `0.00344%`，gamma 差异约为 `0.01–0.17` 个百分点。两种 backend 的剂量结果在当前统计精度下高度一致。

## 3. 性能结果

| 指标 | Arc B580 Level Zero | TITAN RTX CUDA |
|---|---:|---:|
| 10M wall time | 43.79 s | 536.51 s |
| Throughput | 228,365 histories/s | 18,639 histories/s |
| 相对速度 | 12.25x | 1.00x |

Level Zero 10M 运行的详细分解：

| 阶段 | 时间 |
|---|---:|
| Primary kernel | 28.87 s |
| Secondary kernel | 13.67 s |
| 总时间 | 43.79 s |

Level Zero 运行还确认：

- device memory estimate：`5201 MiB`；
- B580 global memory：`12216 MiB`；
- secondary queue capacity：`35,000,000`，未自动缩容；
- secondary queue overflow：0；
- cascade queue overflow：0；
- energy balance error：`4.02e-6`。

历史 CUDA 性能分析表明，CUDA full-plan 的主要瓶颈是 secondary transport，而不是 primary transport。早期 1M 优化记录为：

| 1M 指标 | TITAN RTX CUDA | Arc B580 Level Zero |
|---|---:|---:|
| 总时间 | 55.48 s | 4.57 s |
| Secondary kernel | 52.66 s | 1.38 s |
| Throughput | 18,025 histories/s | 218,895 histories/s |

CUDA 约 95% 的1M运行时间消耗在 secondary kernel；Level Zero 的 secondary kernel 占比明显较低。

## 4. 为什么性能相差较大

### 4.1 不是严格的 backend-only A/B

当前比较同时改变了：

- GPU：TITAN RTX 与 Arc B580；
- runtime：oneAPI CUDA plugin 与原生 Level Zero；
- 操作系统/驱动路径：历史 CUDA 运行包含 WSL 约束；
- 编译目标：CUDA PTX 与 Intel SPIR-V；
- 编译器、plugin 和驱动版本可能不同。

因此不能仅根据 wall time 得出“Level Zero API 本身快 12 倍”的结论。

### 4.2 oneAPI CUDA plugin 与原生 Intel 路径

CUDA 版本不是手写 CUDA kernel。DPC++ 先生成 PTX，再通过 oneAPI CUDA plugin 和 NVIDIA driver 执行。Level Zero 则是 Intel GPU 的原生 oneAPI 路径，编译器、runtime 和硬件结合更紧密。

如果 CUDA 构建没有设置：

```bash
-DCARBON_CUDA_ARCH=sm_75
```

运行时将使用便携 PTX/JIT，而不是明确针对 TITAN RTX 的 AOT 目标。这可能影响寄存器分配、occupancy 和最终机器码质量。
但 `536.51 s` CUDA 运行已有匹配的构建缓存，确认使用 `sm_75`，所以 portable
PTX/JIT 不是这次差距的首要解释。

### 4.3 Work-group 设置不同

代码当前明确使用：

| Backend | Local size |
|---|---:|
| CUDA | 128 |
| Level Zero/OpenCL GPU | 256 |

CUDA secondary kernel 包含大量捕获参数、dose atomic、cascade 逻辑和长粒子轨迹，寄存器压力较高。代码保留 CUDA `local_size=128`，因为256线程通常降低 occupancy；B580 可以使用256线程 work-group。

这会影响并行度，但单独不足以解释全部12倍差距。

### 4.4 Secondary transport 的线程分歧

不同碎片具有不同：

- 粒子种类和质量/电荷；
- 初始能量与方向；
- CT 材料路径；
- 核反应次数；
- track 长度和终止时间。

一个 NVIDIA warp 中只要部分线程仍在输运长轨迹，已经结束的线程就必须等待。虽然 energy sorting 已降低分歧，secondary transport 仍是高度不规则的 workload。

### 4.5 Dose atomic voxel scoring

本次 Level Zero 生产构建使用 FP64 dose atomic，而 `536.51 s` CUDA 运行确认使用
FP32 dose atomic。大量粒子会同时向相同或相邻 voxel 累加，仍可能造成原子竞争，
但 TITAN RTX 的弱 FP64 吞吐不能解释这次 CUDA 慢速，因为 CUDA 实际使用的是FP32。

两端应继续做统一 FP32/FP64 A/B 来量化 atomic 成本，但现有精度不一致理论上更有利
于 CUDA；它不是已观测12倍差距的来源。

### 4.6 WSL 和 CUDA 专用保护

代码中为 CUDA/WSL 路径设置了：

- 较小的 history chunk；
- `local_size=128`；
- device-memory fraction clamp；
- queue 上限；
- 较短 kernel submit，避免 WDDM/TDR 或 WSL 长时间无响应。

这些设置优先保证稳定性，不一定是 TITAN RTX 的最高性能配置。

## 5. 当前可以得出的结论

可以确认：

1. 相同 CT full-plan 物理代码可在 Arc B580 Level Zero 上完整运行。
2. Level Zero 与既有 CUDA 的剂量、gamma、scale 和次级粒子统计高度一致。
3. 当前 B580 Level Zero 运行明显快于历史 TITAN RTX oneAPI CUDA-plugin 运行。
4. 性能差距主要集中在 secondary transport。

不能确认：

1. Level Zero backend 本身一定比 CUDA backend 快12倍。
2. 差异完全来自硬件。
3. 两次运行使用了相同 oneAPI/compiler/plugin 版本和相同热缓存状态。

## 6. 建议的严格 CUDA/Level Zero A/B

应在当前 commit 上重新构建 CUDA-only 版本：

```bash
cmake -S . -B build/oneapi-nvidia-sm75-release \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCARBON_ENABLE_SYCL=ON \
  -DCARBON_BUILD_TESTS=ON \
  -DCARBON_SYCL_TARGETS=nvptx64-nvidia-cuda \
  -DCARBON_CUDA_ARCH=sm_75 \
  -DCARBON_DOSE_FP32=OFF
```

然后使用完全相同的：

- commit；
- oneAPI/compiler/plugin 版本；
- 10M config；
- random seed；
- spot 权重；
- CT grid；
- reaction/cascade package；
- FP64 dose；
- warm-up 和缓存状态。

至少记录两次正式运行，并比较：

- primary/secondary kernel time；
- transported secondary count；
- primary/secondary step count；
- work-group occupancy；
- register usage；
- FP64 与 FP32 atomic A/B；
- `local_size=128/256`；
- `secondary_batch_size`；
- gamma、IDD、NRMSE 和能量闭合。

只有完成该受控实验后，才能把剩余差距分别归因于 GPU 硬件、SYCL CUDA plugin、PTX代码生成、FP64 atomic 或 backend 调度。

## 7. Level Zero 结果文件

本地验证结果位于：

```text
out/ct/level_zero_full_plan/
```

主要文件：

```text
final_10M/run.log
final_10M/dose_10M.mhd
final_10M/dose_10M.raw
final_10M/compare_free_scale/match_metrics.json
final_10M/compare_fixed_scale/match_metrics.json
final_10M/ct_full_plan_level_zero_comparison.png
inputs/input_sha256.txt
```

`out/` 已被 Git 忽略；本文只记录可复核的结论，不提交患者输入或大体积剂量文件。

## 8. 进一步分析：38.35x secondary 时间差如何构成

从 Level Zero 原始 `run.log` 和后来找到的当前代码状态 CUDA 1M日志，可以用输运
工作量一致的记录量化差距：

| 当前代码1M | TITAN RTX CUDA FP32 | Arc B580 Level Zero FP64 | CUDA / Level Zero |
|---|---:|---:|---:|
| 总时间 | 56.4304 s | 4.5684 s | 12.35x |
| Secondary kernel | 52.8035 s | 1.3768 s | **38.35x** |
| 输运带电次级数 | 3,164,377 | 3,158,734 | 1.0018x |
| Secondary transport steps | 13,729,584,726 | 1,899,483,953 | **7.23x** |
| 平均 steps/输运次级 | 4,338.8 | 601.3 | **7.22x** |
| Secondary step throughput | 260.0 Mstep/s | 1,379.6 Mstep/s | 0.188x |

这里有两个相乘的差异：

1. CUDA每个带电次级平均执行约 `7.22x` 更多步骤；
2. 按step数归一化，CUDA表观每步吞吐仍低 `5.31x`。

总step数的 `7.23x` 与每步吞吐的 `5.31x` 相乘，正好得到观测到的 `38.35x`
secondary kernel时间差。后面的profile表明这并非两个独立问题：极少数500k-step
长尾既增加总step数，也让同warp的其他lane长期空转。

### 8.1 已排除 straggling_scale 差异

早期 CUDA 1M配置使用 `straggling_scale=1.0`，当前统一对比使用 `1.2`。另外运行
仅把 Level Zero 参数改为 `1.0` 的1M诊断：

| Level Zero 1M | scale=1.2 | scale=1.0 |
|---|---:|---:|
| Secondary transport steps | 1,899,483,953 | 1,899,700,770 |
| 输运带电次级数 | 3,158,734 | 3,159,410 |
| Secondary kernel | 1.3768 s | 1.3799 s |
| 总时间 | 4.5684 s | 4.5636 s |

步数只变化约 `0.011%`，所以 straggling scale不能解释CUDA的约13.73B steps。

### 8.2 已确认的首要原因：CUDA存在极少数500k-step长尾轨迹

CUDA与Level Zero输运次级数只差 `0.18%`，但secondary steps多 `7.23x`。这表明
差异不是queue容量或产生了更多粒子，而是单条次级轨迹平均循环次数明显更多。

代码中的 secondary 循环曾有 voxel-boundary nudge thrashing 风险，并设置了每粒子
`500000` 步保护。小步路径使用 `nextafter`，在 CT 中还会沿方向移动 `1e-4 mm`。
如果 CUDA 在 DDA face crossing、浮点收缩或边界判断上反复进入异常小步路径，
就可能产生很多几乎不改变剂量的额外循环；这与两端最终剂量高度一致并不矛盾。
CUDA 10M日志也给出31,611,408个次级和137,105,041,136步，平均4,337.2步/次级，
证明该现象可稳定扩展到生产统计量。

为定位这些额外步骤，使用相同 `sm_75`、FP32和物理配置构建
`CARBON_ENABLE_TRANSPORT_PROFILE=ON`，运行10k CUDA：

| CUDA profile 10k | 结果 |
|---|---:|
| 输运带电次级数 | 31,921 |
| Secondary transport steps | 150,977,021 |
| 正常step profile counter | 150,975,868 |
| 两者差值（已进入nudge/continue分支） | 1,153 |
| `>=32768` 最高/overflow直方图桶轨迹 | 264 |

如果最高桶的264条轨迹都运行到代码的500,000步上限，去掉它们后剩余31,657条轨迹
平均为：

```text
(150,977,021 - 264 × 500,000) / (31,921 - 264) = 599.46 steps/track
```

这与 Level Zero 的 `601.34 steps/track` 几乎完全一致。关闭multiple scattering后，
CUDA仍有234条最高桶轨迹；按同样方法去掉后其余轨迹平均603.01步。因此：

- MCS不是500k长尾的必要条件；
- 额外步骤不是大量粒子普遍变慢，而是约0.8%的轨迹触发硬上限；
- `secondary_transport_steps - profile secondary_steps` 只有1,153，说明这些轨迹
  不是在现有 `path_step_mm <= 1e-8` nudge分支中反复循环，而是在正常step分支卡住。

结合代码，最符合数据的机制是：CUDA在某些voxel face上得到一个**略大于
`1e-8 mm`、但小于当前位置float ULP所需位移**的正 `path_step_mm`。它绕过
`min_step_accept`，执行正常沉积/MCS分支，但
`position += direction * path_step_mm` 没有改变至少一个决定下一face的坐标；下一轮
再次得到同一微小步长，直到500,000步保护触发。Level Zero的舍入/FMA路径更可能把
相同残差变成0或落入nudge阈值，因此没有这组长尾。

这是由profile数据强烈支持的根因定位，但要成为逐轨迹的直接证明，还应新增：

- `secondary_step_cap_hits`；
- `secondary_nonprogress_steps`（更新前后position逐位相等）；
- 触发时的粒子种类、能量、位置、方向和限制步长来源；
- 独立的 `secondary_boundary_nudge_continues` counter。

### 8.3 剩余单位step效率差主要也是长尾导致的warp空转

此前把约 `5.3--5.7x` 单位step吞吐差独立归因于代码生成并不充分。500k长尾轨迹
分散在CUDA warp中时，同warp其余约600步结束的线程必须等待长尾线程，导致大量
lane空转。10k profile的三个generation分别只有22,244、8,656和1,021条输入，但
kernel时间仍接近1.15、1.11和0.93 s；工作量相差22倍而时间接近，是“每批由最长
轨迹决定”的直接表现。

因此少量non-progress长尾同时造成：

1. 约 `7.23x` 的计数步骤膨胀；
2. 严重warp divergence，压低表观Mstep/s；
3. 每个64k batch接近固定的约1 s尾延迟；
4. 触发500k上限后把剩余能量局部沉积，带来潜在的小剂量偏差。

在修复non-progress之前，不能把剩余5.31倍主要归因于寄存器溢出、PTX质量或硬件。
修复后再profile，才有意义分解以下次要因素：

- **Warp 长尾分歧**：一个 work-item 完整输运一条轨迹。当前 energy sorting 只按
  四个能量区间分组，无法消除核素、方向、解剖路径、cascade 和 track length 的
  差异；一个 NVIDIA warp 会等待最长轨迹。
- **大型 secondary lambda 的寄存器压力/溢出**：kernel 同时包含 CT DDA、
  stopping power、straggling/MCS、Philox、核反应、cascade、scorer 和队列管理。
  若 PTX 发生 spill，local-memory 成本会在数千次循环中重复，并与 warp divergence
  相互放大。
- **编译目标和代码生成**：oneAPI CUDA plugin 的 PTX 路径与 Intel `spir64`/
  Level Zero 路径不同，最终寄存器分配和控制流质量可能不同。
- **Dose atomic**：voxel热点竞争可能影响性能，但本次CUDA已确认使用FP32。

这些仍是修复长尾后应检查的次级瓶颈。缺少 `ptxas` register/spill报告和
Nsight Compute指标时，不能提前分配其贡献。

### 8.4 不足以单独解释差距的因素

- **Local size 128 vs 256** 会改变 occupancy，但不太可能单独造成 `38.25x`
  secondary 时间差。
- 启用 energy sorting 后，两端 secondary batch 都被代码限制为 `65536`。
- 两端 queue 均无 overflow；显存 clamp 影响容量，不会直接减少实际输运步骤。
- WSL submit/sync 有开销，但 CUDA 的主要时间位于 secondary kernel event 内。
- Primary 阶段没有同等级差距，因此不能简单归结为 B580 对所有计算都比
  TITAN RTX 快。

匹配的CUDA日志和构建时间关系已确认 `sm_75`、FP32和dual-target设置；因此
portable PTX和CUDA FP64不是当前结果的主要解释。

### 8.5 最小闭环实验

下一步不必再重复10M。优先修复和验证float non-progress：

1. 在position更新后检查是否真正前进；若未前进，直接把限制该步的坐标
   `nextafter`到face另一侧，不能只依赖固定 `1e-8 mm` 阈值。
2. 更稳健的方案是按当前位置ULP和方向分量计算最小可表示位移，或让几何位置使用
   double；优先选择显式face snap以控制性能。
3. 增加cap-hit/non-progress计数，CUDA 10k应从约264个cap hit降到0。
4. 验收secondary steps/track应由约4,339降到接近Level Zero的约601，同时保持
   queue统计、能量闭合、IDD以及global/local gamma。
5. 修复后再使用 `ptxas`/Nsight Compute检查
   registers/thread、spill、achieved occupancy、warp/branch efficiency 和 atomic
   throughput，并做local size与FP32/FP64受控测试。

当前最可靠的结论是：profile和步数算术强烈表明CUDA约0.8%的secondary轨迹在正常
step分支发生float non-progress并跑到500k上限；去掉这些轨迹后，CUDA其余轨迹
平均步数与Level Zero几乎一致。这些长尾又造成严重warp空转，因而很可能解释
12.25倍总时间差的大部分。Level Zero API或B580硬件不是首要根因；应先用显式
cap-hit/non-progress counter完成逐轨迹确认并修复几何前进保证，再评价真正的
backend/hardware单位step性能。

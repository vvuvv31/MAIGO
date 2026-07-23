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
- FP64 voxel-dose atomic scoring；
- 相同的 reaction/cascade package；
- 相同的 `ct/code/physical_dose.mhd` Dij·x/matRad physical-dose 参考。

Level Zero 结果使用：

- Git commit：`672e5efd1f5d1f40da03d593550bee1a429a5677`；
- Intel oneAPI DPC++/C++ 2026.1；
- Intel Arc B580；
- Level Zero driver `1.15.38646+4`；
- Intel-only `spir64` 构建；
- `UR_L0_V2_DISABLE_ZE_LAUNCH_KERNEL_WITH_ARGS=1`。

既有 CUDA 结果来自项目历史记录中的 NVIDIA TITAN RTX oneAPI CUDA-plugin 运行。

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

### 4.3 Work-group 设置不同

代码当前明确使用：

| Backend | Local size |
|---|---:|
| CUDA | 128 |
| Level Zero/OpenCL GPU | 256 |

CUDA secondary kernel 包含大量捕获参数、FP64 atomic、cascade 逻辑和长粒子轨迹，寄存器压力较高。代码保留 CUDA `local_size=128`，因为256线程通常降低 occupancy；B580 可以使用256线程 work-group。

这会影响并行度，但单独不足以解释全部12倍差距。

### 4.4 Secondary transport 的线程分歧

不同碎片具有不同：

- 粒子种类和质量/电荷；
- 初始能量与方向；
- CT 材料路径；
- 核反应次数；
- track 长度和终止时间。

一个 NVIDIA warp 中只要部分线程仍在输运长轨迹，已经结束的线程就必须等待。虽然 energy sorting 已降低分歧，secondary transport 仍是高度不规则的 workload。

### 4.5 FP64 atomic voxel scoring

当前生产构建使用 FP64 dose atomic。大量粒子会同时向相同或相邻 voxel 累加，造成原子竞争。TITAN RTX 的 FP64 吞吐较弱，oneAPI CUDA plugin 对大型、分歧严重且包含 FP64 atomic 的 SYCL kernel 也可能产生额外开销。

Level Zero 运行同样使用 FP64 atomic，因此剂量精度口径一致；但两套 driver/compiler 对 atomic 和控制流的代码生成效率可能显著不同。

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
3. 历史 CUDA 构建已经使用最佳 `sm_75` AOT、相同 oneAPI 版本和热缓存。

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

## 8. 进一步代码审计：差距集中在 secondary kernel

继续对照当前实现、历史 CUDA 日志和本机 CUDA 构建缓存后，可以比第4节进一步缩小
问题范围：`12.25x` 不是整个输运流程的普遍差距，而是几乎完全来自 charged
secondary transport。

1M 记录可以拆成：

| 阶段 | TITAN RTX CUDA | Arc B580 Level Zero | CUDA / Level Zero |
|---|---:|---:|---:|
| 总时间 | 55.48 s | 4.57 s | 12.1x |
| Secondary kernel | 52.66 s | 1.38 s | **38.2x** |
| 其余时间 | 2.82 s | 3.19 s | 0.88x |

Primary、初始化、队列准备和结果回传没有数量级差异。真正需要解释的是 secondary
kernel 的约 38 倍差距，而不是泛化地讨论“CUDA 与 Level Zero 哪个更快”。

### 8.1 当前最关键的缺失数据：Level Zero secondary step count

CUDA 1M 优化记录包含约 `1.2746e10` 个 secondary steps、2,979,094 个输运带电
次级，平均约 4,278 步/次级。当前 Level Zero 文档记录了输运次级数和 kernel time，
但没有记录：

- `Steps`；
- `Secondary transport steps`；
- 平均每个次级的步数；
- secondary track-length histogram；
- CT boundary nudge/continue 次数。

因此目前存在两个完全不同的解释：

1. 如果 Level Zero 也执行约 `1.27e10` 步，则其 secondary 吞吐约为
   9.2 billion steps/s，而 CUDA 约为 0.24 billion steps/s。此时应重点检查 CUDA
   kernel 的 warp efficiency、register spill、occupancy 和 oneAPI PTX 代码生成。
2. 如果 Level Zero 的 secondary step 数明显较少，则两端并没有执行相同数量的底层
   工作。应优先检查浮点边界判断、`nextafter`、DDA face crossing 和 nudge 路径，
   而不能把差距归因于硬件或 backend。

代码已经明确记录 secondary 曾出现 voxel-boundary nudge thrashing，并设置每粒子
`500000` 步的保护。小步路径会执行 `nextafter`，在 CT 内还会额外沿方向移动
`1e-4 mm`。不同后端的浮点收缩、舍入和 `nextafter` 实现可能改变进入该路径的次数，
而最终积分剂量仍可保持接近。所以 Level Zero 的 step count 是下一步判断根因的
第一优先级，重要性高于再次重复10M wall-time。

### 8.2 Warp divergence 是 CUDA secondary 的高风险项

当前实现为一个 work-item 完整输运一条次级轨迹。不同碎片具有不同粒子种类、能量、
方向、CT 路径、核反应次数和终止时间。一个 NVIDIA warp 中只要仍有一条长轨迹，
同一 warp 内已经结束的线程就无法贡献有效计算。

现有 energy sorting 只按 `<2 / <10 / <50 / >=50 MeV/u` 分成4档，不能消除：

- 不同核素和电荷导致的 stopping-power/射程差异；
- 不同方向和解剖路径导致的 track-length 差异；
- cascade 是否发生以及发生位置的差异；
- 少量数万到数十万步长尾轨迹。

Arc 的实际 SIMD 宽度和线程调度方式可能让长尾轨迹造成的组内浪费较小。单独的
SIMD/warp 差异未必足以解释38倍，但会放大寄存器压力和随机内存访问的影响。

### 8.3 巨型 secondary lambda 可能在 CUDA 上发生寄存器溢出

Secondary kernel 是一个大型单体 lambda，同时包含 CT DDA、stopping power、
straggling/MCS、Philox状态、核反应采样、fragment cascade、多套 scorer、队列管理
和 summary 写回。大量捕获参数和长生命周期局部变量会增加每线程寄存器需求。

如果 TITAN RTX 的生成代码发生寄存器溢出：

1. 每 SM 可驻留的 warp 数下降；
2. 不足以隐藏不规则全局内存访问；
3. 溢出变量反复访问 local memory；
4. 长轨迹循环将 spill 成本重复数千次；
5. warp divergence 又会进一步降低有效吞吐。

这种模式可以解释“primary 接近、secondary 慢几十倍”，但目前没有保存
`ptxas` register/spill 报告或 Nsight Compute 指标，所以仍是高优先级假设而不是
已确认结论。

### 8.4 两个原有解释需要重新核对运行来源

第4节提出 portable PTX/JIT 和 FP64 atomic 可能拖慢 CUDA。这个方向本身合理，但
当前本机 `build/oneapi-release/CMakeCache.txt` 显示：

```text
CARBON_CUDA_ARCH=sm_75
CARBON_DOSE_FP32=ON
CARBON_SYCL_TARGETS=spir64,nvptx64-nvidia-cuda
```

当前 CUDA 启动日志也报告 `dose_atomic=fp32`。这与本文第1节声称两边都使用 FP64
voxel-dose atomic 不一致。可能的情况包括：

- 536.51 s 来自更早的 FP64 binary；
- 随后在同一 build 目录重新配置成 FP32；
- 文档记录时混用了不同运行的构建设置。

在找到对应 CUDA `run.log`、`CMakeCache.txt` 和 executable hash 之前，不能把
FP64 atomic 或缺少 `sm_75` AOT 当作本次差距的既定原因。即使它们需要做严格A/B，
其优先级也低于先确认两边的 secondary steps。

### 8.5 不太可能单独解释12倍差距的因素

- **Local size 128 vs 256**：会影响 occupancy，但单独不足以解释38倍 secondary
  差距。
- **Secondary batch size**：配置启用 energy sorting 后，代码会把两个后端都限制到
  `65536`，并不是 Level Zero 整队列单次运行而 CUDA 只跑64k。
- **Queue capacity和显存 clamp**：两端均无 overflow；它们影响可分配容量，不直接
  减少实际输运粒子。
- **CT分辨率/查表**：将 CT 从0.5×2×0.5 mm降至2×2×2 mm、体素减少16.07倍后，
  CUDA 1M full plan 端到端只快0.65%，说明 CT voxel数量不是总瓶颈。
- **WSL submit开销**：WSL和频繁同步可能有成本，但 CUDA 的大部分时间已计入
  secondary kernel event，不能只用主机启动开销解释。
- **纯硬件代差**：primary阶段接近，不支持“B580所有计算天然比TITAN RTX快12倍”。

### 8.6 建议的最小定位实验

不需要先重复10M。应在相同 commit 和输入哈希上做100k或1M受控A/B：

1. 保存两端完整 `run.log`、`CMakeCache.txt`、compiler/plugin/driver版本和 executable
   hash。
2. 强制相同 FP32/FP64、seed、CT、spot weights、reaction/cascade package、
   `secondary_batch_size` 和 local size。
3. 首先比较 `Secondary transport steps` 和平均步数/次级。
4. 用 `CARBON_ENABLE_TRANSPORT_PROFILE=ON` 比较 boundary nudge、CT DDA、
   track-length histogram、dose atomics 和 cascade lookup次数。
5. CUDA 使用 `ptxas`/Nsight Compute 记录 registers/thread、local-memory spill、
   achieved occupancy、warp execution efficiency、branch efficiency 和 atomic
   throughput。
6. 若步数一致但 CUDA 仍慢，依次测试 local size 128/256、FP32/FP64、CUDA-only
   `sm_75` AOT，并考虑按核素、能量和预计track length拆分 secondary kernel。
7. 若 CUDA 步数明显更多，先修复 backend-dependent boundary nudge/thrashing，
   再评估调度和硬件性能。

当前证据下，最可能的组合原因是 CUDA warp 长尾分歧与巨型 secondary kernel 的
寄存器压力/溢出；但在取得 Level Zero secondary step count 前，不能排除 CUDA
特有的边界抖动产生了大量额外步骤，也不能把12.25倍差距归因于 Level Zero API
或 B580 硬件本身。

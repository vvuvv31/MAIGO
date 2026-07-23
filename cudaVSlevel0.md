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

既有 CUDA 结果来自项目历史记录中的 NVIDIA TITAN RTX oneAPI CUDA-plugin 运行。
由于对应的 CUDA `run.log`、构建缓存和可执行文件哈希目前不可用，不能确认
`536.51 s` 这次运行使用的是 FP32 还是 FP64 dose atomic，也不能确认其 PTX
目标参数。剂量结果可以比较，但编译参数不能按当前 Level Zero 构建反推。

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
但历史 `536.51 s` 运行缺少匹配的构建缓存，因此目前不能确认它是否属于这种情况。

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

### 4.5 FP64 atomic voxel scoring

本次 Level Zero 生产构建确认使用 FP64 dose atomic。大量粒子会同时向相同或相邻 voxel 累加，造成原子竞争。若历史 CUDA 运行也使用 FP64，TITAN RTX 的 FP64 吞吐和 oneAPI CUDA plugin 对大型、分歧严重 kernel 的代码生成都可能产生额外开销。

但是历史 CUDA 运行的 atomic 精度目前没有原始构建记录，FP64 atomic 只能列为待验证变量，不能作为已确认根因。

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

## 8. 进一步分析：38.25x secondary 时间差如何构成

从 GitHub 取回的分析指出，1M 性能差距几乎全部集中在 charged secondary
transport。现在已从 Level Zero 原始 `run.log` 补齐 step count，因此可以把差距
进一步量化：

| 1M 指标 | TITAN RTX CUDA（历史记录） | Arc B580 Level Zero | CUDA / Level Zero |
|---|---:|---:|---:|
| 总时间 | 55.48 s | 4.5684 s | 12.15x |
| Secondary kernel | 52.66 s | 1.3768 s | **38.25x** |
| 输运带电次级数 | 2,979,094 | 3,158,734 | 0.943x |
| Secondary transport steps | 约 12,746,000,000 | 1,899,483,953 | **6.71x** |
| 平均 steps/输运次级 | 4,278.5 | 601.3 | **7.11x** |
| Secondary step throughput | 242.0 Mstep/s | 1,379.6 Mstep/s | 0.175x |

这里有两个相乘的差异：

1. 历史 CUDA 记录每个带电次级平均执行约 `7.11x` 更多步骤；
2. 即使按 step 数归一化，CUDA 每步仍约慢 `5.70x`。

`7.11 × 5.70 ≈ 40.5`，与观测到的 `38.25x` secondary kernel 时间差接近；少量
偏差来自两端输运次级数量不同。因此 wall time 差距不是单一原因造成，至少同时
包含“执行的底层工作量不同”和“每一步的执行效率不同”。

### 8.1 已排除 straggling_scale 差异

历史 CUDA 1M 配置使用 `straggling_scale=1.0`，当前 Level Zero full-plan 使用
`1.2`。为判断它是否造成步数差，另外运行了仅把 Level Zero 参数改为 `1.0` 的
1M 诊断：

| Level Zero 1M | scale=1.2 | scale=1.0 |
|---|---:|---:|
| Secondary transport steps | 1,899,483,953 | 1,899,700,770 |
| 输运带电次级数 | 3,158,734 | 3,159,410 |
| Secondary kernel | 1.3768 s | 1.3799 s |
| 总时间 | 4.5684 s | 4.5636 s |

步数只变化约 `0.011%`，所以 straggling scale 不能解释 CUDA 的约 12.746B steps。

### 8.2 目前能够确认的首要原因：历史 CUDA 执行了更多步骤

CUDA 的输运次级数并没有更多，反而比 Level Zero 少约 `5.7%`；但 secondary
steps 多 `6.71x`。这表明差异不是 queue 容量或产生了更多粒子，而是单条次级轨迹
平均循环次数明显更多。

代码中的 secondary 循环曾有 voxel-boundary nudge thrashing 风险，并设置了每粒子
`500000` 步保护。小步路径使用 `nextafter`，在 CT 中还会沿方向移动 `1e-4 mm`。
如果 CUDA 历史 binary 在 DDA face crossing、浮点收缩或边界判断上反复进入该
路径，就可能产生很多几乎不改变剂量的额外循环；这与两端最终剂量高度一致并不
矛盾。

但“CUDA 边界抖动”目前仍不是已证实根因。历史 CUDA 数字来自优化记录，没有与
Level Zero 完全匹配的 commit、完整 `run.log`、输入哈希和 executable hash。
因此另一种同样需要保留的解释是：历史 CUDA binary/config 与当前 Level Zero
代码并不完全相同，step 计数口径或输运逻辑曾经改变。必须在同一 commit 上重跑
CUDA，才能在这两种解释之间作出判断。

### 8.3 剩余约 5.70x 的单位 step 效率差

即使暂时接受两端 step 口径一致，CUDA 仍只有约 `242.0 Mstep/s`，Level Zero 约为
`1,379.6 Mstep/s`。代码审计显示以下因素可能叠加：

- **Warp 长尾分歧**：一个 work-item 完整输运一条轨迹。当前 energy sorting 只按
  四个能量区间分组，无法消除核素、方向、解剖路径、cascade 和 track length 的
  差异；一个 NVIDIA warp 会等待最长轨迹。
- **大型 secondary lambda 的寄存器压力/溢出**：kernel 同时包含 CT DDA、
  stopping power、straggling/MCS、Philox、核反应、cascade、scorer 和队列管理。
  若 PTX 发生 spill，local-memory 成本会在数千次循环中重复，并与 warp divergence
  相互放大。
- **编译目标和代码生成**：oneAPI CUDA plugin 的 PTX 路径与 Intel `spir64`/
  Level Zero 路径不同，最终寄存器分配和控制流质量可能不同。
- **Dose atomic**：FP64 atomic 和 voxel 热点竞争可能影响性能，但历史 CUDA
  运行究竟是 FP32 还是 FP64 目前没有可核验的构建记录。

以上是与观测模式一致的高优先级假设，不是已经由 profiler 证明的事实。缺少
`ptxas` register/spill 报告和 Nsight Compute 指标时，不能准确分配这 `5.70x`。

### 8.4 不足以单独解释差距的因素

- **Local size 128 vs 256** 会改变 occupancy，但不太可能单独造成 `38.25x`
  secondary 时间差。
- 启用 energy sorting 后，两端 secondary batch 都被代码限制为 `65536`。
- 两端 queue 均无 overflow；显存 clamp 影响容量，不会直接减少实际输运步骤。
- WSL submit/sync 有开销，但 CUDA 的主要时间位于 secondary kernel event 内。
- Primary 阶段没有同等级差距，因此不能简单归结为 B580 对所有计算都比
  TITAN RTX 快。

此前从另一环境记录的 `CARBON_CUDA_ARCH=sm_75`、`CARBON_DOSE_FP32=ON` 和
dual-target cache 不能证明 `536.51 s` 运行使用了这些设置：当前工作树中不存在与
该运行匹配的 CUDA cache 或日志。本文不再把这些值当作历史运行事实。

### 8.5 最小闭环实验

下一步不必先重复10M；在有 TITAN RTX 的环境中做同 commit、同输入的100k/1M
CUDA A/B 即可快速定位：

1. 保存两端完整 `run.log`、`CMakeCache.txt`、compiler/plugin/driver版本、输入
   SHA-256 和 executable hash。
2. 固定 FP32/FP64、seed、CT、spot weights、reaction/cascade package、
   `secondary_batch_size`、local size 和所有物理参数。
3. 先比较 `Secondary transport steps`、steps/次级、boundary nudge 次数和
   track-length histogram。
4. 若同代码 CUDA 仍执行约 `6.7x` 更多 steps，集中检查 DDA face crossing、
   `nextafter`、浮点编译选项和 nudge/thrashing。
5. 若 steps 已一致但 CUDA 每步仍慢，使用 `ptxas`/Nsight Compute 检查
   registers/thread、spill、achieved occupancy、warp/branch efficiency 和 atomic
   throughput，再做 local size、FP32/FP64 与 `sm_75` AOT 的受控测试。

当前最可靠的结论是：历史 CUDA secondary kernel 的 `38.25x` 时间差中，约
`7.1x` 来自平均 steps/次级更多，剩余约 `5.7x` 来自单位 step 吞吐更低。
前者究竟是旧 binary/config 差异还是 CUDA 路径的边界抖动，后者究竟有多少来自
warp divergence、register spill、atomic 或 PTX 代码生成，都需要同 commit CUDA
重跑和 profiler 数据才能最终归因；现有证据不足以把 `12.25x` 直接归因于
Level Zero API。

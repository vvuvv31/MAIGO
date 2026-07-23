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

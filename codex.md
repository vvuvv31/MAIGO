# MAIGO TPS 90° CT 优化剂量：Codex 工作记录与后续计划

更新时间：2026-07-20（P0–P3、P2 SP LUT、P8 dense MHD 完成；剂量为总 Gy）

工作目录：`/home/v/project/MAIGO`
开发基线 commit：`6797ec9 Add optimized TPS 90 CT GPU plan`

> 交接说明：仓库先从 `7991dba` fast-forward 到开发基线 `80f6a49`，本文所述的
> TOPAS plan 接口、TPS 90° 坐标转换和 GPU 性能优化包含在与本文一起提交的后续
> commit 中。患者 DICOM、由患者 CT 生成的网格和剂量输出只保留在本地，不进入
> 代码提交。

## 1. 用户目标与必须保持的物理含义

目标是读取 `ct/topas` 中被拆成两段运行的 TOPAS PBS 计划，将两段按原顺序连接，
使用 `ct/spotWeight.csv` 中 dose optimization 后的 spot weight 分配 Monte Carlo
histories，在 Intel Arc B580 上完成患者 CT 的 GPU 剂量计算，并输出 MHD/RAW。

以下约束已经按用户确认实现，后续不要改回去：

- 计划是 **TPS 90° 入射**，不是 0° 入射。
- `spots_c_*.txt` 中的 `RotX/RotY` 是把固定束流建立到 TPS 0° 基准方向。
- `run_c_*.txt` 中的 `Patient/RotZ=90°` 才是 TPS 计划角度。
- TOPAS 的两段只是为并行运行而拆分，必须按
  `c_01`（spot 1–459）→ `c_02`（spot 460–917）连接。
- `spotWeight.csv` 与连接后的 917 个 spot 一一对应。不能给每个 spot 等量 histories。
- 本轮正式统计量按原始 91,700,000 histories 减少两个数量级，即 917,000。

## 2. 已完成的功能

### 2.1 拉取仓库

- 执行了最新代码拉取，master 从 `7991dba` fast-forward 到 `80f6a49`。
- 最近三个提交为：
  - `80f6a49 optimized spot weight`
  - `d5c4a3c upload topas ct script`
  - `c9fbc93 Track required runtime physics data`

### 2.2 TOPAS spot plan 与优化权重

增加了 TOPAS `spots_*.txt` 的 L0–L14 解析和多文件连接能力，涉及：

- `include/carbon/topas_spots.hpp`
- `src/topas_spots.cpp`
- `include/carbon/transport_config.hpp`
- `src/config.cpp`
- `src/main.cpp`

实现的规则：

1. 严格按配置中的文件顺序连接 `spots_c_01.txt` 和 `spots_c_02.txt`。
2. 读取一列式 `ct/spotWeight.csv`。
3. 以 `number_of_histories` 作为整份 plan 的总预算，根据优化权重比例分配，而不是
   把这个数当作每个 spot 的 histories。
4. 正权重 spot 至少获得 1 个 history，剩余数使用 largest-remainder 方法分配，
   因此总数严格等于配置值；零权重 spot 跳过。
5. 每个 spot 保留自己的能量、能散、源位置、束流局部坐标基和 emittance 参数，
   并使用确定性的独立随机数流。

当前输入统计：

- spot 总数：917
- 权重总和：25931.5
- 零权重 spot：64
- 实际参与计算的正权重 spot：853
- 正式总 histories：917,000
- 分配后单 spot histories 范围：1–10,423

可只检查计划而不输运：

```bash
build/cpu-make/carbon_mc \
  --config config/beam_ct_optimized_tps_90.yaml --plan-only
```

### 2.3 TPS 90° 坐标转换与 CT 重排

增加了 `validation/scripts/reorient_ct_grid_tps_90.py`。GPU 输运仍固定沿 +Z，先对 CT
做一次无插值轴置换：

```text
GPU (x, y, z) = patient (y, z, x)
TPS 90° 的 patient +X 束流轴 -> GPU +Z
```

这只是离散轴置换，不做旋转插值，因此 density、material ID 和 CCTG material-table
tail 均保持原值。生成文件：

- `ct/grid/patient_ct_tps_90.bin`
- `ct/grid/patient_ct_tps_90.metadata.json`

生成命令：

```bash
python validation/scripts/reorient_ct_grid_tps_90.py
```

新 CT/scoring 网格为：

- shape：`505 × 35 × 417`
- spacing：`0.5 × 2.0 × 0.5 mm`
- voxel 数：7,370,475
- GPU Z 长度：208.5 mm

对应生产配置为 `config/beam_ct_optimized_tps_90.yaml`。配置中保留了 TOPAS patient
translation、`Patient/RotZ=90°`、SAD=450 mm 和原 CT X 轴最小坐标，用于逐 spot
构造与 TOPAS 一致的源位置和方向。

### 2.4 优化权重下的 CT 剂量计算

生产配置启用了：

- CT energy-dependent mass stopping power
- energy straggling
- multiple scattering
- primary attenuation
- direct secondary generation 和 charged-secondary transport
- fragment cascade，最多 2 代
- 逐体素 scoring
- double-precision dose atomic accumulation

3D Gy 使用每个 CT voxel 的 Schneider density 计算 dose-to-medium，不使用统一水密度。
剂量输出已改为**全部 history 的总 Gy**（不再除以粒子数）。若要转换为某个治疗
fraction 的临床绝对 Gy，还需要 TPS spot weight/MU 到实际碳离子数的标定；仅凭这份
单列权重不能推出该绝对归一化。

### 2.5 正式 917k 结果与 MHD 输出

曾用优化前的逐 spot GPU launch 实现完成一次 917,000-history 正式计算，结果未被
后续 smoke benchmark 覆盖：

- 总耗时：4590.4568 s（约 76.5 min）
- 吞吐量：199.762 histories/s
- primary kernel：871.4406 s
- secondary kernel：3691.7352 s
- 总 steps：181,407,308,328
- secondary steps：78,858,617,768
- energy-balance relative error：`9.8986e-6`
- queue overflow：0

剂量结果：

- 非零 voxel：3,681,806 / 7,370,475
- 最大剂量：`1.07505168367e-08 Gy/primary`
- 最大值索引：`(156, 15, 206)`
- 最大值坐标：`(-48, -4, 103.25) mm`
- IDD 最大值深度：125.75 mm

已输出：

- `out/ct/optimized_tps_90_dose.mhd`
- `out/ct/optimized_tps_90_dose.raw`（29,481,900 bytes）

MHD 信息：

- `DimSize = 505 35 417`
- `ElementSpacing = 0.5 2 0.5`
- `Offset = -126 -34 0.25`
- little-endian `MET_FLOAT`
- 单位：Gy/primary
- RAW 中所有值均 finite、无负值，最大值和索引与 CSV 一致

原始 IDD、species 和 sparse voxel CSV 也保存在 `out/ct/optimized_tps_90_*`；其中两个
voxel CSV 分别约 224 MB 和 158 MB。`out/` 被 git ignore，不会出现在 `git status`。

## 3. 已完成的性能优化

慢的主要原因不是 I/O，而是输运步数和 GPU 利用率：旧正式计算约
197,827 steps/primary；secondary 占旧运行时间约 80.4%，平均每个 transported
secondary 约 49,765 steps。与此同时，853 个正权重 spot 被逐个 launch，很多 spot
的 histories 很少，Arc B580 严重欠占用。

按之前讨论的编号，已经实现 1、2、5、6：

### 3.1 优化 1：整份 spot plan 批处理

- 新增 `PrimarySpotBatchEntry`，保存每个 spot 的 flattened history range、seed、
  能量/能散、emittance、source origin 和局部束流基。
- 默认将所有有效 spot 展平成一个 history range，一次调用 SYCL transport。
- primary kernel 对每个 history 二分定位所属 spot，然后使用该 spot 的完整源参数。
- 保留 `--sequential-spots`，只用于正确性/性能 A/B。
- config validation 要求 batch ranges 连续、非空并且严格覆盖总 histories。

这项改造消除了 853 次小 kernel launch 和大量低 occupancy 工作组。

### 3.2 优化 2：persistent secondary workers

- secondary kernel 改为固定 worker pool。
- worker 使用 device atomic counter 动态领取下一个 charged-secondary track，完整输运
  后继续领取，减小不同 track 长度引起的 SIMD 空等。
- backend 标签增加 `+secondary-transport-persistent`。

注意：当前实现是 **track-level persistent scheduling**，还不是真正的 step-level
wavefront。长尾 track 仍会占住执行线程，这是后续还有优化空间的主要位置。

### 3.3 优化 5：减少 dose atomic 次数

- primary 在线程本地合并连续落入同一 depth bin 的能量，再做一次 FP64 atomic。
- primary 在线程本地合并连续落入同一 voxel 的能量，再做一次 FP64 atomic。
- secondary 继续使用 pending voxel aggregation。
- 保留 FP64 全局累积，避免大量 histories 下 FP32 scorer 的累积误差。

### 3.4 优化 6：近 +Z primary 的 CT face fast path

- 新增 `clamp_step_to_ct_faces_near_z_if_needed()`。
- 当 `abs(dz) >= 0.999` 且 proposed endpoint 可证明仍在同一个 CT x/y cell 时，只检查
  z face；否则回退到原来的精确三轴逻辑。
- 加入 endpoint bounds 检查，避免负坐标转整数时的截断错误。
- 当前只用于 primary，secondary 仍走通用路径。

### 3.5 优化后的实测（smoke 9,170）

在 Arc B580 OpenCL backend 上做了 9,170-history smoke：

- elapsed：11.615174 s
- throughput：789.485 histories/s
- primary kernel：1.607736 s
- secondary kernel：7.910649 s
- total steps：1,824,811,995，约 199k/history
- transported charged secondaries：15,907
- secondary steps：794,310,177，约 49.9k/secondary
- energy-balance relative error：`1.32868e-4`
- queue overflow：0

同一 9,170-history 的 sequential-spots A/B 运行超过 3 分钟仍未完成，因此主动终止；
在这种小 histories/spot 场景下，batch 至少快约 15 倍。

### 3.6 P0：正式 917k batch+persistent 基线（已锁定）

配置：`config/beam_ct_p0_batch_persistent_917k.yaml`（输出到
`out/ct/p0_batch_persistent_917k/`，不覆盖优化前结果）。

优化前 917k 备份在 `out/ct/preopt_917k_baseline/`（由旧逐 spot launch 得到）。

#### 性能（Arc B580 OpenCL）

| 指标 | 优化前 (preopt) | P0 batch+persistent | 加速比 |
| --- | ---: | ---: | ---: |
| wall elapsed | 4590.4568 s (~76.5 min) | **144.73381 s (~2.4 min)** | **31.7×** |
| throughput | 199.762 h/s | **6335.7692 h/s** | **31.7×** |
| primary kernel | 871.4406 s | 67.223716 s | 13.0× |
| secondary kernel | 3691.7352 s | 77.30176 s | 47.8× |
| secondary 时间占比 | 80.4% | 53.4% | — |

其他：

- backend：`sycl-gpu+...+spot-batch+...+secondary-transport-persistent+fragment-cascade`
- total steps：181,457,432,330（约 197.9k/history，与优化前 ~197.8k 同量级）
- secondary steps：78,908,909,936（约 49.8k/transported secondary）
- transported charged secondaries：1,585,029
- nuclear interactions：195,136
- queue overflow：0（secondary / cascade）
- energy-balance relative error：`9.8985301e-6`（与优化前 `9.8986e-6` 一致）
- device memory estimate：2020 MiB / budget 5802 MiB（50% of 11605 MiB）
- host max RSS（`/usr/bin/time -v`）：475344 kB
- wall clock with I/O（time）：2:35.62（含 CSV 写出）

说明：算法 steps/history **没有**因 batch 降低；吞吐提升主要来自 launch 合并与
secondary persistent scheduling，以及 atomic 合并 / near-Z fast path。

#### 剂量对比（preopt vs P0，同 seed/config 917k）

工具：`validation/scripts/compare_p0_ct_baseline.py`  
报告：`out/ct/p0_batch_persistent_917k/compare_vs_preopt.json`

IDD（Gy/primary）：

- peak depth：两边均为 125.75 mm
- peak 相对差：−0.0497%
- R80 差：+0.0119 mm
- 积分相对差：−0.0102%
- NRMSE（相对 peak）：0.0217%
- 1%/1 mm 与 2%/2 mm gamma（thr 10%）：**100%**

3D voxel（MHD，505×35×417）：

- 峰值索引：两边均为 `(156, 15, 206)`
- 最大剂量相对差：−0.646%（单 voxel 峰值，MC 噪声量级）
- 3D 积分相对差：−0.0144%
- 非零 voxel：3,681,806 → 3,676,485
- 全局 max 归一 RMSE：0.142%
- ≥10% peak 区 mean |rel|：1.16%
- 2%/2 mm gamma（thr 10%，随机抽样 50k / 182,567）：**99.994%**

结论：**P0 剂量基线通过**。后续 P1–P7 性能改动应以本 917k 结果为对照，而不是优化前
逐 spot 版本。未要求 bit-identical（secondary 调度/FP 归约顺序会变），以 gamma 与
积分指标验收。

产物（本地，git ignore）：

- `out/ct/p0_batch_persistent_917k/{idd,idd_Gy,species,species_Gy,voxels,voxels_Gy}.csv`
- `out/ct/p0_batch_persistent_917k/dose.mhd` + `dose.raw`
- `out/ct/p0_batch_persistent_917k/run.log`、`time.txt`、`compare_vs_preopt.json`
- `out/ct/preopt_917k_baseline/*`（优化前对照，勿删）

重跑命令：

```bash
ONEAPI_DEVICE_SELECTOR='opencl:gpu' build/perf-make/carbon_mc \
  --config config/beam_ct_p0_batch_persistent_917k.yaml --device gpu
```

## 4. 验证状态

已完成：

- GCC CPU build 成功：`build/cpu-make`
- CPU CTest 全部通过
- oneAPI `icpx -O3 -DNDEBUG` SYCL Release build 成功：`build/perf-make`
- Arc B580 OpenCL backend 上完整 SYCL CTest 通过
- 新增 near-Z heterogeneous z-face 测试
- 新增 batched primary 与分开运行 primary 的等价性测试
- `git diff --check` 通过
- 临时 benchmark 使用 `/tmp` 配置和输出，没有覆盖正式 dose

测试命令：

```bash
ctest --test-dir build/cpu-make --output-on-failure
ONEAPI_DEVICE_SELECTOR='opencl:*' \
  ctest --test-dir build/perf-make --output-on-failure
```

设备说明：当前机器 `sycl-ls` 能看到 Intel Arc B580。Level Zero backend 曾返回
unsupported/error 44；本轮可用且通过测试的是 OpenCL GPU backend，运行时使用：

```bash
ONEAPI_DEVICE_SELECTOR='opencl:gpu' build/perf-make/carbon_mc \
  --config config/beam_ct_optimized_tps_90.yaml --device gpu
```

## 5. 本次代码提交范围与本地数据

主要修改文件：

- `README.md`
- `furtherStep.md`
- `include/carbon/ct_grid.hpp`
- `include/carbon/topas_spots.hpp`
- `include/carbon/transport_config.hpp`
- `src/config.cpp`
- `src/io.cpp`
- `src/main.cpp`
- `src/topas_spots.cpp`
- `src/transport_sycl.cpp`
- `tests/carbon_tests.cpp`

本次代码提交新增的关键文件：

- `config/beam_ct_optimized_tps_90.yaml`
- `validation/scripts/reorient_ct_grid_tps_90.py`

以下内容是本地患者输入、生成物或正式结果，不纳入代码 commit：

- `ct/topas/`，其中包括患者 DICOM、TOPAS run/spots 文件和提交脚本
- `ct/grid/patient_ct_tps_90.bin`
- `ct/grid/patient_ct_tps_90.metadata.json`
- git ignored 的 `out/ct/optimized_tps_90_*`

不要删除或覆盖这些文件以及 `ct/spotWeight.csv`。

## 6. 下一步优化计划

下面按收益、风险和依赖排序。建议每一步都单独做 A/B，不要同时改多个物理或输运
路径，否则很难定位剂量差异。

### P0：正式性能/剂量基线 — **已完成（见 §3.6）**

1. ~~用当前 batch + persistent 版本重跑完整 917,000 histories。~~
2. ~~记录 wall time、kernel 时间、steps、队列、overflow、energy balance、显存估计。~~
3. ~~与优化前 917k CSV/MHD 比较 IDD、3D gamma、R80、积分、峰值。~~
4. 基线已锁定；后续改动对照 `out/ct/p0_batch_persistent_917k/`。

### P1：step 分类计数和 profiling — **已完成**

#### 实现

- CMake 选项：`-DCARBON_ENABLE_TRANSPORT_PROFILE=ON` → 定义 `CARBON_TRANSPORT_PROFILE=1`
- 头文件：`include/carbon/transport_profile.hpp`
- 源文件：`src/transport_profile.cpp`
- CT face 路径标签：`clamp_step_to_ct_faces*` 可选 `CtClampPath* path_out`
- SYCL kernel 内 device atomic 计数；**默认 OFF**，正式 dose 用 `build/perf-make`（profile OFF）
- 独立 profile 构建：`build/profile-make`
- 运行结束打印 `Transport profile counters:`（见 `main.cpp`）

计数项（primary/secondary 分离）：

- steps（完整物理步，不含 boundary-nudge continue）
- CT sample、SP table、mass-SP 查表
- face clamp 分类：three_axis / short_step_skip / homogeneous_skip / near_z_*
- dose atomic flush（pending 合并后）
- straggling / MCS / nuclear / cascade
- secondary track step-count histogram（2^b 桶，含 thrash 在内的 loop steps）

构建与 smoke：

```bash
# 生产（无 profile）
cmake -S . -B build/perf-make -DCMAKE_BUILD_TYPE=Release \
  -DCARBON_ENABLE_SYCL=ON -DCARBON_ENABLE_TRANSPORT_PROFILE=OFF \
  -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/2026.1/bin/icpx
cmake --build build/perf-make -j

# profile 诊断构建
cmake -S . -B build/profile-make -DCMAKE_BUILD_TYPE=Release \
  -DCARBON_ENABLE_SYCL=ON -DCARBON_ENABLE_TRANSPORT_PROFILE=ON \
  -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/2026.1/bin/icpx
cmake --build build/profile-make -j

ONEAPI_DEVICE_SELECTOR='opencl:gpu' build/profile-make/carbon_mc \
  --config config/beam_ct_p1_profile_smoke.yaml --device gpu
```

#### 917-history profile smoke 关键发现（Arc B580）

配置：`config/beam_ct_p1_profile_smoke.yaml`（917 histories，同生产物理）
日志：`out/ct/p1_profile_smoke/run.log`

| 计数 | Primary | Secondary |
| --- | ---: | ---: |
| 完整物理 steps | 667,340 | 1,340,923 |
| CT samples / SP lookups | ~115.5M | ~79.3M |
| face three_axis | 114,817,517 | 77,966,992 |
| face short_step_skip | 544,049 | 1,178,892 |
| face homogeneous_skip | 352 | 131,158 |
| face near_z_* | ~122k | 0（secondary 不用 near-Z） |
| boundary_nudge_continues | **114,817,144** | （含在 loop steps） |
| dose atomics（flush 后） | depth 99k / voxel 100k | 136k |
| MCS | 667k | 1.34M |

解释：

1. **Primary 约 99.4% 的循环是 CT face thrash**（nudge continue），不是完整 dE/dx 步。
   `primary_steps`（完整物理）仅 0.67M，而 `primary_ct_samples` ~115M。
2. thrash 路径被记为 `three_axis`，因为 `step_mm <= 1e-6` 时 near-Z / homogeneous
   会提前落到通用 face clamp（见 `ct_grid.hpp`）。
3. near-Z 快路径本身有效，但在 thrash 循环中几乎用不上（~0.1%）。
4. Secondary 同样以 three_axis face 为主；track histogram 显示 **205 条 track 落在
   [32k, 65k) steps**，长尾严重，支持后续 P4 wavefront。
5. Dose atomic 已因 pending 合并降到 ~每完整步 0.15 次量级，**不是当前主瓶颈**。
6. Profile 构建因全局 atomic 计数吞吐会明显下降（smoke ~64 h/s）；**不要用 profile
   构建跑正式 917k dose**。

#### 对后续优先级的影响

| 项 | 调整 |
| --- | --- |
| **P3 DDA / span** | **提到最优先**。消灭 face thrash 与重复 floor/clamp，直接砍 steps/history。 |
| **P2 lookup table** | 仍重要（每完整步都有 mass-SP + log），但完整物理步远少于 thrash 循环；DDA 后再做更划算。 |
| **P4 wavefront** | secondary 长尾 track 已证实，DDA 降 steps 后再做。 |
| **P5–P7** | 不变，优先级低于 P3→P2→P4。 |

### P2：预计算 CT mass-SP lookup table — **已完成**

实现：

- 加载 CCTG mass-SP 因子时，在 host 上对每个 Schneider section × water-SP 能量点
  预计算 `ct_mass_sp_energy_factor(za, I, E)`，上传设备常驻 LUT。
- 布局：section-major，`size = n_sections × table_size`（本患者 25 × ~400）。
- kernel 中只做 `clamp / index / lerp`，不再每步 `log`/Bethe 求 mass factor。
- primary 与 secondary 共用同一 LUT；backend 标签：`+ct-grid-mass-sp-lut`。

精度（917k，相对 P3 on-the-fly mass-SP）：

- IDD peak 深度一致 125.75 mm
- peak 相对差 **+0.003%**
- NRMSE **0.0007%**

性能：kernel 与 P3 同量级（~1.8 s）；首跑 JIT 冷启动可能偏慢，热跑 transport
elapsed **~1.96 s / ~469k h/s**。

### P3：CT integer DDA / homogeneous span — **已完成**

实现（`include/carbon/ct_grid.hpp`）：

- `CtDdaState` + `ct_dda_init` / `ct_dda_step` / `ct_dda_homogeneous_span`
  （Amanatides–Woo）
- face 上 t≈0 时自动进入下一 voxel，消灭 boundary thrash
- 同 material/density 连续 voxel 合并为 span（最多 64 hop）
- primary/secondary 的 `clamp_step_to_ct_faces*` 均走 DDA
- CT 微步接受阈值降到 `1e-8`（避免合法微步被 nudge thrash）
- backend 标签：`+ct-dda`

#### 917-history profile（DDA 后 vs DDA 前）

| 计数 | DDA 前 | DDA 后 |
| --- | ---: | ---: |
| total steps | 194.8M | **2.01M** |
| primary_ct_samples | 115.5M | **0.67M** |
| primary_boundary_nudge | 114.8M | **0** |
| primary_face_three_axis | 114.8M | 251 |

#### 正式 917k（`config/beam_ct_p3_dda_917k.yaml`）

| 指标 | P0 batch+persistent | **P3 + DDA** |
| --- | ---: | ---: |
| elapsed | 144.7 s | **2.078 s** |
| throughput | 6336 h/s | **441,256 h/s** |
| primary kernel | 67.2 s | 0.961 s |
| secondary kernel | 77.3 s | 0.913 s |
| total steps | 181.5B | **2.061B** |
| steps/history | ~198k | **~2248** |
| energy balance | 9.90e-6 | 9.91e-6 |
| overflow | 0 | 0 |

相对 P0 剂量（P0 为 Gy/primary，×917k 对齐总剂量）：

- peak depth 一致 125.75 mm
- peak 相对差 **−0.098%**
- 积分相对差 **~0%**
- NRMSE **0.029%**
- R80 差 **+0.019 mm**

产物：`out/ct/p3_dda_917k/`（含 `dose.mhd`/`dose.raw`，单位 **Gy** 总剂量）。

### 剂量输出：总剂量（不再 /histories）— **已完成**

所有 CSV dose 列改为**全部 history 的总沉积**，不再除以 `number_of_histories`：

- 列名：`dose_Gy`、`energy_deposition_MeV`（去掉 `_per_primary`）
- MHD `DoseUnits = Gy`
- 涉及：`src/io.cpp`、`include/carbon/io.hpp`、相关测试与脚本默认值

说明：这是相对当前模拟统计量的绝对剂量标度；若要临床 fraction Gy，仍需 TPS MU→离子数标定。

### 与 `ct/physical_dose` 的坐标对齐

参考剂量 `ct/physical_dose.mhd`：

- `DimSize = 104 126 35`，`Spacing = 2 2 2`，`Offset = -102.75 -43.55 -814.19`
- 轴为 **patient (X,Y,Z)**（与 DICOM/RAI 一致）；束流沿 **patient X** 拉长

GPU 内部剂量为重排后的：

- `DimSize = 505 35 417`，`Spacing = 0.5 2 0.5`
- `GPU (x,y,z) = patient (y,z,x)`，束流沿 **GPU +Z = patient +X**

因此直接对比两个 MHD 会像“角度错了”。**束流相对 CT 解剖的入射与 physical 一致**；
需要把 GPU 剂量映回 patient 轴后再叠图：

```bash
python validation/scripts/gpu_dose_to_patient_mhd.py \
  out/ct/p2p8_sp_lut_mhd/dose.mhd \
  out/ct/p2p8_sp_lut_mhd/dose_patient_like_physical.mhd \
  --like ct/physical_dose.mhd
```

该映射含 patient X/Y 翻转以匹配参考剂量；与 `physical_dose` 的 3D cosine ≈ **0.93**，
IDD 峰值深度一致（约 bin 46–52）。绝对 Gy 标度仍取决于 MU/离子数标定，不能直接比数值大小。

### P8：直接写 dense MHD/RAW — **已完成**

- API：`write_dense_voxel_dose_mhd()`（`src/io.cpp`）
- 配置：`voxel_dose_mhd_output_file: out/.../dose.mhd`
- 空路径禁用任意可选输出（CSV / MHD）；`voxel_dose_output_file:` 可关掉 sparse CSV
- 布局与旧 `sparse_dose_to_mhd.py` 一致：`DimSize X Y Z`，RAW z-major float32 LE，
  `Offset` = 第一体素中心，单位总 **Gy**（dose-to-medium，用 CT density）
- 生产配置 `config/beam_ct_optimized_tps_90.yaml` 与
  `config/beam_ct_p2p8_sp_lut_mhd.yaml` 默认写 MHD、关闭 sparse voxel CSV

917k 产物示例：`out/ct/p2p8_sp_lut_mhd/dose.mhd` + `dose.raw`（29,481,900 bytes）

### P4：真正的 step-level secondary wavefront

当前 persistent worker 仍让一个线程完整跑完一条 track。下一版可改为：

1. 每个 worker 每次只推进固定 quantum（例如 32/64/128 steps）。
2. 未结束 track 写回 active queue。
3. compact 后进入下一 wave；可选按 species、energy bucket 或 CT section 排序。
4. 结束、逃逸和新生粒子进入各自队列。

这样可限制超长 track 对 SIMD lane 的占用，但会增加 queue traffic 和 compaction 成本。
应对 32/64/128 quantum 做 A/B，并与 track-level persistent 比较。

### P5：kernel 编译期特化

为生产配置生成专用 kernel variant，把运行时恒定分支编译掉，例如：

- CT on、layered/insert off
- voxel scoring on
- secondary/cascade on
- neutral transport off
- mass-SP-e on

可使用模板参数或少量明确的 kernel variants，避免组合爆炸。检查生成代码和寄存器占用，
防止因过度内联降低 occupancy。

### P6：进一步优化 tally

评估 work-group local/tiled tally 或分层 sparse tally：先在 local memory 合并热点 voxel，
再 flush 到 FP64 global scorer。患者 CT 的访问较分散，必须测 hit rate；若 local hash
冲突或 flush 太多，可能不如当前 per-thread pending aggregation。

### P7：可选 FP32 tally + FP64 reduction

只作为可选 fast mode：每批使用 FP32 scorer，批间转换并累积到 FP64。必须与当前
全 FP64 atomic 基线比较最大 voxel 相对误差、IDD、gamma 和统计重复性。不能未经验证
替代最终剂量模式。

### P8：直接写 dense MHD/RAW

增加正式的 dense MHD/RAW writer，直接从内存 dose grid 输出，跳过约 382 MB sparse
CSV 的格式化和后处理。这主要减少收尾时间与磁盘占用，不会解决 kernel 内约 199k
steps/history 的核心瓶颈，因此优先级低于 P2–P4。

## 7. 仅列入 future plan、尚未实现的物理近似

这是用户此前要求“把 3 和 4 写进 future plan”的部分。当前生产配置仍保持
`energy_cutoff_MeV=0.1`、`maximum_relative_energy_loss=0.005` 和 full cascade。

### Future 3：低能 cutoff / 最大相对能损 A/B

- 测试 `energy_cutoff_MeV=1.0`。
- 测试 `maximum_relative_energy_loss=0.01`，必要时再测 0.02。
- cutoff 以下剩余能量局部沉积到当前 voxel。
- 必须比较 3D gamma、R80、D95/D2、积分剂量、峰值位置和能量闭合。
- 通过预定阈值后只能作为 fast-dose 配置，不能自动替代最终剂量配置。

### Future 4：分级物理模式

- `preview-primary`：只输运 primary，用于坐标、范围和快速预览。
- `direct-secondary`：输运直接带电次级，不做 fragment cascade。
- `full-cascade`：当前完整模型，最终报告默认使用。

每个输出必须记录 backend/physics-tier 标签以及关闭的能量通道，避免把 preview dose
误当作最终剂量。

## 8. 建议的统一验收门槛

每个性能改动至少检查：

- 构建和 CPU/SYCL 测试全部通过。
- queue overflow 为 0。
- energy-balance error 不劣于当前统计噪声可解释范围。
- 与固定 seed 的当前 full-physics 基线比较：IDD、R80、积分能量、3D dose、最大 voxel
  和峰值位置。
- 报告 2%/2 mm gamma；若用于最终剂量，再增加更严格 gamma 或临床 ROI 指标。
- 同时报告 wall time、kernel time、histories/s、steps/history 和每个 transported
  secondary 的 steps，避免把减少物理工作量误报为纯 GPU 加速。

交给下一位开发者时，最稳妥的顺序是：**P0–P3、P2、P8 已完成 → P4 step-level
wavefront → P5 kernel 特化 → Future 3/4 物理近似**。坐标和优化权重分配已经明确，
不应在性能重构中改变。

已锁定事实：

- 917k 热跑 transport **~2 s / ~450k+ h/s**（DDA + mass-SP LUT）。
- steps/history **~2.2k**（原 thrash 路径 ~198k）。
- 剂量输出为 **总 Gy**；3D 默认 **dense MHD**（不再写 380MB sparse CSV）。
- 下一步瓶颈更接近 secondary 长尾与 SP/charge 公式本身（P4 / P5）。

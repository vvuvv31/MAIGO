# MAIGO TPS 90° CT 优化剂量：Codex 工作记录与后续计划

更新时间：2026-07-20

工作目录：`/home/v/project/MAIGO`
开发基线 commit：`80f6a49 optimized spot weight`

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
当前结果单位是 `Gy / total sampled primary`。若要转换为某个治疗 fraction 的绝对
Gy，还需要 TPS spot weight/MU 到实际碳离子数的标定；仅凭这份单列权重不能推出该
绝对归一化。

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

### 3.5 优化后的实测

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
在这种小 histories/spot 场景下，batch 至少快约 15 倍。与旧 917k 正式运行的约
200 histories/s 相比，smoke 达到约 3.9 倍吞吐，但两个统计规模不同，**不能把
789 histories/s 当作完整 917k 的已验证吞吐**。优化后的 917k 正式计算尚未重跑。

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

### P0：先锁定新的正式性能/剂量基线

1. 用当前 batch + persistent 版本重跑完整 917,000 histories。
2. 记录 wall time、每个 kernel 时间、总/primary/secondary steps、各队列数量、overflow、
   energy balance 和显存峰值。
3. 与现存优化前 917k CSV/MHD 比较 IDD、3D gamma、R80、积分剂量、最大值和峰值位置。
4. 只有通过该基线，后续性能改动才有可靠的比较对象。

### P1：增加低开销的 step 分类计数和 profiling

为 primary/secondary 分开统计：

- stopping-power/material lookup 次数
- CT sample 次数
- homogeneous skip、near-Z fast path、three-axis fallback 次数
- face clamp 次数和 short-step 次数
- dose atomic flush 次数
- nuclear/MCS/straggling 分支次数
- track step-count histogram 或分位数

profiling counters 应能通过编译选项关闭，正式 dose 不承担原子计数开销。先用这些数据
确认热点，再决定 DDA、查表或 wavefront 的优先级。

### P2：预计算 CT material/species/energy lookup table

将每步重复的能量相关 mass stopping power、截面及相关昂贵函数，预采样成设备常驻的
`material/section × species × energy-bin` 表，kernel 中只做 clamp、索引和线性插值。

实施要点：

- energy grid 要覆盖 plan 的 165–240 MeV/u 以及所有 secondary 能区。
- primary C-12 和 fragment species 可分别使用不同分辨率。
- 先用较密网格建立精度基线，再逐步减小表。
- 验证表插值对 R80、峰值和 fragment dose 的误差。

这是目前最值得优先尝试的单步计算成本优化，因为 batch 并没有降低约 199k
steps/history 的算法工作量。

### P3：secondary 通用 CT traversal 改为 integer DDA / homogeneous span

当前 near-Z fast path 主要帮助 primary，secondary 方向任意且仍执行通用 face 检查。
建议为每条 track 保存当前 voxel integer index 和到下一 x/y/z face 的参数距离，使用
Amanatides-Woo 类 DDA 更新；同 material/density section 的连续 voxel 可合并为 span。

目标：

- 避免每一步重复 floor/除法和三轴边界重算。
- 避免跨面后 epsilon nudge 导致的边界 thrash。
- 必须保持不跨越异质 CT voxel，不允许以性能为由漏采样材料边界。

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

交给下一位开发者时，最稳妥的顺序是：**P0 完整基线 → P1 profiling → P2 lookup
table → P3 DDA → P4 step-level wavefront**。坐标和优化权重分配已经明确，不应在性能
重构中改变。

# Copper minibeam 接入与验证结果

本文记录 `pristine` 分支从 `352fa95` 开始完成的 Copper minibeam GPU
输运、TOPAS 数据提取、多能量剂量验证和水中散射修正。本文中的大体积
TOPAS/GPU 输出保存在本机 `/mnt/sda/wuwei` 和仓库忽略的 `out/` 下，不随
Git 提交；生成配置、分析脚本和汇总数据随代码提交。

## 1. 实现范围

### 1.1 可选编译

minibeam 通过 CMake 选项 `CARBON_ENABLE_MINIBEAM` 编译，默认关闭，因此
普通 CT、水箱和 LET 构建不会装载 Copper 专用事件库，也不会在粒子热路径
增加运行时负担。

本次 NVIDIA 验证固定使用：

```text
CARBON_ENABLE_MINIBEAM=ON
CARBON_DOSE_FP32=ON
CARBON_DOSE_FP64=OFF
CARBON_DISABLE_INTEGRITY_CHECKS=ON
```

完整性检查当前由 CMake 强制关闭；运行时会打印醒目的
`[integrity-checks] DISABLED`。这满足当前开发需求，但正式分发前应重新评估
是否恢复 SHA256 内容验证。

### 1.2 几何和调用参数

GPU 实现使用与 TOPAS 一致的半径 60 mm 圆柱 Copper 外体；有限长矩形 slit
几何保持一致。新增配置参数包括：

- collimator 几何中心到 isocenter 的距离；
- collimator 长、宽、厚；
- slit 长、宽、厚；
- slit 数量、间距和整体偏移；
- collimator 旋转角；
- water entrance 的世界坐标位置。

GPU 支持：

- `absorbing_geometry`：精确直线 aperture/吸收几何，用于 smoke test；
- `copper_em`：Copper 中连续能损、MCS、核衰减和产物输运；
- TOPAS/TPS spot plan、BiGaussian emittance、能散和 IEC 坐标变换；
- 水入口 primary phase-space 输出及诊断计数；
- dense primary track-length fluence MHD 输出。

### 1.3 Copper 和水中物理

实现和接入的主要物理项为：

- Geant4 11.3.2 提取的 C12/Copper stopping power 和 inelastic rate；
- 多离子 Copper stopping power 与 inelastic cross section 表；
- Copper primary 的 Fermi--Eyges/tail MCS，以及独立的 fragment Highland MCS；
- 自建 General Ion Elastic 事件库接口；
- INCL++ Copper 反应事件包和带电产物输运；
- Copper primary 逐步局部能损校正（旧出口 survivor 回补已在第 10 节移除）；
- 水中 Unified EM、energy straggling、primary/secondary inelastic；
- `cinel02_max_secondary_inelastic_generations=2`，允许碎片继续发生一次反应，
  同时保留末代 charged product 的 EM 输运；
- 水中 primary C12 的 step-size-aware Gaussian core/rare-wide-tail MCS。

正式 150/250/300 MeV/u 配置均关闭 GPU discrete Copper elastic。独立相空间
验证表明当前有限事件库会产生过强宽角尾；condensed Copper MCS 仍启用。

## 2. TOPAS reference 和统计口径

TOPAS example 位于 `benchmark/carbonminibeam/`，使用：

- 3 cm × 3 cm PBS field；
- 15 个 slit，宽 0.5 mm、pitch 3.6 mm；
- 250 mm 水箱；
- lateral/depth spacing = 0.1/0.25 mm；
- Geant4 11.3.2、OpenTOPAS 4.2.p3；
- `g4em-standard_opt4`、INCL++、自建 General Ion Elastic 及完整 hadronic 模块；
- 本地 Slurm 192 threads。

比较脚本从 MHD/TOPAS header 自动读取 grid spacing，峰谷曲线使用 1 mm
depth smoothing，中心九个 peak 和八个 valley。所有表中比值均为 GPU/TOPAS，
不做拟合归一化。

## 3. 完整 Copper+water 多能量结果

开启 fragment generation 2 后，原 10M、tail width 0.85 基线为：

| Energy (MeV/u) | Total dose | 2-D L1 | IDD L1 | Lateral L1 | Bragg peak | Bragg valley | Bragg PVDR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 150 | 0.9955 | 3.663% | 0.946% | 1.800% | 1.003 | 1.102 | 0.910 |
| 250 | 0.9902 | 3.919% | 1.001% | 1.798% | 0.988 | 1.056 | 0.936 |
| 300 | 0.9887 | 3.800% | 1.143% | 1.631% | 0.966 | 1.031 | 0.938 |

Bragg depth 为 51.375/124.375/168.625 mm；GPU 对应为
51.375/124.375/168.875 mm。总剂量、射程和 IDD 已接近，但 250/300 的
Bragg PVDR 仍系统偏低。

fragment generation 2 是必要修复。它将 250/300 MeV/u distal total ratio
从 1.146/1.131 改善到 0.992/0.989，将 charged-fragment distal ratio 从
1.192/1.170 改善到 1.032/1.023。该差距不是“secondary Unified EM 未开启”，
而是此前遗漏了 fragment reinteraction。

## 4. 水入口同源 replay 隔离

### 4.1 相空间筛选修正

最初使用 `GenericIon(6,12,6)` 会只保留 charge-state 6 的 C12，并遗漏低能、
宽角 primary。修正为 `GenericIon(6,12,*)` 后，10,000,128 个 incident
histories 得到：

- 1,202,596 个 C12 crossing；
- 1,202,572 个 parent-ID 0 C12 replay histories；
- surviving fraction 0.1202557；
- mean/std energy = 219.00/54.22 MeV/u；
- minimum energy = 0.0041 MeV/u。

GPU 和 TOPAS 均仅 replay 这 1,202,572 个 non-empty histories；若需要 incident
normalization，再共同乘 surviving fraction，不影响 GPU/TOPAS 比值。

### 4.2 tail width 0.85 的高统计 replay

| Metric | GPU/TOPAS or error |
|---|---:|
| Total dose | 0.9969 |
| 2-D L1 | 3.358% |
| IDD L1 | 0.499% |
| Lateral-integral L1 | 1.172% |
| Bragg peak | 0.978 |
| Bragg valley | 1.034 |
| Bragg PVDR | 0.946 |

旧 30,878-primary replay 的 Bragg valley/PVDR `0.808/1.258` 主要是统计噪声，
不能用于模型调参。高统计结果确认稳定误差为 GPU primary 随深度轻微过展宽。

## 5. Geant4 逐深度 primary 约束

TOPAS 在水深 40、80、100、115、124 mm 记录 parent C12 crossing；GPU 用
primary track-length fluence MHD。普通 RMS 会被完整 slit array 的宽度主导，
因此分析使用 3.6 mm 基频、二次谐波和折叠到单 pitch 的 profile。

tail width 0.85 的 primary 基频比为：

| Depth (mm) | 40 | 80 | 100 | 115 | 124 |
|---:|---:|---:|---:|---:|---:|
| GPU/TOPAS fundamental | 0.997 | 0.968 | 0.948 | 0.947 | 0.953 |

误差从约 80 mm 开始累积，105–115 mm 的剂量 peak 低约 3%、valley 高约 9%。
这排除了单纯入口相空间修正，并证明当前 Gaussian core 过宽。

基于上述 Geant4 profile，只测试了一个受约束候选：保持 tail strength 0.5，
将 width 从 0.85 调整为 0.90。该变化保持投影二阶矩，但把更多方差移入更稀有
的宽尾，使主 core 收窄约 3.5%。新基频比为：

| Depth (mm) | 40 | 80 | 100 | 115 | 124 |
|---:|---:|---:|---:|---:|---:|
| GPU/TOPAS fundamental | 0.997 | 0.972 | 0.959 | 0.965 | 0.978 |

对应 250 MeV/u 高统计 replay 改善为：

| Metric | width 0.85 | width 0.90 |
|---|---:|---:|
| 2-D L1 | 3.358% | 3.245% |
| Lateral-integral L1 | 1.172% | 1.002% |
| Bragg peak | 0.978 | 0.982 |
| Bragg valley | 1.034 | 1.009 |
| Bragg PVDR | 0.946 | 0.973 |

同 seed、完整 Copper+water 256k 检查在 250 和 300 MeV/u 均改善 2-D、lateral、
IDD 和 Bragg PVDR，因此正式 250/300 配置采用 width 0.90。150 MeV/u 保持
tail 关闭；把同一 tail 无条件应用到 150 会破坏已匹配的剂量。

注意：三能量完整 10M 表仍对应 width 0.85；width 0.90 已完成 1.20M 同源
water replay 和 250/300 完整 256k paired 验证，但尚未完成新的三能量完整
10M 复验。

## 6. 性能和审计

RTX 2080 Ti、FP32 dose：

| Workload | Histories | Elapsed | Throughput |
|---|---:|---:|---:|
| 250 MeV/u water replay, width 0.90 | 1,202,572 | 27.35 s | 43,970 histories/s |
| 250 MeV/u complete Copper+water | 256,000 | 9.41 s | 27,205 histories/s |
| 300 MeV/u complete Copper+water | 256,000 | 10.82 s | 23,652 histories/s |

对应运行满足：

- Unified-EM missing-domain count = 0；
- queue overflow = 0；
- energy balance residual < 1.8e-5；
- Bragg depth、总能量和 dose outputs 有限；
- FP32 dose scorer，未使用 FP64 fallback。

TOPAS 1,202,572-history replay 用时 29:51、峰值 RSS 4.05 GB；五深度
phase-space 诊断用时 32:35、峰值 RSS 4.51 GB。

## 7. 主要代码和数据改动

- `include/carbon/minibeam_collimator.hpp`：矩形 collimator/slit 几何；
- `include/carbon/transport_config.hpp`、`src/config.cpp`：minibeam 参数、校验和
  optional diagnostics；
- `src/transport_sycl.cpp`：Copper primary/fragment 输运、INCL++、elastic bank、
  水入口相空间、fragment audit 和 water MCS tail；
- `src/tps_source.cpp`、`src/plan_run.cpp`：显式 source pose/direction replay；
- `src/io.cpp`：phase-space CSV 和 primary fluence MHD；
- `data/ion_*_copper_geant4_11_3_2.csv`：Copper 多离子 stopping/rate 表；
- `benchmark/carbonminibeam/`：TOPAS configs、Slurm scripts、提取/比较/绘图脚本；
- `config/beam_minibeam_*`：smoke、full field、multi-energy 和 replay 配置；
- `failed.md`：已否决模型和允许重试条件。

## 8. 已否决路线

详细数值见 `failed.md`，主要包括：

- 单纯缩短 Copper step；
- Copper 段末一次性 Highland；
- 高频窄尾或 width 1.0 的过强 tail；
- 将 250/300 的 tail 无条件用于 150；
- delta aggregate 固定横向 Gaussian relocation；
- 单一位置条件 Copper 能损修正；
- 最后 10 mm 的低能 Highland recovery；
- 每步 Fermi–Eyges 位移叠加；
- 标准 Highland 全程方差差分；
- 单独精确截断 slit 侧壁。

这些路线不得原样重试，除非满足 `failed.md` 中记录的新证据条件。

## 9. 当前限制和后续工作

1. 当前仍是 research/non-production 模式。
2. Copper neutral products 主要作为 beamline energy sink，尚未建立完整中子/光子
   下游剂量模型。
3. 旧 Copper survivor energy-loss scale 已证明不能直接作为局部 stopping scale；
   homogeneous slab 已确认正式三能量配置应使用 `s(E)=1`。water tail 仍是只在
   已验证能量使用的 residual correction，不应外推到其他材料和几何。
4. 需要补做 width 0.90 的 150/250/300 完整 10M 最终验收；150 预期保持不变。
5. 入口 valley 的主要剩余误差来自 Copper 联合位置—能量—角度分布；水中
   250 MeV/u Bragg PVDR 已由 0.946 改善到 0.973。
6. minibeam validation 配置引用本机 `/mnt/sda/wuwei` 下编译后的 Copper INCL++/
   elastic assets；仓库包含可复现的 TOPAS extraction/compile scripts，但不提交这些
   大体积运行产物。

## 10. 结构性 Copper 输运后续（2026-09-18）

根据 water-entry 联合相空间复核，已实现三个不新增自由参数的结构修正：GPU
外体改为与 TOPAS 相同的半径 60 mm 圆柱；primary nuclear transport 改用跨步
累计 optical depth 并在步内真实碰撞位置生成 INCL++ 产物；Copper 能损进入逐步
能量历史，删除出口处只对 survivor 回补能量的逻辑。water tail 参数没有改动。

旧 `0.8663/0.9430/0.9647` 系数直接作为局部 `s(E)` 会复活过多低能 primary，
150/250/300 MeV/u 的 Copper-touched survivor 比变为 `1.030/1.101/1.095`，因此
该迁移已写入 `failed.md` 并否决。使用未缩放 Geant4 stopping 表后，三能量结果为：

| MeV/u | touched count | touched mean E | touched angle RMS | all-primary x RMS |
|---:|---:|---:|---:|---:|
| 150 | 0.955 | 0.955 | 1.032 | 0.997 |
| 250 | 1.009 | 0.971 | 1.015 | 1.004 |
| 300 | 0.988 | 0.981 | 0.953 | 1.004 |

以上均为 GPU/TOPAS。250 MeV/u direct primary 为 `18785/18760`，说明圆柱与 slit
直穿分类已经对齐；其 touched valley fluence/stopping proxy 为 `0.983/1.024`。
随后完成的缩减 homogeneous Copper slab 矩阵包含 150/250/300 MeV/u 和
1/10/60 mm。凡 primary C12 能穿出时，TOPAS 出口平均能量与未缩放 stopping 表
的直接 CSDA 积分仅差 `0.013--0.341 MeV`：例如 250 MeV/u、10 mm 为
`2052.384` 对 `2052.043 MeV`。因此正式配置删除旧 calibration，不再引入新的
局部 `s(E)`。完整 slit 的条件能谱残差应继续从路径长度、散射与 survival 的
联合相关性中定位，不能从 dose/PVDR 或平均出口能量反推 stopping correction。

为控制资源占用，TOPAS slab 矩阵由原计划的 192 CPU/160 GB 缩为 48 CPU/24 GB，
唯一失败点又以 16 CPU/8 GB 单独补跑；全部有效点均已完成。随后只在 RTX 2080 Ti
上运行 250 MeV/u、256k histories 的 1 mm 和 10 mm solid-Cu 对照。结果如下，
各项均为 GPU/TOPAS：

| Cu 厚度 | primary survival | mean E | A0 | A1 | A2 | q68 | q95 | q99 | q99.9 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 mm | 0.99970 | 1.00006 | 0.491 | 0.555 | 0.730 | 0.910 | 0.854 | 0.765 | 0.538 |
| 10 mm | 0.99933 | 1.00130 | 0.700 | 0.706 | 0.724 | 0.874 | 0.853 | 0.824 | 0.688 |

这里 `A0=<x²>`、`A1=<x theta_x>`、`A2=<theta_x²>`。1 mm GPU 运行吞吐为
`44,474 histories/s`，能量残差 `5.86e-9`、Unified-EM missing=0、queue
overflow=0。10 mm 对应为 `43,072 histories/s`、`1.72e-8`、0、0。

该隔离结果把剩余首要误差定位到 Copper MCS：核存活和平均能量已经匹配，但
Gaussian core 偏窄、非高斯尾不足，且薄层的位移—角度相关明显错误。单独把当前
`0.785` Highland scale 放大只能修正 A2，无法同时把 1 mm 的 A0/A1 从
`0.491/0.555` 修到 1，也无法补回 q99.9。因此下一步必须以 slab 的 A0/A1/A2
约束步长不变的 scattering power 和相关位移，并用 q68--q99.9 单独约束 tail；
在此之前不再运行大规模全场剂量，也不调整 water tail。

## 11. Copper MCS 与能损涨落 kernel（2026-09-18）

新增可选 `fermi_eyges_tail` Copper MCS，使用局部 scattering power、相关横向
位移以及步内 Poisson tail event；旧 `highland` 路径保留为默认回退。250 MeV/u
的 1/10 mm slab 中，A0/A1/A2 比从旧模型的
`0.506/0.567/0.730`、`0.703/0.707/0.724` 改善为
`1.087/1.065/1.023`、`1.019/1.013/1.003`，两种厚度的 q68--q99 均在 2% 左右，
q99.9 为 `0.997/1.000`。0.25→0.05 mm 的 10 mm 步长检查变化不超过 1.3%。

Copper primary straggling 开关已真正接入 kernel，并将 stopping 更新为中点预测
校正。250 MeV/u 的 1/10 mm 出口 `mean/std` 从
`2914.052/0.00034`、`2055.055/0.054 MeV` 改为
`2913.873/1.744`、`2051.947/6.142 MeV`；TOPAS 分别为
`2913.891/1.708`、`2052.384/6.001 MeV`。正式三能量 256k 配置已显式启用新
MCS 和 scale=1 straggling，旧模式仍保持默认兼容。

低统计 full-field 趋势中，250/300 MeV/u Bragg PVDR 达到 `0.963/0.985`；150
为 `1.169`，需高统计复核。关闭 250 MeV/u water low-energy core correction 的
配对试验使 Bragg PVDR `0.891→0.818`，已恢复并写入 `failed.md`；因此没有继续
以 water 标量补偿 Copper。

## 11.1 最新精度复核

见 [精度复核与模型选择](docs/minibeam_accuracy_audit_20260918.md)。新增 250 MeV/u、
1/10 mm Copper 的 TOPAS 核弹性关闭对照（Slurm 6432/6433）确认，其宽角分布变化
约 1%，不能解释 GPU 的散射尾缺口。统一到物理 Copper 出口后，1 mm 的位置方差比
由 0.491 修正为 0.506，主要结论不变。另发现 GPU 的出射能谱标准差仅为
0.000340/0.05369 MeV，对照 TOPAS 为 1.708/6.001 MeV；当前 Copper straggling
参数未被 SYCL kernel 使用。上述新结果是诊断证据，尚非新模型的剂量改善认证。
## 12. Copper fragment consistency and boundary follow-up（2026-09-18）

This follow-up deliberately stopped before a 10M full-dose acceptance run.
The primary Copper slab model remains unchanged; the new work isolates the
fragment and slit-edge limitations called out by review.

- Primary and fragment MCS normalizations are now separate.  Primary
  `fermi_eyges_tail` keeps scale 1.0, while Copper fragments retain their
  previous Highland scale 0.785 and honor the global Copper MCS switch.  This
  removes the accidental 27.4% fragment angular-width increase.
- Fragment electronic stopping now uses midpoint predictor/corrector
  integration.  Fragment straggling has an independent switch and remains
  disabled.  The aggregate water-entry widths mix nuclear birth spectra,
  Copper paths, and survival selection, so they are not a straggling
  validation; fixed-species, fixed-energy Copper slabs are still required.
- Both primary and fragment Copper steps now limit actual path length rather
  than only axial advance.  At 0.25 mm maximum path the fitted tail Poisson
  mean is at most 0.0266; the probability of three or more events is about
  3e-6 per step, so the existing two-event cap is negligible in this regime.
- An optional exact straight-ray navigator splits the path at the first
  cylindrical/slit Copper-air boundary.  It does not yet implement a Brownian
  bridge for a Fermi--Eyges displacement that crosses a boundary; therefore it
  is not claimed as complete Urban-style boundary transport.

The new opt-in `minibeam_fragment_phase_space_output_file` produced exactly
48,776 valid charged-fragment water-entry records in the 250 MeV/u 256k smoke
run, matching the transport survivor counter.  Unified EM missing and queue
overflow were zero and the energy residual was 1.72e-5.  TOPAS scores only a
±70 mm transverse plane.  After back-projecting GPU records by 0.02 mm to that
plane and applying the same two-dimensional extent, 46,996 GPU fragments are
compared with 57,685 TOPAS fragments, for a corrected total ratio of 0.815.
The dominant isotope yield ratios (GPU/TOPAS) are p 0.696, d 0.860, t 0.920,
He-3 0.791, and He-4 0.973.  Aggregate exit-spectrum widths combine birth
spectra, paths, and survival selection, so they do not validate fragment
straggling; the switch remains off pending fixed-species slab tests.

The secondary-interaction audit shows why a real fragment+Cu final-state
model is the next physics dependency, but does not by itself apportion the
water-entry yield deficit.  The current terminal approximation
absorbed 46,764 charged children in Copper: C/B/Be/Li/He/p/d/t counts were
110/265/117/298/12720/15823/12788/4643.  The former 35--59 mm value was only
the distance to the downstream axial plane, not remaining Copper thickness,
and is withdrawn.  The audit now separately integrates straight-ahead Copper
material while subtracting finite-slit air and cylindrical side escape.  Mean
values for C/B/Be/Li/He/p/d/t are
41.1/37.4/38.4/32.8/28.9/34.7/32.5/29.6 mm; these still do not predict later
MCS and cannot be converted directly into missing water-entry yield.  No
synthetic continuation was added.
Packages must reproduce the reference list: proton+Cu uses Binary Cascade,
while d/t/He and GenericIon+Cu use INCL++ in the relevant energy domain, with
their documented FTFP/QGSP transitions above it.

A low-cost extraction smoke test completed as local Slurm job 6436 (32 CPUs,
32 GB requested, 50k proton histories, 15 s wall time).  The capture contract
recorded 37,179 proton+Cu interactions and 395,995 products; every event
reported the expected `Binary Cascade` model.  Its 47.3 MB CINPKG04 was also
successfully combined with the existing C-12 INCL++ package.  The reread
package contains projectile keys `(1,1)` and `(6,12)`, 103,791 interactions,
1,678,848 products, and 103,445 exact energy nodes.  This validates the
process-faithful multi-projectile package path, but the smoke package remains
disabled in GPU transport until child collision-point replay and
remaining-Copper product transport are implemented.

The fragment nuclear sampler now uses a persistent optical depth and truncates
EM loss/MCS transport at the sampled collision position instead of testing a
Bernoulli probability after the complete 0.25 mm step.  In the 250 MeV/u 256k
smoke run this changed raw charged survivors from 48,776 to 49,017 and the
matched-scorer count from 46,996 to 47,206 (total TOPAS ratio 0.815 to 0.818).
Terminal interaction count changed from 46,764 to 46,648.  This is a transport
consistency change, not evidence that the missing final states are solved;
the collided child is still terminated at the now-correct collision point.

The straight-boundary on/off comparison was inconclusive as a dose acceptance
test: total dose and IDD changed by less than 0.04 percentage point while
low-statistics local PVDR moved substantially and the segmentation changed RNG
streams.  This candidate is recorded in `failed.md`; future retry requires
displacement-induced crossing handling and matched fragment output first.

## 13. Process-faithful fragment+Cu replay pilot（2026-09-18）

The fragment comparison now uses the same TOPAS scorer plane: GPU records are
back-projected by 0.02 mm, clipped to ±70 mm in both transverse coordinates,
and angles use `atan2(p_transverse,p_longitudinal)`.  The corrected terminal
baseline is 46,996/57,685 = 0.815, not the former un-clipped 0.846; its proton
ratio is 0.696.

A projectile-indexed CINPKG04 compiler now enforces the process actually used
by the reference physics list: proton+Cu is `Binary Cascade`, while d/t/He and
GenericIon+Cu are `INCL++ undefined`.  Slurm jobs 6437/6438 completed in
83/47 s.  The light-energy-complete pilot contains 252,030 interactions,
3,822,354 products, and 251,518 exact energy nodes (384 MB).  Its p/d/t
coverage was extended down to 4.07/2.52/1.82 MeV/u without endpoint clamping.

The runtime option `minibeam_copper_fragment_cascade_generations: 1` now
replays the correlated final state at the sampled fragment collision point,
queues charged products separately, transports them through the remaining
Copper/slit/air geometry, and only then inserts survivors into the water
secondary queue.  Zero preserves the old terminal approximation.  The first
implementation deliberately permits one reaction generation; fragment
straggling remains disabled.

For 250 MeV/u and 256k histories, strict lookup hits were 45,637/46,648
(97.8%).  Misses were projectile/below-domain/gap = 118/165/728; target,
above-domain, and empty-node misses were zero.  Both Copper and water charged
queues had zero overflow, Unified EM missing was zero, and the energy residual
remained 1.72e-5.  After the matched scorer cut, charged-fragment yield became
56,467/57,685 = 0.979.  Dominant isotope yield ratios changed from terminal
baseline to replay as follows: p 0.696→0.966, d 0.860→1.005, t 0.920→0.989,
He-3 0.791→0.906, He-4 0.973→0.972.  For p/d/t, exit mean-energy ratios are
1.002/0.994/0.985 and theta-q95 ratios are 0.947/1.000/0.988.

The first dose comparison above was invalidated by a generation-namespace
bug: Copper cascade products entered water as generation 1, consuming one of
the two water nuclear generations, and charged products born at the cap were
locally deposited instead of receiving terminal EM transport.  Copper and
water generations are now independent; every Copper survivor enters water at
generation 0, and every above-cutoff charged final state is queued for EM even
when its generation guard suppresses further nuclear reactions.

After rerunning both sides of the same-seed 256k A/B, replay improves IDD L1
from 0.934% to 0.619%, lateral-integral L1 from 5.74% to 5.41%, and 2-D L1
from 18.323% to 18.290%.  Bragg peak/valley/PVDR changes only from
1.014/1.154/0.879 to 1.014/1.157/0.877.  Therefore the cascade remains a major
exit-yield correction but is not the remaining Bragg/PVDR solution.

The one-Copper-generation approximation is now quantified rather than assumed
small.  Its ignored downstream optical-depth sums for C/B/Be/Li/He/p/d/t are
0.017/0.228/1.950/1.014/186.91/3031.24/1074.47/221.69, dominated by light
ions.  This is substantial and prevents declaring Copper cascade complete;
it motivates a second Copper generation or a separate convergence test.
Strict lookup misses are also recorded by species and 25 MeV/u bin.

For 45,637 replay hits, actual and selected incident kinetic-energy sums are
13,136,760 and 13,136,810 MeV; summed absolute selection mismatch is only
3,005 MeV (0.023%).  Recorded parent+products+local output is 11,118,704 MeV,
leaving 2,029,141 MeV of absolute kinetic closure difference associated with
reaction Q/residual/untransported channels.  This demonstrates that the small
global run residual is bookkeeping closure, not independent validation of the
local nuclear final state.  Adding Cu-63/Cu-65 nuclear rest masses and all
recorded product/parent rest masses reduces the summed absolute mass-energy
closure difference to 2,568 MeV (about 0.056 MeV per hit), with zero baryon
number mismatches.  Formal three-energy configs remain at Copper
generation zero pending this convergence work.

## 14. Copper cascade generation convergence and replay cleanup（2026-09-18）

The Copper queue now carries its own explicit generation and processes
generation batches up to a configurable cap of three. It remains independent
of the water nuclear-generation counter. Every generation uses the same
midpoint Copper stopping, optional fragment straggling, MCS, exact straight-ray
material splitting, true optical-depth collision point, strict process-faithful
lookup, correlated final state, and queue/energy audits. The deprecated
`enable_terminal_generation_em_transport` key is accepted as a no-op; charged
products above cutoff always receive EM transport and only their later nuclear
hazard is capped.

At 250 MeV/u and the same 256k histories, terminal convergence is:

| Copper cap | final queue | terminal `sum(tau)` | terminal `sum(1-exp(-tau))` |
|---:|---:|---:|---:|
| 1 | 169,001 | 4,517.52 | 4,133.68 |
| 2 | 180,346 | 144.23 | 135.84 |
| 3 | 180,631 | 2.02 | 1.94 |

Strict lookup interactions/hits/misses by reaction generation are
`46648/45637/1011`, `4069/4000/69`, and `108/108/0`; later generations do not
introduce a hidden coverage collapse.

Generation 2/3 are dose-converged at this precision. Total dose ratio is
`1.002109/1.002090`, IDD L1 `0.5808/0.5807%`, lateral-integral L1
`5.4391/5.4398%`, 2-D L1 `18.28669/18.28668%`, plateau PVDR ratio
`0.881661/0.881661`, and Bragg PVDR ratio `0.876834/0.876834`. Queue overflow
is zero and the global energy residual stays `1.73e-5`. Generation 1/2/3
matched-plane fragment yield ratios are `0.9789/0.9609/0.9605`; the decrease
shows that nuclear-generation convergence does not by itself improve the
remaining phase-space agreement. The plateau/Bragg PVDR error is unchanged,
so downstream water/slit-edge isolation remains the next task.

The water-entry converter now has two explicit modes. The historical default
keeps parent-0 C12 only and preserves its metadata keys. `--selection
charged-ions` writes one GPU/TOPAS source pair per isotope plus incident-history
normalization metadata; the 250 MeV/u reference produces 88,563 ion records in
20 isotope groups. These split GPU sources still use the primary kernel and are
not claimed as a production-equivalent mixed-fragment replay; section 15 adds
the separate identity-preserving secondary-queue injection path. Replay
configs now consistently use water tail width 0.90.

A non-minibeam 2,000-history regression compared the deprecated terminal-EM
key absent versus true. Total voxel dose differs by `2.27e-10` relative and
voxel L1 by `4.47e-8`, consistent with FP32 atomic accumulation order; both
runs pass energy accounting (`4.78e-6`) and have no queue overflow. Formal
At that stage the 150/250/300 configs remained Copper-cascade disabled pending
same-source water replay and multi-energy validation. Section 23 records the
completed 10M validation and supersedes that provisional status.
## 15. 同源水入口 mixed-ion 回放基础（2026-09-18）

已实现保持 primary/fragment 不同输运语义的最小回放路径：parent-0 C12
继续由现有 primary replay 运行；其余带电离子（包括 non-parent-0 C12）由
`minibeam_water_entry_secondary_replay_file` 注入正常 secondary queue，水中
generation 重置为 0。fragment-only 运行不启动 primary kernel，且
`number_of_histories` 固定为原始 incident histories，因此 fragment 剂量已按
原始 256k 归一化；primary-only 剂量仍需乘 `30878/256000` 后再相加。

转换器已修正实际计分面漂移：TOPAS 记录位于 world Y=59.980 mm，现沿每条
粒子的三维方向推进到水入口 Y=60.000 mm，同时更新两个横向坐标。250 MeV/u
文件得到 30,878 条 parent-0 C12 和 57,685 条 fragment，后者保留 4 条
non-parent-0 C12。入口 identity CSV 保留 origin、run/event/track/parent ID、
PDG、Z/A、能量、权重、位置与方向；当前严格要求单位权重。

RTX 2080 Ti、FP32 dose 的首次 fragment replay 结果：

| 水中模式 | 输入/最终 queue | secondary kernel | 相对能量残差 | overflow |
|---|---:|---:|---:|---:|
| CINEL03 开启 | 57,685 / 114,162 | 0.486 s | 1.28e-5 | 0 |
| 纯 EM (`enable_inelastic=false`) | 57,685 / 57,685 | 0.207 s | 3.12e-7 | 0 |

纯 EM 首次运行曾因 ion stopping table 只在 `enable_inelastic=true` 时上传而
触发 CUDA illegal address；现已改为“核反应开启或存在 secondary replay”时
上传。修复后核反应/核弹性均为零，queue 不增长。纯 EM 的 untracked
`482.64 MeV` 与入口中不属于当前 18-ion EM 表的少数同位素总动能一致，是
显式覆盖缺口而不是账本丢失。

本阶段只证明入口平面、身份分流、secondary-only 启动、纯 EM 隔离、归一化
和能量记账可运行。尚未完成 GPU 自身 export→replay 剂量闭合，也尚未运行
TOPAS mixed-ion 同源水回放，因此不能据此更新最终 PVDR match 结论。下一步
应先完成这两个闭合，再用多深度位置—角度—能量矩和 peak/valley/PVDR 判断
剩余误差属于水输运还是 Copper/slit-edge。

## 16. 原始 256-spot 束流的 1.024M 纯 EM 全链路对照（2026-09-18）

本节使用原始 `3 cm x 3 cm` PBS 束流，而不是对水入口相空间重采样：256 个
spot 各发射 4,000 histories，总计恰好 1,024,000 histories。GPU 和 TOPAS
都运行完整的 Copper→60 mm air gap→water 链路；Copper 核衰减和水中
inelastic 在 GPU 关闭，TOPAS physics list 仅包含 `g4em-standard_opt4` 与
`g4decay`。剂量均按相同 incident histories 归一化，并以绝对 Gy 直接比较，
没有拟合 scale。

TOPAS 正式结果是本地 Slurm job 6441，使用 192 threads 和 160 GB request，
实际 elapsed 307.157 s、MaxRSS 约 2.92 GB。jobs 6439/6440 分别因缺少
`beam_model.csv` 和误用 2150 MeV beam model 在模拟开始前失败，不进入统计。
GPU 使用 RTX 2080 Ti、FP32 dose，elapsed 9.504 s，吞吐 107,745
histories/s，primary kernel 3.902 s；核反应数和 queue overflow 均为零，
相对能量残差为 `1.63e-5`。

统一 0.1 mm 横向、0.25 mm 深度网格的严格比较如下：

| 指标 | 结果 |
|---|---:|
| total dose GPU/TOPAS | 1.007995 |
| 2-D Pearson / L1 | 0.995510 / 8.8566% |
| IDD Pearson / L1 | 0.999968 / 0.9257% |
| lateral-integral Pearson / L1 | 0.999281 / 3.2545% |
| Bragg depth TOPAS / GPU | 124.375 / 124.625 mm |

代表性 peak/valley 指标（均为 GPU/TOPAS）为：

| depth | peak | valley | PVDR |
|---:|---:|---:|---:|
| 0.375 mm | 1.037 | 1.030 | 1.006 |
| 39.875 mm | 1.030 | 1.006 | 1.023 |
| 79.875 mm | 1.037 | 1.080 | 0.960 |
| 99.875 mm | 1.033 | 1.107 | 0.933 |
| 124.375 mm | 1.062 | 1.097 | 0.968 |

因此纯 EM 下 total dose、IDD、Bragg range 和入口 PVDR 已较好一致；但这不等于
完整剂量已经 match。80--100 mm 的 GPU valley 仍系统性高约 8--11%，PVDR
低约 4--7%，而 2-D L1 明显高于 IDD L1。因为本实验完全移除了核 cascade，
这些残差把当前主要问题进一步定位到水中 EM/MCS 的横向联合演化，而不是
Copper fragment cascade。134.375 mm distal 点剂量统计过低，本次不据其
局部 PVDR 作物理判断。

可复现输入为
`benchmark/carbonminibeam/run_field3cm_em_only_e250_1024k.txt` 和对应 sbatch；
原始结果与图位于
`/mnt/sda/wuwei/minibeam_field3cm_em_only_e250_1024k/`。正式 150/250/300
配置已经恢复原有 histories 与 nuclear switches，未保留本次临时纯 EM 修改。

## 17. 原始束流 12.8M 纯 EM 高统计结果（2026-09-18）

为降低 section 16 中局部 peak/valley 的统计波动，使用相同原始 256-spot
束流将每个 spot 提高到 50,000 histories，总计 12,800,000 histories。物理、
网格和归一化口径完全不变，仍是完整 Copper→air→water 的纯 EM 对照，
没有 dose fitting。

TOPAS job 6442 使用 192 threads，完成时间 3822.14 s、MaxRSS 约 2.99 GB。
RTX 2080 Ti FP32 GPU 完成时间 55.80 s，吞吐 229,387 histories/s，墙钟加速
约 68.5x；核反应与 queue overflow 均为零，相对能量残差 `1.62e-5`。

| 指标 | 1.024M | 12.8M |
|---|---:|---:|
| total dose GPU/TOPAS | 1.007995 | 1.003424 |
| 2-D Pearson | 0.995510 | 0.999564 |
| 2-D L1 | 8.8566% | 2.8091% |
| IDD L1 | 0.9257% | 0.4887% |
| lateral-integral L1 | 3.2545% | 1.1785% |
| Bragg depth TOPAS / GPU | 124.375 / 124.625 mm | 124.625 / 124.375 mm |

高统计局部结果（GPU/TOPAS）为：

| depth | peak | valley | PVDR |
|---:|---:|---:|---:|
| 0.375 mm | 1.036 | 0.950 | 1.090 |
| 19.875 mm | 1.012 | 0.988 | 1.024 |
| 39.875 mm | 1.010 | 1.018 | 0.992 |
| 59.875 mm | 0.997 | 1.028 | 0.970 |
| 79.875 mm | 1.005 | 1.109 | 0.907 |
| 99.875 mm | 0.995 | 1.167 | 0.853 |
| 124.375 mm | 1.004 | 1.033 | 0.972 |

这组结果取代 section 16 的 1.024M 局部 peak/valley 结论。增加统计后，峰剂量
在 20--124 mm 基本保持在 TOPAS 的约 +/-1.5% 内；系统误差清楚集中在 valley：
入口低约 5%，随后随深度过快上升，在 80/100 mm 高 10.9%/16.7%，到 Bragg
附近回落为高 3.3%。因此 PVDR 从入口高 9.0% 转为中深度低最多约 14.7%，
Bragg 处低 2.8%。这种深度相关、非单一比例的变化不能用 dose scale 修正，
且在完全关闭 nuclear transport 后仍存在，进一步支持水中 EM/MCS
位置—角度演化是下一项隔离对象。

高统计输出和图位于
`/mnt/sda/wuwei/minibeam_field3cm_em_only_e250_12800k/`。比较脚本新增
`dose_ratios_vs_depth.png`，直接画出 IDD、peak、valley 和 PVDR 的
GPU/TOPAS 深度比值。

## 18. 12.8M 同源纯 EM 水回放与多平面归因（2026-09-19）

为避免继续重复约一小时的 TOPAS 运行，job 6455 在一次完整
Copper→air→water 纯 EM 运行中同时输出总剂量、electron/non-electron
carrier 剂量和 water-entry parent-0 C12。该入口的 1,622,795 条 C12 随后
作为 GPU/TOPAS 的共同水输入；TOPAS job 6529 在同一次回放中同时记录
40/60/80/100/120 mm 五个平面。GPU 使用同一入口、FP32 dose，10.153 s
完成，能量残差 `2.69e-10`，无核反应和 queue overflow。

同源水回放的绝对剂量结果为：total ratio `1.000019`、2-D L1 `2.5097%`、
IDD L1 `0.1719%`、lateral-integral L1 `1.0424%`，两侧 Bragg depth 均为
`124.625 mm`。固定 periodic regions（peak `<0.25 mm`、shoulder
`0.25--0.9 mm`、valley `0.9--1.8 mm`）比局部极值稳定：

| depth (mm) | peak | shoulder | valley |
|---:|---:|---:|---:|
| 40 | 0.9995 | 1.0001 | 1.0013 |
| 60 | 0.9901 | 0.9996 | 1.0163 |
| 80 | 0.9840 | 0.9954 | 1.0284 |
| 100 | 0.9816 | 0.9901 | 1.0387 |
| 120 | 0.9912 | 0.9988 | 1.0217 |
| 124.375 | 0.9939 | 0.9992 | 1.0129 |

同源入口仍复现 80--100 mm 的 valley excess，因此该残差的主因已从
Copper/slit-edge 输入隔离到水中 transport/scoring。电子 carrier 只占总剂量
`7.2999%`，并且比总剂量更集中于 peak（40--100 mm peak electron fraction
约 `10.08%→7.61%`，valley 为 `7.14%→6.23%`）。carrier 终点剂量不等于
“电子能量原地沉积”的反事实，因而这不能完全排除电子响应；结合下面的 C12
证据，当前优先级仍明确是 water primary MCS。

不依赖粒子 identity 的五平面 C12 分析进一步给出直接证据。100 mm 处
GPU/TOPAS 的角分布 `q68/q95/q99/q99.9` 为
`0.937/1.032/1.230/1.270`，固定 valley C12 fluence 为 `1.0461`，其条件
平均能量仅为 `1.0106`。将相空间与剂量统一限制在 `|x|<=18 mm` 后，80/100
mm valley fluence 应为 `1.0486/1.0497`；此前 `1.0452/1.0461` 是全平面
pitch-folded 口径。因此当前
模型并非整体散射简单偏强，而是中心核逐渐偏窄、非 Gaussian 尾明显偏宽，
从 peak 向 valley 搬运过多 C12；C12 fluence 已能解释固定区域剂量误差的
主要符号与量级。stopping 和总剂量标定继续冻结。

最初的相空间脚本曾把 192-thread TOPAS replay EventID 当作入口文件行号；
该映射在 worker block 间不稳定，所有基于它的入口能量分组和逐轨迹
`A0/A1/A2` 结论均撤回。脚本现只输出无需 identity join 的总体角度、pitch
折叠位置、位置--角度协方差、Fourier 调制度及固定区域通量/条件能谱。
有效结果位于
`/mnt/sda/wuwei/minibeam_water_replay_e250_em12800k/phase_comparison_aggregate/`。
真正的逐轨迹矩必须在以后的一次 omnibus TOPAS 原始全链路运行中，让入口和
所有下游平面共同保留稳定的 run/event/track identity；本轮不再提交 TOPAS。

## 19. 水中区间散射分析与可选相关 kernel（2026-09-19）

同一次 TOPAS replay 的下游平面可用 `(RunID,EventID,TrackID)` 稳定配对；
不能做的只是 `EventID→入口文件行号`。40→100 mm 可配对 1,134,040 条轨迹，
所有相邻区间均没有出口能量高于入口。分析脚本现输出 40→60、60→80、
80→100、100→120 mm 的局部角增量、相对上游直线外推的位移、二者协方差、
完整分位数曲线、上游能量分组和配对存活率，并新增 `Σ S(E)/|u_z|` 穿越量。
fixed regions 与剂量统一到 `|x|<=18 mm`；pitch-folded covariance 只保留为
周期束形指标，不再称作 Fermi--Eyges propagation covariance。

旧模型在 40→60 mm 的能量分组直接显示两个补偿项：120--140 MeV/u 的
variance/q68/q99/q99.9 比为 `0.683/0.687/1.191/1.793`；180--200 MeV/u 为
`1.453/0.976/1.619/2.734`。相应 paired survival 两侧相同，不能用停止筛选
解释。诊断平面改为与 TOPAS surface 一致的 `39.995/59.995/... mm`。

新增默认关闭的 `minibeam_water_primary_mcs_model: fermi_eyges_tail`，当前只
替换 native-water primary C12 路径，不叠加旧 Highland、低能 scale 或
40.5% 尾方差模型。核心使用局部 scattering power 和相关角度--位移采样；
尾部为不截断事件数的沿程 Poisson 过程，事件位置决定剩余漂移。初始参数只用
40→60 mm 能量组约束，60→120 mm 仅作为未参与拟合、但共享轨迹的传播检查，
因此不是统计独立 holdout，也不是唯一或最终 tail 解；汇总 q99.9 仍偏低。

| interval (mm) | angle variance | displacement variance | covariance | q68 | q99 | q99.9 |
|---:|---:|---:|---:|---:|---:|---:|
| 40--60 | 0.985 | 1.016 | 1.007 | 1.020 | 0.998 | 0.938 |
| 60--80 | 1.009 | 1.012 | 1.010 | 1.015 | 0.991 | 0.949 |
| 80--100 | 0.990 | 1.009 | 1.004 | 1.015 | 0.989 | 0.924 |
| 100--120 | 0.978 | 0.992 | 0.986 | 1.004 | 0.980 | 0.896 |

同源 water replay 的固定 C12 valley fluence 比在 40/60/80/100/120 mm 为
`0.999/1.001/1.004/1.011/1.009`。total dose ratio 保持 `1.000019`，IDD L1
`0.1705%`，Bragg bin 仍为 `124.625 mm`；2-D L1 从 `2.5097%` 降至
`2.2834%`，lateral-integral L1 从 `1.0424%` 降至 `0.6793%`。固定 valley
dose 在 40/60/80/100/120/124.375 mm 为
`0.993/0.996/0.998/1.009/1.012/1.012`。

内部散射最大分段从 0.10 mm 改为 0.05 mm 后，四区间 q68/q99 变化均小于
0.25%，q99.9 最大变化 1.1%；固定区域剂量变化约 0.3% 内。关闭 plane
records 后，FP32 dose 相对 L1 仅 `6.01e-8`，属于 atomic accumulation
order 差异。0.05/0.10 mm primary kernel 分别为 6.35/5.77 s；候选仍保持
可选且不进入正式三能量配置。

初版平面诊断曾将整步位移和起末方向线性插值到交点。这对随机过程有偏：路径
比例为 `f` 时会得到 `f²TL` 而非 `fTL` 的角方差，并把晚于平面的 tail event
错误分摊到上游。现已改成条件于已采样终态的 integrated-Brownian Gaussian
bridge；tail 只在其真实事件位置早于平面时进入交点状态。查询 bridge 使用独立
counter-RNG 维度，不改变输运终点。修复前后汇总指标变化不大，但低能分组可见，
例如 40→60 mm、80--100 MeV/u 的角方差比由 `0.851` 修正为 `0.866`。

工程与 sampler 回归已补齐：`CARBON_ENABLE_MINIBEAM=ON/OFF` 两种 CUDA SYCL
构建均成功；固定 160 MeV/u、20 mm 的 500k reference sampler 中，长步三个
解析矩为理论的 `0.9984/0.9992/0.9996`，200×0.1 mm 相对长步为
`1.0024/1.0036/1.0038`。Poisson count 均值 `0.04993`（理论 `0.05`），事件
位置分数均值/方差 `0.5007/0.08321`（理论 `0.5/1/12`），半步 bridge 三个矩
为理论的 `0.9991/0.9980/0.9979`；倾斜入射投影检查为 1。

64 次、256 stable-identity blocks 的轨迹 bootstrap 显示，总体 q99.9 缺口
具有统计意义：40→60 mm 的 TOPAS/GPU q99.9 为
`25.51 [25.06,25.85] / 23.93 [23.63,24.21] mrad`，100→120 mm 为
`39.76 [39.05,40.15] / 35.64 [35.40,36.00] mrad`。以 TOPAS q99.9 为固定
阈值，GPU 超阈概率分别为 `0.00075 [0.00071,0.00079]` 和
`0.00051 [0.00046,0.00056]`，TOPAS 约为 0.001。低能小样本区间更不确定：
60→80 mm、80--100 MeV/u 只有约 32k 配对，TOPAS/GPU q99.9 的 95% 区间
`[58.24,72.74]/[51.58,61.91] mrad` 有重叠。故保留尾部缺口结论，但不把它
简化为统一的 5--10% 修正，更不据此继续调 `9.9/0.0025/2.4`。

冻结参数后的完整 Copper+water、核反应开启 256k single-seed A/B 使用完全
相同的 Copper 模型、cascade cap、输入和 seed；正式配置在运行后恢复 legacy。
相对现有 TOPAS 256k reference：

| MeV/u | model | 2-D L1 | IDD L1 | lateral L1 | total dose ratio |
|---:|:---|---:|---:|---:|---:|
| 150 | legacy | 17.672% | 1.210% | 7.287% | 0.99090 |
| 150 | candidate | 17.923% | 1.215% | 7.078% | 0.99098 |
| 250 | legacy | 18.197% | 1.035% | 5.598% | 0.99210 |
| 250 | candidate | 18.009% | 1.032% | 5.296% | 0.99217 |
| 300 | legacy | 17.224% | 1.813% | 5.327% | 0.98285 |
| 300 | candidate | 17.227% | 1.819% | 5.383% | 0.98273 |

因此 250 MeV/u 的同源收益能部分传到完整链路，但很小；150 的 lateral 改善伴随
2-D 退化，300 基本持平并略退化。局部 Bragg PVDR 比在 150/250/300 分别由
`1.116/0.938/0.962` 变为 `0.988/0.930/0.979`，同样不是统一改善。三能量的
Unified-EM missing 和 queue overflow 均为 0，energy residual 为
`1.6e-5--2.7e-5`。这次筛查否决了直接推广候选，但不否决 250 MeV/u 同源水中
归因；在取得独立 150/300 相空间约束前继续保持可选、冻结参数。

## 20. 冻结水散射候选的 150/300 MeV/u 独立同源回放（2026-09-19）

已使用现有完整链路 water-entry phase space，筛选出 150/300 MeV/u 的
`26,425/34,557` 条 parent-0 C12，在 GPU 与 TOPAS 中只做水中纯 EM 回放。
TOPAS job 6603 使用 192 线程，在一次运行内同时输出全部多平面相空间以及
total/electron/non-electron 三套剂量；申请 60 GB，实际 MaxRSS 约 6.1 GB，
总 wall time 2:06。冻结的 `9.9/0.0025/2.4` 未调整，另用相同入口和 seed
补跑 legacy GPU A/B；未修改 stopping、Copper、fragment 或电子模型。

候选的区间角方差/位移方差/协方差大多落在 TOPAS 的约 ±5% 内。150 MeV/u
四个区间角方差比为 `0.997/0.995/1.013/0.947`，legacy 则随深度从 `0.672`
降到 `0.221`；300 MeV/u 候选除 100--120 mm 的稀有极端事件令方差比为
`1.224` 外，其余为 `0.947--1.045`，legacy 最后一区间只有 `0.488`。
候选 q99 基本接近 1；q99.9 点估计仍有波动，但 64 次 stable-block bootstrap
下各区间置信区间均重叠，当前每区间只有约 19k--31k tracks，不能据此继续调尾。

| MeV/u | total ratio | IDD L1 | Bragg TOPAS/GPU (mm) | fixed region 最大偏差 |
|---:|---:|---:|---:|---:|
| 150 | 1.000009 | 0.3671% | 51.625 / 51.625 | 2.60% |
| 300 | 1.000024 | 0.2696% | 169.375 / 169.125 | 2.82% |

150 MeV/u 的 50 mm fixed-valley ratio 从 legacy `0.869` 改善到 candidate
`0.994`；300 MeV/u 各采样深度 fixed-valley 为 `0.976--1.019`。本轮只有
26k--35k histories，局部极值 PVDR 和 2-D sparse-voxel L1（12--16%）噪声很大，
不用于参数验收。TOPAS electron-carrier 总占比为 `5.33%/8.00%`，分量闭合
L1 为 `2.68e-8/2.75e-8`；结合 C12 fluence 与固定区域剂量，当前不支持优先改
电子响应。

结论：候选的水中结构在未拟合的 150/300 MeV/u 上明显优于 legacy，并非只在
250 MeV/u 过拟合；但先前完整 Copper+water A/B 仍未统一改善，所以正式三能量
配置继续保持 legacy。下一步应先解决 full-chain 上游联合相空间/补偿，再用较高
统计同源回放确认 q99.9 的能量依赖，而不是修改 `0.20/0.90` 或继续从剂量拟合
`9.9/0.0025/2.4`。完整输出位于
`/mnt/sda/wuwei/minibeam_water_multienergy_replay_20260919/`。

## 21. Copper/slit-edge 高统计隔离与步长复核（2026-09-19）

在步长稳定的 Copper FE/tail 模型下，重新满足了旧 `0.05 mm` 步长试验的重试
条件。250 MeV/u、256k、同 seed 对照中，入口 valley fluence/stopping proxy 从
`0.914/0.848` 改善到 `1.004/0.939`，但 touched 三维联合 shape TV 仅从
`5.629%` 变为 `5.591%`；2-D L1 为 `18.323% -> 18.314%`，IDD L1 为
`0.934% -> 0.956%`，lateral L1 为 `5.745% -> 5.884%`，80--100 mm 的固定
valley 还出现退化。因此正式 Copper 最大步长保持 `0.25 mm`，不把入口单点改善
视作整体收益。

更关键的是，复用了 12.8M-history 纯 EM TOPAS water-entry reference，并用当前
GPU Copper/slit 模型生成同统计入口。TOPAS/GPU parent-0 C12 数为
`1,622,795/1,627,171`；GPU/TOPAS 的总产额、平均能量、能谱标准差和角度标准差
分别为 `1.00270/1.00126/0.99824/0.97150`，q68/q95/q99 为
`0.9962/0.9864/0.9745`。位置—能量—外向角三维直方图的 yield L1 为 `1.395%`，
归一化 shape TV 只有 `0.647%`；valley fluence/stopping proxy/mean-energy 为
`0.9795/0.9904/1.0014`。六个 slit-residual bin 的平均能量均在约 0.2% 内，
只剩 `>40 mrad` 稀有宽角率约 5--14% 的小幅不足。

这说明双方都只开 EM 时，当前 Copper FE/tail、slit 几何和能损联合输运已经高度
匹配，不能再把完整链路的大剂量残差归因于 Copper EM/slit-edge，也不应继续调
Copper scattering。256k 全物理样本中的 5--8% sparse-histogram TV 主要受有限
统计或 nuclear survival selection 影响。下一步用现成的 10,000,128-history
TOPAS 全物理 water-entry 数据验证 surviving-C12 条件分布；若仍约 1%，主任务
转回 water candidate、fragment water transport 和完整剂量，而不是 Copper。

该 10,000,128-history 全物理检查随后完成。GPU 用 0.25 mm 水层只记录入口，
耗时 18.26 s、吞吐 547,533 histories/s；Unified-EM missing 与 queue overflow
均为 0，能量残差 `1.65e-5`。TOPAS/GPU surviving C12 数为
`1,202,572/1,207,927`，产额比 `1.00445`；能量均值/标准差、横向位置标准差和
角度标准差比为 `1.00104/0.99773/1.00019/0.97376`，q68/q95/q99 为
`0.9979/0.9914/0.9737`。三维联合 yield L1 为 `1.365%`，归一化 shape TV 为
`0.638%`，与纯 EM 的 `0.647%` 等价。六个位置 bin 的 fraction 误差都小于 1%，
mean-energy 误差除一个 0.62% bin 外均约 0.4% 内；主要剩余项仍只是稀有
`>40 mrad` 尾部不足。

因此 nuclear survival selection 也已排除为完整剂量主误差源。当前证据把后续
工作明确限定到水中：primary 应使用已由三能量同源回放支持但尚未正式推广的
相关 FE/tail 候选；full-physics 剩余差异还需按 primary/fragment 分量隔离，
重点检查 fragment 的水中散射、能损和再反应，而不是继续修改 Copper/slit。
完整输出位于 `/mnt/sda/wuwei/minibeam_copper_entry_e250_full10m_gpu_20260919/`
和 `/mnt/sda/wuwei/minibeam_copper_entry_e250_full10m_joint_20260919/`。

## 22. 水中 FE/tail 三能量 10M full-chain 验收（2026-09-19）

高统计入口确认 Copper/slit 不再是主要误差源后，旧的“上游误差补偿”阻碍已经
解除。冻结 `9.9/0.0025/2.4`，不重新拟合参数，使用现成的三能量 10M TOPAS
reference 运行当前完整 GPU 链路。三次运行均为 FP32、完整性检查关闭；Unified
EM missing 和 queue overflow 为 0，能量残差为 `1.56e-5--2.70e-5`。

| MeV/u | total ratio | 2-D L1 | IDD L1 | lateral L1 | Bragg depth TOPAS/GPU mm | Bragg peak | Bragg valley | Bragg PVDR |
|---:|---:|---:|---:|---:|:---:|---:|---:|---:|
| 150 | 0.99905 | 3.393% | 0.848% | 1.725% | 51.375/51.375 | 1.006 | 0.974 | 1.032 |
| 250 | 0.98632 | 3.481% | 1.390% | 1.692% | 124.375/124.125 | 1.004 | 1.001 | 1.004 |
| 300 | 0.98186 | 3.551% | 1.827% | 1.951% | 168.625/168.875 | 0.961 | 1.005 | 0.956 |

Bragg 的固定区域 peak/valley 比分别为 `1.004/0.988`、`0.989/1.004`、
`0.987/1.004`，比局部极值更稳定。250 MeV/u 的严格同 seed legacy/candidate
A/B 中，candidate 将 2-D L1 从 `3.610%` 降至 `3.481%`，Bragg 局部
peak/valley/PVDR 从 `0.995/1.035/0.962` 改为 `1.004/1.001/1.004`；总剂量和
IDD 基本不变，说明收益来自横向剂量搬运而不是 stopping 补偿。运行时间为
legacy/candidate `108.4/116.8 s`，candidate 的代价约 7.8%。

基于独立同源水回放、上游 10M 联合相空间和本次三能量 full-chain 三层证据，
正式 150/250/300 配置现启用 `minibeam_water_primary_mcs_model:
fermi_eyges_tail`，最大内部分段保持 0.1 mm。入口固定 valley 仍低约 4--7%，
250/300 总剂量仍低约 1.4/1.8%，300 Bragg peak 低约 3.9%；这些不随 primary
MCS A/B 变化，下一步应从 Copper fragment cascade/水中 fragment 分量和绝对
产额记账隔离，不再修改已冻结的 primary scattering 参数。完整输出位于
`/mnt/sda/wuwei/minibeam_fullchain_e250_10m_20260919/`、
`/mnt/sda/wuwei/minibeam_fullchain_e150_10m_candidate_20260919/` 和
`/mnt/sda/wuwei/minibeam_fullchain_e300_10m_candidate_20260919/`。

## 23. Per-energy Copper fragment cascade 的三能量 10M 验收（2026-09-19）

水中 primary MCS 冻结后，剩余最明显误差是 250/300 的绝对剂量低约
1.4/1.8%。250 MeV/u 的 cascade generation 3 A/B 把 total ratio 从
`0.9863` 提高到 `0.9950`，IDD L1 从 `1.390%` 降到 `0.655%`，lateral L1
从 `1.692%` 降到 `1.073%`，2-D L1 从 `3.481%` 降到 `3.155%`；Bragg PVDR
保持 `1.002`，证明改善来自补齐 fragment 产额而非破坏 primary 横向分布。

最初把同一份含 250 MeV/u C12 节点的 shared cascade package 用于三能量是错误
的。它在 300 MeV/u 把 primary Copper products 从 `58.57M` 改成 `40.26M`，使
total ratio 降到 `0.9195`。无需重跑 TOPAS，已从保存的 CINEL02 raw records
离线重编三份 package：每份保留对应能量原始 C12+Cu campaign，只合入共同的
p/d/t/He/Li/Be/B/C11 fragment 节点。SHA256 为：

- 150: `bc67655dc8c862887316238944892f80a8425d8f2b1b166b1277082736b71b52`
- 250: `8294832188136de382ea60d2e279c569f698ba6b651e4150188be2cbd215fdc3`
- 300: `b25b008455c77b5444e192a6e0e605c14a42ccd266fc9af79000841c4f48d41f`

逐能量包、cascade generation 3 和冻结 FE/tail 的最终 10M 结果为：

| MeV/u | total ratio | 2-D L1 | IDD L1 | lateral L1 | Bragg PVDR | Bragg fixed peak/valley |
|---:|---:|---:|---:|---:|---:|:---:|
| 150 | 1.00009 | 3.359% | 0.857% | 1.665% | 1.030 | 1.004/0.989 |
| 250 | 0.99498 | 3.155% | 0.655% | 1.073% | 1.002 | 0.990/1.006 |
| 300 | 0.99546 | 2.975% | 0.614% | 0.901% | 0.955 | 0.989/1.007 |

三次运行的 Unified EM missing、queue overflow 和 baryon mismatch 均为 0，能量
残差为 `1.56e-5--2.70e-5`。generation 3 后预计再反应数相对 10M histories
可忽略；但 lookup miss 在高能量仍存在，300 MeV/u 为 `108,740/2,898,742`
次反应，主要是 proton gap/高能节点。这是下一轮 fragment 数据覆盖工作，而非
继续调 primary MCS。正式 150/250/300 配置现同时启用逐能量 cascade g3 和
water FE/tail。

## 24. p/d/t 中间能区空档补齐与三能量回归（2026-09-19）

冻结 water primary FE/tail、Copper EM、slit 和 stopping 参数后，新增 fragment+Cu
联合 miss 审计，按 `generation × Z/A × reason × 5 MeV/u bin` 同时累计查询次数、
入射动能、碰撞深度和沿碰撞后当前方向的直线剩余 Copper 路径；该量不包含后续
随机散射。300 MeV/u 的 256k 诊断确认旧包
`2746` 次 miss 中 `2295` 次来自能区空档；p/d/t 分别占 `1126/908/261` 次，
对应被终止的入射动能合计约 `1.23e6 MeV`。

Slurm job 6677 使用 64 CPU/48 GB，在 3 分 11 秒内一次完成 56 个定向 TOPAS
提取点。p 使用与参考物理表一致的 Binary Cascade，d/t 使用 INCL++；提取范围
分别为 `295--340`、`295--430`、`295--380 MeV/u`，节点间距 5 MeV/u，并在
两端与旧 campaign 重叠。package 编译器现可复用旧 metadata 的全部 raw source，
且强制比较指定 projectile 的 interaction bytes、对应 product bytes 和 energy
nodes。三份新包均保留各自原始 C12 数据，字节级检查通过。SHA256 为：

- 150: `595ed99eb317f52b15a37212581112622a4dfefa35a51f2564a1091aed40939b`
- 250: `a51626bc9382df308c824aa450767f95519b5fa31989f8a6db54430174ddd8de`
- 300: `3a888762fcae76a03f2a44359ddfd46344e906677688ec7cf68ffe95ba372483`

300 MeV/u 同 seed 256k 中，fragment-Cu miss 从 `2746` 降至 `449`，gap 从
`2295` 降至 0；p/d/t miss 从 `1193/982/288` 降至 `66/75/27`。10M 及三能量
回归结果如下，括号内为旧 per-energy package：

| MeV/u | total ratio | 2-D L1 | IDD L1 | lateral L1 | Bragg peak/valley/PVDR | fixed peak/valley |
|---:|---:|---:|---:|---:|:---:|:---:|
| 150 | 1.00010 (1.00009) | 3.359% (3.359%) | 0.857% (0.857%) | 1.665% (1.665%) | 1.006/0.977/1.030 | 1.004/0.989 |
| 250 | 0.99519 (0.99498) | 3.150% (3.155%) | 0.642% (0.655%) | 1.059% (1.073%) | 1.005/1.003/1.002 | 0.990/1.006 |
| 300 | 0.99603 (0.99546) | 2.960% (2.975%) | 0.578% (0.614%) | 0.867% (0.901%) | 0.963/1.008/0.955 | 0.989/1.007 |

250 已用当前同一 binary、`2026092510` seed 和相同 10,000,128 histories 重跑
旧包/新包严格 A/B。两者 Bragg depth 均为 `124.125 mm`；新包将 2-D/IDD/lateral
L1 从 `3.15535/0.65507/1.07266%` 小幅改善到
`3.14970/0.64191/1.05884%`，但 local peak/PVDR 仅从
`1.004775/1.001911` 变为 `1.004781/1.001784`，没有改变峰形。这里的 local
peak 不是单体素值，而是 1 mm 深度平滑、三个横向体素平均后，再取九个峰的
中位数。
三能量的 Unified
EM missing、queue overflow 和 baryon mismatch 均为 0，能量残差保持
`1.56e-5--2.70e-5`。

结论是补包消除了真实的非物理终止，并小幅改善 250/300 的 total 与 IDD；但
300 的局部 Bragg peak 和 PVDR 基本不变，因此剩余 Bragg 误差不能归因于 p/d/t
空档。正式配置已切到各自 gap-filled package；下一步应做分物种 Bragg 峰形和
水中 fragment 输运诊断，而不是继续扩大中间能区节点或修改已冻结的 primary MCS。

## 25. 300 MeV/u 分物种、分直接出生材料的 Bragg 诊断（2026-09-19）

在不修改 transport 物理、seed 或 cascade package 的条件下，增加了可选的 17 类
FP32 体素计分：primary C12，以及按直接出生于 Copper/water 分开的 secondary
C12、p、d、t、He3、He4、Z>=3 非 C12 和其他带电粒子。`birth_region` 表示当前
带电 track 的直接出生材料；Copper-born 粒子在水中再反应产生的子代归为
water-born，不能把它解释为完整 ancestry。该计分复用
`enable_charged_origin_voxel_scoring`，默认关闭，不增加正式运行负担。

300 MeV/u 使用 gap-filled package、generation 3、seed `202609300` 重跑
10,000,128 histories。诊断总剂量与原同 seed 运行的体素 L1 仅
`2.53e-8`，总和比 `1.00000000010`，说明计分没有改变输运。17 类重建总剂量的
全局相对差为 `8.97e-8`，最大非零体素相对差为 `1.34e-5`。Unified EM missing、
queue overflow 和 baryon mismatch 仍为 0，能量残差 `1.56e-5`。诊断运行时间
`176.0 s`，相同无分量基线 `165.7 s`，约有 6.2% 额外开销。

在 TOPAS Bragg 平面 `168.625 mm`，使用固定区域和 1 mm 深度平滑后：

| 区域 | GPU/TOPAS 总剂量 | primary C12 占 GPU | Copper-born 占 GPU | water-born 占 GPU |
|:---|---:|---:|---:|---:|
| peak | 0.9893 | 81.33% | 2.34% | 16.33% |
| shoulder | 0.9950 | 79.84% | 2.54% | 17.62% |
| valley | 1.0072 | 74.44% | 3.39% | 22.17% |

因此 300 MeV/u 的固定 peak 缺口约 1.07%，而不是局部极值指标显示的 3.7%；
局部 peak 误差主要是峰位/峰宽形状问题。Bragg 附近 fragment 剂量又以
water-born 为主，Copper-born 直接剂量只有 2--3%，所以若后续确认 fragment
输运主导，应先检查水中出生和水中传播，而不是继续修改 Copper package。

现有 TOPAS 分物种参考只有 1,024,000 histories，逐体素噪声较大，但在
Bragg +/-2 mm 和固定横向区域积分后可作方向性检查。peak/shoulder/valley 的
GPU/TOPAS 分量比分别为：primary C12 `1.001/0.984/1.024`，Z=1
`1.038/1.040/1.037`，all-helium `1.026/1.024/1.011`，以及 taxonomy 一致的全部
secondary Z>=3 合计 `0.906/0.954/0.944`。旧 TOPAS scorer 将所有 secondary
carbon isotope 合并，不能把它的 carbon 分量直接与 GPU 的 C12-only 类别比较；
因此这里不再报告会混淆 C10/C11 的单独 secondary-C12 比值。这组低统计结果
提示 primary 横向重分配和重碎片合计都值得继续隔离，但尚不足以据此修改
secondary Highland。

完整输出位于
`/mnt/sda/wuwei/minibeam_component_e300_10m_20260919/`；指标和图位于其
`analysis/metrics.json` 与 `analysis/bragg_source_components.png`。

## 26. 分量口径修正与 secondary C12 FE 候选（2026-09-19）

分量分析现与主剂量比较脚本共享同一套 MHD 坐标和固定 ROI 掩码，明确采用
peak `<0.25 mm`、shoulder `[0.25,0.9) mm`、valley `[0.9,1.8] mm` 的边界规则。
对已有 300 MeV/u 10M 数据重算后，固定 peak/shoulder/valley 总剂量比为
`0.989338/0.994962/1.007215`。TOPAS 的 helium 是全部 Z=2，因此比较改用 GPU
已有的 all-helium map；Bragg +/-2 mm 的三类 ROI 比为
`1.02591/1.02406/1.01085`。

primary 和 secondary 核反应中低于 cutoff 的带电子代仍按原方式把动能局部计入
总剂量，但分量标签现从父粒子改记到 `(child Z,A,water-born)`。同 seed 256k
计分开关 A/B 的总剂量体素 L1 为 `1.09e-8`，17 类重建总剂量的体素 L1 为
`2.48e-7`、全局相对差 `1.10e-9`，未发现负分量体素。这里仍是 immediate-birth
语义；Copper-born 粒子的水中子代归为 water-born，不能代表 Copper ancestry。

另增加默认关闭的 `minibeam_water_secondary_c12_mcs_model`。选择
`fermi_eyges_tail` 时，仅 secondary C12 在均匀水中复用 primary C12 的相关
角度—位移 kernel；p/d/t/He/重碎片继续使用原 Highland。minibeam ON/OFF 的
FP32 构建均通过。300 MeV/u、256k 同 seed full-chain smoke 中，开关前后总剂量
比为 `1.0000018`，Bragg 160--177 mm 积分比为 `1.0000038`，但体素 L1 为
`0.622%`，说明该路径确实改变了少量 C12 的空间搬运。该结果只验证工程接入和
影响范围，尚无同源 secondary-C12 相空间证据，因而没有写入正式三能量配置，
也不能宣称剂量精度改善。

## 27. secondary-C12 FE 同源纯 EM 验证（2026-09-19）

本轮使用同一份 12.8M 原始 histories 产生的 `1,622,795` 条 parent-0 C12 水入口
记录，分别送入 primary path（A）、secondary legacy Highland（B）、secondary
FE 0.10 mm（C）和 secondary FE 0.05 mm（C05）。这是同一 C12 状态经不同代码
路径的诊断，不把输入称为真实核反应 secondary。没有提交新 TOPAS；TOPAS 剂量
和 40/60/80/100/120 mm 平面均复用 job 6529。所有 GPU 运行使用 FP32，核非弹性
和核弹性关闭，输入队列没有增长，Unified EM missing、queue overflow 和核反应
计数均为 0，quality report 无 failure，能量残差为 `1.12e-9--1.67e-9`。

区间内按同一运行的 `(RunID,EventID,TrackID)` 配对，不把 TOPAS EventID 映射到
入口文件行。四个 20 mm 区间的全能量组结果如下：

| 模型 | 角增量方差 GPU/TOPAS | 位移方差 GPU/TOPAS | 位移-角协方差 GPU/TOPAS | q99 GPU/TOPAS |
|:---|:---:|:---:|:---:|:---:|
| B secondary legacy | 1.356--1.477 | 1.379--1.445 | 1.373--1.455 | 1.163--1.204 |
| C secondary FE 0.10 | 0.984--1.013 | 1.015--1.018 | 1.009--1.016 | 0.992--1.007 |
| C05 secondary FE 0.05 | 0.980--1.011 | 1.014--1.018 | 1.008--1.014 | 0.990--1.007 |

在五个绝对平面上，B 的角方差比从 `1.129` 随深度增至 `1.413`，folded-x
方差从 `1.006` 增至 `1.103`，q99 从 `1.049` 增至 `1.184`；C 对应范围为
`0.997--1.015`、`0.999--1.010` 和 `0.991--1.007`。固定 peak/shoulder/valley
通量也由 B 在 120 mm 的 `0.923/0.957/1.112` 改善为 C 的
`0.985/0.989/1.004`。这直接证明 legacy secondary C12 的水中散射增长过快，
而 FE 恢复了与 TOPAS/primary FE 一致的位置-角度联合演化。

C05 的实际 FE 分段数从 C 的 `1,987,503,392` 增至 `3,920,305,655`，不是空
开关。C05/C 的直接 IDD L1 为 `1.45e-5`；各平面角方差、folded-x 方差和协方差
的最大相对变化分别约 `0.212%/0.056%/0.426%`。q99.9 最大点变化约 2.17%，仍在
稀有尾统计敏感范围，不据此调参。关闭平面计分后，相对 C 的总剂量比为
`1.0000000001`、体素 L1 为 `6.25e-8`，确认诊断不扰动输运。

纵向 EM 并未在 primary/secondary 路径间闭合，原因已定位为 stopping 语义而非
散射。primary minibeam 路径在 Unified-EM loss 后应用冻结的
`minibeam_water_primary_stopping_power_scale=0.9958`，secondary C12 不应用该
scale。因而 C 的平均能量 GPU/TOPAS 比从 40 mm 的 `0.99896` 降到 120 mm 的
`0.95238`，Bragg peak 为 `123.875 mm`，比 TOPAS/primary A 的 `124.625 mm`
提前 0.75 mm。剂量汇总为：

| 路径 | total ratio | 2-D L1 | IDD L1 | Bragg TOPAS/GPU mm |
|:---|---:|---:|---:|:---:|
| A primary FE | 1.000019 | 2.283% | 0.1705% | 124.625/124.625 |
| B secondary legacy | 1.000019 | 4.387% | 1.898% | 124.625/123.875 |
| C secondary FE 0.10 | 1.000019 | 3.263% | 1.892% | 124.625/123.875 |
| C05 secondary FE 0.05 | 1.000019 | 3.275% | 1.893% | 124.625/123.875 |

因此 FE 改善了横向分布，但不能也不应通过散射参数补偿 secondary stopping 差异。
150/300 MeV/u 现有 256k 同源数据的低统计筛查给出同样结论：secondary FE 的
平面角方差分别为 TOPAS 的 `0.994--1.016` 和 `0.991--1.039`，folded-x 方差为
`0.995--1.004` 和 `0.999--1.011`，q99 为 `0.979--1.008` 和
`0.978--1.019`，显著优于 legacy；但未统一 stopping 时 IDD L1 仍为
`2.211%/1.789%`，Bragg 分别提前 `0.25/1.25 mm`。26k/35k 样本的 q99.9 和
sparse 2-D L1 不作验收指标。

结论限定为：secondary-C12 FE 的散射部分通过 150/250/300 同源筛查和 250
MeV/u 分步稳定验证；正式三能量配置继续保持该开关关闭。下一步先用真实
secondary-C12 出生谱验证，并隔离统一 C12 stopping/straggling 语义，之后才考虑
He-4 的独立 FE 标定。完整输出、汇总图和 JSON 位于
`/mnt/sda/wuwei/minibeam_secondary_c12_same_source_e250_20260919/`，150/300
筛查位于
`/mnt/sda/wuwei/minibeam_secondary_c12_same_source_multienergy_20260919/`。

## 28. C12 能损语义闭合、FE 复核和真实出生状态回放（2026-09-19）

本节使用修正后的统一 Unified-EM 路径和精确正向 z 边界截断，取代第 27 节中
尚未闭合 stopping 语义及受边界 bug 污染的纵向数字。正式三能量配置、Copper
EM、slit、primary MCS、stopping 和 cascade package 均未修改；全部运行使用
FP32，未提交新 TOPAS。

首先将同一批 `1,622,795` 条水入口 C12 同时送入 primary 和 secondary path，
关闭 MCS、核非弹性与核弹性。secondary 显式启用与 primary 相同的 Unified-EM，
并新增默认值为 1 的诊断参数
`minibeam_water_secondary_c12_post_sample_loss_scale`，其应用位置与 primary
完全相同：对已经采样的整步总能损做乘法，且与 FE 开关解耦。受控结果为：

| 对照 | primary/secondary IDD L1 |
|:---|---:|
| 无 straggling，scale=1 | 0.00827% |
| 有 straggling，scale=1 | 0.02993% |
| 有 straggling，scale=0.9958 | 0.03137% |

120 mm 平面的 secondary-primary 平均能量差仅约 `-0.09` 至 `-0.13 MeV`。
secondary 审计得到 raw/scaled loss 为
`4.18993e9/4.17233e9 MeV`，实测比 `0.99579945`；因此 `0.9958` 确实同时
缩放均值与采样涨落，方差尺度为 `0.9958^2=0.99161764`。把 1 改为 0.9958
后，primary/secondary 的 IDD L1 变化分别为 `1.66163%/1.65991%`，120 mm
平均能量分别增加 `25.8263/25.8179 MeV`。这完成了 C12 能损路径的受控闭合，
也确认不能只给均值补一个 stopping 系数。

闭合能损后，复用已有 TOPAS job 6529 重新检查 secondary C12 散射。四个 20 mm
区间的 GPU/TOPAS 范围为：

| 模型 | 角方差 | 位移方差 | 位移-角协方差 | q99 | q99.9 |
|:---|:---:|:---:|:---:|:---:|:---:|
| legacy Highland | 1.471--1.512 | 1.535--1.552 | 1.516--1.537 | 1.202--1.221 | -- |
| FE 0.10 mm | 0.969--0.988 | 0.995--1.017 | 0.990--1.012 | 0.983--0.994 | 0.898--0.954 |
| FE 0.05 mm | 0.964--0.995 | 0.993--1.016 | 0.988--1.009 | 0.980--0.996 | 0.897--0.951 |

100--120 mm 的配对存活率为 TOPAS `0.868742`、FE010 `0.868199`、FE005
`0.868156`。剂量的 2-D/IDD/lateral L1 分别为 legacy
`4.763/0.182/3.553%`、FE010 `2.310/0.175/0.691%`、FE005
`2.286/0.174/0.654%`。FE010 到 FE005 的 IDD L1 仅 `0.03095%`，固定 ROI
与区间联合矩稳定；细体素 L1 `2.123%` 来自 0.1 mm 横向体素中的随机实现变化，
不作为分步失败判据。TOPAS Bragg 为 `124.625 mm`，三组 secondary 为
`124.375 mm`，仅差一个 0.25 mm 网格。由此 secondary-C12 FE 在相同入口和
统一能损语义下通过散射验证，但正式开关仍保持关闭。

此前 Unified-EM+FE 的周期性剂量尖峰和表观步长不稳定并非 FE 固有问题，而是
secondary z 边界截断忽略了 `0 < dz_step <= 1e-5 mm` 的正距离，使轨迹在边界前
残留后跨越多个深度 bin。改为接受所有正边界距离后，FE010/FE005 IDD L1 从约
`2.65%` 降为 `0.03095%`；连续能损也已固定使用步首 `bin_z` 计分。平面计分
开关的 IDD/体素 L1 约 `4.1e-9/4.5e-8`。

最后增加了真实 C12 queue-birth 快照和内部出生回放。导出记录保留唯一 replay
particle ID、原 incident history、RNG stream、generation、直接出生材料、能量、
权重、真实位置和方向。250 MeV/u、256k full-chain 导出 `4511` 条水中出生 C12，
覆盖 generation 0--2、`z=0.014--249.867 mm`，其中 `644` 条为非正向，且
`221` 条与其他记录共享 incident history；总出生动能为
`586838.968 MeV`。Copper 产物另在水入口输出，共 `58116` 条带电粒子；本次
256k 样本有 17 条其他 carbon isotope，但没有 C12，因此未把 Copper-born C12
混入真实水中出生回放。内部回放使用 4511 个独立 ledger/plane ID，但剂量仍按原
`256000` histories 归一化。纯 EM 回放载入并完成全部 4511 条，queue 不增长，
核/弹性反应、Unified-EM missing 和 overflow 均为 0；总沉积
`586839.041 MeV`，能量残差 `8.42e-8`。这证明任意水内出生深度、反向粒子、
唯一粒子身份和归一化路径已经闭合；由于没有对应 TOPAS 真出生记录，本结果只
是实现/覆盖闭合，不是新的物理 match 声明。

能损与 FE 输出位于
`/mnt/sda/wuwei/minibeam_secondary_c12_energy_path_e250_20260919/`，真实出生
导出和回放位于
`/mnt/sda/wuwei/minibeam_secondary_c12_birth_validation_e250_20260919/`。
下一步可在不改正式配置的前提下，用 full-chain 的真实 secondary-C12 出生谱做
开关 A/B；通过后再单独验证 He-4，不能把 C12 的 FE 参数直接推广到其他物种。

## 29. secondary-C12 full-chain 分阶段 A/B（2026-09-19）

在 full-chain A/B 前先完成了 secondary 公共水路径的双向深度边界修复。反向
粒子恰好位于边界时，现在按运动方向归入即将进入的上游 bin；legacy 非 Unified
路径也不再用 `1e-5 mm` 下限放大真实的微小边界距离。确定性测试覆盖
`边界前/边界上/边界后 × 正向/反向 × legacy/Unified` 共 12 个组合，所有组合的
首个计分 bin 均符合预期。legacy/Unified 总剂量比为 `0.999999921`，能量残差
分别为 `2.12e-8/6.36e-8`，核反应和 overflow 均为 0。结果位于
`/mnt/sda/wuwei/minibeam_secondary_depth_boundary_20260919/`。

新增默认关闭的 `minibeam_water_secondary_c12_enable_unified_em`，只改变均匀水中
secondary C12 的 EM 路径，不改变 p/d/t/He/重碎片。300 MeV/u、256k、同 seed
依次运行：

| 组 | secondary C12 EM | loss scale | MCS |
|:---:|:---|---:|:---|
| A | 正式 legacy 路径 | 1 | Highland |
| B | Unified EM | 1 | Highland |
| C | Unified EM | 0.9958 | Highland |
| D | Unified EM | 0.9958 | FE |

相对已有 10M TOPAS，以 histories 比例做绝对剂量换算而不拟合，结果为：

| 组 | total ratio | 2-D L1 | IDD L1 | lateral L1 | fixed peak/shoulder/valley |
|:---:|---:|---:|---:|---:|:---:|
| A | 1.002270 | 12.277% | 0.706% | 3.127% | 0.9874/0.9897/1.0225 |
| B | 1.002252 | 12.277% | 0.707% | 3.124% | 0.9884/0.9879/1.0227 |
| C | 1.002263 | 12.274% | 0.708% | 3.123% | 0.9874/0.9890/1.0231 |
| D | 1.002226 | 12.269% | 0.705% | 3.101% | 0.9881/0.9879/1.0237 |

256k 的 2-D 与局部极值仍明显受统计波动影响。A→B 和 B→C 对总剂量及固定 ROI
没有可辨识的稳定收益，因此没有扩大为四组高统计。C→D 是已有同源相空间证据
支持的唯一有意义配对，随后使用两个独立 GPU seed 各跑 `10,000,128` histories：

| seed | 模型 | total ratio | 2-D L1 | IDD L1 | lateral L1 | fixed peak/shoulder/valley |
|---:|:---:|---:|---:|---:|---:|:---:|
| 202609300 | C | 0.996018 | 2.9544% | 0.5736% | 0.8679% | 0.98885/0.99516/1.00708 |
| 202609300 | D | 0.996013 | 2.9548% | 0.5732% | 0.8698% | 0.98942/0.99499/1.00727 |
| 202609301 | C | 0.994946 | 3.0133% | 0.6700% | 1.0158% | 0.99174/0.99591/1.00742 |
| 202609301 | D | 0.994952 | 3.0124% | 0.6700% | 1.0166% | 0.99184/0.99581/1.00741 |

FE 会显著重排 secondary-C12 自身的空间剂量：C/D 的该分量体素 L1 为
`31.47%/31.06%`；但它只占总剂量约 `0.547--0.550%`，在 Bragg +/-2 mm 约
`0.97%`。因此 C→D 的总剂量比在两个 seed 分别为 `0.9999945/1.0000064`，方向
相反；2-D、IDD、lateral 和固定 ROI 也没有跨 seed 的一致联合改善。water-born
非 C12 分量的 D/C 总量比为 `0.999910/1.000037`。该量只对 C12 后代变化敏感，
不是严格的 C12 ancestry 绝对计分。

150/250 MeV/u 另做了 256k C/D 回归。D/C 总剂量比分别为
`0.9999765/0.9999746`；两能量均没有大幅退化，但各指标变化处于低统计小量级，
不能作为正式推广证据。所有运行的 Unified-EM missing、queue overflow 和
baryon mismatch 为 0，10M 能量残差为 `1.56e-5`，C/D 的 Copper cascade 计数
在同 seed 下完全一致。

真实出生回放格式也已扩展为保留原 `rng_stream`、water generation 和
`birth_region`；4511 条记录逐条一致，纯 EM 回放能量残差 `6.33e-8`。多 spot/
shard 累加现在合并 `c12_birth_records`、偏移 source history，并重新分配全局唯一
replay particle ID。该回放仍只用于隔离传播，不能把不同代祖先和后代的独立回放
相加当成 full-chain 剂量。

结论是 secondary-C12 FE 的输运结构仍优于 legacy，但当前 full-chain 总剂量收益
受该分量占比限制且未跨 seed 一致；正式三能量配置继续保持 C12-only Unified、
`0.9958` secondary scale 和 secondary FE 全部关闭。高统计输出位于
`/mnt/sda/wuwei/minibeam_secondary_c12_fullchain_cd_e300_10m_seed*_20260919/`。

## 30. FE 先接入全部带电碎片（2026-09-19）

按“先接入、再修正”的开发顺序，公共水中 FE kernel 已从 C12 推广到队列中的
带电离子，并保留单一 YAML 回退开关：

```yaml
multiple_scattering_model: fermi_eyges  # 或 highland
fermi_eyges_species: all_charged        # c12/c12_he4/c12_he4_pdt/all_charged
fermi_eyges_max_segment_mm: 0.1
```

显式选择 `highland` 时也会关闭旧的 minibeam-only primary/secondary FE 路径，
避免表面上选择 Highland、实际 primary 仍使用 FE。150/250/300 三份正式 minibeam
配置现已选择 `fermi_eyges + all_charged`；这表示代码和配置已接入，不表示各碎片
物种已完成 TOPAS 标定。旧 Highland kernel 未删除，可以通过上述一个公共选择器
整体回退。

当前非 C12 离子暂时复用 C12+water 约束得到的 FE core/tail 常数，但运动学使用
各自 `(Z,A)`、beta 和动量。没有把 `0.9958` 扩展到这些路径；通用 secondary
输运保持 loss scale=1。后续按 He-4、p/d/t、其他重碎片的顺序修正物种参数。

minibeam ON/OFF 两种 FP32 构建均通过。minibeam-OFF 的 200 MeV/u、10k-history
full-physics 分阶段 smoke 全部通过质量检查且 queue overflow 为 0。直接 MCS
路由计数如下：

| scope | C12 FE | He-4 FE | p/d/t FE | other FE | Highland | energy residual |
|:---|---:|---:|---:|---:|---:|---:|
| `c12` | 31,582 | 0 | 0 | 0 | 3,882,684 | `3.97e-6` |
| `c12_he4` | 31,504 | 1,361,865 | 0 | 0 | 2,809,671 | `3.74e-6` |
| `c12_he4_pdt` | 31,498 | 1,361,732 | 3,164,238 | 0 | 414,033 | `3.15e-6` |
| `all_charged` | 31,790 | 1,363,806 | 3,160,107 | 476,537 | 0 | `2.92e-6` |

独立 Highland full-physics smoke 的 FE 计数全为 0、Highland 为 3,910,765，
能量残差 `4.59e-6`，证明回退路径仍可运行。不同 scope 改变轨迹、停止和后续核
反应，因此总步数本来就不必相等；这些结果只验收接线、有限数、能量记账与质量
门，不声明 He/p/d/t/重碎片的相空间已经匹配 TOPAS。本轮没有提交新 TOPAS 任务。

另使用正式 250 MeV/u minibeam 配置完成同 seed、256k full-chain FE/Highland
A/B。两组均 `accepted=true`、failures 为空、全部 overflow 为 0；由于配置本身为
research 并带已声明的近似，状态仍为预期的 `non_production`。FE/Highland 的
能量残差为 `1.72e-5/1.73e-5`，Bragg 最大值位于同一深度 bin。FE/Highland
总剂量比为 `1.000106`，IDD L1 为 `0.1367%`，lateral L1 为 `3.7505%`，体素
L1 为 `16.23%`。后两项说明全碎片 FE 已明显改变横向输运，不能把“运行稳定”
误写成“物理已经匹配”；下一轮首先从 He-4 和 p/d/t 的分物种角分布检查这些
C12-derived 参数是否过宽。FE/Highland 耗时为 `14.15/13.62 s`。

同一 full-chain 工况随后完成了逐级 scope 消融。相对全 Highland：

| scope | total ratio | IDD L1 | lateral L1 | voxel L1 | Bragg bin |
|:---|---:|---:|---:|---:|---:|
| `c12` | 1.000053 | 0.0704% | 3.6675% | 15.118% | 498 |
| `c12_he4` | 1.000022 | 0.0977% | 3.7298% | 15.614% | 498 |
| `c12_he4_pdt` | 1.000167 | 0.1317% | 3.7526% | 16.118% | 498 |
| `all_charged` | 1.000106 | 0.1367% | 3.7505% | 16.231% | 498 |

逐级增量的 lateral L1 为 He-4 `0.3583%`、p/d/t `0.1612%`、其余带电碎片
`0.6480%`。因此本次 3.75% 总横向差异主要来自已经验证过的 primary/C12 FE，
不是 p/d/t 的 1.46 亿个 FE 步引起的数值爆炸；非 C12 三组仍产生可测量但较小的
空间重分配。256k 体素增量受统计噪声影响较大，不能据此拟合参数。下一步仍按
He-4→p/d/t→其他碎片做同源相空间验证，但当前没有发现需要撤销接入的稳定性 bug。

## 31. 250 MeV/u all-charged FE、10,000,640 histories full-physics 对照（2026-09-19）

使用正式 `beam_minibeam_field3cm_copper_e250_256k.yaml` 物理设置，将本次 GPU
histories 精确设为 `10,000,640`，与已有 TOPAS 五个 shard 的合计 histories 完全
相同。比较使用绝对 Gy，不进行 history 缩放或剂量拟合；网格从元数据读取为横向
`0.1 mm`、深度 `0.25 mm`，峰谷曲线使用 `1 mm` 深度平滑。运行后正式配置的
`tps_histories_scale` 已恢复为 `0.01`。

GPU 日志确认 `fermi_eyges + all_charged` 生效：全部 secondary MCS 步均进入 FE，
Highland 步数为 0。运行耗时 `135.63 s`，吞吐率 `73,733 histories/s`，能量残差
`1.72e-5`；所有 queue overflow 和 Unified-EM missing-domain 均为 0。质量报告
`accepted=true`、`failures=[]`；由于正式配置仍声明 research approximation，状态为
预期的 `non_production`。

整体绝对剂量结果：

| metric | result |
|:---|---:|
| GPU/TOPAS total dose | 0.995693 |
| 2-D dose L1 / TOPAS | 3.2359% |
| 2-D Pearson | 0.999453 |
| IDD L1 / TOPAS | 0.5955% |
| IDD Pearson | 0.999974 |
| lateral-integral L1 / TOPAS | 1.1468% |
| lateral-integral Pearson | 0.999913 |
| TOPAS / GPU Bragg depth | 124.375 / 124.375 mm |

分深度局部峰谷和 lateral 1-D 指标如下。比值均为 GPU/TOPAS；lateral L1 在
`|x| <= 30 mm` 内计算。入口请求的 `0.5 mm` 对应最近网格中心 `0.375 mm`。

| depth (mm) | peak ratio | valley ratio | PVDR ratio | lateral L1 | Pearson |
|---:|---:|---:|---:|---:|---:|
| 0.375 | 1.0222 | 0.9936 | 1.0288 | 3.0820% | 0.999649 |
| 19.875 | 1.0045 | 1.0110 | 0.9935 | 2.4167% | 0.999642 |
| 39.875 | 0.9892 | 1.0286 | 0.9618 | 2.4871% | 0.999509 |
| 59.875 | 0.9917 | 0.9914 | 1.0003 | 2.5701% | 0.999458 |
| 79.875 | 1.0057 | 1.0031 | 1.0027 | 2.5680% | 0.999323 |
| 99.875 | 0.9964 | 0.9947 | 1.0018 | 2.8029% | 0.999093 |
| 119.875 | 0.9726 | 1.0125 | 0.9607 | 3.0091% | 0.998855 |
| 124.625 | 0.9841 | 0.9784 | 1.0058 | 3.4143% | 0.998515 |
| 134.625 | 1.0211 | 0.9899 | 1.0315 | 4.4975% | 0.994003 |

局部极值在 40 和 120 mm 附近仍显示约 4% 的 PVDR 低估，但固定 ROI 积分更接近：
在 39.875 mm 的 peak/shoulder/valley 比为 `1.0065/0.9990/0.9769`，在
99.875 mm 为 `0.9981/1.0006/0.9981`，在 124.625 mm 为
`0.9931/1.0042/1.0052`。134.625 mm 已位于 distal 低剂量区，局部指标和 4.50%
lateral L1 更易受统计波动影响，不能与 plateau/Bragg 指标等权解释。

本次结果说明 all-charged FE 接入后，250 MeV/u full-physics 的绝对总剂量、IDD、
Bragg 峰位和大部分固定区域剂量已经较好匹配；主要剩余差异仍是约 3.24% 的二维
局部剂量 L1、部分深度的局部峰谷形状，以及尚未逐物种标定的 fragment FE 尾部。
这是一组单 seed 高统计基线，不等同于三能量、多 seed 的最终验收。

输出位于：

```text
/mnt/sda/wuwei/minibeam_e250_allcharged_fe_fullphysics_10000640_20260919/
/mnt/sda/wuwei/minibeam_e250_allcharged_fe_fullphysics_10000640_20260919/comparison_topas/
```

## 32. Primary water loss scale 0.9958 / 1.0 单因素对照（2026-09-19）

在第 31 节的 250 MeV/u、all-charged FE、full-physics 高统计工况上，使用同一
binary、seed 和精确相同的 `10,000,640` histories，仅将
`minibeam_water_primary_stopping_power_scale` 从 `0.9958` 改为 `1.0`。两组均与
同一份 `10,000,640` histories TOPAS 绝对剂量比较，不做拟合。scale=1 运行同样
`accepted=true`、`failures=[]`，所有 queue overflow 为 0；正式 YAML 在运行后已
恢复为 `0.9958`。

| metric | scale=0.9958 | scale=1.0 | better |
|:---|---:|---:|:---|
| GPU/TOPAS total dose | 0.995693 | 0.996131 | 1.0（仅总量） |
| 2-D dose L1 | 3.2359% | 3.6509% | 0.9958 |
| IDD L1 | 0.5955% | 1.1202% | 0.9958 |
| lateral-integral L1 | 1.1468% | 1.2029% | 0.9958 |
| GPU Bragg depth | 124.375 mm | 123.875 mm | 0.9958 |
| TOPAS Bragg depth | 124.375 mm | 124.375 mm | — |

scale=1 将 Bragg 峰提前两个深度体素（`0.5 mm`），说明该 `0.42%` 能损差异会沿
完整射程积累，不能仅凭数值接近 1 而忽略。plateau 中 scale=1 偶尔改善单个局部
指标，例如 79.875 mm lateral L1 从 `2.5680%` 降到 `2.4642%`；但纵向和二维全局
指标一致支持 `0.9958`。119.875 mm 的 lateral L1 则从 `3.0091%` 恶化到
`4.1718%`。Bragg 附近 124.625 mm 的固定 peak/shoulder/valley 比：

```text
scale=0.9958: 0.99315 / 1.00423 / 1.00518
scale=1.0:    0.98634 / 0.99752 / 0.99658
```

因此在当前 Geant4 stopping table、能损采样和 TOPAS 参考下，保留 `0.9958` 的证据
明显强于改回 `1.0`。这仍是一个经验校正；它改善的是完整水程中的纵向演化，不代表
其微观能损均值和方差已经独立验证。

scale=1 输出与图位于：

```text
/mnt/sda/wuwei/minibeam_e250_allcharged_fe_fullphysics_scale1_10000640_20260919/
/mnt/sda/wuwei/minibeam_e250_allcharged_fe_fullphysics_scale1_10000640_20260919/comparison_topas/
```

## 33. 公共 FE 平面诊断修复与 scope 隔离（2026-09-19）

修复了公共 `multiple_scattering_model: fermi_eyges` 绕过旧 minibeam 平面诊断的
问题。公共多子段 FE helper 现在可接收步内观察路径，在包含交点的真实 FE 子段调用
已有 integrated-Brownian 条件 Gaussian bridge；Poisson tail 仍按真实事件位置判断
是否发生在观察平面上游。观察使用独立 RNG dimensions，不改变已采样的输运终态。
primary 和 secondary 公共 FE 均使用同一接口，不再依赖整步位置/角度线性插值。

水中平面 CSV 在保留原字段顺序的基础上追加：

```text
particle_id,rng_stream,atomic_number,mass_number,weight,transport_path
```

primary 的 `particle_id` 使用全局 history，secondary replay 使用唯一 RNG stream，
`transport_path` 明确区分 primary kernel 与 secondary queue；多 shard 累加同时偏移
primary 的 `source_history` 和 `particle_id`。诊断输出由此可以按粒子身份和 Z/A
配对，不再只能假定所有记录都是 primary C12。

使用正式 250 MeV/u full-chain、256k、all-charged FE 做平面计分 on/off 对照：

| check | result |
|:---|---:|
| plane records | 121,603 |
| counts at 20/40/60/80/100/120/124.375 mm | 26,988 / 23,441 / 20,152 / 17,107 / 14,166 / 11,014 / 8,735 |
| duplicate `(particle_id, plane)` | 0 |
| backward records / wrong depth | 0 / 0 |
| scoring-on/off voxel dose L1 | `9.72e-9` |

同一份 1,622,795 粒子 C12 secondary replay 使用公共 C12 FE 后输出 6,127,025 条
平面记录；五个平面的计数为
`1,432,245/1,341,262/1,244,093/1,131,275/978,150`。Z/A 全为 C12，唯一身份
无冲突、无反向记录，平面计分 on/off 的 voxel dose L1 为 `6.00e-8`。两组质量
状态均为预期的 `non_production`，能量账本和 overflow 检查通过。

另用同源 30,878 粒子、纯 EM、无核反应的 primary-only replay 验证旧 minibeam FE
和公共 C12 FE 的实现一致性。两者 total ratio 为 `1.00000000023`、IDD L1
`3.97e-9`、lateral L1 `2.36e-9`、voxel L1 `3.28e-8`。此前显式选择
`multiple_scattering_model: highland` 的测试会按设计同时关闭旧 primary FE，不能
用作旧/新 FE 等价性对照；本次已依据 canonical 配置修正该口径。

最后在同 seed 256k full-chain 中冻结 primary `0.9958`、secondary EM、Copper、
package 和 slit，仅改变 secondary FE scope：

| comparison | total ratio | IDD L1 | lateral L1 | voxel L1 |
|:---|---:|---:|---:|---:|
| 公共 C12 FE / 旧 primary FE + secondary Highland | 0.999993 | 0.0070% | 0.2281% | 0.6547% |
| all-charged FE / 公共 C12 FE | 1.000053 | 0.1181% | 0.7799% | 6.6059% |

因此公共 C12 接线和旧模型已闭合；当前候选相对 C12-only 的主要新增空间变化来自
非 C12 碎片。上述是 GPU/GPU 单 seed 低统计隔离量，不代表 all-charged FE 更接近
TOPAS，也不能据此调整 9.9/0.0025/2.4。下一步应从固定 ROI 剂量贡献最大的物种
开始做同源纯 EM 相空间验证，首选 He-4，再检查 p/d/t 和其他重碎片。

最终代码的 minibeam ON/OFF 两种 FP32 构建均通过；仓库当前没有注册 CTest，独立
sampler 的 500k 样本检查仍满足解析 FE 矩、Poisson 事件率、bridge 半步矩和倾斜
方向旋转检查。最终 256k 单平面 smoke 输出 26,988 条记录，身份与 path 标签检查
全部通过。

诊断输出：

```text
/mnt/sda/wuwei/minibeam_fe_plane_bridge_validation_20260919/
/mnt/sda/wuwei/minibeam_fe_scope_isolation_20260919/
```

## 34. Secondary EM 闭合与分物种 FE minibeam A/B（2026-09-19）

He-4 300 MeV/u、10 mm 水片的旧参考差异已定位为 TOPAS EM table 上限：默认
600 MeV 小于 He-4 的 1200 MeV 总动能。用 10 GeV 上限重建 24-case 矩阵后，
TOPAS/GPU 的平均能损为 `3.50323/3.52421 MeV/u`，出口能谱标准差为
`0.174162/0.174193 MeV/u`。全矩阵最差平均能损误差 `0.765%`，最差谱宽误差
`4.302%`，谱宽比中位数 `1.00314`。因此不增加 He-4 stopping scale；非 C12
secondary 需要 Unified EM 才会真正采样能损涨落。

水出口 GPU 计分改为精确端点的 step-start scorer，24 个 case 均从旧规避式
`thickness-0.005 mm` 改为真实 `thickness`，每例记录 `250,000/250,000`；TOPAS
下游 0.010 mm 真空漂移在分析中沿方向反投影回水面。逐 case 报告包含能损、谱宽、
`Var(theta)`、`Var(x)`、`Cov(x,theta)`、q68/q95/q99/q99.9 与置信区间。

随后以 250/300 MeV/u、同 seed、每组 256k histories 做三组 FP32 full-chain 筛查：

| group | FE parameters | secondary EM |
|:---|:---|:---|
| A | shared C12 `9.9/0.0025/2.4` | legacy |
| B | p/d/t/He-4 water fit | legacy |
| C | p/d/t/He-4 water fit | Unified + straggling |

所有组均 `accepted=true`、overflow=0，primary `0.9958`、Copper/slit/package/seed
完全相同。A→B 仅改变 FE 参数，对目标剂量影响很小：250 MeV/u 的
2-D/IDD L1 为 `13.162/0.6846% -> 13.177/0.6893%`，300 MeV/u 为
`12.252/0.7158% -> 12.253/0.7098%`。B→C 隔离 secondary EM；300 MeV/u 的
IDD/lateral L1 从 `0.7098/3.144%` 变为 `0.6983/3.097%`，但 256k 单 seed
不足以把这个小变化认定为稳定收益。

共享 ROI 口径的关键 C 组比值如下：

| energy | depth | peak | shoulder | valley |
|---:|---:|---:|---:|---:|
| 250 | 39.875 mm | 1.0217 | 1.0125 | 0.9692 |
| 250 | 119.875 mm | 0.9779 | 1.0445 | 0.9820 |
| 250 | Bragg 124.375 mm | 0.9892 | 1.0315 | 1.0046 |
| 300 | 39.875 mm | 1.0145 | 0.9995 | 0.9742 |
| 300 | 119.875 mm | 1.0309 | 1.0053 | 1.0005 |
| 300 | Bragg 168.625 mm | 0.9879 | 0.9910 | 1.0221 |

300 MeV/u C 组 Bragg peak/shoulder/valley 中 primary C12 占
`80.90/78.60/73.87%`，He-4 占 `7.13/7.89/9.88%`，p/d/t 合计约
`4.50/5.08/6.55%`。250 MeV/u 对应 He-4 为 `4.42/5.20/9.04%`。新的
p/d/t/He-4 FE 参数只改变这些次要分量的空间分配，解释了 A→B 对总剂量的效应很小。
当前证据不支持为它启动无差别双 seed 10M A/B，也不支持开发 FE shoulder 或继续
CT 标定。完整 JSON/CSV 与复现脚本位于：

```text
/mnt/sda/wuwei/fe_species_water_em10gev_20260919/
/mnt/sda/wuwei/minibeam_species_fe_ab_20260919/
benchmark/carbonminibeam/summarize_species_fe_minibeam_ab.py
```

为避免用出生能量或 step count 代替目标剂量贡献，另增加了默认关闭的轻量诊断
`enable_minibeam_energy_band_roi_scoring`。它按实际沉积步的步首能量将 p/d/t/He-4
剂量分为 `<50`、`50--300`、`>300 MeV/u`，并直接按共享固定 ROI 和深度累计。
250/300 MeV/u C 组重跑均通过 quality、能量账本和 overflow 检查；该诊断与四物种
component map 在所有入口/40/120/Bragg ROI 的最差闭合误差分别为 `0.052%/0.077%`。

Bragg 处四物种合计的能区剂量比例为：

| energy | ROI | <50 MeV/u | 50--300 MeV/u | >300 MeV/u |
|---:|:---|---:|---:|---:|
| 250 | peak | 53.30% | 46.52% | 0.18% |
| 250 | shoulder | 52.01% | 47.81% | 0.18% |
| 250 | valley | 47.45% | 52.35% | 0.20% |
| 300 | peak | 42.72% | 56.72% | 0.56% |
| 300 | shoulder | 42.58% | 56.81% | 0.61% |
| 300 | valley | 42.83% | 56.59% | 0.58% |

这说明 `>300 MeV/u` 外推区在 Bragg 目标 ROI 中小于 0.7%，不值得优先补高能节点；
`<50 MeV/u` 却占约 43--53%，若后续分物种 A/B 显示可辨识偏差，应优先增加低能
held-out 标定。诊断结果位于：

```text
/mnt/sda/wuwei/minibeam_species_fe_energybands_20260919/
benchmark/carbonminibeam/analyze_minibeam_energy_band_roi.py
```

## 35. Valley 分物种绝对剂量残差归因（2026-09-19）

暂停扩大 FE / CT 标定。primary `0.9958`、Copper、slit 和 CT 设置未改。
点估计保留；没有 per-history ROI 矩或独立 shard，**不确定度标为 unknown**。
不再使用 `|D|/√N` 或 “可分辨 1%” 标记。

可比来源类是 TOPAS 带电核 origin 与 GPU 沉积离子 Z 图。`neutral_origin` 和
`unclassified` 是尚未匹配的 TOPAS 计分，记为缺少对应计分的残差，不能解释为
GPU 缺失物理剂量。各引擎内部 category-vs-total 闭合单独报告，不作为跨引擎
映射验收。Bragg ±2 mm 使用未平滑数组；1 mm 平面继续平滑。

当前 C 组 256k vs TOPAS 1.024M，固定 valley ROI：

| 能量 | 深度 | window | GPU/TOPAS | Δ_total | 可比类 ΣΔ | unmatched 缺计分 | primary_c Δ |
|---:|---|---|---:|---:|---:|---:|---:|
| 250 | 40 mm | 1 mm 平滑 | 0.9639 | −3.608% | −2.785% | −0.822% | −2.659% |
| 250 | Bragg | 1 mm 平滑 | 1.0072 | +0.718% | +0.988% | −0.270% | +1.228% |
| 250 | Bragg ±2 mm | 未平滑 | 1.0098 | +0.977% | +1.257% | −0.280% | +1.273% |
| 300 | 40 mm | 1 mm 平滑 | 0.9781 | −2.188% | −1.395% | −0.793% | −0.672% |
| 300 | Bragg | 1 mm 平滑 | 1.0231 | +2.314% | +2.642% | −0.328% | +2.019% |
| 300 | Bragg ±2 mm | 未平滑 | 1.0237 | +2.370% | +2.721% | −0.352% | +2.374% |

较早 GPU 10M 分量运行只作 auxiliary，不与 C 组合并为同模型多 seed。

```text
/mnt/sda/wuwei/minibeam_valley_residual_20260919/
benchmark/carbonminibeam/analyze_minibeam_residual_attribution.py
```

## 36. Primary-carbon 诊断与同源入口（2026-09-19）

CINEL03 局部沉积补记后重跑 250/300 C 组 256k。quality accepted、overflow=0。
诊断开关相对无开关 C 组剂量 L1 为 `1.1e-8` / `1.4e-8`。修复后能区 vs
component 最差不闭合仍为 `0.052%` / `0.077%`；该量级现在是其余路径/FP32，
不能再用旧 JSON 当修复证据。

新增独立 primary-C12 ROI：局部离子能损、浓缩电子落地/跨 ROI/逃逸、轨迹长度。
本配置水中未启用显式电子包，`delta_landed=0`，浓缩电子留在 `local_ion`。
与 TOPAS 显式电子只能按 production-cut 映射比较。

同源入口（同一 GPU 水核，256k 场的 C12 存活者，不按存活数再归一化）：

| 能量 | valley | GPU-entry/TOPAS-entry | C12 穿越比 | 均能 MeV | rms θ mrad |
|---:|---|---:|---:|---|---|
| 250 | 40 mm | 0.997（−0.30%） | 1.016 | 1527.9 / 1528.4 | 21.16 / 21.37 |
| 250 | Bragg | 1.013（+1.34%） | 1.015 | 355.5 / 356.3 | 23.91 / 24.06 |
| 300 | Bragg | 1.002（+0.21%） | 0.993 | 429.9 / 429.6 | 22.56 / 22.27 |

250 MeV/u 40 mm 全链 Δ_total=−3.61% 在换成 TOPAS 入口后并未消失。该处 C12
穿越、能谱和角度已经接近，因此下一步不是改 primary MCS，而是 C12 局部能损
空间分布与浓缩电子计分。300 MeV/u Bragg 全链 +2.3% 同样不是入口相空间主因。
现有 TOPAS 回放 dose 的 original-history 约定与 GPU 不一致，未当作水核判决。
不确定度仍为 unknown；尚未做 per-history ROI 矩。

```text
/mnt/sda/wuwei/minibeam_primary_c12_diag_20260919/
/mnt/sda/wuwei/minibeam_homologous_entry_20260919/
```

## 37. 同源 C12 水输运：归一化、能损账本、空间计分（2026-09-19）

物理参数冻结。未启用电子包，未改 primary MCS。failed.md 中经验电子展宽和
MCS 扫描路线未恢复。

### 归一化

先前 GPU-entry vs TOPAS-entry 现称为**入口替换敏感性**，不是跨引擎同源。
TOPAS 250 `topas_6363` 打开 empty histories（256000，末尾补 225122 空 history），
DoseToMedium Sum 却是 30878-history 运行的 8.28 倍，不能用 survivor/incident
去硬折。可比文件是 `topas_6364`：empty=false，发射 30878，weight=1，
MultipleUse=1。GPU 同入口 30878 粒子总剂量比为 0.996。300 MeV/u 已有 EM-only
同源（34557 粒子，empty=false）；空 history 约定与 250/6363 不同。不把 TOPAS
多线程 EventID 当源行号。

### 跨引擎同源（固定 valley ROI）

| 设置 | 40 mm | Bragg / ±2 mm |
|---|---|---|
| 250 全物理，同 30878 C12 | GPU/TOPAS=0.9867（−1.33%） | 0.9922 / 1.0016 |
| 300 EM-only，同 34557 C12 | 0.9761（−2.39%） | 1.0037 / 1.0061 |

300 EM-only 40 mm valley 的 C12 穿越 4962/4926，均能 1888.6/1895.6 MeV，存活
0.9057/0.9057，角方差一致。轨迹接近而剂量仍偏 −2.4%。250 全链 −3.61% 明显大于
同源 C12+后代 −1.33%，入口替换敏感性只有 −0.30%。

### 能损账本与空间审计

`unified_em_loss` 同一次 draw 的 continuous/delta 在应用 0.9958 前后分别记录；
RNG 和终态不变。250/40 mm valley：delta 占采样能损 6.95%，after_scale/sampled
=0.99580。`local_total_deposit` 已更名；采样 continuous+delta 与缩放后局部沉积
按 1/0.9958 闭合。本配置无显式电子包，delta 留在局部沉积。

空间审计不改输运：peak/shoulder/valley 的直线跨界与 FE 终点跨界计数均为 0，
整步归属没有把能量搬出 valley。逐 voxel 100% 跨步是索引约定问题，不能当成
ROI 计分误差。能区剩余 0.052%/0.077% 的路径清单见
`energy_band_gap_inventory.json`。剂量不确定度仍为 unknown。

判断：不改 MCS、不启用电子包。300 EM-only 轨迹已接近，40 mm valley 剂量残差
更指向能损分区/计分位置。下一步物理 A/B 只能在能损账本或空间归属上选一项。

```text
/mnt/sda/wuwei/minibeam_homologous_c12_20260919/
/mnt/sda/wuwei/minibeam_primary_c12_ledger_20260919/
```

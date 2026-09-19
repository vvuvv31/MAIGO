# Minibeam 精度复核与下一步模型选择（2026-09-18）

结论：Copper 电磁多重散射的角度—位移联合输运和 Copper 能损涨落均已实现为
显式可选模型，并通过低成本 homogeneous slab 验收。当前证据仍不支持继续调整
单一 Highland 宽度系数，也不支持把新核反应模型当作首选。完整 256k 剂量只作
回归和趋势检查，不替代后续三能量高统计最终验收。

## 当前版本与统计量

检查起点为 `47da9c4` 加工作区已有未提交结构修正（圆柱外体、primary optical
depth、移除出口能量回补），这些改动予以保留。正式 250/300 配置的 water tail
width 已是 0.90；`minibeamresult.md` 中三能量 10M 剂量表对应此前版本，不能作为
当前整个工作区的最终认证。

`out/minibeam_structural_cylinder_optical_e{150,250,300}_scale1_256k/`
的最新已有全场结果如下。各比值为 GPU/TOPAS，未拟合归一化：

| MeV/u | 总剂量比 | IDD L1 | 2-D L1 | 入口 PVDR 比 | Bragg PVDR 比 |
|---:|---:|---:|---:|---:|---:|
| 150 | 0.9747 | 2.577% | 18.014% | 1.295 | 1.114 |
| 250 | 0.9845 | 1.627% | 18.594% | 1.148 | 0.960 |
| 300 | 0.9772 | 2.322% | 17.573% | 1.135 | 0.950 |

256k 在百万体素的低剂量区有明显统计噪声；不能将其 18% 原始 2-D L1 直接与旧
10M 的约 4% 比较，也不能将所有峰谷偏差视为噪声。修模应先看低维相空间分布，
再做多 seed、三能量高统计剂量验收。

## 新完成的过程隔离

TOPAS full-list 铜片基准启用了 `CarbonIonElasticPhysics`，GPU slab 基准关闭了它。
为排除这个不同设置的影响，在本机以 Slurm 运行 250 MeV/u、256k histories、
1/10 mm 铜片；唯一物理变化为移除该模块，其他模块、seed、几何均保留。

| 厚度 | Slurm job | CPU / 内存申请 | 完成时间 | 数据目录 |
|---:|---:|---:|---:|---|
| 1 mm | 6432 | 16 / 8 GB | 10 s | `/mnt/sda/wuwei/minibeam_copper_slab_6432` |
| 10 mm | 6433 | 16 / 8 GB | 64 s | `/mnt/sda/wuwei/minibeam_copper_slab_6433` |

两项均为 `COMPLETED`。移除核弹性后，角方差相对 full-list 为 0.9918/0.9936，
径向角 q99.9 为 1.0025/0.9905。这个量级远小于 GPU 的缺口；不能用开关核弹性
来解释或修正当前铜片散射偏差，也不能把这个消融试验当成关闭 TOPAS 物理的建议。

新脚本 `benchmark/carbonminibeam/compare_copper_slab_phase_space.py` 将 TOPAS
计分面在铜片后的 0.01 mm 真空间隙反投影到物理出口，再比较同一平面。旧表中
1 mm 的位置方差比 0.491 因此修正为 0.506，缺口仍然存在。下表对照 full-list：

| 量 | 1 mm GPU/TOPAS | 10 mm GPU/TOPAS |
|---|---:|---:|
| primary 存活率 | 0.99970 | 0.99933 |
| 平均动能 | 1.00006 | 1.00130 |
| 横向位置方差 | 0.5064 | 0.7026 |
| 位置—角度协方差 | 0.5669 | 0.7073 |
| 投影角方差 | 0.7302 | 0.7237 |
| 径向角 q68 | 0.9101 | 0.8741 |
| 径向角 q99.9 | 0.5381 | 0.6879 |

能谱宽度另有很大差距：1 mm 的 GPU/TOPAS 标准差分别为
**0.000340/1.708 MeV**，10 mm 为 **0.05369/6.001 MeV**。平均能量接近不代表
能损分布正确。能谱涨落对最终 PVDR 的贡献尚未分离，不能根据这一结果宣称它是
主要剂量误差来源。

输出含可复算 JSON 与独立 PNG，位于
`out/minibeam_accuracy_audit_20260918/slab_{1,10}mm/`。

## kernel 与目标模型差异

- `src/transport_sycl.cpp` 的 Copper primary 先沿原方向走完整步，再调用逐步
  Highland Gaussian 改变方向，没有同步采样相关位移，也没有非高斯尾。
  1 mm 铜片只有四个 0.25 mm 步，末端 kick 的位置偏差特别明显。
- `src/detail/sycl_device_math.inc` 中 Highland 对数项依赖当前步长。在固定
  250 MeV/u 下，仅把步长从 0.25 改为 0.05 mm，就会把单位长度角方差乘以
  0.8836。缩短步长同时改变模型本身，因此原样重试不是有效的收敛验证。
- Copper 碎片使用累计段 Highland 方差增量，和 primary 不同；该段没有检查
  `minibeam_copper_enable_mcs`。这个开关目前不能解释为关闭所有 Copper MCS。
- `minibeam_copper_enable_energy_straggling` / `minibeam_copper_straggling_scale`
  仅在配置声明、解析和校验中出现，当前 SYCL 输运没有使用它们。直接改 YAML
  无法补上铜能损涨落。历史 Bohr 试验没有改善低统计尖峰，见
  `docs/archive/minibeam.md` §17.3；若重新实现，应以独立 slab 能谱为验收量。
- primary 的 Copper 能损是步首 stopping 的 Euler 积分。10 mm 出口均值比 TOPAS
  高 2.672 MeV，而高精度 CSDA 表积分已与 TOPAS 均值接近。应验证中点/预测校正
  积分并限制步内相对能损，不应重新缩放正确的 stopping 表。
- Copper 内碎片再反应目前是抽样后将 child 能量置零，没有产生该反应的下一代
  带电产物；水中 `generation=2` 不能自动补上这段 Copper cascade。

TOPAS 的实际运行日志明确给出 **GenericIon → UrbanMsc、DispFlag:1**，
`ionIoni → LindhardSorensen、fluct:1`。不能仅根据 `opt4` 的名字推断碳离子使用
WentzelVI；此运行中 WentzelVI 出现在 proton 等物种上。

## 建议实施顺序

1. **替换 Copper MCS kernel。** 以当前 Geant4 11.3.2 的 Urban 离子散射作为
   对齐目标：提取材料/能量相关输运参数，移植或构建经验证的 core + 非高斯尾
   采样器，联合采样步内位移和角度，并正确处理 true path、几何边界与剩余步。
   Fermi–Eyges 可用于小角 core 的联合协方差，不能单独代替整个尾部模型。
   先验证 150/250/300 MeV/u、1/10 mm，保留额外能量/厚度作为未参与标定的测试。
   再做 slit-edge、water-entry 联合相空间；所有数据只能约束输运，不能用最终
   dose/PVDR 反推自由参数。新模型成立后才重启 0.25/0.10/0.05 mm 步长收敛。
2. **补 Copper 能损分布与积分。** 分开验证确定性积分误差和随机涨落。以 slab
   能量均值、宽度及分位数约束；Bohr 只能作为厚层方差基准，靠近边界或低能时
   需要验证受限损失/显式 delta 分配是否适用。primary 与碎片应使用一致接口。
3. **重新隔离水中误差。** Copper 修复后，用同一份 TOPAS 水入口相空间回放，
   重新验证当前低能 MCS scale=0.20 和 tail=0.90。已有证据显示部分经验修正在
   补偿上游误差；不能同时调整上下游参数，也不能把 250/300 的 tail 推给 150。
4. **按 origin 验证核反应和电子输运。** Copper cascade 的近似需要完善，但先
   用 primary/fragment 分量确定优先级。电子非局域剂量需要能谱和响应约束，
   不再试固定 Gaussian 剂量模糊。暂不整体切换 INCL++，以免同时引入更多变量。

最终必须报告三能量多 seed 的绝对剂量、IDD、分深度 peak/valley/PVDR、射程和
分物种 distal dose，并通过 energy ledger、Unified-EM domain、quality 与 queue
overflow 检查。GPU 开发固定 FP32 scorer 和 sm_75；下述实现与运行均遵守该约束。

## 已实现的 Copper kernel 后续

新增 `minibeam_copper_mcs_model: fermi_eyges_tail`，同时保留默认的 legacy
`highland` 回退路径。新模型使用无步长对数项的局部 scattering power，联合采样
Fermi--Eyges core 的角度与横向位移；宽尾由按真实路径长度发生的 Poisson event
产生，event 位置在步内采样，角度采用窄/宽两分量。参数只由独立 Copper slab 的
A0/A1/A2 和 q68--q99.9 约束，没有使用最终 dose/PVDR。

250 MeV/u、256k histories 的 GPU/TOPAS 结果如下：

| 厚度 | A0 | A1 | A2 | q68 | q95 | q99 | q99.9 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 mm | 1.087 | 1.065 | 1.023 | 1.019 | 1.012 | 1.015 | 0.997 |
| 10 mm | 1.019 | 1.013 | 1.003 | 1.009 | 1.010 | 1.016 | 1.000 |

未参与 250 MeV/u tail 参数约束的 150/300 MeV/u、1 mm A2 比为
`0.981/1.037`，q68--q99 均在 2.6% 内，q99.9 为 `0.905/1.009`；300 MeV/u、
10 mm 的 A0/A1/A2 为 `1.040/1.036/1.028`，q99.9 为 `1.019`。将最大 Copper
步长从 0.25 mm 缩到 0.05 mm 后，250 MeV/u、10 mm 的散射量变化仅
0.3%--1.3%，满足原失败记录的重试条件。

`minibeam_copper_enable_energy_straggling` 现在实际接入 primary Copper kernel。
其方差采用 Copper `Z/A` 的 condensed-loss variance，scale=1；确定性能损同时从
步首 Euler 改为中点预测校正。250 MeV/u 的出口能谱为：

| 厚度 | GPU mean/std (MeV) | TOPAS mean/std (MeV) |
|---:|---:|---:|
| 1 mm | 2913.873 / 1.744 | 2913.891 / 1.708 |
| 10 mm | 2051.947 / 6.142 | 2052.384 / 6.001 |

能谱宽度误差从约 99% 降到 2.1%--2.4%，没有新增拟合 scale。当前实现尚未把同一
涨落接口推广到 Copper fragment，这仍是后续项。

三能量 256k full-field 仅作为回归。相对结构修正前基线，总剂量与 IDD 整体改善；
启用 MCS+straggling 后 250/300 MeV/u Bragg PVDR 为 `0.963/0.985`。150 MeV/u
低统计 Bragg PVDR 为 `1.169`，需要高统计复核，不能据此重新调 Copper 参数。
全部运行使用 FP32 scorer，Unified-EM missing=0、queue overflow=0，能量残差为
`1.6e-5--2.7e-5`。

## 复现

单能量单厚度的过程隔离提交方式（逗号分隔的厚度列表应通过 shell 环境传入，
不要放进 Slurm `--export` 的逗号分隔赋值串）：

```bash
SLAB_ENERGIES_CSV=250 SLAB_THICKNESSES_CSV=1,10 SLAB_DISABLE_ION_ELASTIC=1 \
  sbatch --cpus-per-task=16 --mem=8G --export=ALL \
  benchmark/carbonminibeam/run_copper_slab_matrix.sbatch

python3 benchmark/carbonminibeam/compare_copper_slab_phase_space.py \
  --topas-full /mnt/sda/wuwei/minibeam_copper_slab_6421/e250_t1mm/output/slab_exit.phsp \
  --topas-no-elastic /mnt/sda/wuwei/minibeam_copper_slab_6432/e250_t1mm/output/slab_exit.phsp \
  --gpu out/minibeam_copper_slab_gpu_e250_t1_256k/phase.csv \
  --thickness-mm 1 --output-dir out/minibeam_accuracy_audit_20260918/slab_1mm
```
## Fragment consistency follow-up

The production primary and fragment MCS scales have been decoupled.  A new
opt-in charged-fragment water-entry phase space and a terminal Copper
reinteraction audit were added.  After back-projecting GPU records to the
TOPAS scorer and applying its ±70 mm transverse extent, the 250 MeV/u,
256k-history comparison is 46,996 GPU fragments versus 57,685 TOPAS charged
ions after excluding parent-0 C-12.  The corrected proton ratio is 0.696.
Aggregate exit-spectrum widths cannot isolate transport straggling because
birth spectra, paths, and survival selection are mixed.  The GPU terminally
absorbs 46,764 fragment reinteractions, but the old 35--59 mm audit was only
distance to the downstream plane and is withdrawn; it cannot quantify how
many tertiary products would reach water.  The replacement package must
follow the reference physics list: Binary Cascade for proton and INCL++ for
d/t/He/GenericIon in the relevant energy ranges.  Fixed-species slab tests
remain required before enabling fragment straggling.

The corrected geometry audit now subtracts finite-slit air and cylindrical
side escape from a straight-ahead ray. Mean post-interaction Copper lengths
are 28.9--41.1 mm across populated species. This remains a no-future-MCS
diagnostic and is not a survival prediction. Local Slurm job 6436 independently
validated proton+Cu extraction against the reference physics list: all 37,179
captured interactions identify `Binary Cascade`. Its package and the existing
C-12 `INCL++ undefined` events were successfully combined and reread as a
two-projectile CINPKG04. Runtime replay remains disabled until child collision
position and tertiary Copper transport are implemented.

Fragment inelastic sampling has additionally been changed from a per-step
Bernoulli test at the step end to persistent optical depth with EM/MCS
transport truncated at the sampled collision point. The 250 MeV/u matched
scorer count changes only from 46,996 to 47,206, while 46,648 terminal
interactions remain. This establishes the collision-position interface but
does not substitute for replaying the captured final states.

## Fragment+Cu final-state replay pilot

The former terminal approximation is now an explicit generation-zero mode.
An opt-in one-generation path performs strict `(Z,A,Cu,E)` lookup at the true
collision point, preserves the captured correlated final state, transports
charged tertiary products through the remaining collimator, and audits all
lookup failures and queue overflow.  Proton packages are compiled only from
Binary Cascade records; d/t/He/GenericIon packages accept only INCL++ records.

The first broad pilot exposed a useful validation failure: 32,621/46,648
queries were below the p/d/t package domains.  No endpoint clamp or nearest
fallback was introduced.  A low-energy extraction matrix reduced strict
misses to 1,011 (2.17%): 118 missing-projectile, 165 below-domain, and 728
energy-gap misses.  The corrected ±70 mm scorer result is 56,467 GPU versus
57,685 TOPAS fragments (0.979), compared with 46,996 (0.815) for terminal
absorption.  Queue overflow and Unified-EM missing are zero and the physical
energy residual remains 1.72e-5.  The 256k dose trend improves IDD and lateral
integrals but not 2-D L1 or Bragg PVDR, so this is accepted as a structural
transport correction rather than a completed dose match.

The original dose A/B is withdrawn because Copper generation 1 was reused as
water generation 1 and charged products born at the configured water cap were
locally deposited.  The corrected implementation resets Copper survivors to
water generation 0 and always gives above-cutoff charged terminal products EM
transport.  With both baseline and candidate rerun, terminal/cascade IDD L1 is
0.934/0.619%, lateral-integral L1 is 5.74/5.41%, 2-D L1 is 18.323/18.290%,
and Bragg PVDR is 0.879/0.877.

The remaining-Copper one-generation cap is not negligible: summed ignored
optical depth is about 4517, mainly p/d/t (3031/1074/222).  The corresponding
per-track expected-reaction sum is about 4134. Generation 2 reduces that sum
to about 136, and generation 3 to 1.94. The same-seed 256k generation-2/3 dose
results are converged at the reported precision: total-dose ratios are
1.002109/1.002090, IDD L1 0.5808/0.5807%, 2-D L1 18.28669/18.28668%, and Bragg
PVDR ratio 0.876834 for both. Generation 3 is therefore the diagnostic
convergence point; formal multi-energy configs remain disabled until broader
validation. Per-generation interactions/hits/misses are
46648/45637/1011, 4069/4000/69, and 108/108/0. Per-species and 25 MeV/u miss histograms and event-level
actual/selected/output kinetic-energy ledgers are emitted.
The corresponding rest-mass-aware closure audit is 2,568 MeV summed over
45,637 hits (0.056 MeV/hit) with zero baryon-number mismatches; unlike the
2.03e6 MeV kinetic-only deficit it explicitly includes the Cu target and
recorded residual/product masses.
The Copper cascade overflow counter is also part of the unified quality gate.
## Same-source charged-ion water-entry replay update

The replay contract now keeps parent-0 C12 on the primary transport path and
injects every Copper fragment, including non-parent C12, into the secondary
path at water generation zero.  The TOPAS plane at world Y=59.980 mm is
propagated to the common Y=60.000 mm boundary before coordinate conversion.
For the 250 MeV/u reference this yields 30,878 primary C12 and 57,685 charged
fragments, including four fragment C12.

The first FP32 RTX 2080 Ti fragment-only runs passed queue and energy checks.
With water nuclear transport enabled the secondary queue grew from 57,685 to
114,162 and the relative energy residual was 1.28e-5.  With
`enable_inelastic=false`, the queue remained at 57,685, nuclear inelastic and
elastic counts were both zero, and the residual was 3.12e-7.  This required
decoupling secondary buffer/launch and ion-stopping-table allocation from the
nuclear switch; nuclear hazards are now explicitly guarded.

This is infrastructure validation, not a new dose-match claim.  GPU
export-to-replay closure and a TOPAS replay of the identical charged-ion set
remain required before attributing the plateau/Bragg PVDR residual to water or
to Copper/slit-edge transport.

## Full-chain pure-EM isolation with the original field

The requested high-statistics isolation was run as the original 256-spot,
3 cm x 3 cm PBS source with 4,000 histories per spot (1,024,000 total), not as
a duplicated water-entry phase space. Both engines retained Copper, the air
gap, and water. TOPAS loaded only `g4em-standard_opt4` and `g4decay`; GPU
disabled Copper nuclear attenuation and water inelastic transport.

TOPAS Slurm job 6441 used 192 threads and completed in 307.157 s. Jobs
6439/6440 were source-setup failures before event processing. The matching
FP32 RTX 2080 Ti run completed in 9.504 s at 107,745 histories/s, with no
nuclear events or queue overflow and relative energy residual `1.63e-5`.

With absolute incident-history normalization and no fitted dose scale:

| Metric | Result |
|---|---:|
| dose sum GPU/TOPAS | 1.007995 |
| 2-D Pearson / L1 | 0.995510 / 8.8566% |
| IDD Pearson / L1 | 0.999968 / 0.9257% |
| lateral-integral Pearson / L1 | 0.999281 / 3.2545% |
| Bragg depth TOPAS / GPU | 124.375 / 124.625 mm |

Entrance peak/valley/PVDR ratios are `1.037/1.030/1.006`, so the initial
periodic contrast is already close in this observable. At 79.875 and
99.875 mm the valley ratios rise to `1.080/1.107`, while PVDR falls to
`0.960/0.933`. At the TOPAS Bragg depth, peak/valley/PVDR ratios are
`1.062/1.097/0.968`. The very-low-dose distal point is not used to assign a
model discrepancy at this sample size.

This isolates an important remaining error: removing all nuclear transport
does not remove the depth-dependent lateral broadening mismatch. The excellent
IDD agreement but materially larger 2-D and lateral error, especially the
80--100 mm valley excess, points to water EM/MCS position-angle evolution as
the next model target. It does not establish that Copper/slit-edge transport
is perfect, and it does not justify empirical PVDR retuning.

Inputs are `benchmark/carbonminibeam/run_field3cm_em_only_e250_1024k.txt` and
its sbatch wrapper. Raw outputs and comparison plots are under
`/mnt/sda/wuwei/minibeam_field3cm_em_only_e250_1024k/`.

### 12.8M-history update

The same original 256-spot source was rerun at 50,000 histories per spot,
12.8M total. TOPAS job 6442 completed on 192 threads in 3822.14 s, and the
matching FP32 RTX 2080 Ti run completed in 55.80 s. No nuclear events or queue
overflow occurred; GPU relative energy residual was `1.62e-5`.

Higher statistics reduce 2-D L1 from `8.8566%` to `2.8091%`, IDD L1 from
`0.9257%` to `0.4887%`, and lateral-integral L1 from `3.2545%` to `1.1785%`.
Total dose is GPU/TOPAS `1.003424`. The local conclusion is sharper than at
1.024M: GPU/TOPAS peak remains within about 1.5% from 20 mm through the Bragg
region, but valley dose rises from `0.950` at entrance to `1.109` at 80 mm and
`1.167` at 100 mm before returning to `1.033` at 124.375 mm. Corresponding
PVDR ratios are `1.090`, `0.907`, `0.853`, and `0.972`.

The sign-changing, depth-dependent valley residual survives complete removal
of nuclear transport and cannot be explained by a global dose normalization.
It is now a statistically resolved water EM/MCS phase-space-evolution target.
The 12.8M results and the dedicated `dose_ratios_vs_depth.png` are under
`/mnt/sda/wuwei/minibeam_field3cm_em_only_e250_12800k/`.

## Same-source 12.8M water replay diagnosis

One full-chain pure-EM run collected the parent-0 C12 water-entry phase space,
total dose, and electron/non-electron carrier dose. The resulting 1,622,795
C12 states were then replayed as the identical water source in both engines.
TOPAS job 6529 recorded 40/60/80/100/120 mm phase-space planes in the same
run; no additional TOPAS jobs are required for this diagnosis.

The water-only replay has GPU/TOPAS total dose `1.000019`, IDD L1 `0.1719%`,
2-D L1 `2.5097%`, and identical `124.625 mm` Bragg-bin depth. Fixed-region
valley dose ratios at 40/60/80/100/120/124.375 mm are
`1.001/1.016/1.028/1.039/1.022/1.013`. This reproduces the full-chain
mid-depth residual from a common entrance and assigns its dominant source to
water transport/scoring rather than Copper/slit-edge input.

The electron-carrier partition closes to L1 `2.73e-8` and contributes 7.30%
of total dose. It is more peak-concentrated than total dose, so omitted
explicit GPU electron transport does not explain a GPU valley excess. The
identity-free primary-C12 plane comparison instead shows a shape error in the
water angular kernel: at 100 mm GPU/TOPAS `q68/q95/q99/q99.9` is
`0.937/1.032/1.230/1.270`, while fixed-valley C12 fluence is `1.046`.
The core is too narrow while the tail is too broad; this is consistent with
the present synthetic tail reserving 40.5% of nominal angular variance and
compensating by narrowing the Gaussian core.

The original phase-space analysis assumed that TOPAS MT replay EventID was a
stable entrance-row index. It is not: records become shuffled across worker
blocks. Entrance-conditioned groups and identity-matched moments from that
analysis are invalid and are no longer emitted. The corrected analyzer only
uses aggregate current-plane observables. A future omnibus original-source
run must score entrance and downstream planes together if true per-track
Fermi--Eyges moments are needed. Stopping calibration remains frozen; the
next model target is a step-stable water scattering power with separately
validated core, tail process, and displacement-angle correlation.

### Interval-matched optional water kernel

Downstream TOPAS planes can be joined by `(RunID,EventID,TrackID)` even though
EventID cannot be mapped to an entrance-file row. The corrected analyzer now
reports adjacent-plane angle increments, displacement relative to the upstream
ray, their covariance, energy-group survival, full quantile curves, and an
`S(E)/|u_z|` crossing proxy. Fixed regions use the same `|x|<=18 mm` field as
dose; corrected legacy 80/100 mm valley-fluence ratios are `1.0486/1.0497`.

The optional `minibeam_water_primary_mcs_model: fermi_eyges_tail` replaces,
rather than augments, the legacy Highland/low-energy/tail combination for C12
in native water. It uses local scattering power, correlated displacement, and
an untruncated path-Poisson tail with sampled event locations. Parameters were
constrained from 40--60 mm energy groups only. On 60--120 mm propagation checks,
angle variance, displacement variance, and covariance remain within about
3.5%, 1.2%, and 1.5%; q68/q95 are within 2%, while q99.9 remains 5--10% low.
These intervals share tracks with the fit interval and are not statistically
independent holdout samples. It is therefore a promising candidate, not a
unique or final tail model.

Same-source fixed valley dose ratios at 40/60/80/100/120/124.375 mm improve to
`0.993/0.996/0.998/1.009/1.012/1.012`; 2-D and lateral-integral L1 improve to
`2.2834%` and `0.6793%` without changing total dose, IDD, or Bragg depth.
Internal scattering segments of 0.10/0.05 mm agree in robust interval metrics,
and disabling plane scoring changes FP32 dose by only `6.01e-8` relative L1.
The legacy model remains default pending multi-energy and full-chain nuclear
validation.

### Engineering and diagnostic closure

The first plane implementation linearly interpolated the full-step direction
and displacement. That is not a valid state of the stochastic process: at path
fraction `f` it produces an angular variance proportional to `f^2` rather than
`f`, and it moves Poisson tail events upstream of their sampled locations. The
candidate scorer now samples the integrated-Brownian conditional bridge given
the unchanged endpoint. A tail event contributes to the plane state only when
its location precedes the crossing. The endpoint and transport RNG stream are
unchanged by enabling the diagnostic.

The minibeam ON and OFF CUDA SYCL configurations both build. An independent
500k-sample, 160 MeV/u, 20 mm reference sampler reproduces the analytic angle
variance, displacement-angle covariance, and displacement variance within
0.2%; 200 x 0.1 mm steps agree with one 20 mm step within 0.4%. Poisson count,
uniform event location, half-step bridge, and inclined-direction covariance
checks also pass. This validates the sampler structure, not the fitted physical
constants or a complete outer transport-step convergence.

A 64-replicate, 256-block stable-track bootstrap was added to the existing
TOPAS/GPU interval analyzer. The aggregate q99.9 deficit is statistically
resolved, but low-energy groups have much wider uncertainty. For example,
60--80 mm and 80--100 MeV/u has only about 32k paired tracks; its TOPAS/GPU
q99.9 95% intervals are `[58.24,72.74]` and `[51.58,61.91]` mrad. The later
intervals reuse tracks and are correlated propagation checks, not independent
holdout samples. Parameters remain frozen; hashes and exact outputs are listed
in `docs/minibeam_water_mcs_candidate_20260919.md`.

Frozen-parameter 256k full-chain screening gives only a small 250 MeV/u gain
and no consistent cross-energy improvement: candidate versus legacy 2-D L1 is
`17.923/17.672%`, `18.009/18.197%`, and `17.227/17.224%` for
150/250/300 MeV/u. This is a single-seed screening result, but it is sufficient
to keep the model optional and avoid dose-driven retuning. The formal energy
configs remained on `legacy_highland` at the time of this audit. The later
same-source and three-energy 10M acceptance in `minibeamresult.md` sections
21--23 supersedes this status: formal configs now use the frozen water
`fermi_eyges_tail` model and per-energy Copper cascade packages.

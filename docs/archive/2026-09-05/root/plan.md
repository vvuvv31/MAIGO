# 结论

这不是一个通过“统一缩放 inelastic 截面”“调整 MCS scale”或“给某类碎片乘权重”就能解决的问题。当前偏差呈明显反向：

* C、B、Be、Li 高出约 44%–63%；
* proton 少约 55%；
* Primary C 仍高 12%；
* He 积分接近，但形状不对。

这说明至少有多个误差源同时存在。公开代码中已经能确认若干**确定性的实现错误**，其中最严重的是：

> **次级粒子的 CINEL02 反应回放被放进了 3D voxel scorer 的换体素分支，导致核反应是否发生依赖 scorer 是否启用以及 voxel 大小。**

因此，你使用双方相同的 200×200×800 三维 DoseToMedium scorer，虽然排除了 1D scorer 归一化问题，但当前 GPU 代码反而可能让 **3D scorer 网格本身改变物理过程**。在修复这些实现问题前继续调物理参数，会把程序错误吸收到“拟合参数”里。

另外，当前 YAML 中 CINEL02 package 和 rate CSV 指向仓库外部绝对路径，所以我无法从公开仓库核查实际事件包内容、能量节点和统计量；下面会把这部分列成必须在本地执行的检查。([GitHub][1])

---

# 一、当前代码里优先修复的问题

## P0-1：次级 inelastic 回放错误地依赖 voxel scorer

在 `src/transport_sycl.cpp` 的次级输运循环中，secondary CINEL02 replay 位于类似下面的控制结构中：

```cpp
if (enable_voxel_scoring) {
    if (cur_voxel != pending_sec_voxel) {
        // secondary CINEL02 replay
    }
}
```

这会产生三个后果：

1. 关闭 voxel scorer 后，次级核反应可能完全不回放；
2. 开启 scorer 后，只在 track 进入新 voxel 的步骤才有机会回放；
3. 改变 scorer 分辨率会改变次级反应次数。

当前配置中最大步长是 0.1 mm，而 z voxel 是 0.5 mm。对近轴前向粒子，数量级上可能只有大约每 5 个步骤中的 1 个步骤进入换体素分支。这个 1/5 是基于当前步长和网格的估算，不是所有轨迹的精确比例，但足以造成严重的次级级联抑制。首次次级 track 的 `pending_sec_voxel` 还是无效值，因此 reaction local deposit 在深度总计、物种 3D map 等不同 scorer 间还可能出现归属不一致。([GitHub][2])

**修复原则：**

* 物理推进、反应采样和事件回放必须完全独立于 scorer；
* scorer 只能读取已经产生的 step/event，不得控制是否产生 event；
* 添加回归测试：scorer 开/关、voxel 0.4/0.8/1.0 mm 时，反应计数和 birth spectra 必须保持一致。

这很可能是 proton 不足的一个来源，因为大量 proton 可能来自碎片的后续反应；但修复后也可能同时增加高代重碎片，所以不能只看一次总 IDD 是否“变好”。

---

## P0-2：次级碰撞使用了错误的能量、位置和方向时序

当前次级路径大致是：

1. 用 step 起点能量 `sec_e` 查找 CINEL02 event；
2. 再计算该步电磁能损 `dE`；
3. 在旧位置生成产物或更新 parent；
4. 然后再执行 `sec_e -= dE`；
5. 可能使用 event 更新后的方向推进原本属于碰撞前的位移。

这会混用：

* step 起点状态；
* 电磁 AlongStep 后的碰撞输入状态；
* nuclear PostStep 后的 parent 状态。

TOPAS/Geant4 事件采集的正确语义是：事件输入应是 `AlongStep` 已完成、进入 hadronic `PostStepDoIt` 时的 track 状态；事件输出是 `G4VParticleChange` 返回的 parent 和 products。([GitHub][3])

建议重构成统一的状态机：

```cpp
pre = current_track_state;

// 1. 采样本步反应距离；决定实际推进距离
ds = min(transport_step, sampled_collision_distance);

// 2. 只做碰撞前电磁推进
post_em = advance_em(pre, ds);
// post_em 包含：碰撞点位置、碰撞前方向、碰撞前剩余能量、连续 dE

score_continuous_em(pre, post_em);

if (!collision_at_step_end) {
    state = post_em;
    continue;
}

// 3. 必须用碰撞输入态能量查 event
event = lookup(projectile, target, post_em.energy / A);
require(event.valid);

// 4. reaction local deposit 位于碰撞点
score_reaction_local(
    post_em.position,
    event.process_local_deposit
);

// 5. 所有局部方向都相对于碰撞前入射方向旋转
spawn_products(
    event.products,
    post_em.position,
    post_em.direction
);

// 6. parent continuation 完全由事件包的最终态决定
if (event.parent_alive) {
    state = make_parent_final_state(event, post_em);
} else {
    terminate_track();
}

// 此处绝不能再次执行 energy -= dE
// 也不能用 post-event 方向重新推进碰撞前的 ds
```

Primary 和 secondary 最好共用同一个 `advance_to_collision_state()` 和 `replay_cinel02_event()`，避免两条路径继续漂移。

---

## P0-3：DoseToMedium 中把 NIEL 重复加了一次

TOPAS 写出端分别保存了：

```cpp
process_local_deposit_MeV = change.GetLocalEnergyDeposit();
nonionizing_deposit_MeV   = change.GetNonIonizingEnergyDeposit();
```

GPU 回放端当前却使用：

```cpp
local_deposit + nonionizing_deposit
```

Geant4 的定义中，non-ionizing energy deposit 是 local energy deposit 的一个子量，不是额外独立的一份能量；`G4Step` 会把 local deposit 加入 total deposit，同时把 NIEL 单独保存作分量信息。因此 DoseToMedium 应只增加 `process_local_deposit_MeV`，NIEL 只保留为诊断分量。([GitHub][4])

正确实现：

```cpp
dose_to_medium += event.process_local_deposit_MeV;

// 仅用于诊断/分量输出
diagnostic_niel += event.nonionizing_deposit_MeV;
```

建议加入一个最简单的单元测试：

```text
event.local = 10 MeV
event.NIEL  = 3 MeV

DoseToMedium 增量必须是 10 MeV，而不是 13 MeV。
```

这个修复预期会降低反应相关类别的剂量，尤其是重碎片附近的局部沉积，但它不太可能单独解释 proton 少 55%。

---

## P0-4：级联深度硬编码为 15，与设计目标不一致

当前次级反应判断使用了类似：

```cpp
frag.generation < 15U
```

而仓库方法说明和设计文档把生产级联深度设为约 2，并要求 secondary queue overflow 为零。硬编码 15 会允许很深的非物理或未验证级联。([GitHub][3])

建议增加明确配置：

```yaml
cinel02:
  max_secondary_inelastic_generations: 0
```

定义为：

* `0`：只有 primary C 可以发生 inelastic；所有直接产物只做 EM 输运；
* `1`：直接产物可以再发生一次 inelastic；
* `2`：允许到下一代，作为最终生产候选；
* 不允许生产配置继续使用 15。

第一轮诊断必须用 `0`。因为当前 scorer gating 抑制了级联，而硬编码 15 又可能增加级联，两者可能在总剂量上偶然抵消。

---

## P0-5：事件 lookup miss 被静默吞掉

`cinel02_find_event_device()` 找不到能量范围内的 event 时会返回无效结果，而 primary/secondary 路径目前没有严格报错；该次已经采样到的 inelastic collision 可能退化成一次普通 EM continuation。

运行时还使用约 ±0.51 MeV/u 的能量容差，把范围内多个节点的 events 合并后均匀抽取。若相邻能量节点 event 数量不同，实际插值权重就会受每个节点采样统计量影响，而不是由明确的能量插值规则决定。([GitHub][5])

必须增加以下计数，并在 strict-match 模式中 fail-fast：

```text
rate_covered
collision_sampled
event_lookup_hit
event_lookup_miss
invalid_event_bounds
invalid_product_bounds
parent_status_invalid
```

维度至少包括：

```text
projectile Z/A × target Z/A × collision energy bin × generation
```

生产匹配要求：

```text
event_lookup_miss == 0
invalid_event_bounds == 0
invalid_product_bounds == 0
```

能量节点选择也应改成明确策略：

* 最近能量节点；或
* 两节点按能量线性混合；

不要让“某节点采了更多 events”隐式成为插值权重。

---

## P0-6：次级能量账本不完整

Primary 路径会把 unsupported、neutral、queue overflow 等未输运能量累计到 `untracked_MeV`；secondary 路径对部分 unsupported/neutral 产品只是跳过，对 overflow 也主要记录数量，没有完全记录对应动能。结果是能量可能静默消失，而且无法判断 proton 缺失来自 birth 还是 transport。([GitHub][3])

每个 reaction event 应记录：

```text
collision_input_kinetic_energy
pre_collision_EM_loss
process_local_deposit
NIEL_diagnostic
surviving_parent_kinetic_energy
charged_product_kinetic_energy
neutral_product_kinetic_energy
unsupported_product_kinetic_energy
queue_overflow_kinetic_energy
package_Q_or_closure_residual
```

注意不能简单要求“所有 kinetic energy 相加严格等于输入 kinetic energy”，因为核反应涉及质量差/Q value。应使用事件包定义的 closure residual，并确保 GPU 解码后与包中账本一致。

---

## P0-7：事件包支持的核素域与 GPU 输运域不一致

TOPAS writer 能识别的离子核素范围明显比 GPU stopping-power mapper 更广；GPU 当前的显式粒子表只有大约 17 个核素。未支持核素会被 alias 到同元素默认核素，而 `Z>6` 的情况甚至可能落到索引 0，即 proton stopping-power table。若 package 中出现 N、O 或未显式支持的 C/B/Be/Li 同位素，这会严重改变射程和剂量。([GitHub][4])

必须把 silent alias 改成 strict policy：

```cpp
if (!exact_transport_species_supported(Z, A)) {
    record_unsupported_energy(...);

    if (strict_match_mode)
        abort_with_species_report(Z, A);
}
```

然后二选一：

1. 为 package 中有显著产额/能量的全部 charged isotope 加载准确的 stopping power、MCS 和 straggling；
2. 明确将其设为不输运，并把能量记入 unsupported ledger。

为了达到逐点 1%，不能把重要同位素静默映射到“附近核素”，更不能把 Z>6 映射成 proton。

---

## P1：次级 straggling 配置名与实际行为不一致

虽然配置启用了 secondary energy straggling，但当前代码看起来只对 `Z=6, A=12` 的次级 C12 应用了 packaged fluctuation，p、He、Li、Be、B 仍是确定性能损。停止功率文件路径也被硬编码到特定 Geant4 版本数据。([GitHub][1])

这更可能造成 He 的 NRMSE 9.68%、峰前形状和 range straggling 不匹配，不足以单独解释 40%–60% 的积分偏差。应放在前述 P0 修复之后，通过 mono-ion benchmark 处理。

---

# 二、逐步修复与验证路线

## 阶段 0：冻结一个完全可复现的基线

先不要改参数，建立 validation bundle：

```text
MAIGO commit SHA
编译器与版本
SYCL backend/device
编译 flags
随机种子和 RNG 方案
完整 YAML
TOPAS macro
TOPAS/Geant4 版本
physics list
production cuts
step limits
CINEL02 package
rate CSV
metadata/contract
所有文件 SHA-256
```

当前 package/rate 使用仓库外绝对路径，因此必须确认二者来自同一 capture campaign，而不是“新 package + 旧 rate”。配置和事件包 manifest 至少应核对：

```text
campaign UUID
capture git SHA / executable hash
TOPAS/Geant4 version
physics list
cuts and step limit
primary_only
projectile/target set
energy node set
units
event count per cell
product weights
local-deposit definition
```

仓库已有 package validation、unsupported audit 和 compile 工具，可作为基础，但需要再增加 runtime isotope compatibility audit。([GitHub][4])

同时保留当前 100k 结果和所有 MHD/RAW checksum，作为 `B0`。

---

## 阶段 1：先加诊断，不改变物理结果

对每个 history/反应建立统计，完整 histogram 全量保存，逐事件 trace 只保存固定抽样：

```text
history_id / track_id / parent_track_id
generation
projectile Z/A
target Z/A
E_step_start
E_collision
z_collision
sampled_rate
event_energy_node
event_index
lookup status
parent alive/killed
products by Z/A
birth kinetic energy and angle
local deposit
NIEL
unsupported/neutral/overflow energy
track exit energy
```

另外输出以下 z 分布：

```text
Primary C survival fluence
first-inelastic depth CDF
reaction count by target H/O
birth multiplicity by Z/A and generation
birth kinetic-energy spectrum
dose by species and generation
lateral escape energy
```

还要明确 particle scorer 分类契约：

* Primary C 是“原始 beam track”，还是所有未改变的 C12？
* Secondary C 是否包括 parent continuation？
* `Z=1` 是仅 proton，还是 p+d+t？
* reaction local deposit 归属碰撞前 parent，还是单独的 nuclear-local 类别？
* 各 species map 求和是否与 total map 一致？

相同的 3D grid 只能保证空间网格一致，不能自动保证这些类别语义一致。

---

## 阶段 2：按独立 commit 修复确定性错误

建议顺序：

```text
1. diag/cinel02-strict-counters-and-energy-ledger
2. feat/cinel02-configurable-cascade-depth
3. fix/cinel02-secondary-replay-independent-of-scorer
4. fix/cinel02-secondary-post-em-collision-state
5. fix/cinel02-dose-local-not-local-plus-niel
6. fix/cinel02-strict-isotope-transport-domain
7. test/cinel02-scorer-invariance
8. test/cinel02-deterministic-event-replay
```

每个 commit 都用相同输入和固定 seed 跑 100k，记录相对上一 commit 的：

```text
反应次数
birth multiplicity
各代产物
积分剂量
IDD NRMSE
peak position/dose
energy ledger
```

不要以“单个修复是否让最终 IDD 更好”作为正确性标准。例如：

* 去掉 NIEL 重复计数应降低 reaction-local dose；
* scorer 解耦会增加此前漏掉的 secondary reaction；
* 把 cascade 从 15 限到 0/2 又会降低高代产物。

这些修复可能方向相反，但必须分别正确。

---

## 阶段 3：固定 event index 的无输运回放测试

增加一个测试接口：

```cpp
replay_cinel02_event(
    projectile,
    target,
    energy_node,
    fixed_event_index
)
```

暂时绕过 rate 和 RNG，不做后续 transport。逐字段比较 CPU package decoder 和 GPU device decoder：

```text
parent alive/status
parent final kinetic energy
parent local direction
product count
每个 product 的 PDG/Z/A
product kinetic energy
product local direction
weight
process local deposit
NIEL
```

验收：

```text
固定事件：离散字段完全一致
浮点字段：只允许预先定义的序列化精度误差
无 invalid bounds
无 product 丢失
```

随后对每个 `projectile × target × energy cell` 做 10⁵–10⁶ 次抽样，比较：

```text
multiplicity distribution
parent survival probability
mean product energy
angular moments
mean local deposit
event index frequency
```

这是验证“TOPAS 数据是否被 GPU 原样回放”的最直接步骤。

---

## 阶段 4：只验证 primary reaction rate 和 parent survival

这一阶段：

```text
secondary inelastic = off
product transport = off
reaction local dose = off
仅记录 primary 第一次反应
```

比较 TOPAS 与 GPU：

```text
first-reaction depth CDF
Primary C survival N(z)/N0
collision energy spectrum
H/O target fraction
```

诊断逻辑：

* first-reaction CDF 偏深：总 rate 偏低、单位/密度错误、能量插值错误或 lookup miss；
* CDF 一致但 Primary C survival 偏高：parent continuation 概率/状态错误；
* survival 一致但 Primary C dose 仍高：EM stopping/scoring/local deposit 问题；
* H/O 比例错误：材料 target number density 或 rate basis 错误。

Primary C 当前积分高 12.23%，必须先通过这一步把“反应率”“parent continuation”“连续能损”三者拆开。

---

## 阶段 5：单次反应 birth-level 验证

设定：

```yaml
max_secondary_inelastic_generations: 0
```

primary 反应后只记录直接生成的 products；可以先不输运，或创建后立即停止。

按以下条件比较 TOPAS raw、compiled package 和 GPU：

```text
collision energy
target H/O
collision depth
product Z/A
multiplicity
birth kinetic energy/u
birth polar/azimuth angle
parent alive probability
```

这是定位当前重碎片和 proton 偏差的分水岭：

* B/Be/Li/C 在 birth 时已经高 40%–60%：问题在提取、package 编译、事件选择、target mixture 或分类；
* birth 一致、输运后才高：问题在 stopping/MCS/straggling/FOV/scoring；
* proton birth 已少约 55%：package capture scope 或缺失 creator process；
* proton birth 一致但剂量少：proton transport、角分布、MCS 或 lateral escape。

---

## 阶段 6：逐核素纯 EM 输运矩阵

关闭所有 nuclear reaction，分别注入 package 实际出现的每个 charged isotope。推荐能量点：

```text
10, 50, 100, 200, 300, 400 MeV/u
```

至少对 p、d、t、He、Li、Be、B、C 做：

1. CSDA only；
2. * energy straggling；
3. * MCS；
4. 当前 80×80 mm scorer；
5. 一个明显更大的横向 scorer，仅作诊断。

比较：

```text
range / R80
integral dose
peak depth and dose
IDD NRMSE
lateral RMS
radial cumulative dose
energy escaping scorer
```

当前 scorer 的横向范围是约 80×80 mm。对角度更大的 proton，MCS 或 birth angular spectrum 的差异会转化为有限 FOV 内积分差异；所以要同时报告：

```text
Dose inside 80×80 mm
Dose inside large FOV
Energy escaping phantom/grid
```

当前配置中的 secondary MCS scale 1.40 不应直接继续拟合。先通过 mono-ion 测试确定它是否让 p/He 横向展宽过大。([GitHub][1])

阶段验收建议：

```text
range difference < 1%
integral dose difference < 1%
IDD NRMSE < 1%–2%
radial containment 在预先规定容差内
```

He 当前积分仅差 −6.70%，但 NRMSE 达 9.68%，很可能是 birth 数量与 EM/角输运误差互相补偿，不能把它视为已经正确。

---

## 阶段 7：按 generation 逐级打开 cascade

修完 scorer coupling、状态时序和 isotope transport 后，依次运行：

| 运行 | 允许的物理                          |
| -- | ------------------------------ |
| G0 | Primary inelastic；直接产物只做 EM 输运 |
| G1 | 直接产物可发生一次 inelastic            |
| G2 | 再允许下一代，作为生产候选                  |

每次输出：

```text
reaction count by projectile/target/energy/generation
birth count by Z/A/generation
dose by Z/A/generation
unsupported/neutral/overflow energy
```

判断方式：

* G0 时 B/Be/Li 已过高：primary event package 或直接碎片输运问题；
* G0 正常、G1 突然过高：次级 projectile rate/event package 问题；
* G1 正常、G2 突然过高：深一代级联或 isotope alias 问题；
* 某代 lookup miss 明显：该代 projectile 没有完整 package coverage。

最终生产模式不要超过 G2，除非每一代都单独通过 birth/rate 验证。

---

## 阶段 8：专门闭合 proton 来源

当前 GPU 配置显式关闭了 nuclear elastic。若 TOPAS reference 中启用了 hadronic elastic、neutron transport 或其他过程，GPU 将缺少一部分 recoil proton。仓库方法文档也说明基线模型没有完整覆盖独立强子弹性、neutral transport 以及一般 e/γ/decay 链。([GitHub][1])

需要在 TOPAS 端按 creator process 和 ancestry 分解 proton：

```text
A. primary C ionInelastic 直接产生
B. charged fragment inelastic 产生
C. neutron-induced reaction 产生
D. hadronic elastic recoil
E. decay / capture / stopping / other
```

GPU 端建立相同分类。

根据结果决定修复：

* A 少：修 package extraction、事件选择或 product decoder；
* B 少：修 secondary replay、cascade coverage；
* C 占比大：需要 neutral transport 或经过独立验证的 kerma/event package；
* D 占比大：实现或接入 nuclear elastic；
* birth 数量一致但 FOV 剂量少：修 proton stopping、角分布、MCS 和 lateral escape。

不要直接给 proton 乘约 2.2 的经验权重。这会破坏能量守恒、空间形状和其他束流能量下的预测。

---

# 三、当前各类偏差最可能对应什么

| 当前现象                  | 优先排查                                                                                 |
| --------------------- | ------------------------------------------------------------------------------------ |
| Primary C +12.23%     | NIEL 重复计数；primary reaction rate；parent continuation；C12 stopping/scoring             |
| Secondary C +44.13%   | 直接 birth 分类；parent 与 secondary C 定义；高代 cascade；C isotope alias                       |
| B/Be/Li +49%–63%      | birth multiplicity；secondary cascade rate；exact isotope stopping；reaction-local 归属   |
| He −6.70%，NRMSE 9.68% | birth 与 transport 抵消；He isotope mapping；straggling/MCS；角分布/FOV                       |
| proton −54.53%        | scorer-gated secondary replay；package 中 p birth；缺失 neutron/elastic 来源；proton MCS/FOV |

一个重要预期是：

* **移除 local+NIEL 重复计数**会降低重碎片相关剂量；
* **修复 scorer-gated replay**会提高 secondary reaction 数量，可能增加 proton，也可能增加重碎片；
* **把级联从 15 限到 0/2**会压低未验证的高代碎片；
* 三者不能打成一个 commit，否则无法知道最终变化来自哪里。

---

# 四、建议马上执行的最小实验序列

| Run | 修改                    | 主要输出                       | 通过条件                 |
| --- | --------------------- | -------------------------- | -------------------- |
| B0  | 仅加 counters，不改物理      | 当前完整账本                     | 与原 100k IDD 一致       |
| B1  | cascade=0             | primary rate、直接 birth      | 无 secondary reaction |
| B2  | 修 scorer coupling     | scorer 开/关、不同 voxel        | 物理计数不依赖 scorer       |
| B3  | 修碰撞状态时序               | E/position/direction trace | 无二次扣 dE              |
| B4  | 修 local+NIEL          | local/NIEL 分量              | Dose 只加 local        |
| B5  | strict lookup/isotope | miss/alias/overflow        | 全部为 0                |
| R1  | rate-only             | first collision、survival   | 与 TOPAS <1% 或统计相容    |
| R2  | birth-only            | Z/A、能量、角度                  | 与 package/TOPAS <1%  |
| R3  | mono-ion EM           | range/IDD/radial           | <1%–2%               |
| R4  | G0 full transport     | 直接碎片剂量                     | 定位 birth/transport   |
| R5  | G1                    | 一代 cascade 增量              | 按代闭合                 |
| R6  | G2                    | 最终候选                       | overflow/miss=0      |
| R7  | 加 proton 缺失通道         | p IDD                      | creator-process 闭合   |

---

# 五、最终验收条件

在重新做完整 200×200×800 三维 DoseToMedium 对比前，先满足这些硬门槛：

```text
event lookup miss                 = 0
invalid event/product bounds      = 0
queue overflow count/energy       = 0
silent isotope alias              = 0
uncovered rate cell               = 0
physics dependence on scorer grid = 0
local + NIEL double counting      = 0
species/category reconciliation   = 通过
per-event energy ledger           = 通过
```

然后分层验收：

```text
Primary first-reaction/survival    < 1%
Birth multiplicity and mean energy < 1% 或统计相容
Mono-ion range/integral            < 1%
Mono-ion IDD NRMSE                 < 1%–2%
最终各类别积分/峰值                 先达到 <2%
最终峰前逐点差                     <1%，且包含统计置信区间
```

100k histories 足够定位几十个百分点的系统误差，但对稀有 B/Be/Li 类别的逐点 1% 判据通常不够。最终历史数应由 batch-to-batch 或多 seed 的 95% 置信区间决定：只有当统计半宽明显小于 1% 时，才能判断剩余差异确实是模型误差。

最优先的前三项不是调参，而是：

1. **把 secondary CINEL02 replay 从 voxel scorer 分支中彻底移出；**
2. **按碰撞前 Post-EM 状态重新实现 secondary 事件回放时序；**
3. **DoseToMedium 只加入 process local deposit，不再叠加 NIEL。**

完成这三项并强制 `max_secondary_inelastic_generations=0` 后，再做 rate-only 和 birth-only；这会最快把当前 40%–60% 的偏差拆成“反应率、出生谱、输运、缺失通道”四个独立问题。

[1]: https://raw.githubusercontent.com/vvuvv31/MAIGO/refs/heads/fred/config/beam_400MeVu_cinel02_species_100k_xy04.yaml "https://raw.githubusercontent.com/vvuvv31/MAIGO/refs/heads/fred/config/beam_400MeVu_cinel02_species_100k_xy04.yaml"
[2]: https://raw.githubusercontent.com/vvuvv31/MAIGO/refs/heads/fred/src/transport_sycl.cpp "https://raw.githubusercontent.com/vvuvv31/MAIGO/refs/heads/fred/src/transport_sycl.cpp"
[3]: https://github.com/vvuvv31/MAIGO/raw/refs/heads/fred/src/transport_sycl.cpp "https://github.com/vvuvv31/MAIGO/raw/refs/heads/fred/src/transport_sycl.cpp"
[4]: https://raw.githubusercontent.com/vvuvv31/MAIGO/fred/startup/extensions/CarbonInelasticEventWriter.cc "https://raw.githubusercontent.com/vvuvv31/MAIGO/fred/startup/extensions/CarbonInelasticEventWriter.cc"
[5]: https://github.com/vvuvv31/MAIGO/raw/refs/heads/fred/src/detail/sycl_cinel02_device.inc "https://github.com/vvuvv31/MAIGO/raw/refs/heads/fred/src/detail/sycl_cinel02_device.inc"

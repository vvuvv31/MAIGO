# `38b5150` 非弹性过程第二轮代码审查与修复方案

我按你这轮结果对应的新提交 **`38b5150`** 做了静态复核。该提交主要引入了：

* 95/200/300/400 MeV/u 的能量相关碎片产额；
* projectile “joint channel”；
* 固定比例 neutron kerma；
* step 内非弹性碰撞位置抽样；
* 新的能量与剩余核诊断。

提交本身是 `feat(fred): energy-dependent yields, joint channels, kerma, in-step collisions`。

本文对照的是 De Simoni 等人的 *A Data-Driven Fragmentation Model for Carbon Therapy GPU-Accelerated Monte-Carlo Dose Recalculation*。

先给结论：

> **现在 100 MeV/u 的 IDD 吻合，很可能是多个方向相反的错误相互抵消，而不是非弹性过程已经正确。**
>
> 200–400 MeV/u 的平台剂量随深度越来越高、远端尾部仍偏低、halo 始终偏窄，说明当前同时存在：
>
> 1. 非弹性截面查表错误；
> 2. 事件中人为产生额外能量；
> 3. 碎片组合与论文算法不一致；
> 4. 轻碎片和 target-like fragments 的产生不足或能谱错误；
> 5. 次级碎片 MCS 明显偏小。

这五项必须按顺序处理，不能再用一个 normalization 或 halo scale 一次性拟合。

---

# 一、从新结果判断目前的错误结构

## 1. 100 MeV/u 只是表面吻合

100 MeV/u：

* Bragg peak 前的 IDD 基本吻合；
* Bragg peak 后仍低约 20%–50%；
* halo σ 仍低约 20%–30%；
* z=26.2 mm 的 GPU lateral profile 峰值大约只有 TOPAS 的一半。

这说明主碳离子的电磁部分仍然正常，但：

* 产生的远程带电碎片不足；
* 碎片能量或射程不足；
* 或大量次级能量被放到了局域沉积、neutron/untracked，而不是远距离输运。

因此不能把“100 MeV/u IDD match”当作验收通过。

## 2. 200–400 MeV/u 的平台过量随射程增长

目测新图：

|        能量 | Bragg peak 前最大 IDD 偏差 |  Bragg peak 后 |
| --------: | --------------------: | ------------: |
| 100 MeV/u |           约 −2% 到 +5% |   −20% 到 −50% |
| 200 MeV/u |              最高约 +18% | 约 −25% 到 −32% |
| 300 MeV/u |              最高约 +40% |        约 −20% |
| 400 MeV/u |          最高约 +60%–65% | 约 −10% 到 −17% |

这种随飞行距离和初始能量增长的正偏差，最符合：

* primary inelastic attenuation 偏小，使过多主碳离子继续沉积剂量；
* 每次 inelastic event 又额外产生 target-fragment kinetic energy；
* neutron kerma 在碰撞点被额外加到剂量中；
* target remnant 被经验性地局域沉积。

## 3. halo σ 系统性偏小是另一个独立问题

在 300/400 MeV/u 上游区域，GPU halo σ 比 TOPAS 低约 40%–50%，接近峰区后误差才逐渐减小。

这通常不能由 IDD normalization 解释，更可能来自：

* light fragment multiplicity 太低；
* target fragment 被局域沉积，没有继续输运；
* target-like angular component 被错误抽样；
* secondary fragment MCS 仍使用未标定的 Highland scale；
* neutron/secondary interaction 产生的宽翼没有被建模。

因此纵向 IDD 和横向 halo 必须分开诊断。

---

# 二、当前最新代码中最严重的问题

# P0-1：GPU 非弹性截面使用了错误的能量网格

这是当前必须第一个修复的问题。

`transport_sycl.cpp` 使用以下变量计算 stopping-power 插值位置：

```cpp
minimum_table_energy
inverse_table_step
```

这两个变量来自 stopping-power table。当前 stopping-power table 的网格是：

```text
0.01, 0.11, 0.21, ...
```

也就是：

```text
Emin = 0.01 MeV/u
ΔE   = 0.1 MeV/u
inverse step = 10
```

但非弹性截面表的网格是：

```text
1, 2, 3, ..., 400 MeV/u
```

也就是：

```text
Emin = 1 MeV/u
ΔE   = 1 MeV/u
```

GPU kernel 却用 stopping-power 的 `Emin=0.01` 和 `inverse_step=10` 去索引只有 400 行的 cross-section array。

这会产生近似映射：

|    实际粒子能量 |       GPU 实际访问的截面能量 |
| --------: | ------------------: |
|   5 MeV/u |          约 51 MeV/u |
|  10 MeV/u |         约 101 MeV/u |
|  20 MeV/u |         约 201 MeV/u |
|  30 MeV/u |         约 301 MeV/u |
| ≥40 MeV/u | 被 clamp 到 400 MeV/u |

也就是说：

```text
100 MeV/u → 使用 400 MeV/u 截面
200 MeV/u → 使用 400 MeV/u 截面
300 MeV/u → 使用 400 MeV/u 截面
400 MeV/u → 使用 400 MeV/u 截面
```

同时 `target_h_fraction` 也使用了同样的错误索引。

影响包括：

1. 100 MeV/u 初始非弹性截面被低估；
2. 低于约 40 MeV/u 后，接近射程末端的截面被严重低估；
3. 过多 primary carbon 存活到更深处；
4. 产生的碎片数量不足；
5. H/O target 比例错误，尤其是低能区 O 被过量选择；
6. IDD 平台偏高、fragment tail 和 halo 偏低。

这一个问题已经能够同时解释：

```text
高能平台逐渐过量
+
halo/fragment tail 不足
```

## 正确修改

最简单且 GPU 友好的方案是，在 host 侧把截面和 H 比例重采样到 stopping-power 网格：

```cpp
const auto& transport_energy = stopping_power.energies();

std::vector<float> macro_xs_on_transport_grid(
    transport_energy.size());

std::vector<float> target_h_on_transport_grid(
    transport_energy.size());

for (std::size_t i = 0; i < transport_energy.size(); ++i) {
    const double energy = transport_energy[i];

    macro_xs_on_transport_grid[i] =
        static_cast<float>(
            cross_section.interpolate(energy));

    target_h_on_transport_grid[i] =
        static_cast<float>(
            cross_section.interpolate_target_h_fraction(energy));
}
```

然后 GPU kernel 可以继续使用：

```cpp
minimum_table_energy
inverse_table_step
```

但必须访问新重采样后的数组，而不是原始 1 MeV/u 网格数组。

另一种方案是给 cross-section 独立保存：

```cpp
xs_min_energy
xs_inverse_step
xs_table_size
```

但从 GPU 性能和代码简单性考虑，host 端重采样更合适。

## 必须新增的测试

```cpp
for (float e : {
    1.0F, 5.0F, 10.0F, 20.0F, 40.0F,
    95.0F, 100.0F, 200.0F, 300.0F, 400.0F})
{
    CHECK_NEAR(
        gpu_interpolate_xs(e),
        cpu_cross_section.interpolate(e),
        1.0e-5F);

    CHECK_NEAR(
        gpu_interpolate_target_h(e),
        cpu_cross_section.interpolate_target_h_fraction(e),
        1.0e-5F);
}
```

当前测试只验证 CPU parser/interpolation，没有测试 GPU kernel 实际使用的索引，因此没有发现这个错误。

---

# P0-2：target fragment kinetic energy 被当作“免费额外能量”

当前 event generator 中，`running_e_sum` 只累计 projectile fragment energy。

代码逻辑相当于：

```text
projectile fragment kinetic energy
    必须小于 incident energy

target fragment kinetic energy
    不计入 energy budget
    被当作额外产生的能量
```

代码注释也明确表示 target KE 是 extra。

随后总能量 residual 虽然会计算：

```text
incident
- projectile fragments
- target fragments
- neutron energy
- local deposit
```

但如果 residual 是负数，即生成的总能量已经超过入射能量，事件仍然继续输运和沉积，没有被拒绝。

论文在前面确实说明 FRED 不做完整的非弹性多体动力学求解，但在具体 fragment-set 抽样部分又明确规定：

> 若 projectile 和 target fragments 的总能量超过 projectile energy，就重新抽取 fragment set，直到质量、电荷和能量满足约束。

所以当前实现与论文的具体 event-sampling 规则不一致。

## 为什么它正好造成你的高能 IDD 结果

target fragment 的能量又按以下方式随 beam energy 增长：

```text
Etarget ∝ E95 × Eprojectile / 95
```

因此每次 inelastic event 中凭空增加的 target KE 会随能量增大。

同时高能粒子的路径更长、发生非弹性反应的机会更多，所以累计额外剂量会呈现：

```text
100 MeV/u 小
200 MeV/u 中等
300 MeV/u 大
400 MeV/u 最大
```

这与你现在的平台偏差随能量增长高度一致。

## 正确修改

事件接受条件必须包含所有动能：

```cpp
double total_product_kinetic_MeV = 0.0;

for (const auto& fragment : products) {
    total_product_kinetic_MeV += fragment.kinetic_energy_MeV;
}

total_product_kinetic_MeV += neutron_kinetic_MeV;
total_product_kinetic_MeV += remnant_kinetic_MeV;

const double available_MeV =
    incident_kinetic_MeV + q_value_MeV;

if (total_product_kinetic_MeV > available_MeV + tolerance) {
    reject_entire_event();
    continue;
}
```

论文严格复现模式下，可以先使用：

```text
q_value = 0
available = incident kinetic energy
```

然后把剩余量定义为：

```text
remnant/excitation/untracked neutral energy
```

但不能：

```text
负 residual → 仍然接受事件
```

---

# P0-3：固定 8% neutron kerma 在当前实现中是双重计数

最新提交增加了固定：

```text
neutron kerma fraction = 0.08
```

当前过程相当于：

1. neutron 的完整 kinetic energy 加入 `untracked_energy`;
2. 再把其中 8% 作为局域剂量沉积；
3. 但没有从 neutron/untracked energy 中扣除该 8%；
4. 该 kerma 也没有完整进入同一 energy ledger。

因此：

```text
deposited dose
+
untracked neutron energy
>
neutron initial kinetic energy
```

这是明确的能量双重计数。

论文的 neutron production 数据来自 FLUKA 补充，因为 Ganil 没有 neutron production measurement；论文没有给出“所有 neutron kinetic energy 的 8% 在生成点局域沉积”这一模型。

## 正确修改

第一阶段直接关闭：

```yaml
inelastic_neutron_kerma_fraction: 0.0
```

若后续仍要保留经验 kerma，则至少必须满足：

```cpp
const float local_neutron_deposit =
    kerma_fraction * neutron_energy;

const float escaped_neutron_energy =
    neutron_energy - local_neutron_deposit;
```

并且：

```text
local_neutron_deposit
+
escaped_neutron_energy
=
original neutron energy
```

但固定 8% 在碰撞点沉积仍然没有论文依据。更合理的实现是：

* 显式 neutron transport；
* 预计算 neutron dose-spread kernel；
* 或从相同 TOPAS physics list 提取与能量、深度和材料相关的 neutral-dose response。

---

# P0-4：新加入的 200/300/400 MeV/u isotope-yield knots 不是论文模型

最新提交加入了：

```text
kFredProbHByE
kFredProbCByE
kFredProbOByE
```

然后在 95、200、300、400 MeV/u 之间插值，并人为增加：

```text
neutron
proton
helium
lithium
```

同时减少部分 B/C-like fragments。

但提供的论文中：

* Table 1 是由 95 MeV/u Ganil 数据和 Newton procedure 得到的 fragment production probability；
* 对其他治疗能量进行的是 **energy distribution 和 angular distribution 的 scaling**；
* 文中没有给出 200/300/400 MeV/u 的 isotope probability knots。 

所以当前版本已经不是论文中描述的 FRED 模型，而是：

```text
95 MeV/u FRED Table 1
+
手工 energy-dependent yield adjustment
+
Geant4 cross section
+
固定 neutron kerma
```

这种 hybrid model 最容易出现：

```text
95/100 MeV/u 看起来正常
高能量逐渐失控
```

这与你的新结果完全一致。

## 建议

立即增加两个明确模式：

```yaml
inelastic_model: fred_paper
```

使用：

```text
固定的 95 MeV/u Table 1 probability
Eq. 12 energy-angle distribution
Eq. 13–16 energy scaling
Eq. 21 angular scaling
Kox/ICRU 或明确选择的 cross-section
无固定 neutron kerma
```

以及：

```yaml
inelastic_model: topas_inclxx_matched
```

使用：

```text
与 TOPAS 相同 physics list 提取的
energy-dependent reaction package / species yield / kinematics
```

不要在一个模式中混用 paper probabilities 和手调 TOPAS corrections。

需要注意：论文验证对象是 FLUKA，不是 TOPAS INCL++，因此完全复现论文模型并不保证逐 bin 等于 TOPAS；但论文在 100–300 MeV/u 的整体剂量差异只有约 2.5%，当前 20%–65% 的偏差显然远大于参考模型差异。

---

# P0-5：当前 “joint projectile channel” 不是真正守恒的 joint channel

最新代码的 projectile fragmentation 基本过程是：

1. 从 Table 1 权重抽取一个 leading isotope；
2. 计算剩余 A/Z；
3. 找一个距离剩余 A/Z 最近的 isotope；
4. 剩余量只有在 `Zrem == 0` 时才转换为 neutron；
5. 返回最多两个 charged projectile fragments。

这不是论文描述的 fragment-set sampling，也没有重现 Newton procedure 中的 fragment correlation。

更严重的是，它并不总是满足精确电荷守恒。

例如抽到：

```text
leading fragment = 6He  (A=6, Z=2)
remaining        =      (A=6, Z=4)
```

`nearest_fred_isotope_fitting()` 很可能选：

```text
6Li  (A=6, Z=3)
```

最终：

```text
total A = 12
total Z = 5
```

缺少一个单位正电荷。

但当前 caller 会把剩余变量重新写成：

```text
leftover neutrons
Zrem = 0
```

因此电荷缺失被隐藏。

另一个例子：

```text
7Li + nearest 4He
```

只得到：

```text
A=11, Z=5
```

实际还缺一个 proton，但代码不能再添加第三个 charged remnant。

当前单元测试只检查：

```text
sumA <= 12
sumZ <= 6
```

而不是：

```text
sumA + remnantA == 12
sumZ + remnantZ == 6
```

所以这些不守恒事件仍能通过测试。

## 对 halo 的影响

这个 sampler 通常会生成：

```text
一个轻 leading fragment
+
一个较重 residue
```

而不是论文中平均每 primary 产生约 2–4 个 charged fragments 的多碎片集合。论文明确报告治疗能区平均每个 primary 产生约 2–4 个带电碎片。

结果会是：

* p/d/t/He 等轻碎片 multiplicity 偏低；
* 宽角 target-like fragments 偏低；
* 剂量更多集中在窄的重残核分量；
* halo σ 系统性偏小；
* distal tail 偏低。

## 正确实现

### 方案 A：最接近论文描述

使用 Table 1 CDF 依次抽取 fragment，直到不能再加入：

```cpp
while (remaining_A > 0) {
    isotope = sample_from_table1_cdf();

    if (isotope.A > remaining_A ||
        isotope.Z > remaining_Z) {
        reject_isotope_and_resample();
        continue;
    }

    add(isotope);
    remaining_A -= isotope.A;
    remaining_Z -= isotope.Z;
}
```

若最终：

```text
remaining_Z != 0
```

则拒绝整个 event。

这仍然不能完全恢复论文内部的 Newton correlation，因为论文没有公开完整的 Newton 方程组和最终 joint-channel table；这一点必须明确记录，不能靠猜测补全。

### 方案 B：推荐的 GPU 最终方案

在 CPU 离线枚举所有满足以下条件的 fragment multisets：

```text
Σ Ai = 12
Σ Zi = 6
```

target O：

```text
Σ Ai + Arem = 16
Σ Zi + Zrem = 8
```

然后通过非负优化或最大熵方法寻找 channel weights，使其满足：

```text
Table 1 isotope marginals
charged multiplicity
neutron multiplicity
physical channel constraints
```

生成：

```cpp
struct FragmentChannel {
    uint8_t count;
    uint8_t isotope[kMaxFragments];
    uint8_t origin[kMaxFragments];
    uint8_t remnant_A;
    uint8_t remnant_Z;
    float cumulative_probability;
};
```

GPU 只做一次 CDF lookup。

---

# P0-6：target remnant 被经验性局域沉积，导致平台偏高和 halo 偏低

当前 target loop 中：

* 抽到 heavy target fragment 后就提前结束；
* 剩余 target A/Z 不再作为真实 remnant 输运；
* 而是用经验公式给出一个最多约 8 MeV 的局域能量沉积；
* 未形成真实 secondary track。

这不是论文中的 fragment-set completion 和 conservation 过程。

它会同时导致：

```text
局域平台剂量升高
+
可输运 target fragments 减少
+
halo 变窄
```

正好与当前图形一致。

## 修改方式

定义显式 remnant：

```cpp
struct NuclearRemnant {
    int A;
    int Z;
    float kinetic_energy_MeV;
    float excitation_energy_MeV;
};
```

然后：

* 若 remnant 带电且 CSDA range 大于 cutoff：加入 secondary queue；
* 若 range 小于物理 cutoff：局域沉积其 kinetic energy；
* excitation 单独记录；
* 不能直接按 `Arem` 乘一个经验常数后沉积。

---

# P1-1：Eq. 13–16 只应用于 A ≥ 10 的 projectile fragments

当前代码只有：

```text
projectile fragment
且 A >= 10
```

才调用 projectile energy correlation。

p、d、t、He、Li、Be、B 大部分都只做简单：

```text
E = E95 × Eprojectile / 95
```

没有 Eq. 14–16 的 eventwise correlation。

论文没有给出 `A >= 10` 这个例外。它说明 projectile fragment energy 使用 Eq. 13，并通过相关因子 `k=c(1-R)` 保证同一事件总能量受到约束；target fragment 则从 exponential component 抽样且不使用相关因子。 

这意味着所有 projectile-origin fragments 都应进入同一个 correlation/energy-budget 过程，而不是只有重残核。

## 另一个问题：逐碎片 clip

当前 projectile fragment 超过剩余预算时，会把该 fragment energy 截断到剩余 room。

论文描述的是：

```text
整个 fragment set 不满足 → 重新抽取 fragment set
```

不是：

```text
只把当前 fragment 削短
```

逐碎片 clip 会：

* 改变 Eq. 12 能谱；
* 使后抽样 fragment 系统性低能；
* 压低 distal fragment tail；
* 让结果依赖 fragment 排序。

当前又把 fragments 按 heaviest-first 排序，因此相关结果进一步依赖人为顺序。

## 修复

```cpp
for (const auto& fragment : sampled_order) {
    fragment_energy =
        sample_correlated_projectile_energy(...);

    if (!valid(fragment_energy)) {
        reject_entire_event();
    }
}

if (sum_all_fragment_energy > available_energy) {
    reject_entire_event();
}
```

不要：

```cpp
fragment_energy = min(fragment_energy, remaining_room);
```

---

# P1-2：exponential sampling 的上限产生了二次能量依赖

当前代码先设置：

```text
E95,max = 2 × Eprojectile
```

然后再执行：

```text
Efragment =
E95 × Eprojectile / 95
```

因此最终 support 上限为：

$$
E_{\max}^\text{fragment}
=
\frac{2E_\text{projectile}^{2}}{95}.
$$

例如：

```text
100 MeV/u → 约 211 MeV/u
200 MeV/u → 约 842 MeV/u
300 MeV/u → 约 1895 MeV/u
400 MeV/u → 约 3368 MeV/u
```

对 projectile fragments，部分会被 remaining-room clip；对 target fragments，当前没有完整能量 gate，因此高能 tail 可以直接进入输运。

论文只说明从 95 MeV/u reference distribution 抽样，然后进行一次线性 scaling；没有给出 `E95,max = 2 × current beam energy` 这样的动态截断。论文给出的实验能量阈值是 fragment-dependent，角度从测量的 4°–43° 外推到 0°–180°。

## 正确做法

定义固定的 reference distribution domain：

```text
E95 ∈ [0, E95_reference_max(fragment, target)]
theta95 ∈ [0°, 180°]
```

`E95_reference_max` 必须：

* 来自 Ganil 数据范围；
* 或取一个足够大的固定数值，使截断概率可忽略；
* 不能随当前 beam energy 改变。

之后只做一次：

```cpp
E = E95 * Eprojectile / 95.0F;
```

并在完整事件层面检查 energy budget。

---

# P1-3：mixture 权重与实际抽样域不一致

当前 Gaussian/exponential mixture 权重使用近似无限积分：

```text
Gaussian integral ≈ π A2 σE σθ
Exponential integral ≈ A1 / (αE αθ)
```

但实际 sampling 却有：

```text
E 截断
theta 截断到 180°
E >= 0
Gaussian 反复拒绝负能量
```

因此 mixture probability 与实际使用的 truncated distribution 并不完全一致。

这对 p/d/t 尤其重要，因为论文规定这三种 hydrogen fragments 在 projectile 和 target fragmentation 中都使用完整 Eq. 12 mixture。

## 修复

对每个：

```text
target × isotope × Ereference-domain
```

离线计算：

```text
WGaussian
WExponential
```

并保存：

```cpp
P_gaussian =
    W_gaussian / (W_gaussian + W_exponential);
```

然后 GPU 使用预计算 LUT。

---

# P1-4：secondary fragment MCS 没有使用论文的 `fmcs`

当前 primary carbon MCS 有可配置 scale，但 secondary charged fragment 在调用 Highland 时使用固定 scale，等效于：

```text
fmcs = 1
```

没有按 fragment species、能量和 range fraction 使用标定值。

论文明确说明 Highland 单 Gaussian 项需要乘 `fmcs`，并且该因子随：

* 粒子种类；
* 能量；
* 深度；

变化。论文示例范围约为 1.29–1.43，并且标定时关闭了 nuclear interactions。

这会直接使 secondary fragment lateral spreading 偏小，是当前 halo σ 低 20%–50% 的重要原因之一。

但这一项必须在：

```text
species yield
birth angle
energy spectrum
range
```

都正确之后再标定，否则 MCS 会被迫补偿错误的 event generator。

---

# 三、这次提交中应该保留的改进

并不是 `38b5150` 的所有改动都需要回退。

## 应保留

### 1. step 内核反应位置抽样

现在先计算本 step 是否发生 interaction，并将 step 缩短到 collision distance，再进行碰撞。这比以前总在 step 末端生成碎片合理。

### 2. p/d/t 使用完整 Eq. 12 mixture

当前已经不再简单地把所有 target p/d/t 强制为 exponential，这一方向与论文一致。

### 3. 角度范围扩展到 0°–180°

这符合论文对 Ganil 4°–43° 拟合结果进行全角度外推的说明。

### 4. secondary step-limit 不再把所有剩余能量局域倾倒

新代码增大了 step cap，并在 step-limit 条件下将剩余能量标记为逃逸/未追踪，而不是全部沉积在当前位置。这基本解决了上一版 400 MeV/u distal spike 的主要人工来源。

### 5. 重 target fragment 使用 CSDA range 判断 local stop

这一修改比简单的 `E/SP` range 估算更合理。

---

# 四、按顺序执行的修复方案

# Step 0：冻结基线并增加可切换的 ablation 模式

以 `38b5150` 建立修复分支：

```bash
git checkout fred
git checkout -b fix/fred-inelastic-v3 38b5150
```

增加以下运行开关：

```yaml
inelastic:
  enable_attenuation: true
  enable_projectile_fragments: true
  enable_target_fragments: true
  enable_neutron_kerma: false
  enable_energy_dependent_yields: false
  strict_energy_conservation: true
  strict_az_conservation: true
  secondary_mcs_mode: unscaled
```

每次运行必须输出：

```text
primary carbon fluence vs depth
reaction count vs depth
reaction count by H/O target
projectile fragment KE
target fragment KE
neutron KE
local remnant deposit
neutron kerma deposit
signed energy residual
A residual
Z residual
charged multiplicity
species yield
secondary termination reason
```

## 本步骤硬门槛

```text
negative energy residual count = 0
A/Z closure failures           = 0
product queue overflow         = 0
secondary step limit           = 0
NaN/invalid stopping power     = 0
```

---

# Step 1：修复 GPU cross-section 与 target-H 查表网格

这是第一条代码提交，不应和其他物理修改混合。

推荐提交：

```text
fix(sycl): interpolate inelastic XS on its own energy grid
```

实现 host-side resampling，或使用单独的 cross-section metadata。

## 单元测试

CPU 和 GPU 分别检查：

```text
1, 5, 10, 20, 40, 95, 100, 200, 300, 400 MeV/u
```

要求：

```text
relative XS difference      < 1e-5
target-H fraction difference < 1e-5
```

## 回归测试

先只开启 attenuation：

```yaml
enable_attenuation: true
enable_projectile_fragments: false
enable_target_fragments: false
```

当发生 inelastic 时，直接杀死 primary，但不产生任何 secondary。

输出：

```text
primary C12 survival vs depth
reaction-depth histogram
H/O reaction ratio
```

这样可以把：

```text
cross-section/attenuation
```

与：

```text
fragment energy deposition
```

完全分离。

### 预期变化

修复后：

* 低能末端 reaction count 应显著增加；
* primary survival 应下降；
* 300/400 MeV/u 随深度增长的正 IDD 偏差应缩小；
* H reaction fraction 在低能区应上升；
* 后续 fragment count 和 halo 应有所增加。

---

# Step 2：关闭所有人为额外能量

推荐提交：

```text
fix(fred): enforce full-event signed energy closure
```

修改内容：

1. target fragment energy 加入同一总预算；
2. neutron energy 加入同一总预算；
3. remnant/recoil energy 加入同一总预算；
4. 禁止 negative residual；
5. 禁止逐 fragment clipping；
6. 不满足则拒绝整个 event。

定义统一 ledger：

```cpp
struct InelasticEnergyLedger {
    double incident_MeV;
    double projectile_charged_MeV;
    double target_charged_MeV;
    double neutron_MeV;
    double remnant_kinetic_MeV;
    double excitation_MeV;
    double local_deposit_MeV;
    double escaped_neutral_MeV;
    double numerical_residual_MeV;
};
```

约束：

```cpp
incident
=
projectile_charged
+ target_charged
+ neutron
+ remnant_kinetic
+ excitation
+ local_deposit
+ escaped_neutral
+ numerical_residual;
```

要求：

```text
|numerical_residual| / incident < 1e-5
```

### 预期变化

* 200–400 MeV/u plateau 应明显下降；
* GPU/TOPAS 正误差不应再随能量近似单调扩大；
* 高能结果不应再依赖 rare target-energy outliers。

---

# Step 3：完全关闭固定 neutron kerma

推荐提交：

```text
fix(fred): remove fixed vertex neutron kerma
```

先设置：

```text
kerma fraction = 0
```

并运行以下对照：

| Run | neutron bookkeeping   | neutron dose |
| --- | --------------------- | ------------ |
| A   | neutron KE 全部 escaped | 0            |
| B   | neutron 显式输运或 kernel  | 正确空间分布       |
| C   | 原固定 8% vertex kerma   | 仅用于定位错误      |

如果 A 比 C 明显降低 200–400 MeV/u plateau，则可直接确认固定 kerma 是正偏差来源之一。

不要长期保留 C。

---

# Step 4：回退手工 energy-dependent isotope yields

推荐提交：

```text
revert(fred): restore paper Table-1 yields at all energies
```

在 `fred_paper` 模式中：

```cpp
probabilities =
    target_is_H ? kFredProbH
                : kFredProbO;
```

不要调用：

```text
kFredProbHByE
kFredProbCByE
kFredProbOByE
```

只有：

```text
fragment energy
fragment angle
```

按 Eqs. 13–21 随 beam energy scaling。

## 必须做的 A/B test

四种能量分别运行：

```text
A: fixed Table 1, kerma off
B: energy-dependent knots, kerma off
C: fixed Table 1, kerma on
D: energy-dependent knots, kerma on
```

通过该矩阵可以直接分离：

```text
yield-knot bias
与
kerma bias
```

### 预期变化

如果当前高能偏差主要由 knots 造成：

* 100 MeV/u 几乎不变；
* 200/300/400 MeV/u plateau 明显降低；
* 高能 species composition 更接近 95-MeV reference scaling；
* 结果的能量趋势变得平滑。

---

# Step 5：删除 nearest-remnant joint sampler

推荐提交：

```text
fix(fred): replace nearest-remnant projectile sampler
```

禁止：

```text
one leading isotope
+
one nearest fitting residue
```

第一阶段可以实现论文文字最接近的 constrained sequential sampler：

```text
抽取 isotope
检查是否适合剩余 A/Z
适合则加入
不适合则重抽
完成后要求 A/Z 精确闭合
否则拒绝整个 event
```

测试必须从：

```cpp
CHECK(sumA <= 12);
CHECK(sumZ <= 6);
```

改为：

```cpp
CHECK(sumA + remnantA == 12);
CHECK(sumZ + remnantZ == 6);
```

target H：

```text
A=1, Z=1
```

target O：

```text
sumA + remnantA == 16
sumZ + remnantZ == 8
```

## 最终推荐

离线生成精确 channel LUT，使 GPU 不需要进行复杂 resampling。

## 验收指标

生成至少 \(10^7\) 个薄靶 events：

```text
A closure                         100%
Z closure                         100%
invalid channel                   0
major isotope marginal error      < 2%
all relevant isotope error        < 5%
mean charged multiplicity         合理且稳定
```

论文没有公开完整 Newton joint-channel system，因此无法只靠文章唯一恢复原始 channel weights。需要明确记录你采用的是：

```text
constrained approximation of published marginals
```

而不是声称完全复现未公开的内部 FRED channel table。

---

# Step 6：按论文重新实现 Eq. 12–21

推荐提交：

```text
fix(fred): implement paper Eq12-21 without dynamic truncation
```

## 6.1 Fragment source 与 component

```text
p/d/t:
    projectile 与 target 都抽完整 Eq. 12 mixture

其他 projectile fragments:
    Gaussian component

其他 target fragments:
    Exponential component
```

这与论文描述一致。

## 6.2 固定 reference domain

禁止：

```text
E95,max = 2 × current beam energy
```

改为固定 reference domain，并在该 domain 上正确归一化 mixture。

## 6.3 对所有 projectile fragments 应用 correlation

删除：

```cpp
if (fragment.A >= 10) {
    apply_correlation();
}
```

改为：

```cpp
if (fragment.origin == Projectile) {
    apply_projectile_energy_scaling_and_correlation();
}
```

对于 p/d/t 的 mixed component，先从 Eq. 12 抽出 `E95`，然后只要其 origin 是 projectile，就纳入 eventwise correlation。

## 6.4 不逐 fragment clip

任何 fragment 导致总预算超出：

```text
拒绝整个 event
```

不要修改该 fragment 的能量。

## 6.5 角度

```text
theta95 sampled over 0°–180°
theta = theta95 sqrt(95 / Eprojectile)
```

但 proton 和 neutron 不做 Eq. 21 scaling；target fragments 使用同样规则。

## 薄靶验证

固定 95 MeV/u，分别使用 H/O target，输出每个 isotope 的：

```text
E/A histogram
theta histogram
E–theta 2D histogram
Gaussian/exponential component fraction
projectile/target source
```

先定性重现论文第 7–11 页的分布图，再进入水模体。

---

# Step 7：显式处理 target remnant、Q 和 excitation

推荐提交：

```text
fix(fred): introduce explicit target remnant and excitation ledger
```

删除：

```text
按剩余 A 经验计算、最多 8 MeV 的局域 deposit
```

替换为：

```text
explicit A/Z remnant
remnant kinetic energy
excitation energy
mass-defect Q
```

在没有足够数据前，可以采用两个模式：

## Paper-minimal 模式

```text
Q = 0
excitation = available energy remainder
excitation 不作为剂量沉积，除非有明确局域释放模型
```

## Extended physics 模式

通过 nuclear mass table 计算：

$$
Q =
\left(
M_{\mathrm{initial}}
-
\sum M_{\mathrm{products}}
-
M_{\mathrm{remnant}}
\right)c^2.
$$

若 excitation 需要转化为局域剂量，必须给出明确模型或通过 TOPAS/FLUKA event package 标定。

---

# Step 8：先修 species production，再处理 halo MCS

推荐提交：

```text
fix(fred): apply species-dependent secondary MCS scaling
```

验证顺序：

## 8.1 Birth-angle-only

关闭 secondary MCS：

```text
fmcs = 0
```

查看刚生成后很短距离处的 angular profile，验证 Eq. 12/21。

## 8.2 MCS-only ion beams

分别模拟：

```text
proton
deuteron
triton
helium
Li/Be
B/C
```

关闭 nuclear interactions，只比较单粒子 lateral broadening。

## 8.3 建立 scale LUT

```cpp
fmcs[species_group][energy_bin][range_fraction_bin]
```

初始可按：

```text
H group
He group
Li/Be group
B/C group
```

论文给出的 `fmcs` 示例在 1.29–1.43 范围，但不能简单给所有 fragment 固定 1.35；应按 species、energy 和 depth 标定。

### 预期变化

完成本步骤后：

* core σ 小幅变化；
* halo σ 明显增加；
* 300/400 MeV/u 上游 halo 的 −40% 到 −50% 误差应显著缩小；
* IDD 积分不应出现明显变化。

如果调整 MCS 会显著改变 IDD，说明 secondary stopping/range 或 scoring 仍有错误。

---

# Step 9：按物理层级重新验证

建议严格按以下顺序执行。

| 阶段 | 开启内容                         | 检查目标                              |
| -- | ---------------------------- | --------------------------------- |
| A  | EM + elastic                 | 保持现有基线                            |
| B  | + inelastic attenuation only | primary survival 与 reaction depth |
| C  | + projectile fragments       | distal tail 与前向 fragments         |
| D  | + target fragments           | 低能宽角 halo                         |
| E  | + explicit remnant           | A/Z 与能量闭合                         |
| F  | + neutron model              | neutron dose 的空间分布                |
| G  | + secondary MCS              | core/halo σ                       |
| H  | 100 MeV/u full               | 接近 95-MeV reference               |
| I  | 200 MeV/u full               | 论文主要 benchmark                    |
| J  | 300 MeV/u full               | 高治疗能区                             |
| K  | 400 MeV/u full               | 外推稳定性                             |

每一级都保存：

```text
dose by primary C12
dose by projectile fragments
dose by target fragments
dose by species Z/A
local remnant dose
neutron-associated dose
escaped energy
```

只看总 IDD 无法判断偏差是由哪一个 component 引起的。

---

# 五、建议的验收标准

## 1. 代码和守恒

```text
GPU/CPU XS interpolation error       < 1e-5
GPU/CPU target-H probability error   < 1e-5
A closure failure                    0
Z closure failure                    0
negative energy residual             0
numerical energy residual / Ein      < 1e-5
product queue overflow               0
secondary step-limit count           0
NaN stopping/range                   0
```

## 2. 薄靶 event generator

```text
major isotope yield difference       < 2%
all retained isotope difference      < 5%
energy-spectrum moments              < 5%
angle-spectrum moments               < 5%
```

## 3. 水模体 100–300 MeV/u

论文报告 100–300 MeV/u 总积分剂量差异在 2.5% 内，并使用 \(10^8\) primaries 降低统计涨落。

建议工程验收：

```text
whole-depth integrated dose error    <= 2.5%
Bragg peak position difference       <= 0.5 mm
plateau IDD difference               <= 3%
distal-tail integral difference      <= 5%
core sigma difference                <= 3%
halo sigma difference                <= 10%
```

开发阶段：

```text
100k histories  smoke test
1M histories    每个物理提交
10M histories   halo 与 distal tail
100M histories  论文级最终验证
```

## 4. 400 MeV/u

400 MeV/u 是论文 scaling 的外推范围，但详细水中 pencil-beam validation 主要报告 100–300 MeV/u。 

建议目标：

```text
integrated dose difference           <= 5%
无随深度单调放大的平台偏差
无人工 distal spike
无能量相关手工 normalization
```

---

# 六、建议的提交顺序

```text
1. fix(sycl): use correct energy grid for inelastic XS and target-H lookup

2. test(sycl): compare CPU/GPU XS interpolation at 1–400 MeV/u

3. fix(fred): enforce complete signed event energy closure

4. fix(fred): remove fixed neutron vertex kerma

5. revert(fred): disable unsupported energy-dependent yield knots
                 in paper mode

6. fix(fred): replace nearest-remnant projectile joint sampler

7. test(fred): require exact A/Z closure, not inequality

8. fix(fred): apply Eq13-16 to all projectile fragments
              and reject whole invalid events

9. fix(fred): use a fixed reference domain for Eq12 sampling

10. fix(fred): represent target remnant and excitation explicitly

11. test(fred): add 95-MeV/u H/O thin-target spectrum benchmarks

12. fix(fred): add species/energy/depth secondary fmcs tables

13. test(fred): staged 100/200/300/400-MeV/u water regressions
```

# 七、最优先修改的四项

当前最可能快速改善新结果的顺序是：

```text
第一：修复 cross-section 网格错配
第二：把 target fragment KE 纳入完整 energy budget
第三：关闭固定 8% neutron kerma
第四：关闭手工 200/300/400-MeV/u isotope yield knots
```

预期图形变化：

```text
cross-section 修复
→ primary survival 降低
→ 200–400 平台正偏差下降
→ fragment count 增加

完整能量预算 + 关闭 kerma
→ 高能平台进一步下降
→ 不再随能量产生额外剂量

精确 fragment channels
→ distal charged-fragment tail 增加
→ halo population 增加

secondary fmcs
→ halo σ 增大
→ 解决剩余的 lateral width 偏差
```

所以，当前不能先调 halo scale，也不能根据 100 MeV/u 的表面吻合继续拟合高能 yield。**先修 cross-section 查表和 event energy closure，这两项是代码正确性问题，而不是模型调参问题。**

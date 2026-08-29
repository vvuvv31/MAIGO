# MAIGO `fred` 分支 inelastic 修复执行方案

## 一、结论

我检查时，`fred` 分支最新提交是 `a4bee63`，该提交正是把论文中的 Table 1、能量角度采样、重残核输运和能量 ledger 接入 GPU 的版本。([GitHub][1])

**当前问题的主要来源不是 EM、主碳离子的 stopping power 或 primary MCS，而是 inelastic event generator 本身。** 论文将碳离子输运拆成电离能损、MCS 和碎裂三个部分，而且 MCS 标定是在关闭核反应时完成的；既然你的 EM+elastic 已经吻合，就应先冻结这部分。

从你给出的图看：

* 100–300 MeV/u：Bragg peak 后的 IDD 明显偏低，halo σ 也大多偏低，说明**次级带电碎片的能量、射程或产额不足**，或者大量能量被放进了 `untracked`。
* 400 MeV/u：远端尾部反而大幅偏高且波动异常，说明还存在**次级输运终止时的人工局域能量倾倒**或高能碎片方向/射程处理错误。
* 误差随能量发生符号反转，因此不能通过一个全局 normalization、一个 MCS scale 或一个 halo scale 修好。

仓库自己的 `plan.md` 已经记录了几个红色信号：四套 Table 1 反演误差为 `0.122 / 0.137 / 0.622 / 0.182`，400 MeV/u 的模型 residual 占入射能量约 `9.71%`，71774 次 inelastic 中有 700 次重采样失败，而且重残核射程仍是 `E/SP` 而不是 CSDA 积分射程。([GitHub][2])

---

# 二、最高优先级的代码问题

## P0-1：Eq. 13–16 实现使第一个 projectile fragment 的能量被固定乘以 0.6

当前实现中：

```cpp
R = running_e_sum / running_a_sum / e_per_u;
k = 0.4 * (1 - R);
E_fragment = E95 * E_projectile / 95 * (1 - k);
```

但第一个碎片出现时 `running_a_sum == 0`，于是：

```text
R = 0
k = 0.4
E_fragment = 0.6 × E95 × Eprojectile / 95
```

也就是说，第一个 projectile fragment 无条件丢掉约 40% 的应有动能。当前代码正是这样实现的。([GitHub][3])

论文则说明，`c = 0.4` 的相关修正应当保证 projectile fragment 的平均能量/核子接近入射 projectile 的能量/核子；target fragment 才是不带相关因子的 exponential 抽样。 

这是最可能造成 100–300 MeV/u distal IDD 严重不足的单个代码错误。

---

## P0-2：Eq. 12 的 Gaussian/Exponential 分支选择与论文不一致

论文规定：

* `1H、2H、3H`：不论 projectile fragment 还是 target fragment，都从完整的 Gaussian + exponential 混合分布抽样。
* 其他 projectile fragments：Gaussian。
* 其他 target fragments：exponential。
* projectile fragments 主要向前、能量/核子接近入射离子；target fragments 能量低、角分布更接近各向同性。

当前代码却是：

```cpp
sample_gauss = !is_target && (u_mix < p_gauss);
```

这意味着：

* 所有 projectile fragment 都可能错误地落入低能 exponential 分量；
* 所有 target fragment 都被强制为 exponential；
* target 的 p/d/t 缺少论文要求的 Gaussian 高能前向分量。([GitHub][3])

同时当前代码还把：

```cpp
E95_exponential <= 35 MeV/u
theta <= 90 degree
```

作为硬截断，而论文明确说角分布应外推到完整的 `[0°, 180°]`。([GitHub][3]) 

`35 MeV/u` 的截断会显著缩短 target proton、deuteron、triton 的射程，直接压低 halo 和 distal tail。

---

## P0-3：inelastic 总截面与 H/O 靶核选择来自两套不一致的模型

当前配置使用 Geant4 11.3.2 的水中总宏观 inelastic 截面。对应 CSV 实际已经包含：

```text
macroscopic_cross_section_per_mm
macro_h_per_mm
macro_o_per_mm
target_h_fraction
```

([GitHub][4])

但 `CrossSectionTable` 只保存能量和总截面，CSV 解析器也只读取一个总截面列，H/O partial cross section 和 `target_h_fraction` 都被丢弃。([GitHub][5])

随后 GPU event generator 又用另一套硬编码 H 拟合和近似 Kox 公式重新计算靶核种类。([GitHub][3])

按当前 GPU 公式计算，与 CSV 给出的靶氢概率约为：

|        能量 | 当前 GPU 公式 \(P_H\) | CSV 中 \(P_H\) |
| --------: | ----------------: | ------------: |
| 100 MeV/u |             0.280 |         0.374 |
| 200 MeV/u |             0.260 |         0.343 |
| 300 MeV/u |             0.254 |         0.336 |
| 400 MeV/u |             0.254 |         0.338 |

CSV 对应值可直接在表中看到。([GitHub][4])

因此代码目前系统性地：

* 少采样 H 相互作用；
* 多采样 O 相互作用；
* O 表中中子占比更高，进而让更多能量进入 `untracked`；
* 改变 projectile/target fragment 的种类和角能分布。

论文的水中靶核选择本来就是根据 H/O 的 partial cross section 构造累计概率。

---

## P0-4：在论文已经给出的 Table 1 上又进行了一次未收敛的 runtime inversion

代码中的 `kFredProbH/C/O` 就是论文 Table 1 的 FRED 概率，并且已经建立了对应 CDF。([GitHub][6])

但程序启动时又分别对：

* projectile on H；
* projectile on O；
* target H，`A=1,Z=1`；
* target O，`A=16,Z=8`；

执行 `invert_table1_independent_probs()`。([GitHub][7])

论文描述的是：Table 1 本身就是经过 iterative/Newton procedure 构造出来、用于处理碎片产生相关性的最终概率表；随后事件通过累计分布抽样，并在质量、电荷、能量条件不满足时重新抽样。

尤其是对 `A=1,Z=1` 的 target-H 去反演完整 Table 1，在物理上不可能收敛——氢靶的 target fragment 只能是一个 proton。当前 `tgt_h` 反演误差高达 0.622 正是这个问题的表现。([GitHub][2])

论文没有公开完整的原始 joint-fragment channel 构造过程，因此稳妥做法不是继续增加 runtime Newton 迭代，而是：

1. 先把 Table 1 当作最终模型输入；
2. 在 CPU 离线生成满足 A/Z 约束的 joint event channels；
3. 检验生成后的 inclusive yield 是否重现 Table 1；
4. 把验证后的 channel CDF 打包给 GPU。

---

## P0-5：能量 ledger 把模型缺失能量伪装成了 `untracked`

当前代码自行定义了：

```cpp
base_q_MeV = 12.0 + 0.04 * E_per_u;
```

论文没有给出这一 Q-value 或 excitation 公式。([GitHub][3])

事件生成后，代码又计算：

```cpp
residual = Eincident
         - Echarged
         - Elocal
         - Euntracked;
```

只要 residual 为正，就直接：

```cpp
untracked += residual;
```

([GitHub][7])

因此外层的“会计能量守恒”可以看起来很好，但物理模型中实际上仍有近 10% 的入射能量没有解释。仓库自己的诊断正好报告 400 MeV/u 的 `model residual / E_in ≈ 9.71%`。([GitHub][2])

更严重的是，32 次重采样失败后，代码将整个 incident energy 设为 `untracked`，相当于让这一条 primary 无声消失。([GitHub][3])

---

## P0-6：secondary 达到 3000 步上限后，会把全部剩余能量沉积在当前位置

次级带电粒子的循环条件是：

```cpp
while (... && sec_steps < 3000)
```

([GitHub][7])

循环无论因为 cutoff、异常 stopping power，还是 `sec_steps == 3000` 结束，只要碎片还在 phantom 内，代码就把剩余 `sec_e` 全部加入当前位置的剂量。([GitHub][7])

若使用 `maximum_step_mm = 0.1 mm`，3000 步只对应 300 mm 路径。在 400 MeV/u 下，部分前向轻碎片或高能重碎片完全可能触发此上限。**这很可能是 400 MeV/u 远端尾部突然变成巨大正误差的直接原因之一。**

---

# 三、按顺序执行的修复步骤

## Step 0：固定可复现基线

在任何修改前固定当前状态：

```bash
git fetch origin
git checkout -b fix/fred-inelastic-v2 a4bee63

source /opt/intel/oneapi/setvars.sh

cmake --preset oneapi-nvidia-release
cmake --build --preset oneapi-nvidia-release -j

ctest --test-dir build/oneapi-nvidia-release \
      --output-on-failure
```

仓库的 NVIDIA SYCL preset 和运行方式记录在 README 中。([GitHub][8])

固定以下条件：

```text
random seed
phantom dimensions
voxel size
beam emittance
stopping-power table
elastic configuration
MCS configuration
TOPAS scoring grid
```

保存四组基线：

```text
EM only
EM + elastic
EM + elastic + inelastic attenuation only
current full inelastic
```

每次开发使用 100k histories 做 smoke test；每个重要提交至少用 1M histories 验证。

---

## Step 1：先增加诊断，不先调物理参数

修改：

```text
include/carbon/inelastic.hpp
src/detail/sycl_inelastic_device.inc
src/transport_sycl.cpp
src/io.cpp
```

增加以下 event-level 或 reduction-level 指标：

```text
reaction depth
target nucleus: H / O
projectile A/Z before and after
target A/Z before and after
fragment multiplicity
fragment isotope and projectile/target source flag
E/A and theta by isotope
charged kinetic energy
neutron kinetic energy
recoil/remnant kinetic energy
local excitation/deposit
mass-defect Q
model-unassigned energy
signed numerical residual
resampling count
resampling failure
secondary queue overflow
secondary termination reason
```

必须把现有的单个 `untracked_energy_MeV` 拆开，至少分为：

```cpp
struct InelasticEnergyLedger {
    float incident_kinetic_MeV;
    float charged_kinetic_MeV;
    float neutron_kinetic_MeV;
    float recoil_kinetic_MeV;
    float local_excitation_MeV;
    float q_mass_MeV;
    float escaped_neutral_MeV;
    float model_unassigned_MeV;
    float numerical_residual_MeV;
};
```

### 本步骤通过条件

```text
A/Z closure failure               = 0
product-capacity overflow         = 0
secondary queue overflow          = 0
secondary step-cap termination    = 0
numerical residual / E_in         < 1e-4
model_unassigned / E_in           单独输出，不能隐藏
resample failure                  = 0 in smoke test
```

在这些条件没有达到前，不允许调 MCS、halo scale 或 stopping-power scale。

---

## Step 2：统一总截面和 H/O 靶核选择

### 2.1 扩展数据结构

把：

```cpp
class CrossSectionTable
```

扩展为类似：

```cpp
struct WaterInelasticTable {
    std::vector<double> energy_MeVu;
    std::vector<double> macro_total_per_mm;
    std::vector<double> macro_h_per_mm;
    std::vector<double> macro_o_per_mm;
    std::vector<double> target_h_fraction;
};
```

修改：

```text
include/carbon/cross_section.hpp
src/cross_section.cpp
src/transport_sycl.cpp
```

为 GPU 增加：

```text
macro_total_device
macro_h_device
macro_o_device
target_h_fraction_device
```

### 2.2 event generator 不再自行重算 H/O

将：

```cpp
sample_carbon_inelastic_products_device(...)
```

增加参数：

```cpp
float target_h_fraction
```

并删除 `sycl_inelastic_device.inc` 中硬编码的 `sigma_H` 和 `sigma_O` 公式。

在与总截面完全相同的 incident energy 上插值：

```cpp
const float pH = interpolate(target_h_fraction_table, E_per_u);
target = uniform() < pH ? H : O;
```

### 2.3 明确两种模式，不要混用

建议增加：

```text
inelastic_model: fred_paper
inelastic_model: reference_matched
```

* `fred_paper`：总截面和 H/O partial 都来自论文的 H 数据拟合、Eq. 6 和 Kox scaling。
* `reference_matched`：总截面和 H/O partial 都来自同一套 TOPAS/Geant4 数据。

目前为分析你的 TOPAS 差异，应先使用 `reference_matched`。当前“一套总截面 + 另一套 H/O selector”的混合模式应删除。

### 本步骤单元测试

```text
macro_total == macro_H + macro_O
P_H(100) == 0.374118
P_H(200) == 0.342821
P_H(300) == 0.335628
P_H(400) == 0.337719
```

---

## Step 3：删除 runtime Table 1 inversion，改为离线验证的 event-channel sampler

修改：

```text
src/transport_sycl.cpp
src/fred_table1.cpp
scripts/invert_fred_table1.py
src/detail/sycl_inelastic_device.inc
```

### 3.1 删除或禁用以下运行时反演

```cpp
invert_table1_independent_probs(kFredProbH, 12, 6, ...)
invert_table1_independent_probs(kFredProbO, 12, 6, ...)
invert_table1_independent_probs(kFredProbH, 1, 1, ...)
invert_table1_independent_probs(kFredProbO, 16, 8, ...)
```

`src/fred_table1.cpp` 和 `scripts/invert_fred_table1.py` 暂时只作为离线诊断工具，不参与生产运行。

### 3.2 第一阶段实现

* projectile on H：使用 `kFredProbH`。
* projectile on O：使用 `kFredProbO`。
* target H：直接生成一个 target proton，不进行 Table 1 反演。
* target O：使用受 A/Z 限制的抽样；剩余未发射 A/Z 显式保存为 `NuclearRemnant`，不能隐式消失。

### 3.3 推荐的最终 GPU 实现

在 CPU 离线生成 joint event channels：

```cpp
struct FragmentChannel {
    uint8_t count;
    uint8_t isotope[kMaxFragments];
    uint8_t source[kMaxFragments]; // projectile or target
    int8_t remnant_A;
    int8_t remnant_Z;
    float cumulative_probability;
};
```

离线优化目标：

```text
重现 Table 1 inclusive isotope fractions
精确满足 projectile/target A/Z
重现 multiplicity distribution
不生成非法组合
```

GPU 只做：

```text
uniform RNG
CDF lookup
copy fixed-size channel
kinematics sampling
```

这也更符合论文采用预计算 lookup table、把复杂工作移出 GPU tracking kernel 的设计原则。

### 本步骤通过条件

对每个靶核生成至少 \(10^6\) 个 95 MeV/u 事件：

```text
A/Z closure                        100%
major isotope inclusive error      < 2%
all reported isotope error         < 5%
target-H target fragment           100% proton
resampling failure                 < 1e-5
```

不要继续使用“反演误差 0.12–0.62 但仍允许进入 GPU”的逻辑。

---

## Step 4：严格按论文修复 Eq. 12 分量选择和截断

在 `sycl_inelastic_device.inc` 中替换当前：

```cpp
const bool sample_gauss =
    !is_tgt && (u_mix < p_gauss);
```

为：

```cpp
const bool is_hydrogen_fragment =
    iso.z == 1 && iso.a >= 1 && iso.a <= 3;

Component component;

if (is_hydrogen_fragment) {
    component = sample_normalized_eq12_mixture(params, Emax, rng);
} else if (!is_target_fragment) {
    component = Component::Gaussian;
} else {
    component = Component::Exponential;
}
```

中子应建立独立规则，因为论文指出中子产额不来自 Ganil 测量，而是由 FLUKA 补充；不要默认把中子完全等同于 proton 参数。

### 4.1 正确计算 mixture 权重

不要直接使用无限积分近似。应在实际定义域内计算：

```text
E ∈ [0, Emax]
theta ∈ [0°, 180°]
```

```cpp
W_gaussian =
    A2 * integral_truncated_gaussian_E
       * integral_truncated_gaussian_theta;

W_exponential =
    A1 * integral_truncated_exponential_E
       * integral_truncated_exponential_theta;

P_gaussian = W_gaussian / (W_gaussian + W_exponential);
```

### 4.2 删除非论文硬截断

删除：

```cpp
e95 = min(e95, 35.0F);
theta = min(theta, 1.57F);
```

改为：

```text
Gaussian E: truncated at E >= 0
Exponential E: truncated by current event physical Emax
theta: sampled over 0°–180°
```

指数分布可以使用截断逆 CDF：

```cpp
float sample_truncated_exp(float u, float alpha, float xmax) {
    const float norm = 1.0F - exp(-alpha * xmax);
    return -log(1.0F - u * norm) / alpha;
}
```

注意论文表格中的能量参数是 MeV/u，角度参数是 degree；完成抽样后再统一转换到 rad。

### 本步骤验证

在固定 95 MeV/u、固定 target 和 isotope 条件下，输出：

```text
E/A histogram
theta histogram
E-theta 2D histogram
Gaussian/exponential component fraction
```

先重现论文第 7–9 页的定性分布，再进入水模体测试。

---

## Step 5：修复 Eq. 13–16 的 projectile energy correlation

令：

```text
P   = incident projectile energy per nucleon
x_i = E95_i × P / 95
T   = Σ(j<i) A_j E_j
M   = Σ(j<i) A_j
A_i = current fragment mass number
c   = 0.4
```

按论文公式中 \(j=0\ldots i\) 的定义，当前碎片能量出现在 \(R\) 中，因此需要解一个简单的隐式一次方程：

$$
E_i =
\frac{
x_i\left[(1-c)+
c\frac{T}{(M+A_i)P}\right]
}{
1-
c\frac{x_iA_i}{(M+A_i)P}
}.
$$

实现：

```cpp
float sample_projectile_fragment_Eu(
    float E95,
    float projectile_Eu,
    int fragment_A,
    float previous_total_energy,
    float previous_total_A)
{
    constexpr float c = 0.4F;

    const float x = E95 * projectile_Eu / 95.0F;
    const float total_A =
        previous_total_A + static_cast<float>(fragment_A);

    const float denominator =
        1.0F
        - c * x * static_cast<float>(fragment_A)
          / (total_A * projectile_Eu);

    if (!(denominator > 1.0e-6F)) {
        return -1.0F; // reject and resample
    }

    return x *
           ((1.0F - c)
            + c * previous_total_energy
              / (total_A * projectile_Eu))
           / denominator;
}
```

只有成功得到当前 \(E_i\) 后，才更新：

```cpp
T += A_i * E_i;
M += A_i;
```

不要再：

```cpp
clamp(R, 0, 1)
```

因为超出物理范围的事件应重采样，而不是通过 clamp 改写分布。

### 关键单元测试

```text
E95 = 95 MeV/u
first fragment
P = 100, 200, 300, 400 MeV/u

expected E_fragment/A ≈ P
not 0.6 × P
```

target fragments 继续使用论文规定的无相关修正 scaling：

```cpp
E_target_i = E95_target_i * P / 95;
```

### 预期图像变化

完成 Step 4 和 Step 5 后，应首先看到：

* 100–300 MeV/u 的 Bragg peak 后 IDD 明显抬高；
* projectile-like heavy fragments 射程增加；
* z=26.2、95.2、185.2 mm 等远端 lateral profile 的峰值不再严重偏低；
* halo component 的有效统计量增加。

---

## Step 6：重构 Q、remnant 和能量守恒

### 6.1 删除任意的 Q 公式

删除：

```cpp
base_q_MeV = 12.0F + 0.04F * e_per_u;
```

论文只给出了“若所有碎片能量之和超过 projectile energy，则重新抽样”的现象学约束，没有给出当前这条线性 Q 公式。

### 6.2 分两阶段完成

#### Phase A：结构调试模式

暂时设置：

```text
q_mass = 0
excitation = 0
```

只允许：

```text
charged fragment kinetic
neutral kinetic
explicit remnant/recoil kinetic
```

目标是先证明：

```text
fragment sampler + kinematics + transport
```

能够工作。

#### Phase B：最终物理模式

加入 isotope mass table，计算：

$$
Q =
\left(
M_{^{12}C}+M_\text{target}
-\sum M_\text{products}
-M_\text{remnant}
\right)c^2.
$$

把 excitation/local nuclear deposit 作为独立模型量。由于论文没有公开 excitation 分布，该部分需要：

* 额外实验数据；
* FLUKA/TOPAS event package；
* 或明确标记为经验参数化。

不要把它伪装成数值 residual。

### 6.3 禁止 post-hoc residual hiding

删除：

```cpp
if (residual_MeV > 0) {
    untracked_MeV += residual_MeV;
}
```

改为：

```cpp
if (abs(numerical_residual) > tolerance) {
    reject_event();
}
```

若某部分能量确实由未追踪中子、gamma 或 excitation 携带，必须在 event generator 中显式赋值到对应字段。

### 6.4 重采样失败处理

删除当前：

```text
32 retries failed
→ primary energy entirely untracked
→ continue simulation
```

验证构建中应直接：

```text
increment fatal counter
mark run invalid
```

生产构建只能回退到一个已经离线验证过的物理 channel，不能让 primary 无声消失。

---

## Step 7：在 step 内抽样真实的核反应位置

当前代码先：

1. 移动完整 `step_mm`；
2. 扣除完整 step 的 EM deposit；
3. 再用 `1-exp(-Σ step)` 判断是否发生 inelastic；
4. 在 step 终点生成碎片。([GitHub][7])

改为 residual optical depth：

```cpp
if (!has_tau) {
    tau_remaining = -log(uniform());
}

const float optical_depth_step =
    macro_xs(E_mid) * step_mm;

if (tau_remaining > optical_depth_step) {
    transport_em(step_mm);
    tau_remaining -= optical_depth_step;
} else {
    const float collision_distance =
        tau_remaining / macro_xs(E_mid);

    transport_em(collision_distance);

    generate_inelastic_event(
        energy_at_collision,
        position_at_collision,
        direction_at_collision);

    terminate_primary();
}
```

这样可避免：

* 反应点系统性向下游移动；
* 使用 step 末端而不是碰撞点能量；
* inelastic 概率对 `maximum_step_mm` 的非物理依赖。

### Step convergence 测试

分别运行：

```text
maximum_step_mm = 0.50
maximum_step_mm = 0.20
maximum_step_mm = 0.10
maximum_step_mm = 0.05
```

要求：

```text
reaction-depth mean shift < 0.1 mm
IDD integral change       < 0.5%
distal-tail integral      稳定收敛
```

---

## Step 8：修复 secondary range 和终止逻辑

### 8.1 用积分 CSDA 射程代替 `E/SP`

当前重 target fragment 使用：

```cpp
range_mm = fragment_energy / stopping_power_at_current_energy;
```

这不是 CSDA range。([GitHub][3])

仓库已经实现了：

```cpp
csda_range_mm()
csda_range_mm_device()
csda_energy_after_distance_MeVu()
```

可以直接复用。([GitHub][9])

为每个带电 isotope 预计算：

```text
cumulative_range[species][energy]
inverse_range_to_energy[species][range]
```

local-stop 判据改为：

```cpp
const float range =
    csda_range_mm_device(species_table, E_per_u, A);

local_stop =
    range < physical_local_cutoff_mm;
```

### 8.2 验证模式禁止 Z² fallback

当前缺少 species-specific stopping-power table 时，会回退到：

```cpp
Z² / 36
```

([GitHub][7])

增加：

```text
strict_fragment_transport: true
```

在严格模式下，只要某一 isotope 缺少 SP/range 表就终止验证，不允许静默 fallback。

### 8.3 明确 secondary termination reason

定义：

```cpp
enum class SecondaryTermination {
    EnergyCutoff,
    RangeExhausted,
    EscapedPhantom,
    GeometryBoundary,
    InvalidStoppingPower,
    StepLimit,
    QueueOverflow
};
```

只有：

```text
EnergyCutoff
RangeExhausted
```

可以把剩余能量局域沉积。

以下情况不得局域沉积：

```text
StepLimit
InvalidStoppingPower
QueueOverflow
```

`StepLimit` 在验证中必须为零。

### 8.4 暂时关闭 secondary fragmentation

论文指出 secondary fragmentation 对总剂量贡献很小，默认只模拟 primary fragmentation。

因此在 primary fragmentation 通过前，保持：

```text
secondary_fragmentation = false
```

避免把 cascade 的额外问题混入当前调试。

---

## Step 9：按层次验证，不能直接只看最终 IDD

### 9.1 验证顺序

| 阶段 | 开启内容                   | 主要检查                              |
| -- | ---------------------- | --------------------------------- |
| A  | EM + elastic           | 保持当前已吻合基线                         |
| B  | + attenuation only     | primary survival 和 reaction depth |
| C  | + projectile fragments | distal IDD、前向 fragment range      |
| D  | + target fragments     | 近反应点 halo、宽角低能 dose               |
| E  | + explicit remnant/Q   | 总能量、局域核能沉积                        |
| F  | + secondary MCS        | lateral core/halo                 |
| G  | full 100 MeV/u         | 首个完整验收                            |
| H  | full 200 MeV/u         | 中能量验收                             |
| I  | full 300 MeV/u         | 高能量验收                             |
| J  | full 400 MeV/u         | 外推与稳定性测试                          |

100 MeV/u 应先通过，因为论文在该能量取得最好的整体剂量一致性。论文的详细单 pencil-beam 验证范围是 100–300 MeV/u，而不是 400 MeV/u。

### 9.2 每阶段必须输出

```text
primary survival vs depth
inelastic reaction depth
dose by fragment Z/A
dose by projectile/target origin
charged multiplicity
E/A by isotope
theta by isotope
local excitation/recoil dose
neutral escaped energy
secondary termination counts
```

仅看总 IDD 无法区分：

```text
截面错误
species yield 错误
fragment energy 错误
fragment range 错误
scoring 错误
```

### 9.3 统计量

论文的最终对比使用了 \(10^8\) 个 primaries 来减小 MC 波动，并报告 100–300 MeV/u 全深度积分剂量差在 2.5% 以内。

建议：

```text
100k   快速回归
1M     每次物理提交
10M+   distal halo 与最终图
```

最终统计量应由不确定度决定，而不是固定只看 100k。

---

## Step 10：物理事件生成正确后再校准 fragment MCS

不要修改已经匹配的 primary MCS。

论文中的 MCS scale 是在关闭核反应的单粒子模拟中标定的，并且因 ion species、能量和深度而异；论文列出的示例范围约为 1.29–1.43，而不是所有 secondary 一律使用 1.0。

当前 secondary transport 调用 Highland 时没有使用主输运的 `multiple_scattering_scale`，相当于固定为自身默认处理。([GitHub][7])

推荐分组：

```text
H group:  p, d, t
He group
Li/Be group
B/C group
```

对每组做“单 isotope、无核反应、固定 E/A”的水中 lateral benchmark，得到：

```text
fragment_mcs_scale[species]
```

但只有在下列条件已经通过后才能开始：

```text
species yield correct
E/A distribution correct
angle-at-birth distribution correct
CSDA range correct
energy ledger correct
```

否则 MCS 会被迫补偿出生角度和射程错误。

---

## Step 11：修复 sigma 拟合脚本，避免把统计噪声当成物理误差

当前 `generate_plots_matplotlib.py` 强制：

```text
sigma_halo = 1.8 × sigma_core + exp(delta)
```

并在 halo weight 小于 1% 或双 Gaussian 改善小于 0.8% 时退化为单 Gaussian。([GitHub][10])

同时 sigma relative error 没有基于 dose、halo weight 或拟合不确定度做有效性 mask。([GitHub][10])

修改为同时输出：

```text
double-Gaussian sigma_core
double-Gaussian sigma_halo
halo fraction
fit covariance / bootstrap uncertainty
direct RMS radius
r68
r95
linear profile residual
log-profile residual
```

建议判据：

```text
dose(z) > 0.5% of maximum
halo weight > 2%
effective halo counts > threshold
relative sigma uncertainty < 20%
```

否则该深度的 halo σ 标为无效，而不是画出几十个百分点的随机振荡。

此外，当前验证脚本允许 peak 6%、ROI 8%、full integral 10% 的误差，这对于判断 FRED 复现是否成功过于宽松。([GitHub][11])

---

## Step 12：最后再做 GPU 优化

物理通过后再优化性能：

1. Table 1 joint channels 离线生成；
2. Eq. 12 的 truncated mixture normalization 离线预计算；
3. 按 target × isotope × energy 建立小型 CDF LUT；
4. 固定长度 fragment channel，避免动态分配；
5. 使用 SoA secondary queue；
6. 分离 projectile-like 和 target-like fragment kernel，减少 warp/sub-group divergence；
7. 把 event diagnostics 编译为可关闭选项；
8. CPU 与 GPU 使用同一套 counter-based RNG 索引约定；
9. 对 100、300 MeV/u 分别报告：

   * primary/s；
   * complete histories/s；
   * average charged secondaries；
   * queue occupancy；
   * overflow；
   * kernel time breakdown。

论文指出 fragmentation 相对主 tracking 是稀有事件，但每个 primary 平均会产生约 2–4 个带电碎片，因此性能关键在 secondary histories 和队列，而不是在 GPU 内进行复杂迭代求解。

---

# 四、最终验收标准

## Event generator

```text
A/Z closure                         100%
invalid channel                     0
resample failure                    < 1e-5
product/queue overflow              0
model-unassigned energy             有明确物理解释
numerical residual / E_in           < 1e-4
step-cap local-deposit              0
```

## 100–300 MeV/u

论文级目标：

```text
total integrated dose error         <= 2.5%
Bragg peak position shift           <= 0.5 mm
entrance/plateau IDD error          <= 3%
distal integrated-tail error        <= 5%
core sigma error                    <= 3%
halo sigma error                    <= 10%
```

其中最后四项是推荐的工程验收条件，不是论文原文给出的逐项指标。

## 400 MeV/u

400 MeV/u 应作为外推验证，不能参与最初参数拟合。论文说明能量 scaling 用于覆盖最高约 400 MeV/u，但详细 pencil-beam 剂量验收报告的是 100–300 MeV/u。 

建议临时目标：

```text
integrated dose error               <= 5%
no artificial distal spike
no secondary step-cap termination
no energy-dependent fudge factor
```

还要注意：论文使用 FLUKA 作为参考，而你现在使用 TOPAS INCL++。即使完全正确复现 FRED，局部 fragment spectrum 和远端 halo 也不一定逐 bin 等于 TOPAS；不要为了强行吻合 TOPAS 而破坏论文模型的一致性。

---

# 五、建议的提交顺序

```text
1. chore(fred): add inelastic event diagnostics and hard validation gates

2. fix(fred): use consistent H/O partial cross sections for target sampling

3. fix(fred): remove unconverged runtime Table-1 inversion
              and add validated channel sampler

4. fix(fred): implement Eq12 source-dependent component sampling
              and remove 35-MeV/u / 90-degree caps

5. fix(fred): solve Eq13-16 projectile energy correlation correctly

6. fix(fred): separate Q, remnant, neutral, model residual,
              and numerical residual ledgers

7. fix(fred): sample nuclear collision inside transport step

8. fix(fred): use per-isotope CSDA range and explicit
              secondary termination reasons

9. test(fred): add 95-MeV/u thin-target and staged water benchmarks

10. fix(plots): add statistically valid halo fitting and uncertainties

11. perf(fred): precompute event/CDF LUTs and optimize secondary queue
```

**最先应看到明显效果的三个修改是：**

```text
Eq. 13–16 第一个碎片 0.6 能量问题
Eq. 12 错误分支和 35 MeV/u 截断
secondary 3000-step 后剩余能量局域倾倒
```

这三项分别最可能对应你图中的：

```text
100–300 MeV/u distal underdose
halo sigma 偏小
400 MeV/u distal overdose
```

[1]: https://github.com/vvuvv31/MAIGO/commits/fred/ "https://github.com/vvuvv31/MAIGO/commits/fred/"
[2]: https://raw.githubusercontent.com/vvuvv31/MAIGO/fred/plan.md "https://raw.githubusercontent.com/vvuvv31/MAIGO/fred/plan.md"
[3]: https://raw.githubusercontent.com/vvuvv31/MAIGO/fred/src/detail/sycl_inelastic_device.inc "https://raw.githubusercontent.com/vvuvv31/MAIGO/fred/src/detail/sycl_inelastic_device.inc"
[4]: https://github.com/vvuvv31/MAIGO/raw/refs/heads/fred/data/c12_inelastic_cross_sections_water_geant4_11_3_2.csv "https://github.com/vvuvv31/MAIGO/raw/refs/heads/fred/data/c12_inelastic_cross_sections_water_geant4_11_3_2.csv"
[5]: https://raw.githubusercontent.com/vvuvv31/MAIGO/fred/include/carbon/cross_section.hpp "https://raw.githubusercontent.com/vvuvv31/MAIGO/fred/include/carbon/cross_section.hpp"
[6]: https://raw.githubusercontent.com/vvuvv31/MAIGO/fred/include/carbon/detail/fred_fragmentation_data.hpp "https://raw.githubusercontent.com/vvuvv31/MAIGO/fred/include/carbon/detail/fred_fragmentation_data.hpp"
[7]: https://raw.githubusercontent.com/vvuvv31/MAIGO/fred/src/transport_sycl.cpp "https://raw.githubusercontent.com/vvuvv31/MAIGO/fred/src/transport_sycl.cpp"
[8]: https://raw.githubusercontent.com/vvuvv31/MAIGO/fred/README.md "https://raw.githubusercontent.com/vvuvv31/MAIGO/fred/README.md"
[9]: https://raw.githubusercontent.com/vvuvv31/MAIGO/fred/include/carbon/stopping_power.hpp "https://raw.githubusercontent.com/vvuvv31/MAIGO/fred/include/carbon/stopping_power.hpp"
[10]: https://raw.githubusercontent.com/vvuvv31/MAIGO/fred/scripts/generate_plots_matplotlib.py "https://raw.githubusercontent.com/vvuvv31/MAIGO/fred/scripts/generate_plots_matplotlib.py"
[11]: https://raw.githubusercontent.com/vvuvv31/MAIGO/fred/scripts/validate_metrics.py "https://raw.githubusercontent.com/vvuvv31/MAIGO/fred/scripts/validate_metrics.py"

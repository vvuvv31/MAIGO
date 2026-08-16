> **Archived.** Historical Geant4 straggling notes. Not a runtime spec.

# Geant4 里的 Energy Straggling 是怎么处理的

Geant4 处理 **energy straggling（能损涨落）** 的核心思路可以概括成一句话：

> **先把大的能量转移作为离散的 δ-electron 显式产生，再对 production cut 以下的连续电离能损做随机涨落采样。**

所以它并不是简单地算一个 Bethe–Bloch 的平均 \(dE/dx\)，然后每一步都乘一个随机数。Geant4 把整个问题拆成了 **continuous loss + discrete loss** 两部分。

---

## 1. Production cut 把碰撞分成两类

假设一次电离碰撞转移给电子的能量是 \(T\)，Geant4 有一个 δ-electron 的 production threshold \(T_\mathrm{cut}\)。

对于

\[
T<T_\mathrm{cut},
\]

这些“软碰撞”不会逐个生成电子，而被并入连续能损：

\[
\left(\frac{dE}{dx}\right)_\text{soft}
=
n_\text{at}
\int_0^{T_\mathrm{cut}}
T\frac{d\sigma}{dT}\,dT.
\]

而

\[
T>T_\mathrm{cut}
\]

的碰撞则作为 **离散过程**，显式产生 δ-electron。

因此你最后看到的整个 energy-loss distribution 的长尾，实际上有两方面来源：

\[
\boxed{
\text{energy straggling}
=
\text{continuous-loss fluctuation}
+
\text{explicit }\delta\text{-ray production}
}
\]

---

## 2. 每一个 Geant4 step 先算平均能损

假设粒子在一步中走了 \(\Delta x\)。

Geant4 根据预先建立的 \(dE/dx\)、range 和 inverse-range tables 算出这一步的**平均连续能损**

\[
\overline{\Delta E}.
\]

当一步中的相对能损很小时，可以近似理解成

\[
\overline{\Delta E}
\simeq
\frac{dE}{dx}\Delta x.
\]

如果一步中能量变化比较大，则不是简单用起点的 \(dE/dx\)，而是利用 range/inverse-range table 来算。

但这里得到的只是

\[
\boxed{\overline{\Delta E}}
\]

还不是 Monte Carlo 中这一条 track 真正损失的能量。

接下来才进入 **fluctuation model**。

---

## 3. Geant4 根据 step 是“厚”还是“薄”采用不同的涨落方式

Geant4 的相关接口是 `G4VEmFluctuationModel`，常见实现包括：

- `G4UniversalFluctuation`
- `G4IonFluctuations`
- `G4AtimaFluctuations`
- `G4PAIModel`
- `G4PAIPhotModel`

其中标准能损涨落主要由 Universal/相关标准 fluctuation model 实现；对于离子、PAI、ATIMA 则有专门处理。

关键点不是简单判断“几何厚度”，而是看这一步包含的有效碰撞是否足够多。

---

## 4. 碰撞很多：Gaussian / Bohr straggling

如果平均连续能损足够大，Geant4 认为这一步包含了很多独立的小碰撞。

对于 restricted energy loss，可用类似条件描述：

\[
\overline{\Delta E}>\kappa T_c,
\]

同时

\[
T_\max\le 2T_c.
\]

这里：

- \(T_c\)：δ-electron production cut 对应的能量
- \(T_\max\)：一次碰撞最大可能的电子能量转移
- \(\overline{\Delta E}\)：这一步的平均连续能损

满足条件时，根据中心极限定理，energy loss distribution 已经接近 Gaussian。

Geant4 使用类似 Bohr straggling variance 的表达式：

\[
\Omega^2 =
2\pi r_e^2m_ec^2
N_\mathrm{el}
\frac{Z_h^2}{\beta^2}
T_\max s
\left(
1-\frac{\beta^2}{2}\frac{T_c}{T_\max}
\right).
\]

然后本质上采样

\[
\Delta E
\sim
\mathcal N
\left(
\overline{\Delta E},\,\Omega^2
\right).
\]

所以对**足够厚的 absorber / 足够长的 step**：

\[
\boxed{\text{straggling}\approx \text{Gaussian}}
\]

---

## 5. 薄层：不是直接调用一个 Landau random

这一点经常被误解。

如果前面的 Gaussian 条件不满足，Geant4 的标准 fluctuation model **并不是简单写成**

```cpp
loss = LandauRandom();
```

而是构造一个简化的原子碰撞模型。可以把原子近似成两个 excitation level：

\[
E_1,\qquad E_2
\]

再加上一类 ionisation collision。

可以把模型想象成：

```text
charged particle
        |
        | step Δx
        |
        +-- excitation E1
        +-- excitation E1
        +-- excitation E2
        +-- ionisation E
        +-- ionisation E
        ...
```

每种碰撞的次数都不是固定的。

---

## 6. 激发碰撞次数用 Poisson 分布采样

Geant4 对两类 excitation 定义宏观截面

\[
\Sigma_1,\qquad \Sigma_2.
\]

一步中的平均碰撞数为

\[
\langle n_i\rangle
=
\Delta x\,\Sigma_i.
\]

真正的碰撞数则随机采样：

\[
n_i\sim
\operatorname{Poisson}
(\Delta x\Sigma_i).
\]

于是 excitation contribution 为

\[
\Delta E_\mathrm{exc}
=
n_1E_1+n_2E_2.
\]

因此即使两个粒子：

- 初始能量一样
- 材料一样
- path length 一样

也会因为

\[
n_1,n_2
\]

不同产生不同能损。

---

## 7. Ionisation 的单次能量转移本身也随机

对于 production cut 以下的 ionisation，Geant4 使用近似

\[
g(E)\propto \frac{1}{E^2}.
\]

具体可以写成

\[
g(E)=
\frac{E_0T_\mathrm{up}}
{T_\mathrm{up}-E_0}
\frac1{E^2}.
\]

其中

\[
E_0 < E < T_\mathrm{up},
\]

而 \(T_\mathrm{up}\) 通常由 δ-ray production threshold 或 kinematic \(T_\max\) 限制。

首先 ionisation 次数也是

\[
n_3\sim
\operatorname{Poisson}
(\Delta x\Sigma_3).
\]

然后每一次 ionisation 的 loss 从 \(1/E^2\) 分布随机抽。

Geant4 使用 inverse-transform sampling：

\[
E_j=
\frac{E_0}
{1-u_j
\frac{T_\mathrm{up}-E_0}
{T_\mathrm{up}}},
\]

其中

\[
u_j\sim U(0,1).
\]

因此

\[
\Delta E_\mathrm{ion}
=
\sum_{j=1}^{n_3}E_j.
\]

最终

\[
\boxed{
\Delta E=
\Delta E_\mathrm{exc}
+
\Delta E_\mathrm{ion}
}
\]

所以这里其实有**两重随机性**：

\[
\boxed{
\begin{array}{c}
\text{碰撞次数涨落}\\
n_i\sim\mathrm{Poisson}
\end{array}}
\]

加上

\[
\boxed{
\begin{array}{c}
\text{每次碰撞能量转移涨落}\\
E\sim1/E^2
\end{array}}
\]

这就是 Geant4 thin-absorber straggling 的核心。

---

## 8. 那 Landau distribution 在哪里？

更准确地讲：

**Geant4 标准 energy-straggling 模型并不是简单在 Gaussian / Landau 两个函数之间切换。**

对于薄 absorber，它做上面这种微观碰撞的随机采样；当进入经典 Landau theory 的适用区域时，最终得到的能损分布会**自然逐渐接近 Landau 形状**。

所以可以画成：

```text
very thin
   |
   |   asymmetric
   |   long high-E tail
   |      ______
   |     /      \_____
   |____/             \________
   |
   +-------------------------- ΔE
          Landau-like

increasing thickness

             ↓

         nearly Gaussian
             ___
           /     \
         /         \
________/           \________
```

不是：

```text
thin  -> TRandom::Landau
thick -> TRandom::Gaus
```

这么直接。

---

## 9. Production cut 会参与 straggling

如果你改变 δ-electron cut：

\[
T_c
\]

你实际上改变了：

```text
一次能量转移

0 ---------------- Tc ---------------- Tmax
|                  |                    |
continuous         δ-rays explicitly produced
```

cut 小：

\[
T_c\downarrow
\]

意味着更多大能量碰撞被作为 δ-ray 显式追踪。

cut 大：

\[
T_c\uparrow
\]

意味着更多碰撞被吸收到 continuous loss + fluctuation 中。

因此理论上：

\[
\text{continuous straggling}
\]

本身会随 cut 改变。

不过显式 δ-rays 和 continuous part 合起来以后，**最终物理 observable 应该尽可能不敏感于合理范围内的 cut 和 step size**。

---

## 10. Step size 同样很重要

对一个 step：

\[
\overline{\Delta E}
\approx
\frac{dE}{dx}\Delta x.
\]

所以

\[
\Delta x\downarrow
\quad\Longrightarrow\quad
\overline{\Delta E}\downarrow.
\]

于是一步可能从 Gaussian regime 进入 thin-absorber regime。

可以直观理解为：

```text
large step
    ↓
many collisions
    ↓
Gaussian approximation

small step
    ↓
few collisions
    ↓
Poisson excitations + ionisations
    ↓
asymmetric straggling
```

这也是为什么做 silicon sensor、TPC、gas detector、薄膜或 microdosimetry 时，**step limit / cuts 不能完全和 energy-loss spectrum 分开考虑**。

---

## 11. 如果是重离子，还有额外处理

对于 proton/hadron/ion，尤其是低速重离子，事情会复杂一些，因为除了普通 Bethe-Bloch 型能损之外，还涉及：

- effective charge
- charge exchange
- ion-specific straggling
- stopping-power corrections

Geant4 为这类情况提供 `G4IonFluctuations`，另外 ATIMA 使用 `G4AtimaFluctuations`。

标准能损框架仍然是：

> 先计算平均能损，再进行 fluctuation sampling。

但是 variance 和离子有效电荷等会有专门处理。

---

## 12. 可以把 Geant4 的整体逻辑抽象成

```cpp
// 一个 charged-particle step

// 1. 根据 production cut 计算 restricted stopping power
meanLoss = RestrictedDEDX(E, Tcut) * stepLength;

// 2. T > Tcut 的碰撞
//    -> 显式产生 delta electrons
SampleDeltaRays();

// 3. T < Tcut 的连续部分加入 straggling
if (thickAbsorberCondition) {

    actualLoss = Gaussian(meanLoss, bohrVariance);

} else {

    n1 = Poisson(meanN1);
    n2 = Poisson(meanN2);
    n3 = Poisson(meanN3);

    excitationLoss = n1*E1 + n2*E2;

    for (int i=0; i<n3; ++i)
        ionisationLoss += sampleFrom1OverE2();

    actualLoss = excitationLoss + ionisationLoss;
}

// 4. 更新 primary energy / local energy deposit
```

虽然真实源码会复杂很多，但**物理逻辑基本就是这样**。

---

## 13. 最值得记住的结论

Geant4 的 energy straggling 可以概括为：

1. 用 **production cut** 把硬碰撞和软碰撞分开。
2. 大能量转移通过显式 δ-ray 产生处理。
3. cut 以下的连续能损先计算平均值。
4. 对平均连续能损做随机 fluctuation sampling。
5. 碰撞多时趋向 **Gaussian / Bohr straggling**。
6. 薄层时采用基于 **Poisson 碰撞数 + 随机单次能量转移** 的模型。
7. 在合适区域，最终分布会自然呈现 **Landau-like** 长尾，而不是直接调用固定 Landau 分布。
8. `production cut` 与 `step size` 都会影响连续/离散能损的划分，因此在薄探测器问题中尤其值得检查。

---

## 参考文档

- Geant4 Physics Reference Manual — Energy Loss
- Geant4 Physics Reference Manual — Energy Loss Fluctuations
- Geant4 Release Notes（关于 fluctuation model 选择与相关更新）

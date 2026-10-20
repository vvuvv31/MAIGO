# II. 材料与方法

**GPU Monte Carlo: From Broadbeam to Minibeam**（[English Version](mm.md)）。

---

## II.A. 计算框架、材料表征与束流源建模

### II.A.1. 输运架构概述与研究范围

本研究开发了跨平台 GPU 蒙特卡罗输运引擎 **MAIGO**，在统一的临床笔形束扫描（pencil-beam scanning, PBS）框架中实现质子与碳离子的宽束（broadbeam）及多狭缝微束（minibeam radiation therapy, MBRT）剂量计算。其核心将凝聚历史带电粒子电磁输运与离散核反应耦合，并显式追踪多代带电核碎片 [1–4]。微束计算将铜多狭缝准直器及下游空气间隙的上游输运与患者体素输运分阶段执行，在患者入口传递位置、方向、能量及物种的联合相空间状态。该结构面向亚毫米峰谷剂量调制的计算需求，结合显式准直器建模 [5] 与分阶段 GPU 调度，将宽束患者输运扩展至微束照射。

引擎支持质子（$^1\mathrm{H}$, $Z=1, A=1$）与碳-12 离子（$^{12}\mathrm{C}$, $Z=6, A=12$）主束输运，并追踪所有受支持通道产生的带电核碎片与核弹性反冲核。物理计分量包括介质吸收剂量（dose to medium, $D_m$）以及可按需开启的剂量平均电子线性能量传递（dose-averaged electronic LET，记为 $L_d$）。本研究以物理剂量、电子 LET 及计算性能为评价终点。由入射粒子种类（$^1\mathrm{H}$ 与 $^{12}\mathrm{C}$）、模体介质（均匀水模体与异质患者 CT）以及照射模式（宽束与铜准直微束）正交组合，共构成八种标准发布配置（release presets）。在患者全计划验证中，各病例完整保留其临床束流光学模型、束斑权重、摆位几何、准直器参数及处方剂量归一化；默认 CT 微束预设配置则作为独立的异质肺部模体基准算例。

### II.A.2. 体素化材料表征与离线物理数据库

患者 CT 图像经离线预处理转换为规则三维笛卡尔体素网格，包含物理坐标原点、体素尺寸、质量密度 $\rho$ 及离散组织材料编码。CT 值（Hounsfield Unit, HU）依据 Schneider 化学计量学标定方法 [6] 映射至连续质量密度与 25 种人体组织元素组成；连同平均激发能 $I = 75\text{ eV}$ 的水参考材料（$\mathrm{Water\_75eV}$），系统共定义 26 种基础电磁介质组成。MAIGO 根据各体素的组织元素组成与连续质量密度计算电磁阻止、密度效应修正及核反应截面，使组织组成和密度共同参与异质介质输运。

所有底层物理截面与响应函数均基于 Geant4 11.3.2 与 OpenTOPAS 4.2.3 [7,8] 离线提取，并封装为紧凑的模块化二进制数据包：
1. **连续密度离子电磁数据包（`continuous26`）**：涵盖 26 种介质的非受限电子阻止本领、能损涨落方差参数及 Sternheimer 密度效应系数。运行时由同一阻止响应构造连续慢化近似（CSDA）射程及反射程映射。
2. **束线材料电磁数据包（`BLEM0001`）**：针对铜（$\mathrm{G4\_Cu}$）与上游空气（$\mathrm{G4\_AIR}$）独立提取的非受限电子阻止本领及全运动学上限能损涨落方差，完整覆盖主粒子及铜核反应产生的各类带电碎片。
3. **多重库仑散射（MCS）数据包**：预提取水、25 种 Schneider 组织、铜及空气中各带电物种的 Urban 多重散射参数、输运自由程与辐射长度，并配对提供高斯 Fermi–Eyges 散射参数。
4. **核反应截面表与相关末态事件库**：包含宏观弹性与非弹性核反应率表（`SCHNRATE` 与 `SCHN2RAT` v3）、分同位素核弹性散射动量转移样本、重靶反冲核专用阻止本领表（`ELRSP001` 与 `LERSP001`），以及替代 GPU 在线核内级联计算的核非弹性相关末态事件库（`CINPKG04`）。

初始化时检查物理表的入射粒子身份 $(Z, A)$、材料组成及能量覆盖范围；模拟在所需通道的数据覆盖域内执行。每次计算记录物理数据版本、运行软件版本、整数束斑粒子分配及随机数种子，以复现相同的源与物理配置。

### II.A.3. 笔形束扫描源模型与束斑权重优化

临床扫描治疗计划由射野、能量层及离散扫描束斑序列定义，每个束斑包含标称动能、等中心平面横向坐标及粒子权重。随能量变化的相空间束流光学模型给出源平面处的横向位置标准差 $(\sigma_x, \sigma_y)$、角散度标准差 $(\sigma_{\theta_x}, \sigma_{\theta_y})$、位置—角度协方差参数以及高斯能量展宽 $\sigma_E$。模拟时，先在束斑局部坐标系内对耦合的横向位置与方向余弦进行联合抽样，再结合二极扫描磁铁虚拟源轴距（SAD）、机架／治疗床旋转角及等中心偏移量，将粒子状态刚体变换至患者世界坐标系。离子能量统一以单核子动能（$\mathrm{MeV/u}$）表征；对于以总动能（$\mathrm{MeV}$）存储的计划输入，按质量数 $A$ 精确换算。

物理剂量逆向优化以单位入射粒子的剂量影响矩阵 $\mathbf{A}$ 和非负束斑粒子权重 $\mathbf{w} \ge \mathbf{0}$ 表示体素剂量 $\mathbf{d} = \mathbf{A}\mathbf{w}$，并根据各病例的靶区覆盖与危及器官（OAR）保护目标求解权重。最终权重在 MAIGO 与 TOPAS 全计划前向计算中固定为同一个向量，使两引擎比较采用共同的照射源。重新优化的计划采用 $4\text{ Gy}$ 物理剂量处方；保留治疗计划系统（TPS）权重的碳离子宽束计划采用原计划累计剂量标度。

---

## II.B. 均匀水模体与异质 CT 中的离子输运物理

### II.B.1. 非受限电子阻止本领、连续密度效应与 CSDA 射程

每条带电粒子轨迹维护其三维空间坐标 $\mathbf{r}$、单位方向向量 $\mathbf{u}$、总动能 $E$、核身份 $(Z, A)$ 以及确定性随机数流状态。以比动能 $e = E/A$（$\mathrm{MeV/u}$）为自变量的电磁物理表根据粒子精确物种、当前体素材料组成及实际质量密度 $\rho$ 进行查询。MAIGO 采用非受限线性电子阻止本领（unrestricted linear electronic stopping power）$S_e(E; \rho)$，包含在完整运动学允许区间 $[0, T_{\max}]$ 内转移给电离次级电子的平均能量。物种分辨的原生阻止本领曲线保留了各离子的有效电荷（$Z_{\mathrm{eff}}$）演化与高阶修正。

在异质 CT 介质中，实际体素密度 $\rho$ 处的非受限电子阻止本领通过结合参考密度 $\rho_0$ 处的线性质量密度缩放与连续 Sternheimer 密度效应修正求得：

$$
S_e(E; \rho) = \frac{\rho}{\rho_0}\, S_e(E; \rho_0) - C_{\mathrm{mat}}(\rho)\,\frac{Z_{\mathrm{eff}}^2}{\beta^2}\left[\delta(X; \rho) - \delta(X; \rho_0)\right],
\tag{1a}
$$

式中 $\beta = v/c$ 为相对论速度比，$X = \log_{10}(\beta\gamma)$，$\delta(X; \rho)$ 为随实际密度连续插值的 Sternheimer 密度效应函数。固定局部材料组成与密度时，非受限 CSDA 射程 $R(E)$ 与有限真实路径 $\ell$ 上的平均电子能损 $\overline{\Delta E}(E, \ell)$ 满足：

$$
R(E) = \int_0^E \frac{\mathrm{d}E'}{S_e(E')}, \qquad
\overline{\Delta E}(E, \ell) = E - R^{-1}\!\left(\max\!\left[R(E) - \ell,\, 0\right]\right).
\tag{1b}
$$

阻止本领、射程及反射程采用同一分段线性响应。参考密度下的 CSDA 射程 $R_0(E)$ 在每个分段线性阻止本领节点区间 $[E_i, E_{i+1}]$（阻止本领由 $S_i$ 线性变化至 $S_{i+1}$）内采用倒数解析积分构造：

$$
\Delta R_i = \int_{E_i}^{E_{i+1}} \frac{\mathrm{d}E'}{S_e(E')} = \frac{E_{i+1} - E_i}{S_i}\,\frac{\ln(1 + q_i)}{q_i}, \qquad q_i = \frac{S_{i+1} - S_i}{S_i}.
\tag{1c}
$$

当区间相对斜率满足 $|q_i| < 10^{-3}$ 时，算法自动切换为四阶 Taylor 级数展开，以消除单精度浮点数（FP32）下的相减抵消误差；区间内的反射程映射 $R_0^{-1}$ 则通过指数函数 $\mathrm{expm1}$ 解析反演，最低能量节点 $E_0$ 以下采用规定的 $\sqrt{E}$ 低能延拓积分。当粒子动能低于密度效应起始阈值（即 $\delta(X; \rho) = \delta(X; \rho_0) = 0$）时，实际密度射程精确简化为线性缩放 $R(E; \rho) = (\rho_0 / \rho) R_0(E)$；高于起始阈值时，则在对数能量空间采用复合 8 点 Gauss–Legendre 求积公式计算倒阻止本领残差积分，并通过带区间保护的 Newton–二分混合迭代求解反射程 $R^{-1}$（相对收敛容差为 $2\times 10^{-6}$）。

在水与患者 CT 输运步中，先计算步首线性能损估计值 $\Delta E_{\mathrm{lin}} = S_e(E)\,\ell$ 并与数据包记录的原生线性能损阈值 $\xi_{\mathrm{lin}} E$ 比较。当相对能损未超过 $\xi_{\mathrm{lin}}$ 时，非离子模型（包括质子）直接采用 $\overline{\Delta E} = S_e(E)\,\ell$，采用 Geant4 `G4VEnergyLossProcess` 的短步规则；重离子模型（$Z > 1$）则保留中点能量预测—校正法，即在预测中点能量 $E - \tfrac{1}{2}S_e(E)\ell$ 处重新计算阻止本领以反映有效电荷的变化。当步能损超过 $\xi_{\mathrm{lin}} E$ 或路径耗尽剩余射程时，统一调用式（1b）的 CSDA 射程反演。常规 18 物种离子注册表以外的重靶反冲核则调用专属的电子加核总阻止本领表（`ELRSP001`），详见第 II.B.6 节。

水与患者 CT 采用局部电离能量沉积近似：非受限电子能损计入当前输运体素，次级 $\delta$ 电子及韧致辐射／原子弛豫光子不作为独立轨迹输运。该处理将电子能量转移纳入离子凝聚历史步进；剂量评价覆盖头部与低密度肺部病例，并采用微束横向剖面分析峰谷结构。

### II.B.2. 主粒子能损涨落（Energy-Loss Straggling）

主粒子（质子与 $^{12}\mathrm{C}$）的能损涨落采用移植自 Geant4 11.3.2 `G4UniversalFluctuation` 与 `G4IonFluctuations` 的统一随机抽样引擎 [8]。对于静止质量为 $M$、速度为 $\beta c$、Lorentz 因子为 $\gamma$ 的入射离子，单次二体碰撞转移给自由电子的运动学最大能量为：

$$
T_{\max} = \frac{2 m_e c^2 \beta^2 \gamma^2}{1 + 2\gamma(m_e/M) + (m_e/M)^2},
\tag{2}
$$

其中电子静能 $m_e c^2 = 0.51099891\text{ MeV}$。由于 MAIGO 采用无次级电子产生截断的非受限凝聚历史方案，$T_{\max}$ 直接作为能损涨落抽样的能量转移上限。对于离线按参考截断能量 $T_c$ 导出的 `continuous26` 介质表，运行时将当前密度下的参考单位长度方差 $V_{\mathrm{ref}}$ 解析还原为全运动学上限 $T_{\max}$ 对应的非受限单位长度方差 $V_{\mathrm{full}}$：

$$
V_{\mathrm{full}} = V_{\mathrm{ref}} \, \frac{T_{\max}/\beta^2 - T_{\max}/2}{T_{\max}/\beta^2 - \min(T_c, T_{\max})/2},
\tag{3}
$$

式中 $V_{\mathrm{ref}}$ 已包含局部质量密度缩放与原生电荷涨落乘子。将 $V_{\mathrm{full}}$ 乘以当前步接受的真实路径长度 $\ell$，即得该步的能损方差 $\sigma_E^2$。上游铜与空气（`BLEM0001`）则在离线提取时直接按 $T_{\max}$ 制表，无需运行时截断转换。

统一涨落引擎依据粒子身份与比动能自动选择物理抽样分支：
1. **Universal 涨落分支（适用于全部能区的 $^1\mathrm{H}$ 以及动能高于切换阈值 $E > (10\text{ MeV}/m_p)\,Z M$ 的 $Z > 1$ 离子）**：在厚吸收层条件（$\overline{\Delta E} \ge 10\,T_{\max}$）下，依据均值标准差比 $\mathrm{SN} = \overline{\Delta E}/\sigma_E$，当 $\mathrm{SN} \ge 2$ 时从 $[0, 2\overline{\Delta E}]$ 上的双侧截断高斯分布抽样，当 $\mathrm{SN} < 2$ 时从形状参数 $k = \mathrm{SN}^2$ 的矩匹配 Gamma 分布抽样。在薄层条件（$\overline{\Delta E} < 10\,T_{\max}$）下，遵循 Urban 双能级原子模型将能损分解为原子激发（$a_1$）与连续电离（$a_3$）两部分。当平均电离碰撞次数 $a_3 > 8$ 时，为保证数值稳定性，将电离能损代数拆分为高斯抽样的密集软碰撞分量与均值受限（$p_3 = 8 a_3 / (8 + a_3) \le 8$）的显式 Poisson 尾部；Poisson 尾部的单次碰撞能损在 $[\alpha E_0, T_{\max}]$ 区间内按 $1/E^2$ 律逐次抽样。
2. **低能 Ion 涨落分支（适用于转换阈值以下的 $Z > 1$ 离子）**：当单步平均能损满足 $\overline{\Delta E} > 0.2 E$ 时，先对离子方差施加步内速度减慢修正；随后依据信噪比 $\mathrm{SN} = \overline{\Delta E}/\sigma_E$ 分别从截断高斯分布（$\mathrm{SN} \ge 2$）、Gamma 分布（$0.1 < \mathrm{SN} < 2$）或 $[0, 2\overline{\Delta E}]$ 均匀分布（$\mathrm{SN} \le 0.1$）中抽取实际能损。

抽样能损受当前可用动能 $E$ 约束；耗尽剩余 CSDA 射程的终止步直接沉积剩余动能。八种配置在患者／水模体及铜准直器中均保留主粒子能损涨落，以描述主束射程歧离与 Bragg 峰展宽。带电次级碎片采用有限路径平均电子能损（$\Delta E = \overline{\Delta E}$），同时继续抽样核反应与多重散射。

### II.B.3. 步长竞争选择、体素边界跨越与轨迹终止

在每个输运步起始处，引擎先计算当前体素组成与密度下的电磁响应，并综合各项物理与几何约束确定步长。设 $f$ 为物种特异的射程步长分数，$r_f$ 为最终射程参数，$\eta$ 为单步允许的最大平均能损分数。当剩余 CSDA 射程满足 $R(E) > r_f$ 时，电磁步长建议值 $\ell_{\mathrm{EM}}$ 由非线性射程导引项与最大能损分数项共同约束：

$$
\ell_{\mathrm{EM}} = \min\!\left[
f R(E) + r_f(1 - f)\!\left(2 - \frac{r_f}{R(E)}\right),\;
R(E) - R\!\left((1 - \eta)E\right)
\right].
\tag{4}
$$

当粒子慢化至 $R(E) \le r_f$ 时，电磁建议步长直接取为剩余射程 $\ell_{\mathrm{EM}} = R(E)$，并在该步结束时终止轨迹。CT 宽束快速输运采用 $\eta = 0.05$，微束患者内输运则采用更严格的 $\eta = 0.01$，以精细解析微束横向高梯度剂量结构。

电磁步长建议 $\ell_{\mathrm{EM}}$ 与区域强制最大步长上限 $\ell_{\max}$（表 1）、体素或材料几何边界距离 $\ell_{\mathrm{geom}}$、多重散射限制及核相互作用随机距离 $\ell_{\mathrm{nuc}}$ 进行竞争。

**表 1.** 各输运区域的引擎强制最大离子步长（$\ell_{\max}$）与主要多重库仑散射（MCS）模型。

| 输运区域 | 最大离子步长 $\ell_{\max}$ | 主要多重库仑散射处理 |
|---|---:|---|
| 水／CT 宽束 | $1.0\text{ mm}$ | 耦合高斯 Fermi–Eyges 模型 |
| CT 微束（患者体积内） | $0.5\text{ mm}$ | 主粒子：数据驱动 Urban；带电次级粒子：分段 Fermi–Eyges（子段上限 $0.1\text{ mm}$） |
| 水微束（水模体积内） | $0.5\text{ mm}$ | 主粒子与带电次级粒子均采用数据驱动 Urban |
| 微束铜狭缝准直器 | $0.05\text{ mm}$ | 铜专用 Urban MCS + 离散核弹性散射 |
| 微束狭缝空气（CT 碳微束预设） | $0.2\text{ mm}$ | 空气专用 Urban MCS |

几何步长与多重散射的耦合方式取决于所采用的散射理论：
- 在 **Fermi–Eyges 输运模式**下，步长直接沿几何直线度量，横向偏移在每个散射子段末端更新。
- 在 **Urban 输运模式**下，算法严格区分微观弯曲真实路径长度 $\ell$ 与几何直线推进弦长 $g$。先在三维欧氏空间计算到体素交界面或准直器狭缝侧壁的几何距离，并将其反演为允许的最大真实路径 $\ell_{\mathrm{geom}}$；随后在真实路径 $\ell = \min(\ell_{\mathrm{EM}}, \ell_{\max}, \ell_{\mathrm{geom}})$ 上计算连续电子能损与核光学深度增量。若离散核碰撞发生在更短的真实路径 $\ell_{\mathrm{nuc}} < \ell$ 处，则将真实步长截断至 $\ell_{\mathrm{nuc}}$，并按缩短后的真实步重新计算几何推进距离与散射角偏转，最后提交粒子状态更新。

连续电离能损计入步起始所在体素，而离散核反应的局部能量沉积计入碰撞顶点所在体素。到达患者或水模体网格外边界的轨迹即刻终止，其剩余动能记入边界逃逸能量账目。总动能降至全局截断阈值 $E_{\mathrm{cut}} = 0.1\text{ MeV}$ 以下的轨迹在介质内原地终止，剩余动能作为零路径末端沉积局部计入吸收剂量，但不计入有限轨迹长度的 LET 累积量。

### II.B.4. Fermi–Eyges 多重库仑散射模型

在宽束配置以及专用 CT 微束配置的患者内带电次级碎片输运中，多重库仑散射采用耦合高斯 Fermi–Eyges 理论 [9] 建模。对于长度为 $L$、局部线性散射功率恒为 $\mathcal{T}$ 的均匀输运子段，粒子在两个正交横向平面上的投影偏转角 $\theta$ 与横向位移 $y$ 服从二元正态分布，其协方差矩阵为：

$$
\boldsymbol{\Sigma}_{\mathrm{FE}} =
\begin{pmatrix}
\langle \theta^2 \rangle & \langle \theta y \rangle \\
\langle \theta y \rangle & \langle y^2 \rangle
\end{pmatrix}
= \mathcal{T}
\begin{pmatrix}
L & L^2/2 \\
L^2/2 & L^3/3
\end{pmatrix}.
\tag{5}
$$

算法在粒子瞬时局部坐标系下通过 Cholesky 分解对两个正交横向轴的 $(\theta_x, x)$ 与 $(\theta_y, y)$ 进行联合抽样，再旋转回全局笛卡尔坐标系，从而精确保持末端横向位移与出射方向余弦之间的相空间相关性。当单个输运步超过设定的散射分段上限（如 CT 微束患者内次级碎片的 $0.1\text{ mm}$ 分段上限）时，自动将该步细分为多个均匀散射子段，并利用已接受的步能损估算各子段中点动能以求取 $\mathcal{T}$。

有效线性散射功率 $\mathcal{T}$ 由 Rossi–Greisen 高斯核心项与标定大角单次散射尾部的等效二阶矩叠加而成：

$$
\mathcal{T}(E) = \left(\frac{E_s Z}{\beta p c}\right)^2 \frac{1}{X_0(\rho)} + \lambda_{\mathrm{tail}}(\rho)\,\left(\frac{E_{\mathrm{tail}} Z}{\beta p c}\right)^2,
\tag{5b}
$$

式中 $p$ 为相对论动量，$X_0(\rho)$ 为由 Schneider 组织组成与体素密度确定的局部辐射长度，$\lambda_{\mathrm{tail}}(\rho) = \lambda_{\mathrm{tail}, \mathrm{water}} \, [X_{0,\mathrm{water}} / X_0(\rho)]$ 按逆辐射长度之比缩放名义尾部散射率。对于轻离子（$^1\mathrm{H}$、$^2\mathrm{H}$、$^3\mathrm{H}$ 与 $^4\mathrm{He}$），有效核心散射能量 $E_s(E/A)$ 及尾部参数由 $50$、$150$ 和 $300\text{ MeV/u}$ 均匀水层 Geant4 标定值分段线性插值给出（区间外取端点值）；对于 $^{12}\mathrm{C}$ 及采用同组标定的重碎片，统一采用标定常数 $E_s = 9.9\text{ MeV}$、水中名义尾部率 $\lambda_{\mathrm{tail},\mathrm{water}} = 0.0025\text{ mm}^{-1}$ 及尾部散射能量 $E_{\mathrm{tail}} = 2.4\text{ MeV}$。强相互作用核弹性散射则作为独立的离散随机过程单独处理（第 II.B.6 节）。

### II.B.5. 宏观核反应截面与弹性—非弹性竞争机制

水与 CT 介质中的离散核反应包含相干核弹性散射（$\mathrm{el}$）与非弹性核碎裂反应（$\mathrm{in}$）。对于反应通道 $c \in \{\mathrm{el}, \mathrm{in}\}$，动能 $E$ 处的宏观线性反应率（单位路径反应概率，$\mathrm{mm}^{-1}$）由局部组织各靶元素 $t$ 的原子数密度 $n_t(\rho)$ 与微观截面 $\sigma_{c,t}(E)$ 求和得到：

$$
\Sigma_c(E; \rho) = \sum_{t} n_t(\rho)\,\sigma_{c,t}(E) = \rho \sum_{t} \left(\frac{\Sigma_{c,t}}{\rho}\right)_{\!\mathrm{tab}}\!(E), \qquad
P(t \mid c, E) = \frac{n_t(\rho)\,\sigma_{c,t}(E)}{\Sigma_c(E; \rho)},
\tag{6}
$$

其中 $(\Sigma_{c,t}/\rho)_{\mathrm{tab}}$ 为预计算的组成相关质量反应率表（在 $0.1$–$460.1\text{ MeV/u}$ 区间内按 $0.5\text{ MeV/u}$ 均匀划分为 921 个能量节点，覆盖全部 25 种 Schneider 组织与水），$P(t \mid c, E)$ 为给定通道 $c$ 时与靶元素 $t$ 发生反应的条件概率。非弹性分靶反应率严格掩膜限制在二进制截面表声明的有效入射粒子—靶核 $(Z, A, t)$ 能量域内。

水与患者介质中的主粒子及带电次级碎片采用统一的无量纲核光学深度（optical depth）$\tau$ 追踪算法：

$$
\tau = -\ln U, \qquad \Sigma_{\mathrm{tot}}(E; \rho) = \Sigma_{\mathrm{el}}(E; \rho) + \Sigma_{\mathrm{in}}(E; \rho),
\tag{7}
$$

式中 $U \sim \mathcal{U}(0, 1)$ 为每条轨迹独立抽取的均匀随机数。在真实路径长度为 $\ell$ 的候选步上，光学深度消耗量为 $\Delta\tau = \Sigma_{\mathrm{tot}}\,\ell$。若 $\Delta\tau < \tau$，则本步不发生核反应，剩余光学深度更新为 $\tau \leftarrow \tau - \Delta\tau$ 并跨体素边界和内核续跑阶段持续保留；若 $\Delta\tau \ge \tau$，则将真实步长截断至核反应位置 $\ell_{\mathrm{nuc}} = \tau / \Sigma_{\mathrm{tot}}$ 并重新抽样新的光学深度。到达碰撞顶点后，先按相对概率 $\Sigma_{\mathrm{el}}/\Sigma_{\mathrm{tot}}$ 与 $\Sigma_{\mathrm{in}}/\Sigma_{\mathrm{tot}}$ 选择弹性或非弹性通道，再根据对应通道的分靶概率分布 $P(t \mid c, E)$ 抽样具体靶核 $t$。

为减少主粒子输运期间对全局内存截面表的冗余查询，主粒子非弹性通道引入了逐轨迹上界反应率缓存（majorant rate cache）$\widehat{\Sigma}_{\mathrm{in}}$。当粒子跨越材料或密度边界、动能降至缓存求值能量的 $80\%$ 以下，或刚刚完成一次候选核反应判定后，自动刷新该缓存。在候选非弹性碰撞点处，计算连续慢化后的真实非弹性率 $\Sigma_{\mathrm{in}}(E_{\mathrm{post}})$，并以接受率 $\Sigma_{\mathrm{in}}(E_{\mathrm{post}}) / \widehat{\Sigma}_{\mathrm{in}}$ 执行拒绝抽样；被拒绝的候选事件视为虚拟空碰撞（null collision），原主粒子保持慢化后状态继续输运。

### II.B.6. 核弹性散射与靶核反冲输运

八种标准配置均开启核弹性散射。当在比动能 $e = E/A$ 处选定弹性散射通道及靶元素 $t$ 后，算法确定包含 $e$ 的相邻能量节点 $[e_0, e_1]$，并按两节点对当前弹性率的贡献权重随机选择一个节点。随后从该节点预存的 512 个弹性样本中均匀抽取一条记录，获取具体靶同位素质量 $m_t$ 及归一化四动量转移平方 $q^2 / q_{\max}^2 = \tfrac{1}{2}(1 - \cos\theta_*)$（其中 $\theta_*$ 为质心系极角散射角）。结合均匀抽样的方位角 $\phi \sim \mathcal{U}(0, 2\pi)$，利用自然单位制（$c = 1$）下的精确相对论二体运动学求解实验室系反冲核动能 $E_R$ 与散射主粒子出射动能 $E'_p$：

$$
s = (m_p + m_t)^2 + 2 m_t E, \qquad
p_* = \frac{m_t \sqrt{E(E + 2 m_p)}}{\sqrt{s}}, \qquad
E_R = \frac{p_*^2 (1 - \cos\theta_*)}{m_t}, \qquad
E'_p = E - E_R,
\tag{8}
$$

式中 $m_p$ 为入射粒子静止质量（严格遵循各物理包的标定质量约定，例如适用路径中 $^{12}\mathrm{C}$ 采用的 $12\text{ u}$ 约定），$p_*$ 为质心系三维动量模长。散射离子与反冲靶核的出射动量向量由当前入射方向坐标系旋转至全局笛卡尔坐标系。

若反冲核身份 $(Z_R, A_R)$ 属于常规 18 物种带电离子注册表，则直接压入次级粒子队列，参与完整的电磁慢化、多重散射与核反应输运。对于常规注册表以外的重靶反冲同位素（如组织中的 $^{12}\mathrm{C}$–$^{40}\mathrm{Ca}$ 或 $^{16}\mathrm{O}$ 反冲核），引擎调用专属的全离子反冲阻止本领包（`ELRSP001`），采用非受限电子加核总阻止本领 $S_{\mathrm{tot}}(E) = S_e(E) + S_n(E)$ 及其派生 CSDA 射程进行连续慢化输运；该组反冲核采用连续慢化模型。在计算剂量平均电子 LET（$L_d$）时，这类重反冲核查询配套的纯电子阻止本领表（`LERSP001`），从而保证核阻止本领 $S_n(E)$ 不混入电子 LET 的分子与分母。初生动能 $E_R \le E_{\mathrm{cut}} = 0.1\text{ MeV}$ 的反冲核直接在碰撞顶点局部沉积能量。

### II.B.7. 核非弹性相关末态抽样与带电碎片输运

非弹性核反应采用离线生成的 Geant4 核内级联相关末态事件库（`CINPKG04`）进行整事件抽样，将多体末态关联保留在适于 GPU 查询的数据表示中。事件库按精确入射物种 $(Z, A)$、靶元素 $t$ 及离散单核子入射能量节点 $e_k$ 索引：其中质子主粒子库由 Geant4 Binary Cascade 模型（`QGSP_BIC_HP`）提取，$^{12}\mathrm{C}$ 主粒子库由 Liège 核内级联模型（`FTFP_INCLXX`）提取，患者内次级碎片库则基于 `G4IonBinaryCascadePhysics` 与 `G4EmStandardPhysics_option4` 联合提取。

当比动能为 $e = E/A$ 的粒子发生非弹性反应且位于有效通道域内的相邻能量节点 $[e_0, e_1]$ 之间（最大允许节点间距为 $5\text{ MeV/u}$）时，算法按线性随机插值概率选择上节点 $e_1$：

$$
P(e_1 \mid e) = \frac{e - e_0}{e_1 - e_0}, \qquad P(e_0 \mid e) = 1 - P(e_1 \mid e),
\tag{9}
$$

随后从所选节点的事件池中均匀抽取一个完整核反应末态事件，并保留其记录的产物能量与动量关联。式（9）使所选节点入射能量的期望等于查询能量 $e$。节点选择造成的入射能量偏移及反应 Q 值残差分别记入能量账目。

抽中事件后，先对整个事件的全部出射产物统一施加同一个随机方位角旋转 $\Phi \sim \mathcal{U}(0, 2\pi)$，再将所有产物方向向量从入射粒子局部坐标系旋转至世界坐标系。这一整体旋转机制完整保留了单次碎裂事件内的产物多重性（multiplicity）及能量—角度联合关联。事件记录的过程局部沉积能量直接计入反应顶点体素；动能高于截断阈值 $E_{\mathrm{cut}} = 0.1\text{ MeV}$ 的受支持带电产物则携带其精确 $(Z, A)$、动能、方向向量、代数计数 $g \leftarrow g + 1$ 及确定性子代随机数身份进入次级输运队列；低于截断阈值的带电产物直接局部沉积。

水与患者 CT 中的次级非弹性级联采用最大代数 $g_{\max} = 2$。达到该上限或按物种策略关闭非弹性分支的带电碎片继续执行连续慢化、多重散射及允许的核弹性散射，直至停止或逃逸。$^6\mathrm{Be}$ 残余核在产生时终止，其动能单独记为未追踪能量。中子与退激 $\gamma$ 光子等中性核产物不继续输运，其动能及未追踪反应余额分别计入能量账目，且不计入患者剂量。

---

## II.C. 多狭缝铜准直微束（Minibeam）的扩展建模

### II.C.1. 多狭缝准直器几何与上游相空间显式生成

微束配置在宽束患者输运框架上游引入了铜多狭缝准直器、狭缝内空气通道及准直器后至患者体表的空气间隙。几何模型精确定义了铜准直器的外廓尺寸、沿束流方向厚度、狭缝数量、狭缝开口宽度、中心节距（pitch）、狭缝长度、阵列横向相位偏移、旋转角，以及源平面、准直器出口、CT 包围盒、患者皮肤表面与等中心之间的相对轴向坐标。准直器固定在射野坐标系中，不随扫描束斑平移。

全计划临床对比实现了两套微束准直器设计：
1. **质子微束准直器**：采用 $60\text{ mm}$ 厚铜块，包含 $15$ 条平行狭缝，单缝宽度 $0.5\text{ mm}$、中心节距 $3.6\text{ mm}$、狭缝长度 $50\text{ mm}$。
2. **碳离子微束准直器（RT07575 头部计划）**：采用 $100\text{ mm}$ 厚铜块，包含 $17$ 条狭缝，单缝宽度 $0.7\text{ mm}$、中心节距 $3.0\text{ mm}$、狭缝长度 $100\text{ mm}$。针对 RT07575 计划中 $90^\circ$ 与 $270^\circ$ 对穿交火射野，两个射野保留各自独立的狭缝阵列定位，其中 $270^\circ$ 射野的名义中心狭缝阵列相对于 $90^\circ$ 射野设置了半节距（$1.5\text{ mm}$）的横向相位交错。

MAIGO 从准直器上游入口逐历史输运主粒子及带电核产物，经过空气狭缝、铜隔板和下游空气间隙生成患者入口相空间。入口分布同时包含直接穿缝主粒子、经铜侧壁散射出射的主粒子，以及铜与上游空气中产生的带电核碎片，由此将准直器引起的束流调制和次级粒子成分共同传递至患者计算。在铜内停止或被吸收的原始历史同样计入总入射历史数 $N$，用于绝对剂量归一化。

### II.C.2. 数据驱动 Urban 多重库仑散射与真实—几何路径转换

由于微束峰谷剂量比（PVDR）与半影锐度对狭缝边缘处的单次及多次大角散射尾部高度敏感，MAIGO 在所有微束上游铜／空气主粒子输运及患者内主粒子输运中，实现了改编自 Geant4 `G4UrbanMscModel` [8] 的数据驱动 Urban 多重库仑散射算法。在模体／患者体积内部，水微束预设与通用微束管线的带电次级粒子同样采用 Urban 散射；而专用质子与碳离子 CT 微束管线则采用混合散射策略：患者内主粒子采用 Urban 散射（外层步长上限 $0.5\text{ mm}$），患者介质内的带电次级粒子（含从上游产生并进入患者 CT 的带电碎片）则采用共享的分段 Fermi–Eyges 散射算子（外层步长上限 $0.5\text{ mm}$，独立散射分段上限在发布预设中为 $0.1\text{ mm}$）。

离线提取的 Urban 物理包（`ct_copper.bin` 与 `air.bin`）按材料与粒子物种存储了第一输运自由程 $\lambda_{\mathrm{tr},1}(E)$、辐射长度 $X_0$ 及角分布形状参数（在水、25 种 Schneider 组织及铜中覆盖 52 种带电离子，在空气中覆盖质子与 $^{12}\mathrm{C}$）。Urban 输运通过真实路径长度 $\ell$ 与几何推进弦长 $g$ 的映射耦合散射和几何推进。在局部输运自由程 $\lambda_{\mathrm{tr},1}$ 近似恒定的条件下，正向路径转换及其精确逆映射表示为：

$$
g(\ell) = \lambda_{\mathrm{tr},1}\left(1 - e^{-\ell/\lambda_{\mathrm{tr},1}}\right), \qquad
\ell(g) = -\lambda_{\mathrm{tr},1}\ln\!\left(1 - \frac{g}{\lambda_{\mathrm{tr},1}}\right),
\tag{10}
$$

并同时包含输运自由程随能量显著变化时的 Geant4 原生能量相关修正分支。当粒子靠近铜—空气狭缝侧壁、准直器出口或体素交界面（欧氏距离为 $d_{\mathrm{geom}}$）时，先将几何推进距离限制为 $g \le d_{\mathrm{geom}}$，再按式（10）反演出允许的最大真实路径 $\ell_{\mathrm{geom}}$。连续电子能损与核光学深度均按真实路径 $\ell$ 计算，使边界推进与路径积分采用同一散射几何表示。

在真实步长最终确定后（已计入可能的离散核碰撞步长缩短），从复合 Urban 角分布中抽取极角散射余弦 $\cos\theta$——该分布将小角高斯核心与非高斯幂律单次散射尾部相融合，其角度展宽依赖于归一化散射厚度 $\ell / \lambda_{\mathrm{tr},1}$、局部辐射长度 $X_0$ 及步前／步后动能均值。耦合的横向位移在粒子局部坐标系中抽样并受几何安全半径约束，随后变换至世界坐标系。边界跨越标记与射程步长调节状态跨输运步及阶段调度交接持续保留。

### II.C.3. 铜与空气中的非受限电子能损与涨落

穿行于铜准直器与上游空气中的带电粒子查询专属的 `BLEM0001` 束线电磁数据包，该包提供覆盖铜核反应全部带电碎片物种的非受限电子阻止本领 $S_e(E)$ 及在运动学上限 $T_{\max}$ 处直接提取的两类原生涨落方差。当试探线性能损不超过当前动能的 $2\%$（即 $S_e(E)\,\ell \le 0.02 E$）时，平均能损直接按步首阻止本领线性计算；超过 $2\%$ 时则调用同一响应构造的 CSDA 射程反演（式 1b），最低能量节点以下采用规定的 $\sqrt{E}$ 阻止本领延拓积分。

铜与空气中的主粒子采用第 II.B.2 节的共享 Universal／Ion 涨落引擎，使用原生粒子质量、电荷、材料激发能及 $T_{\max}$ 参数；带电次级碎片采用有限路径平均电子能损。上游电子能损按局部沉积近似处理，记入束线能量账目，并与患者计分网格内的剂量分开。铜输运步长上限为 $0.05\text{ mm}$，CT 碳微束狭缝空气中的上限为 $0.2\text{ mm}$。

### II.C.4. 铜准直器中的核相互作用与次级碎片级联

四种微束预设均显式开启铜核弹性散射与铜核非弹性碎裂。主粒子在铜中的核碰撞距离由铜弹性与非弹性总宏观反应率 $\Sigma_{\mathrm{tot},\mathrm{Cu}} = \Sigma_{\mathrm{el},\mathrm{Cu}} + \Sigma_{\mathrm{in},\mathrm{Cu}}$ 抽样，经电子连续慢化后在候选碰撞点按实际分通道反应率判定反应类型。若发生铜核弹性散射，从铜同位素角分布样本中抽取动量转移并应用第 II.B.6 节的相对论两体运动学（式 8）：出射束流离子继续输运，而转移给重铜靶核的反冲动能直接记入上游束线能量账目，不在患者剂量中计分。

铜非弹性核碰撞从铜专用相关末态事件库（`cascade.cinpkg`）中按精确入射粒子 $(Z, A)$ 与单核子能量 $e = E/A$ 抽样。反应产生的直接带电产物及记录的带电续行产物共享同一个事件随机方位角旋转，并继续穿过剩余铜厚度与下游空气间隙。铜准直器内部的后续次级核非弹性反应受独立的上游代数上限约束（标准微束配置设为 $g_{\max,\mathrm{Cu}} = 3$）；达到代数上限后停止进一步非弹性分支，存活粒子继续执行电磁慢化与弹性散射。铜内局部顶点沉积、中性核产物及剩余未追踪反应能量统一保留在束线能量平衡账目内。

上游铜碎片在进入患者前的散射输运按入射主粒子管线分别配置：
- **碳离子微束**：铜内产生的带电碎片采用独立提取的分物种铜 Urban 多重散射表与分物种铜核弹性散射数据（`minibeam_copper_fragment_native_transport`）。
- **质子微束**：上游铜产生的次级带电碎片（如氘、氚、$\alpha$ 粒子等）采用基于 Highland 公式的高斯多重散射角度更新，并结合非受限电子阻止本领慢化。

穿过准直器与空气间隙到达患者入口平面的所有存活主粒子及上游带电核产物，将其完整的位置—方向—能量—物种联合相空间状态 $(\mathbf{r}, \mathbf{u}, E, Z, A)$、代数标记及确定性随机数身份无缝交接至下游患者输运阶段。

---

## II.D. GPU 并行实现、内存调度与物理量计分

### II.D.1. SYCL/CUDA 执行管线、队列重排压缩与计数器式随机数

MAIGO 采用 C++20 与 Khronos SYCL 实现，通过 oneAPI Unified Runtime CUDA 后端支持 NVIDIA GPU，通过 Level Zero 后端支持 Intel Xe GPU。设备内核预先编译为目标架构的原生机器码，运行时关闭即时编译。质子宽束、碳离子宽束、质子微束和碳离子微束采用四条编译期特化主粒子管线，并为自定义离子配置提供通用管线；各管线共享支持多物种带电核碎片的次级输运内核。

针对微束模拟中因狭缝准直器吸收、多代核反应分支及轨迹寿命差异导致的 GPU 线程束分化（warp/subgroup divergence），微束管线将粒子输运解耦为通过预分配全局内存队列衔接的多个顺序执行阶段：
1. **上游束线主粒子阶段**：追踪从束流源出发穿过铜狭缝准直器与空气间隙的主粒子，将到达患者入口的存活主粒子写入入口检查点队列，并将铜中产生的次级带电产物推入独立的上游子代队列。
2. **上游铜碎片级联阶段**：在铜准直器与空气中继续追踪最多三代铜带电核碎片。
3. **患者内主粒子输运阶段**：从患者入口恢复存活主粒子并在体素化水或 CT 网格中执行慢化与核反应输运；在适用配置下采用基于计数排序（counting-sort）的槽位重排算法（`bucket_order`）按空间／材料局部性重排活跃轨迹索引，以提高物理表查询的局部性。
4. **患者内次级碎片阶段**：以固定步数分段（`kSecondarySegmentSteps`）推进所有患者内产生及上游传入的带电次级粒子，并在活跃轨迹数低于尾部阈值时执行队列压缩以回收空闲线程。

设备队列在内核启动前预分配，硬件原子计数器记录各阶段的队列占用与完成情况。

设备端采用计数器式 Philox-4x32-10 伪随机数发生器。随机抽样由批次种子、束斑索引、主历史编号、子代轨迹身份及单调递增的计数器确定，使每条轨迹的随机数流独立于线程调度和队列重排。连续批次复用 GPU 上下文及常驻显存的材料与物理表，将初始化和物理表上传的成本分摊至整次计算。

### II.D.2. 最大余数法整数束斑分配与绝对剂量归一化

设单次照射中第 $j \in \{1, \dots, M\}$ 个扫描束斑的物理粒子权重为 $W_j > 0$，全计划总物理入射粒子数为 $W = \sum_{j=1}^M W_j$。对于总模拟历史数为 $N$ 的蒙特卡罗计算，各束斑的整数模拟粒子数 $n_j$ 按最大余数法（largest-remainder method）由 $N W_j / W$ 严格分配：

$$
n_j = \left\lfloor \frac{N W_j}{W} \right\rfloor + \mathbb{I}\!\left(j \in \mathcal{R}_{N - \sum_k \lfloor N W_k / W \rfloor}\right), \qquad \sum_{j=1}^M n_j = N,
\tag{11a}
$$

其中 $\mathcal{R}_K$ 为小数余数最大的前 $K$ 个束斑下标集合。该方法保证全计划整数历史数之和为 $N$，并将余数分配给距下一整数最近的束斑。全局分配随后拆分至各 GPU 和统计批次；配对 TOPAS 比较采用参考计算的整数束斑粒子数清单。

粒子输运过程中，体素沉积能量 $\Delta E$（单位 $\mathrm{MeV}$）通过 GPU 32 位单精度浮点数（`FP32`）原子加操作累加至显存数组。设第 $i$ 个体素的质量密度为 $\rho_i$（$\mathrm{g/cm^3}$）、体积为 $V_i$（$\mathrm{mm^3}$）、累计沉积能量为 $E_i$（$\mathrm{MeV}$），则体素质量 $m_i$（$\mathrm{kg}$）及绝对介质吸收剂量 $D_i$（$\mathrm{Gy}$）由下式给出：

$$
m_i = \rho_i V_i \times 10^{-6}\text{ kg}, \qquad
D_i = \frac{E_i \times 1.602176634 \times 10^{-13}\text{ J/MeV}}{m_i}\,\frac{W}{N}.
\tag{11b}
$$

当整次模拟拆分为 $B$ 个统计批次或多张 GPU 卡（第 $b$ 批包含 $N_b$ 个历史，$\sum_{b=1}^B N_b = N$）时，每个批次输出均统一乘以全局归一化因子 $W / N$；因此在主机端以双精度（FP64）合并批次文件时，只需对各批次剂量直接求和 $D_i = \sum_{b=1}^B d_{i,b}$。

### II.D.3. 剂量平均电子 LET 的轨迹长度估计量

剂量平均电子线性能量传递（$L_d$）采用基于非受限电子阻止本领的轨迹长度估计量（track-length estimator）与吸收剂量同步计分。对于体素 $i$ 内真实路径长度为 $\ell_k > 0$ 的第 $k$ 段有限长带电离子输运步，令 $S_{e,k}$ 表示在步前动能与当前体素实际密度 $\rho_i$ 处求值的非受限线性电子阻止本领（单位 $\mathrm{MeV/mm}$，数值上等价于 $\mathrm{keV/\mu m}$）。引擎在每个体素内同步累积两个可加矩 $A_i$ 与 $B_i$，其比值即为体素剂量平均电子 LET $L_{d,i}$：

$$
A_i = \sum_{k \in i} S_{e,k}^{\,2}\,\ell_k, \qquad
B_i = \sum_{k \in i} S_{e,k}\,\ell_k, \qquad
L_{d,i} = \frac{A_i}{B_i}.
\tag{12}
$$

式（12）的权重为期望电子能损 $S_{e,k}\,\ell_k$，使 LET 计分采用与离子输运一致的阻止响应。$S_{e,k}$ 与 $L_{d,i}$ 均为实际体素密度下的线性阻止量，单位为 $\mathrm{keV/\mu m}$。常规注册表内的离子查询 `continuous26` 非受限电子阻止表，额外重靶反冲核查询 `LERSP001` 纯电子阻止表；LET 的两个矩均按电子阻止本领构造。

水模体或患者 CT 计分网格内所有总动能高于 $0.1\text{ MeV}$ 截断阈值的已输运带电离子（包括主粒子、次级核碎片及显式输运的反冲核）均对 $A_i$ 和 $B_i$ 贡献。上游铜准直器与空气中的轨迹段、核反应顶点局部能量沉积、零路径截断终止沉积以及未追踪中性产物能量均不计入 LET 累积量。两个矩 $A_i$ 与 $B_i$ 在 GPU 端分别通过 `FP32` 原子加操作累积并独立保存至文件，在跨射野、统计批次及多 GPU 设备按双精度完成全局求和后，最终作一次除法求得 $L_{d,i}$。分母 $B_i = 0$ 表示该体素无有限长带电轨迹穿过，在后续统计分析中保留其零分母标识。

---

## II.E. 临床全计划参考算例与基准验证设计

### II.E.1. 临床病例队列、体素网格与模拟历史数

临床全计划评价采用两例头部 CT——**RT07575**（Head 1）和 **RT06423**（Head 2）——以及一例肺部 CT——**20022516**（Lung 1）。头部病例评价复杂解剖结构中的计划剂量一致性，肺部病例评价低密度异质介质中的输运。宽束与铜准直微束、质子与碳离子的配对计算历史数列于表 2。各组 MAIGO 与 TOPAS 计算共享 CT 网格、Schneider 材料映射、束流光学、束斑位置与能量、最终权重、摆位、准直器几何及绝对剂量归一化。

**表 2.** 各临床病例与照射模式下全计划剂量验证的模拟入射粒子历史数（$N$）。

| 临床病例编号 | 解剖部位 | 质子宽束 | 碳-12 宽束 | 质子微束 | 碳-12 微束 |
|---|---|---:|---:|---:|---:|
| **RT07575** | 头颈部 1（Head 1） | $250,626,500$ | $129,638,170$ | $1,000,000,000$ | $1,000,000,000$ |
| **RT06423** | 头颈部 2（Head 2） | $124,266,340$ | $151,087,660$ | $1,000,000,000$ | $1,000,000,000$ |
| **20022516** | 胸部／肺部（Lung 1） | $361,050,622$ | $177,173,040$ | $3,000,000,000$ | —$^{\mathrm{a}}$ |

$^{\mathrm{a}}$— 表示本表未报告该病例／模态的配对数据。默认 CT 碳微束配置对应独立的肺部基准模体。

全计划剂量网格的体素尺寸为 $0.5 \times 0.5 \times 2.0\text{ mm}^3$，网格维数、物理原点和轴顺序由各计划的参考头文件定义。三维剂量与 LET 在共同的患者物理坐标、原生网格及计划绝对剂量尺度下逐体素比较。

### II.E.2. 独立 TOPAS 参考模拟与物理模型对齐

TOPAS 参考剂量由 OpenTOPAS 4.2.3／Geant4 11.3.2 [7,8] 的 `DoseToMedium` 计分器前向计算。物理列表包含标准电磁过程（`g4em-standard_opt4`）、强子与离子非弹性反应（`g4h-phy_QGSP_BIC_HP`、`g4ion-inclxx`）、强子弹性散射（`g4h-elastic_HP`）、停止过程（`g4stopping`）及衰变过程（`g4decay`、`g4radioactivedecay`）。微束参考通过 `AllChargedUrbanPhysics` 为带电粒子配置 Urban 散射；碳离子参考通过封装 `G4IonElasticPhysics` 的 `CarbonIonElasticPhysics` 为 `GenericIon` 注册核弹性过程。

各计划的电磁表能量上限、次级粒子产生截断及步长设置随参考输入记录。肺部质子微束参考采用 $0.05\text{ mm}$ 生产截断与 $10\text{ GeV}$ 电磁表上限，三份碳离子宽束参考采用 $600\text{ MeV}$ 电磁表上限。参考剂量由最终计划权重驱动的独立 TOPAS 全计划输运生成，TOPAS 显式输运超过生产截断的次级电子与光子。MAIGO 的局部电离沉积、次级平均能损、核末态抽样和中性产物处理按第 II.B–C 节定义。

### II.E.3. TOPAS 剂量平均 LET 双矩参考计分

TOPAS 参考采用与式（12）一致的双矩 LET 计分器，分别累积 $A_i = \sum_k w_k S_{e,k}^2 \ell_k$ 与 $B_i = \sum_k w_k S_{e,k} \ell_k$，其中 $w_k$ 为轨迹统计权重，计分阈值为 $0.1\text{ MeV}$ 总动能。计分器查询预先提取的只读非受限电子阻止表；该表按参考模拟的 Geant4 11.3.2 物理列表、粒子种类、Schneider 组织组成、密度和平均激发能生成，使两个引擎采用共同的 LET 定义与计分终点。

为解析 Bragg 峰附近的阻止本领梯度，碳离子宽束 TOPAS LET 参考对轻离子与 `GenericIon` 采用 `dRoverRange = 0.005` 和 `finalRange = 0.1 nm`，质子保持其参考步长控制。RT07575 的单复本配对 LET 比较采用 $25,927,634$ 个入射历史，MAIGO 与 TOPAS 使用相同的整数束斑分配。独立复本的 LET 统计不确定度按第 II.F.4 节估计。

---

## II.F. 定量评价指标、统计不确定度与计算性能分析

### II.F.1. 三维局部 Gamma 通过率与剂量学指标

采用以 TOPAS 参考剂量 $D_{\mathrm{ref}}(\mathbf{r})$ 为基准的三维局部（local）Gamma 指数 [10] 定量评价 GPU 计算剂量分布 $D_{\mathrm{GPU}}(\mathbf{r}')$ 的空间与剂量一致性：

$$
\gamma(\mathbf{r}) = \min_{\mathbf{r}'} \sqrt{
\frac{\|\mathbf{r}' - \mathbf{r}\|^2}{\Delta r^2} +
\frac{\left[D_{\mathrm{GPU}}(\mathbf{r}') - D_{\mathrm{ref}}(\mathbf{r})\right]^2}{\left[\delta D_{\mathrm{ref}}(\mathbf{r})\right]^2}
}, \qquad
\delta D_{\mathrm{ref}}(\mathbf{r}) = \left(\frac{\Delta D_{\%}}{100}\right) D_{\mathrm{ref}}(\mathbf{r}),
\tag{13}
$$

式中 $\Delta r$ 为距离一致性（DTA）容差，$\Delta D_{\%}$ 为相对于参考点局部剂量的百分比容差。GPU 剂量场通过三线性插值在参考点周围半径 $\Delta r$ 的球域内搜索，以计算亚体素距离容差下的 Gamma 指数。参考评价点为 `BODY` 轮廓内超过剂量阈值的体素；搜索时采用完整 GPU 剂量场，包括 `BODY` 边界外的邻近体素。表 3 列出评价域、局部判据及搜索步长。

**表 3.** 宽束与微束模式的三维局部 Gamma 评价域、验收判据及亚体素空间搜索步长。

| 照射模式 | `BODY` 内参考剂量评价域阈值 | 报告的局部 Gamma 判据（$\Delta D_{\%}/\Delta r$） | 空间插值搜索步长 |
|---|---|---|---:|
| **宽束（Broadbeam）** | $D_{\mathrm{ref}}(\mathbf{r}) > 0.10\, D_{\mathrm{ref},\max}$ | $3\%/3\text{ mm}$、$1\%/1\text{ mm}$、$3\%/0\text{ mm}$ | $0.10\text{ mm}$ |
| **微束（Minibeam）** | $D_{\mathrm{ref}}(\mathbf{r}) > 0.4\text{ Gy}$（$4\text{ Gy}$ 处方的 $10\%$） | $3\%/0.2\text{ mm}$、$3\%/0.1\text{ mm}$、$3\%/0\text{ mm}$ | $0.02\text{ mm}$ |

评价点满足 $\gamma \le 1$ 时通过。$3\%/0\text{ mm}$ 单独定义为同位置局部剂量差检验 $|D_{\mathrm{GPU}}-D_{\mathrm{ref}}|/D_{\mathrm{ref}} \le 0.03$。宽束以局部 $1\%/1\text{ mm}$ 通过率 $>99\%$ 为预定验收标准。微束采用 $3\%/0.2\text{ mm}$ 与 $3\%/0.1\text{ mm}$ 判据及 $0.02\text{ mm}$ 搜索步长，使空间容差与亚毫米狭缝尺度相适应。不同深度的横向剂量剖面评价峰宽、谷值与 PVDR，深度剂量剖面评价远端跌落，DVH 指标评价靶区与 OAR 剂量；同位置剂量差分布与批次不确定度共同表征逐体素一致性。

### II.F.2. 三维 LET Gamma 评价与一维空间剖面分析

剂量平均电子 LET（$L_d$）的三维局部 Gamma 评价将式（13）中的剂量替换为 $L_{d,\mathrm{GPU}}(\mathbf{r}')$ 与 $L_{d,\mathrm{ref}}(\mathbf{r})$，局部容差定义为 $\delta L_{d,\mathrm{ref}}(\mathbf{r}) = (\Delta L_{\%}/100)\,L_{d,\mathrm{ref}}(\mathbf{r})$。评价域取为 `BODY` 轮廓与参考剂量掩膜 $D_{\mathrm{ref}}(\mathbf{r}) > 0.10\,D_{\mathrm{ref},\max}$ 的交集。宽束 LET 比较报告 $3\%/3\text{ mm}$、$1\%/1\text{ mm}$ 及 $3\%/0\text{ mm}$ 局部通过率（非零距离判据采用 $0.10\text{ mm}$ 搜索步长对 GPU LET 场作三线性插值）。参考剂量掩膜内若出现 GPU LET 分母为零（$B_i = 0$）的体素，在 Gamma 统计中严格保留为未通过点。

LET 空间分析采用固定几何深度的横向剖面与沿束轴的深度剖面。RT07575 碳离子宽束的深度从中心射线进入 `BODY` 的位置起算，按原生 $0.5\text{ mm}$ 间距取样。中心射线剖面表征局部 LET 结构，$2.5 \times 10\text{ mm}^2$ 横向 ROI 汇总剖面表征区域深度趋势。每个深度切片的 ROI LET 按 $\sum_{i\in\mathrm{ROI}}A_i/\sum_{i\in\mathrm{ROI}}B_i$ 计算。

### II.F.3. 吸收剂量的统计不确定度估计

GPU 端逐体素剂量统计不确定度由 $B$ 个统计独立、束斑源比例匹配的全计划批次估计。对于 $B$ 个等统计量批次，设第 $b$ 批已按全局归一化因子 $W/N$ 缩放的体素剂量贡献为 $d_{i,b}$，总剂量为 $D_i = \sum_{b=1}^B d_{i,b}$，则 $D_i$ 的无偏样本方差估计为：

$$
\widehat{\sigma}_{D_i}^{\,2} = \frac{B}{B - 1} \left[ \sum_{b=1}^B d_{i,b}^{\,2} - \frac{D_i^{\,2}}{B} \right].
\tag{14}
$$

当跨设备吞吐量均衡分配导致各批次历史数 $N_b$ 存在轻微差异时，定义批次历史数权重 $a_b = N_b / N$、全计划等效批次剂量 $Y_{i,b} = d_{i,b} / a_b$ 及权重平方和 $q = \sum_{b=1}^B a_b^2$，此时采用加权批次方差估计量：

$$
\widehat{\sigma}_{D_i}^{\,2} = \frac{q}{1 - q} \left[ \sum_{b=1}^B a_b Y_{i,b}^{\,2} - D_i^{\,2} \right],
\tag{15}
$$

当各批次等大小时（$a_b = 1/B$），式（15）精确退化为式（14）。

TOPAS 不确定度采用逐历史计分方差或匹配先导模拟估计。逐历史方差按归一化因子换算至计划剂量尺度；三份碳离子宽束参考采用 $N_{\mathrm{pilot}}=300,000$ 个历史的先导计算，保持参考几何、物理列表、能谱和相对束斑权重，并按 $\widehat{\sigma}_{D_{\mathrm{ref},i}}=\widehat{\sigma}_{\mathrm{pilot},i}\sqrt{N_{\mathrm{pilot}}/N_{\mathrm{full}}}$ 外推至完整历史数。独立复本间方差用于交叉核验。

在给定的固定评价域 $\mathcal{R}$（包含 $|\mathcal{R}|$ 个体素）内，整体蒙卡统计精度通过相对标准误的均方根（RMS）表征：

$$
u_{\mathrm{RMS}} = 100 \sqrt{ \frac{1}{|\mathcal{R}|} \sum_{i \in \mathcal{R}} \frac{\widehat{\sigma}_{D_i}^{\,2}}{D_{\mathrm{ref},i}^{\,2}} } \;\; (\%),
\tag{16}
$$

并同时报告体素相对标准误的分位数及低于指定误差限（如 $1\%$ 或 $2\%$）的体素比例。鉴于 GPU 与 TOPAS 模拟彼此统计独立，逐体素剂量差 $\Delta D_i = D_{\mathrm{GPU},i} - D_{\mathrm{ref},i}$ 的随机方差由 $\widehat{\sigma}_{\Delta D_i}^{\,2} = \widehat{\sigma}_{D_{\mathrm{GPU},i}}^{\,2} + \widehat{\sigma}_{D_{\mathrm{ref},i}}^{\,2}$ 合成。

### II.F.4. 剂量平均 LET 的比值方差估计

由于剂量平均 LET $L_{d,i} = A_i / B_i$ 是由同一批粒子轨迹同步累积的两个相关随机变量之比，其统计误差评估保留分子与分母之间的协方差 $\mathrm{Cov}(A_i, B_i)$。为在主计算总统计量 $N$ 下估计 GPU LET 统计误差，辅助 GPU 计算将总历史数 $N$ 拆分为 $K$ 个统计独立、近等统计量且源比例匹配的全计划分组（碳离子宽束基准取 $K = 20$）。设第 $b \in \{1, \dots, K\}$ 组在体素 $i$ 累积的两个矩分别为 $A_{i,b}$ 与 $B_{i,b}$，总估计量为 $L_{d,i} = \sum_{b=1}^K A_{i,b} \big/ \sum_{b=1}^K B_{i,b}$，则基于一阶 Taylor 展开（Delta 方法）的分组比值方差估计量为：

$$
\widehat{\sigma}_{L_{d,i}}^{\,2} = \frac{K}{K - 1} \, \frac{\sum_{b=1}^K \left( A_{i,b} - L_{d,i}\,B_{i,b} \right)^2}{\left( \sum_{b=1}^K B_{i,b} \right)^2}.
\tag{17}
$$

当 TOPAS 参考模拟同样包含多个保存了配对分子／分母矩的独立复本时，按式（17）同等计算参考 LET 的逐体素标准误。在固定参考剂量掩膜 $D_{\mathrm{ref}} > 0.10\,D_{\mathrm{ref},\max}$ 内，分别报告逐体素 LET 相对标准误、区域 RMS 不确定度、分位数及达标体素比例。

### II.F.5. 计算吞吐量评测与运行时完整性审计

各型 GPU 硬件平台——包括 NVIDIA GeForce RTX 2080 Ti（`sm_75`）、NVIDIA RTX A6000（`sm_86`）、NVIDIA L4（`sm_89`）以及 Intel Arc B580（`bmg-g21`）——均采用禁用运行时 JIT 的原生 AOT 编译可执行程序独立评测并分别报告（编译耗时不计入运行时间）。性能评测严格区分两项端到端墙钟时间（wall-clock time）指标：
1. **冷启动耗时（Cold-start latency）**：从操作系统启动新进程至首个批次结果落盘保存的完整耗时，涵盖配置解析、GPU 设备初始化、物理数据包上传至显存、内核计算、运行审计及文件输出。
2. **稳态吞吐率（Steady-state throughput，$\mathrm{histories/s}$）**：同一常驻进程内后续连续批次的总模拟入射 histories 除以累计端到端墙钟时间（包含显存回传、运行审计与文件输出）。

通过分阶段内核计时器进一步分解主粒子输运、铜／空气上游输运与带电次级碎片输运的时间占比，并在完全匹配的历史数、源、批次大小和计时边界下对比纯剂量计分（$D_m$）与剂量加 LET 同步计分（$D_m + L_d$）的边际开销。在大统计量多 GPU 协同计算中，总历史数按各 GPU 实测稳态吞吐率成比例分配以平衡各卡剩余完成时间，同时严格维持整次模拟统一的全局最大余数束斑分配（式 11a）。多核 CPU TOPAS 单节点吞吐率按总模拟 histories 除以节点累计计算墙钟时间计算，并明确区分集群 56 线程节点与本地 96 线程工作站配置。

每批计算检查轨迹完成、队列容量、物理表覆盖及能量收支，并将检查结果与剂量和 LET 一同保存。检查要求所有已启动轨迹完成、队列零溢出、物理表查询位于覆盖域内，以及剂量和 LET 两个矩为有限非负值。能量账目分别记录连续沉积、核顶点沉积、截断末端沉积、边界逃逸、上游损失、中性产物及反应残差，用于核对各物种和输运阶段的能量收支。

---

## 参考文献

1. Jia X, Schümann J, Paganetti H, Jiang SB. GPU-based fast Monte Carlo dose calculation for proton therapy. *Phys Med Biol.* 2012;57(23):7783–7797. [doi:10.1088/0031-9155/57/23/7783](https://doi.org/10.1088/0031-9155/57/23/7783).
2. Qin N, Pinto M, Tian Z, et al. Initial development of goCMC: a GPU-oriented fast cross-platform Monte Carlo engine for carbon ion therapy. *Phys Med Biol.* 2017;62(9):3682–3699. [doi:10.1088/1361-6560/aa5d43](https://doi.org/10.1088/1361-6560/aa5d43).
3. De Simoni M, Battistoni G, De Gregorio A, et al. A data-driven fragmentation model for carbon therapy GPU-accelerated Monte-Carlo dose recalculation. *Front Oncol.* 2022;12:780784. [doi:10.3389/fonc.2022.780784](https://doi.org/10.3389/fonc.2022.780784).
4. Lee H, Shin J, Verburg JM, et al. MOQUI: an open-source GPU-based Monte Carlo code for proton dose calculation with efficient data structure. *Phys Med Biol.* 2022;67(17):174001. [doi:10.1088/1361-6560/ac8716](https://doi.org/10.1088/1361-6560/ac8716).
5. Feng H, Holmes JM, Vora SA, et al. Modelling small block aperture in an in-house developed GPU-accelerated Monte Carlo-based dose engine for pencil beam scanning proton therapy. *Phys Med Biol.* 2024;69(3):035003. [doi:10.1088/1361-6560/ad0b64](https://doi.org/10.1088/1361-6560/ad0b64).
6. Schneider W, Bortfeld T, Schlegel W. Correlation between CT numbers and tissue parameters needed for Monte Carlo simulations of clinical dose distributions. *Phys Med Biol.* 2000;45(2):459–478. [doi:10.1088/0031-9155/45/2/314](https://doi.org/10.1088/0031-9155/45/2/314).
7. Perl J, Shin J, Schümann J, Faddegon B, Paganetti H. TOPAS: an innovative proton Monte Carlo platform for research and clinical applications. *Med Phys.* 2012;39(11):6818–6837. [doi:10.1118/1.4758060](https://doi.org/10.1118/1.4758060).
8. Agostinelli S, Allison J, Amako K, et al. Geant4—a simulation toolkit. *Nucl Instrum Methods Phys Res A.* 2003;506(3):250–303. [doi:10.1016/S0168-9002(03)01368-8](https://doi.org/10.1016/S0168-9002(03)01368-8).
9. Eyges L. Multiple scattering with energy loss. *Phys Rev.* 1948;74(10):1534–1535. [doi:10.1103/PhysRev.74.1534](https://doi.org/10.1103/PhysRev.74.1534).
10. Low DA, Harms WB, Mutic S, Purdy JA. A technique for the quantitative evaluation of dose distributions. *Med Phys.* 1998;25(5):656–661. [doi:10.1118/1.598248](https://doi.org/10.1118/1.598248).

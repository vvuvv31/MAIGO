# II. 材料与方法

**GPU Monte Carlo: From Broadbeam to Minibeam**（[English Version](mm.md)）。

本研究开发了跨平台 GPU 蒙特卡罗引擎 **MAIGO**，面向质子与碳离子笔形束扫描（PBS），将宽束患者输运扩展至铜准直微束。共享输运核心结合凝聚历史电磁输运、离散核反应及多代带电碎片追踪 [1–4]。微束扩展显式模拟粒子穿过多狭缝准直器与空气间隙的过程 [5]，并将位置—方向—能量—物种联合状态交接至患者输运，使准直器引起的束流调制与核产物共同参与亚毫米峰谷剂量结构的计算。

材料相关截面与响应函数由 Geant4 11.3.2 和 OpenTOPAS 4.2.3 [7,8] 离线提取，并在输运时查询。研究涵盖质子（$^1\mathrm{H}$, $Z=1, A=1$）与碳-12 离子（$^{12}\mathrm{C}$, $Z=6, A=12$）在水和患者 CT 中的输运，计分量为介质吸收剂量（$D_m$）与剂量平均电子 LET（$L_d$）。

---

## II.A. 离子输运物理

以下共享模型描述离子在水与异质患者组织中的慢化、散射及核反应；铜与空气中的材料相关扩展见第 II.B 节。

### II.A.1. 连续能量损失与涨落

带电离子的连续慢化由当前动能 $E$、局部材料组成与质量密度 $\rho$ 下的非受限电子阻止本领 $S_e(E;\rho)$ 描述。阻止响应包含物种相关的有效电荷与密度效应修正，参考水的平均激发能为 $75\text{ eV}$。对于真实步长 $\ell$，平均能损与连续慢化近似（CSDA）射程满足

$$
\begin{aligned}
R(E) &= \int_0^E \frac{\mathrm{d}E'}{S_e(E')}, \\
\overline{\Delta E}(E,\ell)
&= E-R^{-1}\!\bigl(\max\{R(E)-\ell,0\}\bigr).
\end{aligned}
\tag{1}
$$

质子与重离子的短步均采用中点能量预测—校正法：在 $E-\tfrac{1}{2}S_e(E)\ell$ 处求取阻止响应，再乘以真实步长 $\ell$，以计入步内阻止本领的变化。当试探能损超过原生线性能损分数阈值，或当前步耗尽剩余射程时，按式（1）通过 CSDA 射程反演计算平均能损。

主粒子能损涨落由改编自 Geant4 `G4UniversalFluctuation` 与 `G4IonFluctuations` 模型的统一引擎抽样 [8]。对于静止质量为 $M$、速度为 $\beta c$、Lorentz 因子为 $\gamma$ 的入射离子，单次碰撞转移给自由电子的最大能量为

$$
T_{\max} = \frac{2 m_e c^2 \beta^2 \gamma^2}{1 + 2\gamma(m_e/M) + (m_e/M)^2},
\tag{2}
$$

其中 $m_e$ 为电子静止质量。非受限输运包含直至 $T_{\max}$ 的电子能量转移，对应 Geant4 方差表达中的能量转移截断 $T_c=T_{\max}$。因此，真实步长 $\ell$ 上的基础能损方差为

$$
\sigma_{E,0}^2=
2\pi r_e^2m_ec^2N_{\mathrm{el}}
\frac{Z_h^2}{\beta^2}\,T_{\max}\ell
\left(1-\frac{\beta^2}{2}\right),
\tag{3}
$$

其中 $r_e$ 为经典电子半径，$N_{\mathrm{el}}$ 为局部电子数密度，$Z_h$ 为以元电荷为单位的入射粒子电荷数。式（3）给出非受限基础方差；低能 Ion 模型进一步采用原生有效电荷与电荷涨落修正。

质子及较高速度的离子采用 Universal 涨落模型：薄吸收层中的能损由随机原子激发与电离分量组成，厚吸收层则依据均值与标准差之比采用截断高斯或矩匹配 Gamma 抽样。低能重离子采用 Ion 涨落模型，包含有限步长减速修正及相应的高斯、Gamma 或均匀分布抽样分支。每个主粒子步根据平均能损和非受限方差执行一次全能损抽样；带电次级碎片采用有限步平均能损，同时保留随机多重散射与核反应。

实际能损受可用动能约束，耗尽射程的终止步沉积剩余动能。全部电离能损局部计入当前水或患者 CT 体素。该处理在凝聚历史离子输运中保留了主粒子射程歧离与 Bragg 峰展宽。

### II.A.2. 多重库仑散射

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
\tag{4}
$$

算法在粒子瞬时局部坐标系下通过 Cholesky 分解对两个正交横向轴的 $(\theta_x, x)$ 与 $(\theta_y, y)$ 进行联合抽样，再旋转回全局笛卡尔坐标系，由此实现式（4）给出的角度—位移协方差。当单个输运步超过设定的散射分段上限（如 CT 微束患者内次级碎片的 $0.1\text{ mm}$ 分段上限）时，自动将该步细分为多个均匀散射子段，并利用已接受的步能损估算各子段中点动能以求取 $\mathcal{T}$。

有效线性散射功率 $\mathcal{T}$ 由 Rossi–Greisen 高斯核心项与标定大角单次散射尾部的等效二阶矩叠加而成：

$$
\mathcal{T}(E) = \left(\frac{E_s Z}{\beta p c}\right)^2 \frac{1}{X_0(\rho)} + \lambda_{\mathrm{tail}}(\rho)\,\left(\frac{E_{\mathrm{tail}} Z}{\beta p c}\right)^2,
\tag{5}
$$

式中 $p$ 为相对论动量，$X_0(\rho)$ 为由 Schneider 组织组成与体素密度确定的局部辐射长度，$\lambda_{\mathrm{tail}}(\rho) = \lambda_{\mathrm{tail}, \mathrm{water}} \, [X_{0,\mathrm{water}} / X_0(\rho)]$ 按逆辐射长度之比缩放名义尾部散射率。对于轻离子（$^1\mathrm{H}$、$^2\mathrm{H}$、$^3\mathrm{H}$ 与 $^4\mathrm{He}$），有效核心散射能量 $E_s(E/A)$ 及尾部参数由 $50$、$150$ 和 $300\text{ MeV/u}$ 均匀水层 Geant4 标定值分段线性插值给出（区间外取端点值）；对于 $^{12}\mathrm{C}$ 及采用同组标定的重碎片，统一采用标定常数 $E_s = 9.9\text{ MeV}$、水中名义尾部率 $\lambda_{\mathrm{tail},\mathrm{water}} = 0.0025\text{ mm}^{-1}$ 及尾部散射能量 $E_{\mathrm{tail}} = 2.4\text{ MeV}$。强相互作用核弹性散射则作为独立的离散随机过程单独处理（第 II.A.3 节）。

微束主粒子的多重散射采用 Urban 模型的 GPU 实现 [8,11]，按材料和粒子物种查询从 Geant4 提取的输运自由程、辐射长度及角分布参数。模型联合抽样角偏转与相关横向位移，并计入粒子真实路径与几何推进距离之间的差异。

在 CT 组织界面及铜—空气边界附近，几何步长受边界距离约束，能损与核反应概率则沿对应的真实路径计算，从而保持组织界面和多狭缝准直器中散射与几何输运的一致性。铜内产生的带电碎片在铜中同样采用分物种 Urban 输运，具体见第 II.B.2 节。

### II.A.3. 核相互作用

主离子与带电次级碎片采用统一的光学深度算法抽样核弹性和非弹性反应。对于通道 $c \in \{\mathrm{el},\mathrm{in}\}$，宏观反应率由局部靶元素组成与密度确定：

$$
\begin{aligned}
\Sigma_c(E;\rho) &= \sum_t n_t(\rho)\sigma_{c,t}(E), \\
\tau &= -\ln U, \\
\Delta\tau &= [\Sigma_{\mathrm{el}}(E;\rho)+\Sigma_{\mathrm{in}}(E;\rho)]\,\ell.
\end{aligned}
\tag{6}
$$

其中 $n_t$ 为靶元素 $t$ 的原子数密度，$\sigma_{c,t}$ 为对应微观截面，$U$ 为 $(0,1)$ 上的均匀随机数，$\ell$ 为真实步长。剩余光学深度跨材料边界保留，直至到达核反应顶点，再按各分量反应率选择反应通道与靶核。主粒子非弹性候选反应采用缓存反应率，并根据慢化后的实际反应率进行空碰撞拒绝校正。

核弹性末态根据从 Geant4 提取的入射粒子—靶核特异的同位素及角分布生成 [8]。抽样质心系散射角和均匀方位角后，通过相对论两体运动学求解出射离子与反冲核。属于常规带电碎片集合的反冲核参与电磁与核反应输运；其余靶反冲同位素采用非受限电子加核阻止本领 $S_{\mathrm{tot}}=S_e+S_n$ 及对应 CSDA 射程进行连续慢化，其电子 LET 按 $S_e$ 计分。

非弹性反应从 Geant4 相关末态事件库中抽取完整事件，事件库按入射物种、靶元素和单核子入射能量索引 [8]。质子主粒子库采用 Binary Cascade（`QGSP_BIC_HP`），碳离子主粒子库采用 INCLXX（`FTFP_INCLXX`），次级碎片库采用 `G4IonBinaryCascadePhysics`。对于节点间能量，先按线性插值概率选择相邻能量节点，再从该节点均匀抽取完整事件。全部产物施加同一个随机方位角旋转，以保留事件多重性及能量—角度关联。节点选择引起的能量偏移和反应 Q 值残差分别计入能量账目。

动能高于 $0.1\text{ MeV}$ 的受支持带电产物进入次级输运，低于截断阈值的带电产物及事件记录的顶点局部能量在反应位置沉积。水与患者 CT 中的次级非弹性分支最多追踪至第二代，此后带电碎片继续慢化、多重散射及允许的核弹性散射。中子、退激光子等中性产物及输运集合以外的残余物种记为未追踪能量，不计入患者剂量。

### II.A.4. 步长控制与边界输运

在每个输运步起始处，引擎先计算当前体素组成与密度下的电磁响应，并综合各项物理与几何约束确定步长。设 $f$ 为物种特异的射程步长分数，$r_f$ 为最终射程参数，$\eta$ 为单步允许的最大平均能损分数。当剩余 CSDA 射程满足 $R(E) > r_f$ 时，电磁步长建议值 $\ell_{\mathrm{EM}}$ 由非线性射程导引项与最大能损分数项共同约束：

$$
\ell_{\mathrm{EM}} = \min\!\left[
f R(E) + r_f(1 - f)\!\left(2 - \frac{r_f}{R(E)}\right),\;
R(E) - R\!\left((1 - \eta)E\right)
\right].
\tag{7}
$$

当粒子慢化至 $R(E) \le r_f$ 时，电磁建议步长直接取为剩余射程 $\ell_{\mathrm{EM}} = R(E)$，并在该步结束时终止轨迹。CT 宽束输运采用 $\eta = 0.05$，微束患者内输运采用 $\eta = 0.01$。

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

---

## II.B. 微束输运扩展

### II.B.1. 多狭缝准直器几何

微束配置在宽束患者输运框架上游引入了铜多狭缝准直器、狭缝内空气通道及准直器后至患者体表的空气间隙。几何模型精确定义了铜准直器的外廓尺寸、沿束流方向厚度、狭缝数量、狭缝开口宽度、中心节距（pitch）、狭缝长度、阵列横向相位偏移、旋转角，以及源平面、准直器出口、CT 包围盒、患者皮肤表面与等中心之间的相对轴向坐标。准直器固定在射野坐标系中，不随扫描束斑平移。

全计划临床对比实现了两套微束准直器设计：

1. **质子微束准直器**：采用 $60\text{ mm}$ 厚铜块，包含 $15$ 条平行狭缝，单缝宽度 $0.5\text{ mm}$、中心节距 $3.6\text{ mm}$、狭缝长度 $50\text{ mm}$。
2. **碳离子微束准直器（RT07575 头部计划）**：采用 $100\text{ mm}$ 厚铜块，包含 $17$ 条狭缝，单缝宽度 $0.7\text{ mm}$、中心节距 $3.0\text{ mm}$、狭缝长度 $100\text{ mm}$。针对 RT07575 计划中 $90^\circ$ 与 $270^\circ$ 对穿交火射野，两个射野保留各自独立的狭缝阵列定位，其中 $270^\circ$ 射野的名义中心狭缝阵列相对于 $90^\circ$ 射野设置了半节距（$1.5\text{ mm}$）的横向相位交错。

### II.B.2. 铜与空气中的输运

穿行于铜准直器与上游空气中的带电粒子查询材料相关的束线电磁物理表，该表提供覆盖铜核反应带电碎片物种的非受限电子阻止本领 $S_e(E)$ 及在运动学上限 $T_{\max}$ 处直接提取的原生涨落方差。当试探线性能损不超过当前动能的 $2\%$（即 $S_e(E)\,\ell \le 0.02 E$）时，平均能损直接按步首阻止本领线性计算；超过 $2\%$ 时则调用同一响应构造的 CSDA 射程反演（式 1），最低能量节点以下采用规定的 $\sqrt{E}$ 阻止本领延拓积分。

铜与空气中的主粒子采用第 II.A.1 节的共享 Universal／Ion 涨落引擎，使用原生粒子质量、电荷、材料激发能及 $T_{\max}$ 参数；带电次级碎片采用有限路径平均电子能损。上游电子能损按局部沉积近似处理，记入束线能量账目，并与患者计分网格内的剂量分开。铜输运步长上限为 $0.05\text{ mm}$，CT 碳微束狭缝空气中的上限为 $0.2\text{ mm}$。

铜准直器中显式模拟核弹性散射与核非弹性碎裂。主粒子在铜中的核碰撞距离由铜弹性与非弹性总宏观反应率 $\Sigma_{\mathrm{tot},\mathrm{Cu}} = \Sigma_{\mathrm{el},\mathrm{Cu}} + \Sigma_{\mathrm{in},\mathrm{Cu}}$ 抽样，经电子连续慢化后在候选碰撞点按实际分通道反应率判定反应类型。若发生铜核弹性散射，从铜同位素角分布样本中抽取动量转移并应用第 II.A.3 节的相对论两体运动学：出射束流离子继续输运，而转移给重铜靶核的反冲动能直接记入上游束线能量账目，不在患者剂量中计分。

铜非弹性核碰撞从铜专用相关末态事件库中按精确入射粒子 $(Z, A)$ 与单核子能量 $e = E/A$ 抽样。反应产生的直接带电产物及记录的带电续行产物共享同一个事件随机方位角旋转，并继续穿过剩余铜厚度与下游空气间隙。铜准直器内部的后续次级核非弹性反应受独立的上游代数上限约束（标准微束配置设为 $g_{\max,\mathrm{Cu}} = 3$）；达到代数上限后停止进一步非弹性分支，存活粒子继续执行电磁慢化与弹性散射。铜内局部顶点沉积、中性核产物及剩余未追踪反应能量统一保留在束线能量平衡账目内。

铜与空气中的主粒子采用第 II.A.2 节的 Urban 散射模型。质子与碳离子微束中，铜内产生的带电碎片在铜中均采用共享输运模型，结合独立提取的分物种 Urban 多重散射表、铜核弹性散射及非受限电子阻止本领。碎片响应按其自身核身份 $(Z, A)$ 与能量选取。

### II.B.3. 与患者输运的耦合

MAIGO 从准直器上游入口逐历史输运主粒子及带电核产物，经过空气狭缝、铜隔板和下游空气间隙生成患者入口相空间。入口分布同时包含直接穿缝主粒子、经铜侧壁散射出射的主粒子，以及铜与上游空气中产生的带电核碎片，由此将准直器引起的束流调制和次级粒子成分共同传递至患者计算。在铜内停止或被吸收的原始历史同样计入总入射历史数 $N$，用于绝对剂量归一化。

穿过准直器与空气间隙到达患者入口平面的所有存活主粒子及上游带电核产物，将其完整的位置—方向—能量—物种联合相空间状态 $(\mathbf{r}, \mathbf{u}, E, Z, A)$、代数标记及确定性随机数身份交接至下游患者输运阶段。

患者内主粒子采用 Urban 模型，描述狭缝调制剂量梯度附近的角偏转与横向位移。CT 微束的带电次级粒子采用分段 Fermi–Eyges 散射，包括从上游产生并进入患者的带电碎片。两类粒子的外层步长上限均为 $0.5\text{ mm}$，次级粒子的独立散射分段上限为 $0.1\text{ mm}$。水微束与通用微束输运对主粒子及带电次级粒子均采用 Urban 散射。这些配置使用第 II.A.2 节定义的共同散射算子。

---

## II.C. 剂量与 LET 计分

### II.C.1. 吸收剂量与计划归一化

设单次照射中第 $j \in \{1, \dots, M\}$ 个扫描束斑的物理粒子权重为 $W_j > 0$，全计划总物理入射粒子数为 $W = \sum_{j=1}^M W_j$。对于总模拟历史数为 $N$ 的蒙特卡罗计算，各束斑的整数模拟粒子数 $n_j$ 按最大余数法（largest-remainder method）由 $N W_j / W$ 分配：

$$
\begin{aligned}
h_j &= \frac{NW_j}{W},
& K &= N-\sum_{k=1}^M\lfloor h_k\rfloor, \\
n_j &= \lfloor h_j\rfloor+\mathbf{1}_{\{j\in\mathcal{R}_K\}},
& \sum_{j=1}^M n_j &= N.
\end{aligned}
\tag{8a}
$$

其中 $h_j$ 为理想历史数，$K$ 为向下取整后待分配的历史数，$\mathcal{R}_K$ 为小数余数 $h_j-\lfloor h_j\rfloor$ 最大的前 $K$ 个束斑下标集合，$\mathbf{1}$ 为示性函数。该方法使全计划历史数之和保持为 $N$，且各束斑分配与理想值相差小于一个历史。全局分配随后拆分至各 GPU 和统计批次；配对 TOPAS 比较采用参考计算的整数束斑粒子数清单。

粒子输运过程中，体素沉积能量 $\Delta E$（单位 $\mathrm{MeV}$）通过 GPU 32 位单精度浮点数（`FP32`）原子加操作累加至显存数组。设第 $i$ 个体素的质量密度为 $\rho_i$（$\mathrm{g/cm^3}$）、体积为 $V_i$（$\mathrm{mm^3}$）、累计沉积能量为 $E_i$（$\mathrm{MeV}$），则体素质量 $m_i$（$\mathrm{kg}$）及绝对介质吸收剂量 $D_i$（$\mathrm{Gy}$）由下式给出：

$$
\begin{aligned}
m_i &= \rho_iV_i\times 10^{-6}\,\mathrm{kg}, \\
D_i &= \frac{E_i\times 1.602176634\times 10^{-13}\,\mathrm{J/MeV}}{m_i}
\frac{W}{N}.
\end{aligned}
\tag{8b}
$$

当整次模拟拆分为 $B$ 个统计批次或多张 GPU 卡（第 $b$ 批包含 $N_b$ 个历史，$\sum_{b=1}^B N_b = N$）时，每个批次输出均统一乘以全局归一化因子 $W / N$；因此在主机端以双精度（FP64）合并批次文件时，只需对各批次剂量直接求和 $D_i = \sum_{b=1}^B d_{i,b}$。

### II.C.2. 剂量平均电子 LET

剂量平均电子线性能量传递（$L_d$）采用基于非受限电子阻止本领的轨迹长度估计量（track-length estimator）与吸收剂量同步计分。对于体素 $i$ 内真实路径长度为 $\ell_k > 0$ 的第 $k$ 段有限长带电离子输运步，令 $S_{e,k}$ 表示在步前动能与当前体素实际密度 $\rho_i$ 处求值的非受限线性电子阻止本领（单位 $\mathrm{MeV/mm}$，数值上等价于 $\mathrm{keV/\mu m}$）。引擎在每个体素内同步累积两个可加矩 $A_i$ 与 $B_i$，其比值即为体素剂量平均电子 LET $L_{d,i}$：

$$
\begin{aligned}
A_i &= \sum_{k\in i}S_{e,k}^{\,2}\ell_k, \\
B_i &= \sum_{k\in i}S_{e,k}\ell_k, \\
L_{d,i} &= \frac{A_i}{B_i},\qquad B_i>0.
\end{aligned}
\tag{9}
$$

式（9）的权重为期望电子能损 $S_{e,k}\,\ell_k$，使 LET 计分采用与离子输运一致的阻止响应。$S_{e,k}$ 与 $L_{d,i}$ 均为实际体素密度下的线性阻止量，单位为 $\mathrm{keV/\mu m}$。常规带电碎片集合内的离子查询材料相关的非受限电子阻止表，额外靶反冲核查询纯电子反冲阻止表；LET 的两个矩均按电子阻止本领构造。

水模体或患者 CT 计分网格内所有总动能高于 $0.1\text{ MeV}$ 截断阈值的已输运带电离子（包括主粒子、次级核碎片及显式输运的反冲核）均对 $A_i$ 和 $B_i$ 贡献。上游铜准直器与空气中的轨迹段、核反应顶点局部能量沉积、零路径截断终止沉积以及未追踪中性产物能量均不计入 LET 累积量。两个矩 $A_i$ 与 $B_i$ 在 GPU 端分别通过 `FP32` 原子加操作累积并独立保存至文件，在跨射野、统计批次及多 GPU 设备按双精度完成全局求和后，最终作一次除法求得 $L_{d,i}$。分母 $B_i = 0$ 表示该体素无有限长带电轨迹穿过，在后续统计分析中保留其零分母标识。

### II.C.3. TOPAS 参考计分

TOPAS 介质吸收剂量采用 `DoseToMedium` 计分器计算。TOPAS 参考采用与式（9）一致的双矩 LET 计分器，分别累积 $A_i = \sum_k w_k S_{e,k}^2 \ell_k$ 与 $B_i = \sum_k w_k S_{e,k} \ell_k$，其中 $w_k$ 为轨迹统计权重，计分阈值为 $0.1\text{ MeV}$ 总动能。计分器查询预先提取的只读非受限电子阻止表；该表按参考模拟的 Geant4 11.3.2 物理列表、粒子种类、Schneider 组织组成、密度和平均激发能生成，使两个引擎采用共同的 LET 定义与计分终点。

---

## II.D. GPU 并行实现

### II.D.1. 输运特化与队列调度

MAIGO 采用 C++20 与 Khronos SYCL 实现，通过 oneAPI Unified Runtime CUDA 后端支持 NVIDIA GPU，通过 Level Zero 后端支持 Intel Xe GPU。设备内核预先编译为目标架构的原生机器码，运行时关闭即时编译。质子宽束、碳离子宽束、质子微束和碳离子微束采用四条编译期特化主粒子管线，并为自定义离子配置提供通用管线；各管线共享支持多物种带电核碎片的次级输运内核。

针对微束模拟中因狭缝准直器吸收、多代核反应分支及轨迹寿命差异导致的 GPU 线程束分化（warp/subgroup divergence），微束管线将粒子输运解耦为通过预分配全局内存队列衔接的多个顺序执行阶段：

1. **上游束线主粒子阶段**：追踪从束流源出发穿过铜狭缝准直器与空气间隙的主粒子，将到达患者入口的存活主粒子写入入口检查点队列，并将铜中产生的次级带电产物推入独立的上游子代队列。
2. **上游铜碎片级联阶段**：在铜准直器与空气中继续追踪最多三代铜带电核碎片。
3. **患者内主粒子输运阶段**：从患者入口恢复存活主粒子并在体素化水或 CT 网格中执行慢化与核反应输运；在适用配置下采用计数排序（counting-sort）算法按空间／材料局部性重排活跃轨迹索引，以提高物理表查询的局部性。
4. **患者内次级碎片阶段**：以固定输运步数分段推进所有患者内产生及上游传入的带电次级粒子，并在活跃轨迹数低于尾部阈值时执行队列压缩以回收空闲线程。

设备队列在内核启动前预分配，硬件原子计数器记录各阶段的队列占用与完成情况。

### II.D.2. 随机数生成与批次执行

设备端采用计数器式 Philox-4x32-10 伪随机数发生器。随机抽样由批次种子、束斑索引、主历史编号、子代轨迹身份及单调递增的计数器确定，使每条轨迹的随机数流独立于线程调度和队列重排。连续批次复用 GPU 上下文及常驻显存的材料与物理表，将初始化和物理表上传的成本分摊至整次计算。

多 GPU 计算按各设备实测稳态吞吐率成比例分配历史数，以平衡完成时间。式（8a）的全局整数束斑分配在设备与批次拆分后保持不变，各历史采用独立随机数流，结果按第 II.C 节的剂量归一化与 LET 双矩合并方法汇总。

### II.D.3. 能量收支与输运完成检查

每批计算检查轨迹完成、队列容量、物理表覆盖及能量收支，并将检查结果与剂量和 LET 一同保存。检查要求所有已启动轨迹完成、队列零溢出、物理表查询位于覆盖域内，以及剂量和 LET 两个矩为有限非负值。能量账目分别记录连续沉积、核顶点沉积、截断末端沉积、边界逃逸、上游损失、中性产物及反应残差，用于核对各物种和输运阶段的能量收支。

初始化时检查入射粒子身份、材料组成及物理表能量覆盖。每次计算记录物理数据版本、软件版本、整数束斑分配及随机数种子，以复现相同的源与物理配置。

---

## II.E. 临床验证

### II.E.1. 临床病例与治疗计划

临床全计划评价采用两例头部 CT——**RT07575**（Head 1）和 **RT06423**（Head 2）——以及一例肺部 CT——**20022516**（Lung 1）。头部病例评价复杂解剖结构中的计划剂量一致性，肺部病例评价低密度异质介质中的输运。宽束与铜准直微束、质子与碳离子的配对计算历史数列于表 2。各组 MAIGO 与 TOPAS 计算共享 CT 网格、Schneider 材料映射 [6]、束流光学、束斑位置与能量、最终权重、摆位、准直器几何及绝对剂量归一化。

**表 2.** 各临床病例与照射模式下全计划剂量验证的模拟入射粒子历史数（$N$）。

| 临床病例编号 | 解剖部位 | 质子宽束 | 碳-12 宽束 | 质子微束 | 碳-12 微束 |
|---|---|---:|---:|---:|---:|
| **RT07575** | 头颈部 1（Head 1） | $250,626,500$ | $129,638,170$ | $1,000,000,000$ | $1,000,000,000$ |
| **RT06423** | 头颈部 2（Head 2） | $124,266,340$ | $151,087,660$ | $1,000,000,000$ | $1,000,000,000$ |
| **20022516** | 胸部／肺部（Lung 1） | $361,050,622$ | $177,173,040$ | $3,000,000,000$ | —$^{\mathrm{a}}$ |

$^{\mathrm{a}}$— 表示本表未报告该病例／模态的配对数据。默认 CT 碳微束配置对应独立的肺部基准模体。

全计划剂量网格的体素尺寸为 $0.5 \times 0.5 \times 2.0\text{ mm}^3$，网格维数、物理原点和轴顺序由各计划的参考头文件定义。三维剂量与 LET 在共同的患者物理坐标、原生网格及计划绝对剂量尺度下逐体素比较。

临床扫描治疗计划由射野、能量层及离散扫描束斑序列定义，每个束斑包含标称动能、等中心平面横向坐标及粒子权重。随能量变化的相空间束流光学模型给出源平面处的横向位置标准差 $(\sigma_x, \sigma_y)$、角散度标准差 $(\sigma_{\theta_x}, \sigma_{\theta_y})$、位置—角度协方差参数以及高斯能量展宽 $\sigma_E$。模拟时，先在束斑局部坐标系内对耦合的横向位置与方向余弦进行联合抽样，再结合二极扫描磁铁虚拟源轴距（SAD）、机架／治疗床旋转角及等中心偏移量，将粒子状态刚体变换至患者世界坐标系。离子能量统一以单核子动能（$\mathrm{MeV/u}$）表征；对于以总动能（$\mathrm{MeV}$）存储的计划输入，按质量数 $A$ 换算。

物理剂量逆向优化以单位入射粒子的剂量影响矩阵 $\mathbf{A}$ 和非负束斑粒子权重 $\mathbf{w} \ge \mathbf{0}$ 表示体素剂量 $\mathbf{d} = \mathbf{A}\mathbf{w}$，并根据各病例的靶区覆盖与危及器官（OAR）保护目标求解权重。最终权重在 MAIGO 与 TOPAS 全计划前向计算中固定为同一个向量，使两引擎比较采用共同的照射源。重新优化的计划采用 $4\text{ Gy}$ 物理剂量处方；保留治疗计划系统（TPS）权重的碳离子宽束计划采用原计划累计剂量标度。

### II.E.2. TOPAS 参考模拟

参考模拟采用 OpenTOPAS 4.2.3／Geant4 11.3.2 [7,8]，计分器按第 II.C.3 节定义。物理列表包含标准电磁过程（`g4em-standard_opt4`）、强子与离子非弹性反应（`g4h-phy_QGSP_BIC_HP`、`g4ion-inclxx`）、强子弹性散射（`g4h-elastic_HP`）、停止过程（`g4stopping`）及衰变过程（`g4decay`、`g4radioactivedecay`）。微束参考通过 `AllChargedUrbanPhysics` 为带电粒子配置 Urban 散射；碳离子参考通过封装 `G4IonElasticPhysics` 的 `CarbonIonElasticPhysics` 为 `GenericIon` 注册核弹性过程。

各计划的电磁表能量上限、次级粒子产生截断及步长设置随参考输入记录。肺部质子微束参考采用 $0.05\text{ mm}$ 生产截断与 $10\text{ GeV}$ 电磁表上限，三份碳离子宽束参考采用 $600\text{ MeV}$ 电磁表上限。参考剂量由最终计划权重驱动的独立 TOPAS 全计划输运生成，TOPAS 显式输运超过生产截断的次级电子与光子。MAIGO 的局部电离沉积、次级平均能损、核末态抽样和中性产物处理按第 II.A–B 节定义。

为解析 Bragg 峰附近的阻止本领梯度，碳离子宽束 TOPAS LET 参考对轻离子与 `GenericIon` 采用 `dRoverRange = 0.005` 和 `finalRange = 0.1 nm`，质子保持其参考步长控制。RT07575 的单复本配对 LET 比较采用 $25,927,634$ 个入射历史，MAIGO 与 TOPAS 使用相同的整数束斑分配。独立复本的 LET 统计不确定度按第 II.E.6 节估计。

### II.E.3. 剂量一致性评价

采用以 TOPAS 参考剂量 $D_{\mathrm{ref}}(\mathbf{r})$ 为基准的三维局部（local）Gamma 指数 [10] 定量评价 GPU 计算剂量分布 $D_{\mathrm{GPU}}(\mathbf{r}')$ 的空间与剂量一致性：

$$
\begin{aligned}
\varepsilon_D &= \frac{\Delta D_{\%}}{100},\qquad \Delta r>0, \\
\gamma(\mathbf{r})
&= \min_{\substack{\mathbf{r}'\\\|\mathbf{r}'-\mathbf{r}\|\le\Delta r}}
\left[
\frac{\|\mathbf{r}'-\mathbf{r}\|^2}{(\Delta r)^2}
+\frac{\bigl[D_{\mathrm{GPU}}(\mathbf{r}')-D_{\mathrm{ref}}(\mathbf{r})\bigr]^2}
{\bigl[\varepsilon_D D_{\mathrm{ref}}(\mathbf{r})\bigr]^2}
\right]^{1/2}.
\end{aligned}
\tag{10}
$$

式中 $\Delta r>0$ 为距离一致性（DTA）容差，$\varepsilon_D=\Delta D_{\%}/100$ 为相对于参考点局部剂量的无量纲剂量容差。GPU 剂量场通过三线性插值在参考点周围半径 $\Delta r$ 的球域内搜索，以计算亚体素距离容差下的 Gamma 指数。参考评价点为 `BODY` 轮廓内超过剂量阈值的体素；搜索时采用完整 GPU 剂量场，包括 `BODY` 边界外的邻近体素。表 3 列出评价域、局部判据及搜索步长。

**表 3.** 宽束与微束模式的三维局部 Gamma 评价域、验收判据及亚体素空间搜索步长。

| 照射模式 | `BODY` 内参考剂量评价域阈值 | 报告的局部 Gamma 判据（$\Delta D_{\%}/\Delta r$） | 空间插值搜索步长 |
|---|---|---|---:|
| **宽束（Broadbeam）** | $D_{\mathrm{ref}}(\mathbf{r}) > 0.10\, D_{\mathrm{ref},\max}$ | $3\%/3\text{ mm}$、$1\%/1\text{ mm}$、$3\%/0\text{ mm}$ | $0.10\text{ mm}$ |
| **微束（Minibeam）** | $D_{\mathrm{ref}}(\mathbf{r}) > 0.4\text{ Gy}$（$4\text{ Gy}$ 处方的 $10\%$） | $3\%/0.2\text{ mm}$、$3\%/0.1\text{ mm}$、$3\%/0\text{ mm}$ | $0.02\text{ mm}$ |

评价点满足 $\gamma \le 1$ 时通过。零距离终点定义为同位置相对剂量差 $\eta_D(\mathbf{r})=\lvert D_{\mathrm{GPU}}(\mathbf{r})-D_{\mathrm{ref}}(\mathbf{r})\rvert/D_{\mathrm{ref}}(\mathbf{r})$；$3\%/0\text{ mm}$ 检验以 $\eta_D\le 0.03$ 为通过条件。宽束以局部 $1\%/1\text{ mm}$ 通过率 $>99\%$ 为预定验收标准。微束采用 $3\%/0.2\text{ mm}$ 与 $3\%/0.1\text{ mm}$ 判据及 $0.02\text{ mm}$ 搜索步长，使空间容差与亚毫米狭缝尺度相适应。不同深度的横向剂量剖面评价峰宽、谷值与 PVDR，深度剂量剖面评价远端跌落，DVH 指标评价靶区与 OAR 剂量；同位置剂量差分布与批次不确定度共同表征逐体素一致性。

### II.E.4. LET 一致性与空间剖面

剂量平均电子 LET（$L_d$）的三维局部 Gamma 评价将式（10）中的剂量替换为 $L_{d,\mathrm{GPU}}(\mathbf{r}')$ 与 $L_{d,\mathrm{ref}}(\mathbf{r})$，无量纲容差定义为 $\varepsilon_L=\Delta L_{\%}/100$，局部绝对容差为 $\varepsilon_L L_{d,\mathrm{ref}}(\mathbf{r})$。零距离终点采用同位置相对 LET 差。评价域取为 `BODY` 轮廓与参考剂量掩膜 $D_{\mathrm{ref}}(\mathbf{r}) > 0.10\,D_{\mathrm{ref},\max}$ 的交集。宽束 LET 比较报告 $3\%/3\text{ mm}$、$1\%/1\text{ mm}$ 及 $3\%/0\text{ mm}$ 局部通过率（非零距离判据采用 $0.10\text{ mm}$ 搜索步长对 GPU LET 场作三线性插值）。参考剂量掩膜内若出现 GPU LET 分母为零（$B_i = 0$）的体素，在 Gamma 统计中计为未通过点。

LET 空间分析采用固定几何深度的横向剖面与沿束轴的深度剖面。RT07575 碳离子宽束的深度从中心射线进入 `BODY` 的位置起算，按原生 $0.5\text{ mm}$ 间距取样。中心射线剖面表征局部 LET 结构，$2.5 \times 10\text{ mm}^2$ 横向 ROI 汇总剖面表征区域深度趋势。每个深度切片的 ROI LET 按 $\sum_{i\in\mathrm{ROI}}A_i/\sum_{i\in\mathrm{ROI}}B_i$ 计算。

### II.E.5. 剂量不确定度

GPU 端逐体素剂量统计不确定度由 $B$ 个统计独立、束斑源比例匹配的全计划批次估计。对于 $B$ 个等统计量批次，设第 $b$ 批已按全局归一化因子 $W/N$ 缩放的体素剂量贡献为 $d_{i,b}$，总剂量为 $D_i = \sum_{b=1}^B d_{i,b}$，则 $D_i$ 的无偏样本方差估计为：

$$
\widehat{\sigma}_{D_i}^{\,2} = \frac{B}{B - 1} \left[ \sum_{b=1}^B d_{i,b}^{\,2} - \frac{D_i^{\,2}}{B} \right].
\tag{11}
$$

对于不等历史数批次，定义历史数占比 $a_b=N_b/N>0$ 和全计划等效批次剂量 $Y_{i,b}=d_{i,b}/a_b$，满足 $D_i=\sum_b a_bY_{i,b}$。在批次独立、源比例匹配且单历史方差相同的条件下，$\operatorname{Var}(Y_{i,b})=v_i/N_b$，总剂量方差估计为：

$$
\begin{aligned}
\widehat{\sigma}_{D_i}^{\,2}
&= \frac{1}{B-1}\sum_{b=1}^B a_b\bigl(Y_{i,b}-D_i\bigr)^2 \\
&= \frac{1}{B-1}\sum_{b=1}^B
\frac{\bigl(d_{i,b}-a_bD_i\bigr)^2}{a_b}.
\end{aligned}
\tag{12}
$$

式（12）要求 $B>1$，当 $a_b=1/B$ 时化为式（11）。

TOPAS 不确定度采用逐历史计分方差或匹配先导模拟估计。逐历史方差按归一化因子换算至计划剂量尺度；三份碳离子宽束参考采用 $N_{\mathrm{pilot}}=300,000$ 个历史的先导计算，保持参考几何、物理列表、能谱和相对束斑权重，并按 $\widehat{\sigma}_{D_{\mathrm{ref},i}}=\widehat{\sigma}_{\mathrm{pilot},i}\sqrt{N_{\mathrm{pilot}}/N_{\mathrm{full}}}$ 外推至完整历史数。独立复本间方差用于交叉核验。

在给定的固定评价域 $\mathcal{R}$（包含 $|\mathcal{R}|$ 个体素）内，整体蒙卡统计精度通过相对标准误的均方根（RMS）表征：

$$
\begin{aligned}
u_i &= 100\frac{\widehat{\sigma}_{D_i}}{D_{\mathrm{ref},i}}, \\
u_{\mathrm{RMS}}
&= \sqrt{\frac{1}{\lvert\mathcal{R}\rvert}\sum_{i\in\mathcal{R}}u_i^2}.
\end{aligned}
\tag{13}
$$

$u_i$ 和 $u_{\mathrm{RMS}}$ 均以百分数表示，并报告体素不确定度分位数及相对标准误低于 $1\%$ 或 $2\%$ 的体素比例。鉴于 GPU 与 TOPAS 模拟彼此统计独立，逐体素剂量差 $\Delta D_i = D_{\mathrm{GPU},i} - D_{\mathrm{ref},i}$ 的随机方差由 $\widehat{\sigma}_{\Delta D_i}^{\,2} = \widehat{\sigma}_{D_{\mathrm{GPU},i}}^{\,2} + \widehat{\sigma}_{D_{\mathrm{ref},i}}^{\,2}$ 合成。

### II.E.6. LET 不确定度

由于剂量平均 LET $L_{d,i} = A_i / B_i$ 是由同一批粒子轨迹同步累积的两个相关随机变量之比，其统计误差评估保留分子与分母之间的协方差 $\mathrm{Cov}(A_i, B_i)$。为在主计算总统计量 $N$ 下估计 GPU LET 统计误差，辅助 GPU 计算将总历史数 $N$ 拆分为 $K$ 个统计独立、源比例匹配的全计划分组，各组历史数为 $N_b$、占比为 $a_b=N_b/N>0$（碳离子宽束基准取 $K = 20$）。设第 $b \in \{1, \dots, K\}$ 组在体素 $i$ 累积的两个矩分别为 $A_{i,b}$ 与 $B_{i,b}$，总估计量为 $L_{d,i} = \sum_{b=1}^K A_{i,b} \big/ \sum_{b=1}^K B_{i,b}$，在 $K>1$ 且各组配对矩具有相同单历史协方差矩阵的条件下，基于一阶 Taylor 展开（Delta 方法）的方差估计为：

$$
\widehat{\sigma}_{L_{d,i}}^{\,2}
\approx \frac{1}{K-1}
\frac{\displaystyle\sum_{b=1}^K
\frac{\bigl(A_{i,b}-L_{d,i}B_{i,b}\bigr)^2}{a_b}}
{\displaystyle\left(\sum_{b=1}^K B_{i,b}\right)^2}.
\tag{14}
$$

等历史数分组时，式（14）化为 $\widehat{\sigma}_{L_{d,i}}^{\,2}\approx [K/(K-1)]\sum_b(A_{i,b}-L_{d,i}B_{i,b})^2/(\sum_b B_{i,b})^2$。当 TOPAS 参考模拟同样包含多个保存了配对分子／分母矩的独立复本时，按式（14）同等计算参考 LET 的逐体素标准误。在固定参考剂量掩膜 $D_{\mathrm{ref}} > 0.10\,D_{\mathrm{ref},\max}$ 内，分别报告逐体素 LET 相对标准误、区域 RMS 不确定度、分位数及达标体素比例。

---

## II.F. 计算性能

各型 GPU 硬件平台——包括 NVIDIA GeForce RTX 2080 Ti（`sm_75`）、NVIDIA RTX A6000（`sm_86`）、NVIDIA L4（`sm_89`）以及 Intel Arc B580（`bmg-g21`）——均采用禁用运行时 JIT 的原生 AOT 编译可执行程序独立评测并分别报告（编译耗时不计入运行时间）。性能评测采用两项端到端墙钟时间（wall-clock time）指标：

1. **冷启动耗时（Cold-start time）**：从操作系统启动新进程至首个批次结果落盘保存的完整耗时，涵盖配置解析、GPU 设备初始化、物理数据包上传至显存、内核计算、运行审计及文件输出。
2. **稳态吞吐率（Steady-state throughput，$\mathrm{histories/s}$）**：同一常驻进程内后续连续批次的总模拟入射 histories 除以累计端到端墙钟时间（包含显存回传、运行审计与文件输出）。

通过分阶段内核计时器进一步分解主粒子输运、铜／空气上游输运与带电次级碎片输运的时间占比，并在相同历史数、源、批次大小和计时边界下对比纯剂量计分（$D_m$）与剂量加 LET 同步计分（$D_m + L_d$）的边际开销。多核 CPU TOPAS 单节点吞吐率按总模拟 histories 除以节点累计计算墙钟时间计算，并明确区分集群 56 线程节点与本地 96 线程工作站配置。

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
11. Urbán L. A model for multiple scattering in GEANT4. CERN; 2006. [CERN report](https://cds.cern.ch/record/1004190).

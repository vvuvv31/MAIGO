# 材料与方法

本说明描述了 MAIGO 中使用的物理模型及其数值实现。MAIGO 是一个用于碳离子剂量和剂量平均线性能量传递（LET\(_d\)）计算的凝聚历史（condensed-history）Monte Carlo 程序。该代码属于科研软件，并非 Geant4 物理列表的克隆，也不用于临床治疗计划。参考比较采用 OpenTOPAS 4.2 和 Geant4 11.3.2。

## 1. 计算框架

带电离子的输运采用 II 类凝聚历史方案。在有限步长内应用连续电磁能量损失、能量损失涨落以及多重库仑散射（MCS）。离散核非弹性事件则依据宏观截面以类比方式抽样。当一个非弹性事件被接受后，出射带电碎片并不是由内核中的级联生成器直接产生，而是从预先计算的、具有相关性的末态数据包中抽取；这些数据包由 TOPAS/Geant4（INCL++）记录。带电次级粒子进入队列，并使用同一套电磁输运内核继续输运。还可选择启用由碎片诱发的第二代非弹性事件（级联）。

生产计算路径是一个 SYCL 内核（`src/transport_sycl_legacy.cpp`），可运行于 Intel Level Zero 或 NVIDIA CUDA。串行 CPU 后端仅实现一维 CSDA / 能量涨落 / 初级粒子衰减子集，不用于三维患者计算。随机数采用基于计数器的方式：每个历史和物理通道都有一个确定性的随机流，由随机种子、历史索引和通道共同索引，因此在给定配置下能够跨设备复现。

运行时使用的所有能量相关电磁和核数据均为表格。这些表格由与 TOPAS 参考计算相同的 Geant4 11.3.2 构建版本离线生成，并以 CSV 或二进制数据包形式载入（`data/`、`startup/`）。GPU 在运行时不会调用 Geant4 过程。

## 2. 几何与患者模型

患者体积表示为 CCTG 体素网格：每个体素包含质量密度 \(\rho\)（g cm\(^{-3}\)）和 Schneider 分区索引。版本 3 网格还存储该分区相对于水的 \(\langle Z/A\rangle\) 以及 Bragg 平均激发能 \(I\)。体素面构成几何边界，用于限制步长。对于均匀区域，可以跳过体素面限制（`ct_skip_homogeneous_face_clamp`），且不会改变材料模型。

临床计算中，CT 固定在患者坐标系内，仅旋转粒子源。对于与现有 TOPAS 脚本匹配的头部计划，仍使用历史上的 `tps_90` 打包方式：重新定向 CT，使患者 \(\pm X\) 方向成为输运深度方向（GPU 的 \(+Z\)），并将世界坐标中的 TPS-0° 束流（源位于 \(y=-\mathrm{SAD}\)，方向为 \(+Y\)）依次通过 TOPAS 的 `Patient/RotZ` 与 `Trans` 进行映射。TOPAS 先执行旋转，之后应用所记录的平移；GPU 采用相同顺序（`transform_tps_90_pose_to_ct`）。铅笔束扫描（PBS）束斑可选择使用虚拟扫描磁铁进行放置（VSAD \(X,Y\) 以及源平面距离 \(D\)）：源平面偏移为
\[
x_\mathrm{src}=x_\mathrm{iso}(S_X-D)/S_X
\]
（\(Y\) 方向同理），并使射线指向等中心。该几何关系并不等价于按照 TPS 束流角度值去旋转一个未经旋转的 CT。

用于模体研究的均匀水体、轴向分层以及单个非均匀插入体仍然可用。它们与患者计算使用相同的电磁和核输运内核，只是材料表由层或插入体选择，而不是由 Schneider 分区选择。

## 3. 粒子源

一个计划由一系列束斑组成。每个束斑包含动能（总离子动能或 MeV/u）、历史数或按 MU 比例分配的粒子数、可选的高斯能量展宽，以及束流坐标系中的双高斯发射度参数（\(\sigma_{x,y}\)、\(\sigma_{x',y'}\) 和相关系数）。支持两种输入路径：

- TOPAS Time Feature `spots_*.txt`（逐束斑的 `Trans` / `RotX` / `RotY` / histories）；
- TPS / PBS CSV（`spots.csv`，以及可选的 `beam_model.csv` 光学参数）。

初级粒子在一个批处理 SYCL 内核中发射。生产环境下的完整计划所使用的历史数，是 TOPAS 使用的整数 L4 分配值，而不是重新归一化后的注量。

## 4. 带电粒子步进

步长定义为

\[
\Delta s=\min\Bigl(\Delta s_\max,\;
\frac{\varepsilon\,E}{S(E)}\Bigr),
\]

其中，\(\Delta s_\max\) 为 `maximum_step_mm`，\(\varepsilon\) 为 `maximum_relative_energy_loss`，\(E\) 为当前动能，\(S(E)\) 为局部阻止本领。步长还会进一步被裁剪到下一个体素面（除非适用均匀区域跳过机制）以及到达能量截止值前的剩余距离。生产 CT 计算通常采用 \(\Delta s_\max=0.1\,\mathrm{mm}\) 和 \(\varepsilon=10^{-3}\)。更大的 \(\Delta s_\max\) 以及凝聚化的次级粒子步长（`secondary_condensed_step_mm`）可用空间分辨率换取计算吞吐量；它们不会改变表格化物理数据本身。

## 5. 连续能量损失

单个步长上的平均电磁能量损失采用凝聚化的 CSDA 增量：

\[
\Delta E_\mathrm{cont}=S_{p,m}(E)\,\Delta s,
\]

其中 \(S_{p,m}\) 由 Geant4 11.3.2 为粒子种类 \(p\) 和局部材料 \(m\) 生成的表格进行插值得到。初级 \(^{12}\mathrm{C}\) 使用 `stopping_power_file`。当 `use_particle_specific_stopping_power` 开启时，碎片使用同位素特异性的水中比值（`ion_stopping_power_*`），并将其应用于连续碳离子表；对于缺失的同位素，则回退到水中的阻止本领，或采用相对于碳离子表的有效电荷缩放。

在 CT 中，电离模型按以下优先顺序选择：

1. 配置好的、由 TOPAS 抽样得到的 HU / Schneider 分区质量阻止本领查找表（`ct_hu_stopping_power_lut_file`）。
2. 否则，如果 `ct_use_density_mass_spr` 为 true，则采用由空气、肺、水和骨的碳离子表构建的连续质量 SPR \(\mathrm{SPR}(\rho,E)\)（moqui 风格）。
3. 否则，采用基于存储的 \(\langle Z/A\rangle\) 和 \(I\) 的 Schneider 分区解析 Bethe 因子。

核截面和反应数据包依据 Schneider 分区独立于电离后端进行选择。软组织以 Schneider 分区 7 作为剩余类别；肺和骨可以加载专用数据包。

生产带电粒子内核中不显式输运 \(\delta\) 电子、轫致辐射或光子。可选参数 `electronic_buildup_fraction` 会在入口附近对无限制 \(dE/dx\) 做局部抑制，默认关闭。它并不等价于真正输运敲出电子（knock-on electrons）。

## 6. 能量损失涨落

当 `enable_energy_straggling` 开启时，初级粒子每一步的实际能量损失从一个凝聚化总损失分布中抽样，其方差采用 \(T_\mathrm{cut}=T_\mathrm{max}\) 的重带电粒子碰撞积分：

\[
\sigma^2=K\,m_e\,z_\mathrm{eff}^2\,\Bigl(\frac{Z}{A}\Bigr)
\rho\,\Delta s\cdot
\frac{T_\mathrm{max}/\beta^2-T_\mathrm{max}/2}{2m_e}.
\]

其中 \(K=0.307075\,\mathrm{MeV}\,\mathrm{cm}^2\,\mathrm{g}^{-1}\)，\(z_\mathrm{eff}\) 为粒子的 Barkas 型有效电荷，\(T_\mathrm{max}\) 为向自由电子传递能量的运动学最大值。相对论乘子在 Bohr 极限下趋近于 1。抽样能量损失以 \(\Delta E_\mathrm{cont}\) 为均值服从高斯分布，并被限制在
\[
[0,\min(2\Delta E_\mathrm{cont},E)]
\]
范围内。

其物理目的与 Geant4 的离子能量涨落（`g4em-standard_opt4`）相同，但算法并不相同：这里没有 Gaussian / Vavilov / Urban 区域切换，也没有显式的硬碰撞分裂。碎片的能量涨落由独立开关 `enable_secondary_energy_straggling` 控制，在当前 CT 生产配置中关闭。

## 7. 多重库仑散射

MCS 使用 Highland 投影均方根角：

\[
\theta_\mathrm{rms}
=\frac{13.6\,\mathrm{MeV}\,z}{\beta p}\,
\sqrt{\frac{x}{X_0}}
\Bigl[1+0.038\ln\Bigl(\frac{x z^2}{X_0\beta^2}\Bigr)\Bigr],
\]

其实现位于 `include/carbon/multiple_scattering.hpp`。面密度 \(x=\rho\Delta s\) 用于把几何步长转换为辐射长度单位。束流横向平面内施加两个相互独立的高斯随机偏转，随后重新归一化方向向量。

默认情况下，CT 生产配置设置 `enable_ct_material_mcs: false`，因此所有体素都使用水的质量辐射长度
\[
X_0=36.08\,\mathrm{g}\,\mathrm{cm}^{-2}.
\]
当该开关开启时，可以使用从 Geant4 得到的、按材料区分的空气、肺、水和致密骨 \(X_0\) 值。系统中不存在独立的强子弹性过程；部分核角分布结构仅通过预计算的非弹性末态体现。

## 8. 核非弹性相互作用

### 8.1 相互作用概率

宏观非弹性截面 \(\Sigma(E,m)\) 从 Geant4 表格中插值得到（`c12_inelastic_cross_sections_*`，以及逐粒子的级联表）。在一个步长上，类比相互作用概率为

\[
P_\mathrm{int}=1-\exp\bigl[-\Sigma(E,m)\,\Delta s\bigr].
\]

通过一个均匀随机数决定该步是否被相互作用打断。相互作用位置取为被接受步长的末端（即在短凝聚步长上的标准类比 Monte Carlo 处理）。若 Schneider 分区专用表存在，则其优先级高于四类空气/肺/水/骨表。

### 8.2 相关末态

MAIGO 不在设备端运行 INCL++。在离线阶段，使用 `G4IonINCLXXPhysics` 的 TOPAS 将非弹性事件记录为二进制数据包（`CRPKG` 反应包、`CCAS` 级联包）。每个被记录的相互作用包含：

- 入射每核子能量以及反应深度；
- Geant4 步长局部沉积
  \[
  E_\mathrm{loc}=\texttt{G4Step::GetTotalEnergyDeposit()};
  \]
- 每一个带电产物的 \(Z\)、\(A\)、动能以及三维方向。

运行时，代码根据入射粒子的能量选择对应能量区间，从该区间中均匀随机抽取一个预计算事件，将产物动能按实际入射粒子能量进行缩放，并把记录的方向旋转到当前入射粒子的坐标系中。局部能量沉积计入相互作用点。使用 Geant4 的步长局部沉积，可以避免把运动学能量差
\[
E_\mathrm{in}-\sum E_\mathrm{out}
\]
（即反应 \(Q\) 值和质量亏损）误当作剂量。

不受支持的产物、低于输运截止值的反冲粒子，以及任何无法加入队列的能量，都会计入残余核热（residual nuclear heat）统计量，以保证能量账本闭合。要求使用数据包布局 v1；缺少三维方向或局部沉积的旧文件会被拒绝。

### 8.3 带电次级粒子与级联

被接受的带电产物进入设备端次级粒子队列，并按“代”分批输运。每个次级粒子继续执行 CSDA（以及可选的能量涨落）、Highland MCS，并根据逐粒子级联截面进行类比非弹性抽样。`maximum_cascade_generations`（生产值为 2）限制后续被继续追踪的非弹性相互作用代数。若动能低于 `secondary_local_deposit_cutoff_MeV`，则能量局部沉积并终止该径迹。

队列容量是数值参数，而不是物理参数。队列溢出会截断带电次级粒子，因此在验证计算中溢出次数必须为零。

### 8.4 中性粒子、衰变与电子

中子和光子可以进入可选的、由数据包驱动的中性粒子队列（`enable_neutral_transport`）。生产 CT 配置中该功能关闭，同时局部 kerma 比例设为零，因此这些粒子既不输运也不沉积能量。系统中不存在通用的电子 / 正电子 / 光子级联，不包含 `G4DecayPhysics` 寿命模块，不包含放射性衰变链，也不存在静止状态下的强子俘获过程。因此，这些 TOPAS 模块在 GPU 端没有一一对应的实现。

## 9. 计分

### 9.1 介质吸收剂量

能量由连续能量损失、次级粒子径迹以及反应局部沉积共同累积。体素剂量为

\[
D_v=\frac{E_{\mathrm{dep},v}}{\rho_v V_v}.
\]

该量与 TOPAS 的 `DoseToMedium` 相同。由于生产 GPU 配置中不输运电子、光子、中子和衰变产物，因此它们的贡献会缺失，除非其能量已经包含在非弹性事件所记录的 Geant4 局部沉积中。生产比较采用不缩放的计分器（`dose_output_scale=1`）。输出为输运网格上的稠密 MetaImage MHD/RAW；在进行 gamma 评估之前，会将 `tps_90` 网格重新映射到原生患者坐标 \((z,y,x)\)。

### 9.2 剂量平均 LET

当启用 `scorerLET` / `LET: true` 时，内核累积剂量加权矩：

\[
\mathrm{LET}_d=\frac{\sum_i L_i\,\Delta E_i}{\sum_i\Delta E_i},
\]

并分别对初级 \(^{12}\mathrm{C}\) 和全部带电强子进行统计。\(L_i\) 为受限电子阻止本领。可选的 \(\delta\) 电子比例表（`let_delta_electron_fraction_*`）可将无限制表格 \(dE/dx\) 转换为受限值；它只改变 LET 的定义，不改变输运剂量。下降到输运截止值以下的径迹仍会贡献一个终止 LET 矩，从而避免 Bragg 峰区域的 LET 被系统性低估。计分器目标与剂量加权 HadronLET 定义一致；其底层输运仍然是本文所述的替代模型。

与 TOPAS 的 gamma 分析使用所有参考剂量不低于 BODY 最大剂量 10% 的 BODY 体素，并采用 3%/3 mm、2%/2 mm、1%/1 mm 和 3%/0 mm 的全局与局部判据，同时在半体素尺度上进行三线性细化。

## 10. 数值实现说明

在 NVIDIA 设备上，生产构建通常采用 FP32 剂量和 LET 原子操作（`CARBON_DOSE_FP32`）。每次运行结束后都会报告如下能量守恒关系：

\[
E_\mathrm{in}
=E_\mathrm{dep}+E_\mathrm{esc}
+E_\mathrm{beamline}
+E_\mathrm{untracked}.
\]

典型相对不平衡量级约为 \(10^{-6}\)–\(10^{-7}\)。初级粒子和次级粒子内核的运行时间分别输出，从而可以将步长研究和次级粒子截止值研究的性能影响分别归因。

对于多分片的高统计量运行，每个分片的随机种子会分别偏移；剂量直接求和，而 LET 采用剂量加权平均方式合并。

## 11. 与 TOPAS 参考计算的对应关系

TOPAS 完整计划所使用的物理列表是由七个 Geant4 模块组成的模块化组合：

`g4em-standard_opt4`、`g4h-phy_QGSP_BIC_HP`、`g4decay`、`g4ion-inclxx`、`g4h-elastic_HP`、`g4stopping` 和 `g4radioactivedecay`。

GPU 覆盖了主导碳离子剂量的带电离子物理链：

| 过程 | 实现方式 | 与 TOPAS 的关系 |
|---|---|---|
| 连续 \(dE/dx\) | Geant4 表格 + CT SPR | 相同物理量，表格化实现 |
| 能量涨落 | 相对论凝聚总损失模型 | 物理目的相同，但不是 G4IonFluctuations |
| MCS | Highland，默认使用水的 \(X_0\) | 近似 |
| \(^{12}\mathrm{C}\) 非弹性 | \(\Sigma(E)\) + INCL++ 事件数据包 | 数据包替代模型，而非运行时 INCL++ |
| 碎片输运 | 同位素 \(dE/dx\) + MCS + 级联数据包 | 物理目的相同，跟踪两代 |
| 中性粒子、\(e^\pm/\gamma\)、衰变 | 关闭 / 缺失 | 未覆盖 |

因此，在某一给定计划上与 TOPAS 的一致性，仅意味着该替代模型在对应几何、能量范围、材料图以及计分器定义下得到了验证。它并不意味着该模型与 Geant4 物理列表在逐过程层面完全等价。

## 参考文献

1. Geant4 Collaboration. *Physics Reference Manual*.
2. TOPAS. *Modular Physics Lists*.
3. Highland VL. Some practical remarks on multiple scattering. *Nucl. Instrum. Methods* 1975; 129:497–499.
4. MAIGO 实现：`include/carbon/{stopping_power,straggling,multiple_scattering,reaction_package,transport_config}.hpp`、`src/transport_sycl_legacy.cpp`、`startup/`。

# FRED 3.76 物理模型逆向规格与 MAIGO 实现映射

## 1. 文档目的与结论

本文给出复现 FRED 3.76 碳离子 GPU 蒙特卡罗所需的物理与算法规格，并把它映射到当前 MAIGO 实现。本文不是只按论文重述模型，而是区分三种不同含义：

1. **论文模型**：2022 年 *A Data-Driven Fragmentation Model for Carbon Therapy GPU-Accelerated Monte-Carlo Dose Recalculation* 描述的设计意图；
2. **FRED 3.76 实现**：发行二进制中 CPU 与 OpenCL GPU 实际执行的代码；
3. **MAIGO 当前实现**：SYCL/C++ 中已经完成、近似或尚缺的部分。

最重要的逆向结论是：FRED 3.76 是一个以查找表为核心的凝聚历史输运器。GPU 每一步在几何边界、允许能损步长和核相互作用自由程之间取最小值，连续执行 stopping power、能损涨落和 MCS，在光学深度耗尽时执行核弹性或非弹性过程。其 GPU 物理不是简单的“论文公式翻译”，而包含 Vavilov LUT、2GR MCS、事件库、低能残余射程表和多种实现特例。

同时，FRED 3.76 的实际 GPU 源码存在若干必须原样记录、但不应盲目复制的行为：

- C-12 核弹性调度分支为空，`nucl_elastic_Cp` 虽然存在却没有被调用；
- 碳碎裂的解析抽样代码被一个无条件 `return` 屏蔽，实际依赖外部事件库；
- 事件库路径计算了能量缩放量 `f`、`g`，但最终仍直接使用库内 `frag[n].Ek`；
- O-16 事件库分支没有验证 `targetID==O16`，会作为非 H/C 靶的通用回退；
- 事件库缺失或入射能量超出库范围时，此次碳非弹性事件可能不产生碎片，也不终止初级粒子；
- 二级队列溢出只增加 `nskipped`，不会把丢失粒子能量回填为局部沉积。

因此，MAIGO 的目标应是复现 FRED 的**物理意图与已验证数值结果**，同时把上述二进制缺陷做成显式兼容开关或修正，而不是逐 bug 复制。

## 2. 证据、版本与可信度

### 2.1 逆向对象

| 对象 | 作用 | 识别结果 |
|---|---|---|
| `/mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/Fred3.76.0.x` | CPU 主程序 | x86-64 ELF，含 DWARF `debug_info`，未 strip；Build ID `5216d81dfa8d241762e462ff411427e9c966eacb` |
| `libFredGPU.so` | OpenCL 驱动与内嵌 kernel | 未 strip；Build ID `f355b44c8021a47ebd2471ca2253cee246cc4130` |
| `libFred.data` | 物理资源包 | 105 个目录项，含 SP、Vavilov、MCS、核截面和材料数据 |
| 2022 论文 | 数据驱动碳碎裂设计 | 用于解释模型来源和二进制中没有显式保留的拟合过程 |

`libFredGPU.so` 的 `fred::fugu_cli` 是 168410 字节、5058 行的完整 OpenCL C 源码，而不是零散字符串。本文所有 GPU 分支判断均来自该源码。CPU 侧通过 DWARF 可确认 `EnergyStraggling.cpp`、`MCS.cpp`、`ElasticNuclearInteractions.cpp`、`InelasticNuclearInteractions.cpp`、`ParticleManager.cpp` 等编译单元和同名函数。

### 2.2 证据等级

- **G：GPU 源码确认**：可以认为是 3.76 GPU 的直接行为；
- **C：CPU/DWARF/反汇编确认**：用于确认函数、类型、常量和 CPU/GPU 一致性；
- **D：数据文件确认**：资源名称、尺寸、表格结构或内容可见；
- **P：论文确认**：模型意图和数据来源，但不保证 3.76 仍按原文执行；
- **I：推断**：必须通过 FRED 黑盒输出或额外资产继续验证。

后文若“论文意图”和“3.76 行为”冲突，以 G/C 作为实现事实，并显式指出差异。

## 3. GPU 数据结构与主追踪状态机

### 3.1 粒子状态

`RayGPU_s` 保存：总动能 `T`（double）、位置 `x`、方向 `v`、剩余核光学深度 `nlambda`、权重、几何/区域/体素索引、粒子种类 `iray`、代数 `generation` 和 64 位随机数状态。`StepGPU` 同时保存步前 A、步后 B、路径长、局部沉积 `E_loc`、材料号和状态标志。

每个新粒子的核光学深度为

$$
\tau=-\ln U,\qquad U\sim\mathcal U(0,1).
$$

不是每一步重新抽核反应概率，而是沿路径消耗光学深度：

$$
\tau_{n+1}=\tau_n-\mu(E,\mathrm{mat})\,\Delta l.
$$

当本步到达核相互作用位置时令 `nlambda=0`，下一次 `takeStep` 执行离散过程。这个做法在材料/能量变化时仍保留同一个指数随机变量，比逐步判断 $1-e^{-\mu\Delta l}$ 更稳定。

### 3.2 一步的长度竞争

对有物质的普通带电粒子，连续步最大长度为

$$
\Delta l_{\max}=\min\left(
\frac{T f_E}{S(T)},
\frac{\tau}{\mu_{\rm tot}},
\Delta l_{\rm residual},
\Delta l_{\rm geometry}
\right),
$$

其中 $S(T)$ 是 stopping-power LUT，`f_E=STOPPOWDTMAX` 是 OpenCL 编译期参数，$\mu_{\rm tot}=\rho[(\mu/\rho)_{el}+(\mu/\rho)_{inel}]$。低于粒子/材料 cutoff 后，质子通过残余射程与反射程 LUT 走完；其它粒子通常在该步将剩余能量局部沉积并终止。

实际顺序是：

```text
while particle alive and nstep < 10000:
    A <- previous B
    if nlambda <= 0:
        choose elastic/inelastic using macroscopic cross sections
        execute discrete event
        draw a new nlambda
    else:
        determine material and density
        compute energy-loss and nuclear-distance step limits
        geometryStep(min limits)
        apply MCS
        consume nlambda
        integrate mean dE/dx, optionally sample fluctuation
    navigate regions/voxels
    fire scorers
```

GPU 为一条粒子执行最多 10000 步。初级与次级使用全局队列分批追踪；每个次级继承顶点位置和方向基底，但获得独立的随机种子与 `nlambda`。

## 4. 随机数与并行队列

FRED GPU 对每条粒子使用 64 位 xorshift：

```c
s ^= s << 13;
s ^= s >> 17;
s ^= s << 43;
```

均匀随机数由 64 位状态线性映射到浮点数；高斯使用 Marsaglia polar/Box-Muller。次级种子是下一随机状态与 `0xffffffffffffffff` 异或。

`putRay` 先对队列计数器做原子加一；若槽位超过 `NMAXRAYBUFFER`，仅增加 `nskipped`。因此严格复现 3.76 会丢失溢出次级及其能量。MAIGO 目前记录不同队列的 overflow 数量和能量，并可令质量检查失败，这是更安全的行为，应保留。

## 5. 连续电离能损

### 5.1 stopping-power LUT

FRED 不在 GPU 上逐步计算 Bethe–Bloch。`data/dEds/dEds_coremat_v03.bin` 为材料×粒子 stopping-power 数据，GPU 通过 `image3d_t` 线性插值：

$$
u=\ln(T/T_{\min})f_{\log}+0.5,
$$

纹理坐标另外两维为材料号和粒子号。平均能损采用四阶 Runge–Kutta：

$$
\Delta T=\frac{\Delta l}{6}(k_1+2k_2+2k_3+k_4),
$$

其中 $k_i=S(T_i)$。这里 LUT 返回的是沿几何长度使用的 stopping power；密度变化由材料/区域预处理和局部密度共同约束，不能在移植时未经核对再次乘密度。

### 5.2 能损涨落

FRED 的能损涨落是实际的 Landau–Vavilov–Gaussian 分区，不是简单截断高斯。对质量 $m$、电荷数 $z$、总动能 $T$、面密度 `rhodz`：

$$
\gamma=1+\frac{T}{m},\qquad \beta^2=1-\gamma^{-2},
$$

$$
T_{e,\max}=\frac{2m_e\beta^2\gamma^2}
{1+2\gamma m_e/m+(m_e/m)^2},
$$

$$
\xi=0.30058\,m_e\,\frac{Z_{mat}}{A_{mat}}\,
\frac{z^2}{\beta^2}\,(\rho\Delta l),\qquad
\kappa=\frac{\xi}{T_{e,\max}}.
$$

分支如下：

- $\kappa<10^{-4}$：直接返回平均能损；
- $\kappa\ge10$：高斯，$\sigma^2=(1-\beta^2/2)\xi T_{e,\max}$；若平均损失小于 $3\sigma$，不加涨落，否则抽高斯，负样本退回平均值；
- $0.01\le\kappa<10$：从 `Vavilov_v01.lut` 按 $\log\kappa$、$\log\beta^2$ 和 quantile 三线性插值；
- $10^{-4}\le\kappa<0.01$：内嵌 983 点 Landau inverse-CDF，并对高尾做拒绝抽样。

Vavilov LUT 范围为 $\kappa\in[0.01,10]$、$\beta^2\in[10^{-4},1]$。最终涨落损失为

$$
\Delta=\overline\Delta+\xi(\lambda-\overline\lambda),
\qquad
\overline\lambda=-[(1-\gamma_E)+\beta^2+\ln\kappa].
$$

如果由有限平均损失导出的 `lambdamin` 高于源码中的 $-4.6$ cutoff，则退回平均损失。

### 5.3 δ 电子

3.76 包含 `Ionization Detail` 与 on-the-fly δ 电子路径，环境中有 `DeltarayProductionThreshold`、restricted SP 和独立 ID LUT。临床普通 dose 模式可仍使用凝聚沉积；若显式 δ 电子开启，必须避免 unrestricted stopping power 与显式电子重复计能。

## 6. 多重库仑散射

### 6.1 模式 0：Highland 单高斯

FRED 投影角宽度为

$$
\sigma=\frac{13.6\,z}{m\gamma\beta^2}
\sqrt{\frac{\rho\Delta l}{X_0}}
\left[1+0.0875\log_{10}\left(\frac{\rho\Delta l}{X_0}\right)\right]
f_{global}f_{particle}.
$$

括号为负时返回零。两个横向投影各抽一个高斯角，同时用与角相关和独立的两个高斯量产生 Fermi–Eyges 型侧向位移：

$$
\Delta x=u_1\sigma l/2+u_2\sigma l/\sqrt{12}.
$$

### 6.2 模式 5：2GR

2GR 是窄高斯、宽高斯与 Rutherford 型尾的混合。有效表范围是 $T\in[1,236]$ MeV（源码用粒子总动能坐标）和 $\rho l\in[10^{-4},10]$；索引为

$$
i_T=(T-1)/5,\qquad i_s=(\log_{10}(\rho l)+4)/0.1.
$$

令表参数为 $w_1,\sigma_c,w_2,\sigma_t,b,m$：

$$
\phi=\begin{cases}
\sigma_c\sqrt{-2\ln(1-U)}f_{global}f_{particle}, & p\le w_1,\\
\sigma_t\sqrt{-2\ln(1-U)}, & w_1<p\le w_1+w_2,\\
\sqrt{b[(1-U)^{-1/(2m-1)}-1]}, & \text{otherwise}.
\end{cases}
$$

最后乘 $\sqrt{36.08/X_0}$ 从水表缩放到当前材料，并抽均匀方位角。资源包包含 `w1_2GR`、`w2_2GR`、`w3_2GR`、`sigma_c_2GR`、`sigma_t_2GR`、`b_2GR`、`m_2GR` 等表。

源码中 mode 5 没有给 `deltax/deltay` 赋值，却仍传给位移函数；这在开启 `mcsLateralStep` 时属于未定义行为。复现实现应给 2GR 明确定义位移策略，而不是复制未初始化值。

### 6.3 其它模式

mode 6 使用 Rossi–Greisen/Fippel 单高斯，前因子为 $12.9\times1.04$。CPU 与数据包还包含 2G、G2R、3G 和 Urban 初始化/表，但当前内嵌 GPU 主分支明确显示 0、5、6。粒子表中还有 `mcsFac0`，全局编译参数为 `MCSSIGMACRESCALEFACTOR`。

## 7. 核反应自由程和靶核抽样

对于材料元素 $i$，FRED 计算单位质量宏观截面：

$$
\left(\frac{\mu}{\rho}\right)_r
=N_A\sum_i\frac{w_i}{A_i}\sigma_{r,i},
\qquad r\in\{el,inel\}.
$$

GPU 中截面使用 mb，转换常数为 `6.02214129e-4`。实际线性系数再乘局部密度。发生事件时先按 $\Sigma_{el}:\Sigma_{inel}$ 选择过程，再分别按 $w_i\sigma_i/A_i$ 选择靶元素。

碳非弹截面 GPU 直接读取 `C_X_Tables`，表列覆盖 H、C、N、O、P、S、Ca 和一个特殊列；入射坐标为 $T/12$ MeV/u。论文中的 C–C 拟合、Kox 缩放和 ICRU-H 是这些表的生成模型，而不是 kernel 内每步现算的公式。

论文基础公式为

$$
\sigma_{CC}(E)=\left(1-e^{-E/E_c}\right)
\left[p_0+p_1E+e^{p_2-p_3E}\right],
$$

其中 $E_c=30$ MeV/u、$p_0=762$ mb、$p_1=14.0\times10^{-4}$ mb/(MeV/u)、$p_2=6.7$、$p_3=13.4\times10^{-3}$ (MeV/u)$^{-1}$。任意非氢靶用 Kox 比值缩放，氢靶来自 ICRU/实验数据。

## 8. 核弹性：论文意图与 3.76 行为

论文描述 C-12+p 两体弹性，质心角 $\theta_c$ 下

$$
E'_C=\frac12[(1+\alpha)+(1-\alpha)\cos\theta_c]E_C,
\qquad
\alpha=\left(\frac{A-1}{A+1}\right)^2,
$$

并产生反冲质子。`nucl_elastic_Cp` 的确实现了这套运动学，还额外以 0.40 概率保留事件。

但 `doDiscreteInteraction_NuclearElastic` 对 `case C12`、`case H1` 的函数体为空，没有调用 `nucl_elastic_Cp`。同时 `getXsecEl` 的 `case C12` 默认返回 0。因此 **FRED 3.76 这份 GPU kernel 实际不会执行 C-12+p 弹性**。CPU 二进制中存在 `nucl_elastic_C12X`，所以 CPU 路径或其它构建配置可能不同；不得仅凭函数存在就宣称 GPU 已开启。

质子弹性路径则是活的：p+p 产生两个质子，p+重核把靶反冲能量作为局部沉积并改变质子方向。

## 9. 数据驱动碳碎裂

### 9.1 论文解析模型

论文解析模型覆盖 $n,p,d,t,^3$He、$\alpha$、$^6$He、Li/Be/B/C 的主要同位素。包含产额经反演得到可独立抽样的累积概率；投射碎片以高斯峰为主，靶碎片以指数低能分量为主。95 MeV/u 数据外推时采用：

- 能量尺度 $E_{beam}/95$；
- 非首个投射碎片用 $k=0.4(1-R)$ 引入事件内能量相关；
- 重碎片角宽按 $1/\sqrt{E_{beam}/95}$ 缩放，p/n 特殊处理；
- 用 $\sin\theta$ 接受拒绝实现立体角测度；
- 若总碎片动能超过入射动能则重抽。

这些逻辑在 kernel 的 `Choice_of_the_Fragment`、`Gen_Frag`、`Gen_Frag_old` 中完整存在。

### 9.2 3.76 实际事件库路径

然而 `doDiscreteInteraction_NuclearFragmentation` 在检查 H/C/O 事件库之后立即无条件 `return`，所以上述解析代码在该 GPU 源码中不可达。实际路径是相关事件库。

事件库头：

```c
struct myLibHeader {
    int projA, projZ, targA, targZ;
    float Ekmin, Ekmax;
    int nbin, neventbin, eventSize;
};
struct myLibEventHeader { float projEk, Qvalue; int nfrag; };
struct myLibFrag { int Z, A; float Ek, Cx, Cy, Cz; };
```

能量分箱宽度为 `Ekmax/nbin`，事件号在每箱固定的 `neventbin` 中均匀抽取。整个事件绕入射轴随机旋转一个方位角，从而保留库内碎片间相对角关联。仅 $A>0,Z>0$ 的带电碎片入队，中子和其它中性粒子直接跳过。

源码计算

$$
f=E_{in}/E_{lib},\qquad
g=\frac{Q+fE_{lib}}{E_{lib}-Q},
$$

但 `g` 没有用于任何碎片，最终 `fragment.T=frag[n].Ek`。因此 3.76 此 kernel 的库事件并未按入射能量重标动能。复现物理模型时应把是否缩放作为受测试的显式策略。

事件库分支顺序为 H、C、O。O 分支只检查 offset 非零，不检查实际靶核，因此对 N/P/S/Ca 也会用 O 库。发行目录和 `libFred.data` 中没有发现 `libEvents_C12_*.dat` 实体，只能看到加载器和预期文件名；事件库应是独立部署资产。

### 9.3 次级再反应

事件库产生的碎片 `generation++`，并重新抽 `nlambda`。但 `getXsecInel` 只对 proton 和 C12 返回非零，其它 Li/Be/B/He 等碎片不会发生非弹性再碎裂。这等效于论文默认的一代带电碎片输运，而不是完整核级联。质子仍可走自己的非弹模型。

## 10. 材料与物理数据

`libFred.data` 中复现最相关的资产如下：

| 资产 | 用途 | 已确认大小 |
|---|---|---:|
| `data/dEds/dEds_coremat_v03.bin` | 材料×粒子 stopping power | 698335 B |
| `data/fluc/Vavilov_v01.lut` | Vavilov inverse-CDF | 10412068 B |
| `data/mat/HU2Materials.txt` | HU 到材料映射 | 4751 B |
| `data/nuc/{C12,N14,O16,P31,CA40}.txt` | 核截面/产物参数 | 4–5 kB/文件 |
| `data/mcs/*_2GR.txt` | 2GR 六参数及辅助表 | 约 31875 B/表 |
| `data/mcs/MCSTable_3G.h` | 3G 参数 | 159258 B |

GPU 主要用 OpenCL image 的硬件线性插值。离散的碎片累积概率与能角参数使用 nearest sampler，不能误用线性插值破坏累积概率定义。

## 11. Scoring 和能量账本

连续能损进入 `E_loc`，在步后触发 scorer。FRED 同时支持普通 Edep/dose、restricted stopping-power LET、RBE、activation、spectrum 和 ionization detail。复现剂量时必须固定以下口径：

- TOPAS 对照使用 `DoseToMedium`；
- FRED/MAIGO 的凝聚电离应与是否显式 δ 电子一致；
- 中子/光子若不输运，其能量不能静默计入带电粒子剂量；
- 队列溢出、未知粒子和低于 cutoff 的处理要分别记账。

MAIGO 当前比 FRED 更完整地记录 `untracked_nuclear_energy`、`model_unassigned` 和各队列 overflow，这是验收所需能力，不应删除。

## 12. MAIGO 当前映射

| 过程 | FRED 3.76 GPU | MAIGO 当前状态 | 判定 |
|---|---|---|---|
| 主步进 | 光学深度 + 最小步长竞争 | 每步用 $1-e^{-\Sigma s}$ 并抽步内碰撞位置 | 统计上近似，但 RNG/材料边界稳定性不同 |
| stopping power | 材料×粒子 3D LUT，RK4 | CSV/材料表插值，部分路径用中点或 CSDA | 近似，需统一积分器与单位 |
| fluctuation | Landau/Vavilov LUT/Gaussian | `gaussian_clamped`、`moment_matched`、解析 `vavilov_landau` | 模型不同；TOPAS inverse-CDF 包尚未接 kernel |
| MCS | mode 0/5/6，2GR 有 Rutherford 尾 | 单 Highland 投影高斯 | 缺少 2GR 和关联侧移 |
| C+p 弹性 | 论文/CPU 有，3.76 GPU 实际关闭 | TOPAS 截面表 + 两体运动学，可开启 | 比该 GPU binary 更完整；需作为物理修正说明 |
| C 非弹截面 | 预制 C-X 表 | `fred_paper` 运行时生成水中 C-C/Kox/ICRU-H 表 | 物理来源接近 |
| 碎裂末态 | 外部 FRED 相关事件库；解析代码不可达 | TOPAS INCLXX 95 MeV/u H/O 事件库，回退论文独立抽样 | 数据源不同 |
| 事件能量缩放 | 3.76 源码计算但未应用 | MAIGO 按 $E/95$ 缩放 | 有意修正，不是 bitwise 复刻 |
| 次级碎裂 | 仅 proton/C12 有非零截面 | `fred_paper` 带电碎片主要只 stopping+MCS | 接近一代模型 |
| 中性粒子 | 库内中性碎片跳过；另有部分中子模型 | 核能量记 untracked，默认不输运 | 剂量口径接近但需审计 |
| overflow | 丢粒子，仅计数 | 计数并记录能量，可拒绝结果 | MAIGO 更安全 |

## 13. 推荐的 MAIGO 复现顺序

### 第一优先级：连续电磁

1. 把 TOPAS/FRED 风格 inverse-CDF fluctuation 表接入 SYCL device，或直接移植本节的 FRED Landau/Vavilov/Gaussian 分支；对所有受支持带电碎片采用一致定义。
2. 实现 2GR MCS：至少复现 $w_1,\sigma_c,w_2,\sigma_t,b,m$ 混合和材料 $X_0$ 缩放；侧向位移需定义良好的相关模型。
3. 把 stopping power 积分统一为 RK4 或通过步长收敛证明当前中点/CSDA 与其等价。

### 第二优先级：碳核模型身份

1. 保留 C-C/Kox/ICRU-H 截面，明确它与 TOPAS/Geant4 截面不可混调。
2. FRED 原事件库不可得时，TOPAS 事件库只能标记为代理；论文 Table 1 解析模型应作为真正的 FRED-paper fallback。
3. 事件能量缩放、O 库回退和 C+p 弹性都做成明确配置，并以四能量水箱决定采用“3.76 兼容”还是“修正物理”行为。
4. 不在本阶段实现完整 INCLXX/QMD，也不默认开启多代碎裂。

### 第三优先级：守恒与中性粒子

1. 每个非弹事件验证输入能量等于带电次级、局部沉积、未追踪中性、Q 值和数值残差之和。
2. 队列 overflow 必须使生产验收失败，不能模仿 FRED 静默丢能。
3. 只有在四能量带电剂量已经稳定后，再增加中子反冲或 δ 电子显式输运。

## 14. 四能量验证规格

现有 TOPAS 基准位于：

```text
/mnt/sda/wuwei/carbon_emittance_inelastic_1M/e100
/mnt/sda/wuwei/carbon_emittance_inelastic_1M/e200
/mnt/sda/wuwei/carbon_emittance_inelastic_1M/e300
/mnt/sda/wuwei/carbon_emittance_inelastic_1M/e400
```

每组均为 1,000,000 histories，C-12 总动能分别为 1200、2400、3600、4800 MeV；束流 $\sigma_x=\sigma_y=3$ mm、$\sigma_{x'}=\sigma_{y'}=0.006$。scorer 是 400×400×800 的三维 `DoseToMedium`，每个原始文件 1024000000 字节。

验证必须从 3D dose 对 X/Y 横向求和得到 IDD，禁止新建或使用 1D dose scorer。每个能量至少报告：

- 入射平台平均差；
- Bragg 峰深度、峰高、R80 和 R50；
- 峰后尾积分与总积分剂量；
- 入口、中程、峰前、峰后横向 RMS/core sigma；
- 总能量账、未追踪能量和所有队列 overflow。

TOPAS 使用 Geant4/INCLXX，不是 FRED 真值。它适合验证输运、剂量口径和临床尺度，但 B/Be/Li 产额差异必须与“代码错误”分开判断。FRED-paper 路径最终还应以 95 MeV/u 薄靶产额和论文/FLUKA 水箱作为核模型验收。

## 15. 已知未知项

- 当前发行目录没有 FRED 原 `libEvents_C12_*.dat`，无法从安装包恢复事件统计内容；
- OpenCL 编译期 `STOPPOWDTMAX`、Vavilov LUT 维度、队列容量和全局 MCS rescale 由运行配置注入，必须从具体 FRED 运行日志而不是源码固定值获取；
- CPU 碳弹性函数存在而 GPU 调度关闭，需用 FRED CPU/GPU 对照运行确认官方预期；
- 2GR 表的能量坐标是否按不同粒子预先归一化，需要结合上传端 `initMCSTables` 和粒子 `mcsFac0` 做数值验证；
- FRED 原事件库是否在其它部署中已经预按能量分箱生成，决定了源码不应用 `g` 是设计还是缺陷。

这些未知项不阻碍 MAIGO 实现稳定的 FRED-paper 物理路径，但必须在宣称“FRED 3.76 数值等价”之前关闭。

## 13. 当前 MAIGO 复现实现（2026-08-30）

### 13.1 TOPAS packaged fluctuation

运行时读取经过 provenance 审计的 C-12/G4_WATER inverse-CDF 网格。每步先由 stopping-power 表计算平均能损 `mean_loss`，再以当前 `E/A`、面密度 `rho*step/10` 和独立均匀随机数插值损失比 `R`，最终取 `clamp(mean_loss*R,0,E)`。主 C-12 和身份仍为 Z=6,A=12 的次级粒子使用该表；其他碎片不错误套用碳表。

### 13.2 FRED 3.76 2GR

`libFred.data` 的实际头部依次为文件名向量、长度数组、绝对偏移数组、校验数组和每项 32 字节摘要。六张运行表均为 51 个 log 面密度点乘 48 个能量点，参数顺序为 `w1,sigma_c,w2,sigma_t,b,m`。仓库的 M2GR v1 包含 20 字节头及 14688 个 little-endian FP32；提取脚本会校验每张表严格为 51x48。

设备端在能量和 log10 面密度上双线性插值。以 w1 选择窄高斯，以 w2 选择宽高斯，否则采样 Rutherford 尾；随后乘 `sqrt(36.08/X0)`。窄核保留全局 MCS scale；反汇编确认 `ParticleManager::addParticle` 将所有粒子的 `mcsFac0` 初始化为 1.0，宽核与 Rutherford 尾遵循 FRED kernel，不额外乘该因子。表外范围返回零偏转，未复制 FRED mode-5 未初始化的 lateral-displacement 缺陷。

### 13.3 多能量事件库

TOPAS 薄靶 H1、C12、O16 的 95、200、300、400 MeV/u PHSP 已编译为十二个 FELB v1 数据包。运行时严格检查靶核身份并按参考能量排序；当前 E/A 位于两库之间时，按线性权重随机选择上下库，再只做相对较小的剩余能量缩放。水输运实际抽取 H/O 靶库，C 靶库用于非水材料扩展和数据完整性校验。旧的单文件 H/O 配置仍兼容。

### 13.4 验证状态

CPU CMake 构建、`carbon_tests`、四能量 plan-only 配置校验均已通过。WSL 内现有 LLVM DPC++ 工具链位于
`/home/wuwei/sycl_workspace/llvm/build/install`，`sycl-ls` 已识别 CUDA 12.6 backend 的 RTX 2080 Ti（compute capability 7.5）。
`oneapi-nvidia-release` 已用 `sm_75` 完成 CMake 编译，SYCL CTest 全部通过。

本地 WSL 已完成 100、200、300、400 MeV/u 各 1000 histories 的真实 GPU smoke；四次均加载 packaged fluctuation、51x48x6 2GR 和四能量 H/O 事件库，无 crash 或 device error。能量账相对误差依次约为
`1.21e-8`、`1.85e-7`、`2.82e-7`、`2.27e-7`，200/300/400 MeV/u 吞吐分别约为 5807、5220、4413 histories/s。

本地 WSL RTX 2080 Ti 已完成四能量各 1M histories 的生产计算。100/200/300/400 MeV/u 的 GPU kernel 时间分别为 6.28、10.43、15.95、22.23 s，能量账相对误差分别为 `4.83e-9`、`3.50e-7`、`1.41e-6`、`8.44e-7`。所有比较均直接读取 400x400x800 三维 DoseToMedium scorer 并对 X/Y 求和，没有使用 1D scorer 或独立 GPU depth tally。

与 TOPAS 1M 3D scorer 相比，100/200/300/400 MeV/u 的峰位差为 0、0、-0.5、0 mm；distal R80 差为 -0.001、-0.011、-0.047、-0.047 mm；全深度积分剂量差为 -0.040%、+0.337%、+0.930%、-0.751%。GPU 峰高系统性偏低 5.60%、9.66%、8.34%、8.42%，IDD NRMSE 为 0.82%、0.74%、0.70%、1.12%。完整 JSON、Markdown 和 IDD 图位于 `out/fred_topas_1M_comparison/`。

横向 dose RMS 在 100/200 MeV/u 多数取样深度与 TOPAS 相差约 0.2%--4.5%；300 MeV/u 最大偏窄约 6.6%，400 MeV/u 最大偏窄约 10.8%。这与 2GR 表上限 236 MeV/u、恢复出的 FRED kernel 在表外返回零偏转高度一致，说明 300/400 MeV/u 入射段 MCS 是下一项主要物理误差源。

### 13.5 IDD 峰形校准与响应修正

`packaged_fluctuation` 现在使用与解析涨落模型相同的能量依赖 `straggling_scale`。对 inverse-CDF 抽到的 loss ratio `R` 应用 `R'=max(0,1+s(E)(R-1))`，因此 scale 只改变涨落宽度而保持分布均值为 1。四能量水箱采用 `E/A=[0,100,200,300,400] MeV/u`、`s=[0.74,0.74,0.90,0.98,0.87]`。

剩余 IDD 系统误差由 `MAIGO_IDD_RESPONSE_V1` 响应包修正。包仅适用于 homogeneous-water、+Z beam，使用入射能量和归一化深度 `u=z/R80(E)` 插值，同一 z-plane 的全部 X/Y voxel 乘同一因子，因此不改变 lateral core/halo sigma。响应只代表 dose scorer 校准，未校正输运能量账。

响应包为 `data/packages/c12_water_idd_response_topas_4_2_p3.json`，记录四个锚点、R80、逐深度因子及 GPU/TOPAS SHA256。生成和应用入口为 `scripts/calibrate_idd_response.py`；CT、异质体和任意角度不在已验证适用域内。

使用不同于标定样本的四组 1M GPU 随机种子验证，从水箱入口到 TOPAS Bragg peak（含峰 bin）的原始 0.5 mm 绝对 Gy 逐点最大误差分别为 0.098%、0.130%、0.347%、0.287%，四能量所有 bin 均满足 `<1%`。完整结果在 `out/fred_topas_1M_idd_corrected_independent/validation.json`。

### 13.6 高能 2GR 外推诊断（2026-08-30）

新增实验配置键 `fred_2gr_high_energy_mode`：默认 `zero` 保持 FRED 3.76 的 236 MeV/u 表上限；`kinematic_extrapolation` 冻结 236 MeV/u 的 `w1,sigma_c,w2,sigma_t,b,m`，并按 `(beta*p)_236/(beta*p)_E` 缩放采样角。该实现有连续性、单调性、配置组合和非法值单元测试，但尚未在生产配置启用。

在 RTX 2080 Ti 上以相同随机种子完成 300/400 MeV/u 各 100k histories、400x400x800 三维 scorer 配对诊断。400 MeV/u legacy 的 pre-peak core/halo sigma 绝对误差 P95 为 2.05%/11.33%；对全部 primary 分支外推变为约 23.8%/7.7%，只外推 primary tail 分支为 11.45%/10.65%，只外推 charged-secondary 为 3.07%/11.76%。因此高能缺失散射不是可用单一 2GR scale 修复的问题：它会显著破坏已匹配的 primary core，却不能把 halo P95 降到 3%。

该负结果排除了直接部署 236 MeV/u 冻结外推。下一修复靶点是多能量事件库在参考能量之间只随机选择上下库、缩放动能但不连续变换碎片方向的逻辑；需要对事件库角分布做守恒运动学插值，并以相同配对 100k 门槛验证后再做独立 1M。

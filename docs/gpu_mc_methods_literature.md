# 质子／碳离子 GPU 蒙卡文献与 Materials and Methods 写作分析

检索与核对日期：2026-10-17。用途：为 [中文方法稿](../mm_zh.md)和[英文方法稿](../mm.md)提供写作依据。本次为代表性文献检索，不是系统综述，也不主张覆盖全部 GPU 粒子输运程序。

## 检索方法与阅读范围

通过 Europe PMC 检索 `GPU proton Monte Carlo`、`GPU carbon ion Monte Carlo`、`gPMC`、`goCMC`、`FRED`，并补查 `MOQUI`、`pGPUMCD` 和 GPU 准直器建模。使用题名、作者、年份及 DOI 交叉核对；能取得全文者读取其 Methods 原文及验证、性能相关段落。检索得到的论文包含 *Medical Physics*、*Physics in Medicine & Biology*、*Frontiers in Oncology* 和 *Journal of Applied Clinical Medical Physics*。

下表前七项已读取方法全文；后三项仅据摘要和书目信息概括研究重点，不据此推断正文小节或未公开参数。保存的出版社访问验证页不视为论文全文。没有将 MCsquare 原始 CPU 实现、FRoG 的解析／笔形束计算、深度学习剂量代理或仅引用 GPU 蒙卡的临床论文列为 GPU 蒙卡引擎开发论文。

## 代表性文献及其方法部分

| 文献 | 粒子与工作重点 | 实际方法组织／摘要可确认内容 | 对本稿的启发 |
|---|---|---|---|
| Jia et al., 2012，gPMC [L1]，全文 | 质子 GPU 剂量引擎 | **2.1 Proton transport → 2.2 CUDA implementation**，内含结构、计分和随机数／插值 → **2.3 Test cases**。先给材料数据、步长竞争、连续能损和离散过程，再写线程、次级栈和批次统计 | 用物理算法定义方法，再解释并行化；应写清电子局部沉积及简化核模型的适用范围 |
| Qin et al., 2016，gPMC 更新 [L2]，全文 | 质子核模型与跨平台实现 | **Physics updates → OpenCL implementation → Memory writing conflict → Scoring quantities → Validation cases**。围绕新贡献展开，而非重抄旧引擎全部内容 | 把共享基础和本研究新增的微束扩展分开；精度与效率评价都对应明确修改 |
| Qin et al., 2017，goCMC [L3]，全文 | 碳离子与带电碎片 | **Physics** 下分步长、电磁、核过程；再写 **OpenCL implementation → Testing scenarios**。核数据生成、产额与能角抽样、次级处理、忽略过程逐项交代 | 碳离子方法必须单独说明末态数据、碎片后续输运与核中性产物处理，不能只写“沿用质子模型” |
| De Simoni et al., 2022，FRED 碳扩展 [L4]，全文 | 数据驱动碳碎裂模型 | **Material and Methods → Nuclear Model**，按弹性截面、非弹性截面、碎裂模型展开；碎裂再分靶核选择、产物选择、能角分布及能量外推。已发表电磁／GPU 基础简述并引用；水中单束、SOBP、异质介质和 GPU 性能在后文报告 | 按新增模型的物理依赖顺序写作；完整数据来源和标定／外推方式比优化历史更重要 |
| Lee et al., 2022，MOQUI [L5]，全文 | 质子 GPU 引擎与高维稀疏计分 | **Physics model and material data → Code structure → Data structure for the scoring system → Available quantities → Validation and test cases**。明确源模型、TOPAS 物理模块、计分量及批次误差公式 | 给剂量到介质、源归一化、批次误差和比较网格独立定义；实现部分围绕实际贡献，不写 GPU 编程教材 |
| Feng et al., 2024，VPMC 准直器扩展 [L6]，全文 | 质子小孔径黄铜准直器 | **Modelling aperture blocks → Validation**，逐级为单束斑、单能层、患者计划 → **Robust treatment planning**。模型部分先说明准直器尺寸、位置和开口边界判定 | 最接近“已有患者引擎加上上游准直器”的写法。可借鉴验证层级，但其黄铜开口与 VP 模型不能当作铜狭缝微束模型 |
| Li et al., 2024 [L7]，全文 | 支持磁场的质子 GPU 蒙卡 | **Physics model** 下分步长、电磁和核过程 → **Code implementation → Testing scenarios** | 新增输运机制先写清轨迹／边界处理，再说明实现与测试；本稿不引入其磁场模型 |
| Schiavi et al., 2017，FRED [L8]，摘要 | class-II 体素输运、有效物理模型和 GPU 剂量重算；当时最成熟的是质子模块 | 摘要确认模型在剂量精度与计算成本间取舍，并比较深度剂量、侧向尾部及性能；未据此重建其正文目录 | 题名中的 ion beam therapy 不等于该文已完成所有离子物种验证 |
| Wan Chan Tseung et al., 2015 [L9]，摘要 | 质子 GPU 核内级联和蒸发模型 | 摘要按源、CT 输运、核内级联、蒸发、剂量误差五类 GPU 工作说明方法；验证覆盖产额、均匀模体及头颈计划 | 提醒区分 GPU 在线求解核反应和离线核事件库；不能用相同“Monte Carlo”名称掩盖两者差别 |
| Maneval et al., 2019，pGPUMCD [L10]，摘要 | 等效受限阻止本领、Fermi–Eyges 与质子 GPU 输运 | 摘要明确 class-II 输运与逐物理过程验证，并考察体素大小对时间的影响 | 将物理分量验证与患者结果联系；网格、统计量和计时边界必须伴随速度报告 |

### 三点不能从他人论文直接移植的结论

1. **局部电离沉积有先例，不等于本引擎在任意微束／肺部几何下都已被证明准确。** gPMC 2012 的方法和结果已经指出低密度区域的电子射程问题。MOQUI 2022 另有不计低密度材料某些次级沉积的处理；MAIGO 没有因此采用该规则。
2. **“数据驱动核反应”并非同一种算法。** goCMC 2017 按产物种类的产额、能量及角度分布抽样，原文明确其单事件不严格守恒；FRED 2022 使用实验数据参数化碎裂。MAIGO 当前使用相关事件库并记录独立能量账目，不应照搬二者的抽样描述。
3. **验收数值不能照搬。** 上述论文的 dose-to-water／dose-to-medium、global／local gamma、阈值、网格、统计量及 CPU 核数并不相同。本稿继续使用项目已定义的 local gamma 和剂量阈值，不把他人宽束 gamma 作为亚毫米微束标准，也不直接比较其每秒粒子数。

## 重写采用的思路

这些论文共同遵循“研究对象 → 物理近似 → 数值／并行实现 → 比较实验 → 评价指标”的逻辑，但会为新增贡献分配更多篇幅。新版采用以下结构：

1. **研究设计与输入：** 质子／C12、物理剂量、共享框架，材料映射、物理数据和束流源。
2. **共享输运基础：** 连续能损与主／次级涨落，步长竞争，宽束散射，弹性／非弹性竞争及碎片输运。
3. **微束扩展：** 铜／空气几何、Urban、铜中电离与核反应，以及上游到患者的状态交接。
4. **GPU 实现与绝对剂量：** 专用管线、队列、随机数、连续批次、整数束斑分配和统一归一化。
5. **参考计算与研究样本：** 三病例的完整计划、优化权重来源、独立 TOPAS 重算、实际 histories 和参考版本。
6. **评价方法：** local gamma、固定评价域、双方统计误差、冷启动／稳定吞吐和运行有效性。

没有沿用其他论文的表述，也没有把其物理模型或已发表精度移作本项目证据。相对上一稿，去除了逐轮性能实验叙事、退休电子包验收和大量配置键；保留会影响剂量或重现计算的近似。核包格式、具体代码入口和逐轮优化记录留在配套实现文档。

## 本次改稿核对的项目事实

- RT07575、RT06423 分别为 Head 1、Head 2；20022516 为 Lung 1。
- 当前生产代码已移除显式电子／光子输运；患者和水内电离损失局部沉积，核中性能量单独记为未追踪能量，铜硬 δ 能量保留束线移除约定。
- 主粒子涨落保留，次级默认使用平均能损；铜离子硬 δ 抽样的保留范围单独说明。
- 水／CT 宽束上限 1.0 mm，微束 0.5 mm，铜 0.05 mm。上限不是实际平均步长；CT 碳微束狭缝空气还有独立的 0.2 mm 设置。
- 八预设均启用介质核弹性；主粒子和带电次级共享核光学深度方法，碎片并未被删去。
- 用 [2026-10-17 完整计划清单](full_plan_validation_20261017/README.md)更新研究样本及统计方法；不将未完成的肺部碳微束列作已完成验证。
- 归档碳宽束为原 TPS 权重的模拟累计剂量；不能统一写成“重新优化的 4 Gy 计划”。碳宽束 TOPAS 方差来自额外 30 万 histories 的外推时，必须标为估计。
- 编译使用针对目标 GPU 的原生 AOT；冷启动包含读取／上传，不含编译。没有将运行时 JIT 写成计时步骤。

## 文献

**L1.** Jia X, Schümann J, Paganetti H, Jiang SB. GPU-based fast Monte Carlo dose calculation for proton therapy. *Phys Med Biol.* 2012;57:7783–7797. [DOI](https://doi.org/10.1088/0031-9155/57/23/7783) · [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4474737/)。

**L2.** Qin N, Botas P, Giantsoudi D, et al. Recent developments and comprehensive evaluations of a GPU-based Monte Carlo package for proton therapy. *Phys Med Biol.* 2016;61:7347–7362. [DOI](https://doi.org/10.1088/0031-9155/61/20/7347) · [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC5378691/)。

**L3.** Qin N, Pinto M, Tian Z, et al. Initial development of goCMC: a GPU-oriented fast cross-platform Monte Carlo engine for carbon ion therapy. *Phys Med Biol.* 2017;62:3682–3699. [DOI](https://doi.org/10.1088/1361-6560/aa5d43) · [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC5730973/)。

**L4.** De Simoni M, Battistoni G, De Gregorio A, et al. A data-driven fragmentation model for carbon therapy GPU-accelerated Monte-Carlo dose recalculation. *Front Oncol.* 2022;12:780784. [DOI](https://doi.org/10.3389/fonc.2022.780784) · [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8990885/)。

**L5.** Lee H, Shin J, Verburg JM, et al. MOQUI: an open-source GPU-based Monte Carlo code for proton dose calculation with efficient data structure. *Phys Med Biol.* 2022;67:174001. [DOI](https://doi.org/10.1088/1361-6560/ac8716) · [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9513828/)。

**L6.** Feng H, Holmes JM, Vora SA, et al. Modelling small block aperture in an in-house developed GPU-accelerated Monte Carlo-based dose engine for pencil beam scanning proton therapy. *Phys Med Biol.* 2024;69:035003. [DOI](https://doi.org/10.1088/1361-6560/ad0b64) · [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11009986/)。

**L7.** Li S, Cheng B, Wang Y, et al. A GPU-based fast Monte Carlo code that supports proton transport in magnetic field for radiation therapy. *J Appl Clin Med Phys.* 2024;25:e14208. [DOI](https://doi.org/10.1002/acm2.14208) · [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10795429/)。

**L8.** Schiavi A, Senzacqua M, Pioli S, et al. Fred: a GPU-accelerated fast-Monte Carlo code for rapid treatment plan recalculation in ion beam therapy. *Phys Med Biol.* 2017;62:7482–7504. [DOI](https://doi.org/10.1088/1361-6560/aa8134)。本次仅核对摘要，未取得可读方法全文。

**L9.** Wan Chan Tseung H, Ma J, Beltran C. A fast GPU-based Monte Carlo simulation of proton transport with detailed modeling of nonelastic interactions. *Med Phys.* 2015;42:2967–2978. [DOI](https://doi.org/10.1118/1.4921046)。本次仅核对摘要。

**L10.** Maneval D, Ozell B, Després P. pGPUMCD: an efficient GPU-based Monte Carlo code for accurate proton dose calculations. *Phys Med Biol.* 2019;64:085018. [DOI](https://doi.org/10.1088/1361-6560/ab0db5)。本次仅核对摘要。

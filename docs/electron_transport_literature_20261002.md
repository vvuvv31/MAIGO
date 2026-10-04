# 次级电子与级联加速：文献检索与本仓库建议

检索日期：2026-10-02。结论：优先研究**材料相关的短程联合响应（Macro Monte Carlo）＋界面附近的细步电子输运＋现有能量加权单分支估计器**。这比继续保存完整轨迹库更有希望解决加载、常驻内存和随机访存问题，但尚无本项目完整 CT 的加速或精度保证。

## 检索证据与范围

实际查询 Europe PMC 的论文元数据/摘要、Crossref DOI 元数据，并读取可取得的 Europe PMC 全文 XML。查询响应、原文及提取文本保存在 `out/electron_literature_20261002/`。这是针对方法选型的定向检索，不是系统综述；没有找到的路线不能据此断言不存在。

已读全文：penMRT（PMC9322281）、DPM/PRIMO（PMC6307123）、自适应 proton MMC（PMC6734301）、2026 VHEE MMC（PMC13285896）。1992 MMC 和 1995 PENELOPE 原始论文仅核实书目信息；其方法讨论同时参考上述可读全文。DPM 2000、EGSnrc 2000、gDPM 2011、VPMC 2022 的主要证据来自摘要。部分全文接口返回无正文或请求失败，不视为读过全文。

## 关键文献

### 1. Macro Monte Carlo：保存局部输运规律，而非完整历史

- Neuenschwander H, Born EJ (1992). *A macro Monte Carlo method for electron beam dose calculations*. [DOI:10.1088/0031-9155/37/1/007](https://doi.org/10.1088/0031-9155/37/1/007).
- Zhu C et al. (2026). *Macro Monte Carlo dose calculation for very high energy electron (VHEE) radiotherapy*. [DOI:10.1002/mp.70539](https://doi.org/10.1002/mp.70539), [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13285896/).

局部 MC 在球或薄层中生成位移、能量损失、出射方向及二次粒子等分布；全局模拟抽样这些分布，跨过多个细步。2026 工作分别处理连续输运、电子发射和光子发射通道，说明压缩模型并不必然要求把全部次级能量原地沉积。

2026 工作采用 0.2–25 MeV 的球响应（半径 0.5–3 mm）及 25–250 MeV 的 2 mm 薄层响应；最高报告相对 EGSnrc 27 倍剂量计算效率。患者 CT 的 global 2%/1 mm 通过率 >94%，2%/2 mm >97%，10% 剂量阈值。**这些是高能电子束和毫米尺度验收，既不是 250 MeV/u 碳束，也不能证明本项目峰谷 1% 或 0.2 mm gamma。** 其电子总能截止 0.6 MeV 对应约 89 keV 动能，不能误读成 0.6 MeV 动能，更不能直接照搬截止值。

本项目可借鉴其局部到全局的结构，并为低能 delta electrons 重建更细的响应；不能照搬球尺寸、截止或独立边缘分布而丢掉位移—方向—能量关联。

### 2. DPM / PENELOPE / EGSnrc：无需完整轨迹库的混合电子输运

- Sempau J, Wilderman SJ, Bielajew AF (2000). *DPM, a fast, accurate Monte Carlo code optimized for photon and electron radiotherapy treatment planning dose calculations*. [DOI:10.1088/0031-9155/45/8/315](https://doi.org/10.1088/0031-9155/45/8/315).
- Baró J et al. (1995). *PENELOPE: An algorithm for Monte Carlo simulation of the penetration and energy loss of electrons and positrons in matter*. [DOI:10.1016/0168-583X(95)00349-5](https://doi.org/10.1016/0168-583X(95)00349-5).
- Kawrakow I (2000). *Accurate condensed history Monte Carlo simulation of electron transport. I. EGSnrc, the new EGS4 version*. [DOI:10.1118/1.598917](https://doi.org/10.1118/1.598917).

大量软碰撞合并为能损和多重散射步骤，硬碰撞显式抽样。PENELOPE 的 random-hinge、EGSnrc 的边界穿越和步长修正，对位移与方向关联及界面准确性很有参考价值。这类方法主要依赖材料物理表，而非逐历史 genealogy。

DPM 2000 的约 5 mm 长步有专门的跨异质算法，不等于随意跳过边界；其比较为 1 mm 边长体素、相对最大剂量约 1%。2018 [DPM/PRIMO 验证](https://doi.org/10.1186/s13014-018-1188-6)中患者案例相对 PENELOPE 的加速约 6–12 倍，但空气输运存在最高约 24% 的差异。不能把它直接当成本项目低密度区域的验收依据。

本项目用途：作为无需巨大 replay 库的界面输运路径，或完全独立的小表电子输运候选。完整重写成本高于短程响应原型，需要与现有 TOPAS cut、停止本领及散射设置对齐。

### 3. 自适应 MMC：边界判断必须覆盖横向邻域

Kueng R et al. (2019). *Adaptive step size algorithm to increase efficiency of proton macro Monte Carlo dose calculation*. [DOI:10.1186/s13014-019-1362-5](https://doi.org/10.1186/s13014-019-1362-5), [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6734301/).

其质子算例中，距离平行骨/肺界面 0.1 mm 时，不限制宏步可出现 >5% 剂量差异；限制后恢复到约 1%。患者 CT 相对非自适应 MMC 效率最高提高 3.4 倍。文中的 0.5 mm 侧向阈值是该质子算法的经验条件，**不作为电子路径阈值直接使用**。

对我们的启示：只检查中心线或响应终点材质不足。候选宏步需要考虑整个局部响应域的材料和密度；靠近界面缩小响应域，再交给细步路径。球内响应需要以首次出球为停止条件并输出出射状态，不能把无限介质的最终沉积分布简单截断或按端点重抽。

### 4. penMRT：与谷区误差最相关的证据

Keshmiri S et al. (2022). *A high-resolution dose calculation engine for X-ray microbeams radiation therapy*. [DOI:10.1002/mp.15637](https://doi.org/10.1002/mp.15637), [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9322281/).

全文明确讨论低能电子 random-hinge 输运对谷区的重要性，以及 C1/C2 对精度与速度的权衡。penMRT 和 Gate 的峰内结果相近时，谷区仍可相差 1–7.5%；作者认为可能来自物理模型差异。其研究对象是 X 射线微束，不能直接转用到质子/碳束，但它说明整体 gamma 或峰值吻合不足以约束电子远尾与谷剂量。

全文提到一种卷积电子核的混合算法曾报告 600 倍加速，同时指出其仅输出分离的宏观峰/谷剂量图，不能完整解析复杂交叉微束分布。故不据此宣称普通径向核卷积能够满足本项目。

### 5. gDPM：级联调度可以分开，但并非新的物理压缩

Jia X et al. (2011). *GPU-based fast Monte Carlo simulation for radiotherapy dose calculation*. [DOI:10.1088/0031-9155/56/22/002](https://doi.org/10.1088/0031-9155/56/22/002).

把光子和电子分开调度，减少不同物理路径导致的线程分歧，并采用 GPU RNG/插值。论文报告 Tesla C2050 对 2.27 GHz Xeon 约 69–87 倍加速；这不是对当前 GPU 实现的增益，更不是对 100 线程 TOPAS 的增益。本仓库已有电子任务队列，单纯再加一层队列不是新方案；新价值主要在将来加入光子 continuation 时保留分粒子调度。

### 6. VPMC：可借鉴思想，不能直接照搬次级近似

Shan J et al. (2022). *Virtual particle Monte Carlo: A new concept to avoid simulating secondary particles in proton therapy dose calculation*. [DOI:10.1002/mp.15913](https://doi.org/10.1002/mp.15913).

用预生成 PDF 和虚拟粒子代替运行时产生全部次级质子，降低 GPU 分歧。摘要明确说明电子、重离子和核碎片剂量原地沉积，中性粒子按逃逸忽略；患者比较采用 2%/2 mm 和 10% 阈值。其平均 2.84 s 使用四张 Ampere GPU，不能与我们的单卡 minibeam 直接比较。

这套局部沉积近似不适合作为本项目 carbon minibeam 的电子/碎片替代。电磁电子级联和碳核碎裂级联是两个问题；电子响应拟合不能取代后者，碎片仍要保留种类、能量、方向和后续输运。

## 对当前实现的具体判断

当前 `electron_packet_transport.hpp` / `electron_energy_lineage.hpp` 已按能量份额选择沉积、父电子继续、子电子或光子分支；它并没有显式追踪整棵电子树。统计能量 W 与被抽样电子的动能不同，选中分支后不能再次乘分支概率。

对于线性平均剂量，分支能量为 e_i、总量为 E 时，按 p_i=e_i/E 选一支，并继续携带统计权重 W，有 E[W f_I]=W∑p_i f_i；这是单分支估计的基本理由，要求各分支条件输运正确且能量账闭合。它不保留完整事件的粒子数/剂量涨落，也不直接保证 LET 或微剂量学的非线性统计。该解释来自当前估计器结构，不把它冒称为 VPMC 原文算法。

因此主要瓶颈仍是被选中分支要读取大 replay bank、恢复状态和逐步处理。当前 carbon bank 常驻分配约 93.6 GiB。已有 0.65 MeV 均匀肺模型仅约 0.85 MiB，但只是一个能量/材料点，不能推断全覆盖包大小。

源码复核更正：当前驱动会回放同材料内已有的 PDG22 轨迹；需要在材料界面或有限源耗尽后重新抽样光子 continuation 时，才进入显式未追踪能量项，并非所有光子出生立即终止。CT 路径对核次级带电粒子的电子损失覆盖也尚未完整。新响应须保留已支持的光子回放及缺覆盖记账；纯带电响应实验不能自称与完整现有 packet 范围一致，也不能把未追踪光子能量转为局部沉积。

## 推荐实施顺序（建议，尚未运行新实验）

1. **短程联合响应原型优先。** 在 TOPAS 中对单能电子生成局部有界域内的沉积及首次出域状态，条件为材料、密度、入射能量、域尺寸。联合抽样位置、方向、剩余能量和分支类型，保留核心与稀有远尾；同一电子输运模型可供 proton/carbon 共用，产生端能谱与角谱仍分别处理。
2. **界面路径同步设计。** 宏步只用于经过验证的同质域；响应域缩小到不足以加速时使用已验证的细步处理。原 replay 可在小型验证夹具中当参考，但正式 fallback 若仍强制加载完整 93.6 GiB 库，就没有解决部署问题。最终需要按需局部数据或小表物理输运路径。
3. **保留能量包估计器，比较等精度效率。** 平均剂量可以用能量权重选一个后继；需要时对谷区相关的稀有通道采用有权重补偿的分层/分裂，不能删尾。比较峰/谷方差与时间的乘积（或 1/(tσ²)），同时报告冷启动、内存、电子核和完整计划时间，不能只比较 samples/s。
4. **独立验证再替换。** 材料覆盖 Water_75eV、肺、软组织、骨及其必要密度；电子能量从实际 cut 到离子源所需上限，不以 250 MeV/u 作为电子能量。测试平行/垂直接口、薄层和返回散射，以及能量/密度插值。随后测试真实 carbon/proton CT minibeam 的峰谷约 1%、局部差异及 1/2/3%/0.2 mm、3%/0 mm gamma；低剂量阈值不能排除需检查的谷区。

所有 GPU 剂量计分与原子累加继续 FP32；检查输运/EM 审计、能量账、缺失覆盖与队列溢出。参数化 pilot 的范围和结果见 [已有报告](electron_parametric_20261002.md)。

本轮只检索文献和形成选型依据，没有更改正式输运、运行新的 MC，或测得新的 CT 加速比。

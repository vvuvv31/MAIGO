# 八小时 CT broadbeam 吞吐优化（已完成）

授权时段：北京时间 2026-10-09 22:01:34 至 2026-10-10 06:01:34。最后一小时用于最终验收、恢复稳定代码和报告。没有自动 commit/push。

只优化 proton/carbon CT broadbeam；两者关闭电子输运，使用现有电子能量局部沉积路径。独立带电碎片、离子涨落、连续26电磁包、FP32剂量原子操作、核反应和显式未追踪中性能量出口保留。其他六个默认预设不改变物理行为。

最终在同一0.5mm步长约束下，proton218,321→234,665/s（+7.486%），carbon66,067→83,638/s（+26.596%）；四组配对均提速，候选患者TOPAS local1%/1mm gamma分别99.901487%、99.393756%。两者每臂80M剂量、64M计时。原始代码漏执行步长上限；包含该修复的最终吞吐低于原始代码，完整原始对照和同一步长对照见下表。

## 剂量和测量条件

- 使用原 RT07575 **头部完整计划**，原 CT、束流源、权重、摆位和归一化一起保留。不是默认肺部微束基准。
- 患者内历史 TOPAS 剂量参考 ≥10% 最大剂量：proton 320,771 个体素，carbon 351,014 个体素。
- local 1% / 1 mm，搜索 0.1 mm，严格 >99%。无剂量拟合或配准。CT 的原计分间距保留：患者轴顺序为 2 / 0.5 / 0.5 mm。
- 最终每粒种、每臂 8,000 万入射粒子，四个独立种子计划，每计划20批、前4批只从吞吐计时中排除；剂量、权重和审计覆盖所有批次。每臂计时统计为6,400万粒子。
- ABBA+BABA交错运行，连续批次共用同一个 GPU context/packages。最大余数分配覆盖整次计划；逐spot种子匹配。
- 每次计划的batch剂量求和后只按历史入射粒子数归一化一次；四次已归一化计划剂量取平均，不能相加再作为计划剂量。
- RTX 2080 Ti，本地运行，时钟/温度未锁定。观察到84–85°C热降频；不宣称通用GPU加速率。无动态硬件停顿计数（NCU权限受限）。历史TOPAS参考没有逐体素方差，结果不代表新TOPAS模拟验收。

## 保留的优化

1. Carbon CT关闭电子时的专用主粒子stage17/18：删除不可达出生工作，复用经条件认证的CT/计分cell；带电碎片复用同一输运步的电磁点。
2. Carbon主离子/带电碎片delta能损使用step-start一阶矩，省略邻近能量delta斜率求值。名义rate/mean/variance与涨落保留。属于明确的步内物理近似。
3. Proton主离子、carbon主离子/碎片省略restricted stopping的步内斜率求值。保留名义stopping/range、完整终止射程及离子电荷修正。属于明确近似；carbon的初步gamma曾下降约0.207个百分点，最终需重新验收。
4. 仅proton CT electron-off保留不可变核数据device residency（313,301,059字节），完整初次验证和预算计数不变，mutable粒子/RNG/剂量不缓存。
5. 仅proton CT electron-off预计算实际CT `(section,density_bits)` 的当前连续26材料字段（17,617,029字节）。RT07575共3967个实际元组；GPU求值、不量化密度、不用响应密度节点，不添加缺数据沉积替代。
6. 共享不可变CCTG v3 host读取结果，以及按grid owner和精确体素体积复用FP64计分质量。水使用同一精确公式的常量质量。输入路径/stat变化及体积变化失效，保持完整几何检查。

这些优化的增量比值不能相乘当作最终加速率；最终值必须来自同一轮完整对照。

## 必须保留的0.5 mm步长修复

本轮检查发现：非Urban CT broadbeam原先只用range、mean-loss和几何face等限制，未明确执行配置中的maximum_step_mm。现在在主离子及带电碎片的face/nuclear竞争之前加入CT专用上限，并在每批quality报告中记录 `ct_broadbeam_ion_step_cap=0.5`。水与微束路径保持原行为。

实际连续26包327,600个组件状态、18species/26materials的非轴向合法几何见证中，54,180个可能状态超过0.5mm，最大0.757576mm。这是可能状态证明，不是运行时命中比例。

初步proton16M/arm交错比较：未修正的已优化控制418,534/s → 修正后244,070/s（ratio0.583153）；TOPAS gamma99.944820% →99.954173%。完整16M配对中primary有效能损调用增加68.529%、secondary增加14.546%。Carbon同期132845→83932/s，gamma99.590045→99.602010%，primary/secondary调用增加40.088%/36.081%。此成本必须如实报告，不能为吞吐恢复错误步长。

为分离两类效应，保留原始goal-start二进制的最终对照，同时准备“goal-start源码只加相同0.5mm修复”的独立控制。后者仅在实验输出目录构建，不覆盖当前源码或发布binary。

## 已完成的实现/数据检查

- 完整五类管线及两个选择性构建均完成；当前完整binary SHA256：`0e48d8898315aa79788325cb3ab6ae7084166414d2311c30dd5fd13b7f6025e0`。
- 其余六个默认预设各40k/arm匹配GPU回归通过：source/discrete/fullEM/energy/completion/coverage/zero-overflow匹配，剂量L1仅FP32原子顺序舍入。**不是TOPAS验收或性能结论**。
- 两个CT broadbeam full/selective各40k/arm匹配通过，并核验0.5mm标记。小统计gamma不能作为最终验收。
- 旧接口拒绝检查通过；数据manifest173资产/8预设通过。
- 核cache的cold/warm、路径/stat失效、预算拒绝恢复、context move/destruction检查通过。
- CT精确材料cache实际GPU生命周期、预算拒绝恢复与匹配剂量检查通过。
- CCTG host owner/stat/truncation/version/recovery及计分质量volume/geometry/density-floor/water公式检查通过。

## 已撤回候选

质子碎片专用delta、mean-loss5%→10%、carbon核驻留、跨步4-lane随机数cache、Gaussian配对随机数、C12有界moment剪枝、carbon材料字段cache、primary pending FP64→FP32、native Gamma power、inverse Poisson、Poisson内部Philox批量、carbon次级非弹性代数2→1，均未给出稳定有用收益，已恢复。默认代数仍为2。失败证据保留在 `failed.md` 和本轮out目录，不留生产分支或备用实现。

## 最终结果

每项均为80M/arm剂量、64M/arm计时；源码和最终五管线binary保持一致。以下两组对照分别交错测量，不拼接不同轮次的吞吐。

### 与原始启动版本比较（包含必须的步长修复）

| 粒种 | 原始启动版本 /s | 最终版本 /s | 速度比 | 原始TOPAS gamma | 最终TOPAS gamma |
|---|---:|---:|---:|---:|---:|
| Proton | 375,146 | 236,871 | 0.631410 | 99.893694% | 99.899305% |
| Carbon | 103,785 | 83,635 | 0.805845 | 99.748728% | 99.374099% |

### 同样执行0.5mm上限时的优化收益

控制为原始启动源码只加入与最终版本相同的CT0.5mm上限及审计标记，只有五处文件差异；独立实验目录构建，未覆盖当前发布源码。

| 粒种 | 仅修复步长 /s | 最终优化 /s | 收益 | 四组速度比 | 最终TOPAS gamma |
|---|---:|---:|---:|---|---:|
| Proton | 218,321 | 234,665 | +7.486% | 1.062821/1.080643/1.080288/1.075696 | 99.901487% |
| Carbon | 66,067 | 83,638 | +26.596% | 1.266260/1.267816/1.264101/1.265675 | 99.393756% |

原始代码未明确执行0.5mm上限，所需工作量更少。最终版本含物理约束修复；应同时报告原始速度变化和相同步长下的优化收益。所有四项候选高统计量剂量门槛均严格>99%，每项80批的quality/completion/coverage/energy/zero-overflow和source/spotseed检查通过，步长标记均为0.5。

- 原始启动对照/proton：候选能量残差最大6.62982265e-07；64M timed/80M dose；正收益配对0/4。
- 原始启动对照/carbon：候选能量残差最大3.9702345e-06；64M timed/80M dose；正收益配对0/4。
- 同一步长对照/proton：候选能量残差最大6.0522872e-07；64M timed/80M dose；正收益配对4/4。
- 同一步长对照/carbon：候选能量残差最大3.9498016e-06；64M timed/80M dose；正收益配对4/4。

0.5mm cap、源文件hash、full/selective一致性、六个未改默认的匹配回归、接口和manifest均已验证。保留其余六个默认行为；不将小统计实现回归称为TOPAS验收。

原始结果见 `out/broadbeam_night_20261009/final/final_summary.json`、四个 `*_80M/summary.json` 和各自逐批文件。没有自动commit/push。

## 剩余成本（同一步长80M实测）

Carbon候选的带电碎片kernel合计526.554s，占transport loop944.058s的55.8%；proton为23.730s/325.450s（7.3%）。Carbon每个原初history约674次带电碎片能损调用，proton约8.50次，不能仅从primary/s判断两种任务成本相同。

同一步长下Carbon控制→候选：primary有效能损调用277.995亿→278.226亿，secondary538.486亿→539.260亿，调用数量基本相同；secondary kernel672.104→526.554s，整个loop1196.084→944.058s。收益主要表现为相近工作次数的执行成本降低。上述组件总计包含cold/warmup，与表中的64M warm交错吞吐分开；不把queue/scoring余量误称为精确primary kernel时间，也不据此声称硬件stall来源。

## GPU采样误差补充

从同一步长的四个独立20M计划估计80M合并剂量的GPU标准误差。候选患者mask内体素相对标准误差中位数：proton0.582%、carbon0.519%；95百分位分别1.558%、1.480%。控制对应统计几乎相同，未观察到以显著增加这项噪声换取吞吐的现象。

患者mask剂量总和相对TOPAS偏差：proton控制+0.02374%、候选+0.01729%；carbon控制−0.56065%、候选−0.61378%。候选GPU剂量总和标准误差分别约0.00196%、0.00223%（以参考总和为分母）。这些是GPU抽样误差估计，只有四个重复、方差自由度3，逐体素估计仍不稳定；缺少TOPAS方差，不能把模型偏差统称为随机噪声。验收仍使用实际合并剂量gamma，不用此误差估计替代剂量门槛。详见 `final/GPU_sampling_uncertainty.json`。

全部原始证据：`out/broadbeam_night_20261009/`，包括启动源码/二进制、每次命令、逐批日志/quality/energy/dose、matched checks、summary、journal和源文件hash。实际结束：2026-10-10T06:01:56.434333+08:00；持续至少八小时。

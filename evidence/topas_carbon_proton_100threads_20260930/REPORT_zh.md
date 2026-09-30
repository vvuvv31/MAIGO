# RT07575 carbon/proton TOPAS 100线程smoke

## 结果

同一AMD EPYC9965节点ps、同一Slurm分配21230中顺序运行（carbon→proton），同一TOPAS4.2.p3/Geant4 11.3.2及PBS扩展binary、相同100物理核心。每端20万原发粒子，各自优化后完整spot权重按最大余数法整数分配。

|粒子|原发数|外部进程wall|完整进程吞吐|TOPAS Execution Real|Execution口径吞吐|进程峰值内存|
|---|---:|---:|---:|---:|---:|---:|
|C12|200,000|263.959s|757.69/s|260.285s|768.39/s|11.93GiB|
|Proton|200,000|93.101s|2148.20/s|89.893s|2224.87/s|11.09GiB|

- 完整进程墙钟：proton/carbon **2.8352倍**。
- TOPAS Execution：proton/carbon **2.8955倍**。
- TOPAS四阶段Real之和吞吐：碳764.37/s，proton2188.82/s。外部进程wall额外包含退出清理等开销；都不含Slurm等待。
- 两端日志明确100工作线程、源实际200000 histories、正常退出0；剂量非零、有限、非负，网格417×505×35一致；输入和binary hash复核通过。

## 输入与物理

- carbon使用当前已优化碳CT计划输入：`/mnt/sda/wuwei/carbon_fe_gamma_20260929/latest_6481909/spots.csv`，853行；200k整数分配后837非零spot（16个极小权重舍入为0），C12总能量1.98–2.88GeV，即165–240MeV/u。
- proton使用MATLAB优化导出的`benchmark/protonCT/RT07575_proton_GPU/inputs/spots_optimized.csv`，870行、540个原非零spot；200k整数分配后538非零spot，能量92.93–126.10MeV。
- 不选最快spot或单能量；每个原始CSV行保留。beam模型使用各自原表，proton继续继承插值后的carbon光学参数。
- 相同CT和Schneider HU文件，原几何/束线光学；DoseToMedium Sum单scorer，关闭DIJ和两个LET scorer。TOPAS native完整7模块（opt4、QGSP_BIC_HP、ion-INCLXX、elastic_HP、decay/stopping/radioactivedecay）；未把GPU的Gaussian/core+tail FE、local-electron或neutral-sink近似移植到TOPAS。
- 原脚本默认600MeV表不覆盖高总动能碳源。已取消该尝试21229并保留`rejected_em600/`，两端正式smoke显式EMRangeMax10GeV，日志确认10GeV。不调整stopping scale或材料。
- CT按原Schneider转换文件生成，无使用G4_WATER(78eV)替换Water_75eV的GPU水参考；本轮是两端TOPAS CT吞吐，不是GPU/TOPAS水剂量校准。

## 计时限制

**这是单次20万smoke，不是大计划稳态吞吐。** TOPAS Execution还包含物理表准备等固定开销，不能等同于独占粒子追踪kernel时间。proton Execution平均CPU忙碌核数约9.88，碳约68.17（User+Sys / Real）；配置100线程不表示整个Execution阶段持续100核满载。
因此2.84倍是本次含固定开销的smoke比值，不能用它覆盖此前大计划线程归一化估算，也不能证明扩展到100线程只有该效率。不同source能谱/计划、二级粒子工作量均属于此次计划对比。

Slurm core binding最初包含SMT siblings，preflight21228在TOPAS启动前被assert拦下；改为按socket/core辨识100物理核心并限制每核心一个逻辑CPU。两端继承同一CPU affinity68–167。Slurm账本显示200逻辑CPU是其按core分配保留SMT sibling的方式，进程可用CPU和TOPAS workers均100；没有运行200TOPAS线程。

剂量计分仍出现已知Geant4导航未计分警告：carbon 9668 hits、6.66209e-05MeV，proton 299 hits、1.9912e-08MeV。正常退出与非零剂量不构成本轮物理/gamma验收；没有进行新gamma比较。此处TOPAS原生binary输出Float64，是原CPU scorer格式，本轮没有构建/运行GPU FP64 scorer。

## 文件

- `prepare.py`、`run.slurm`、`execute.py`：输入构造、100核Slurm和顺序执行。
- `analyze.py`、`manifest.json`、`summary.json`：实际阶段计时、检查及hash。
- 原始配置、source CSV、CPU affinity、log、time峰值内存与剂量：`/mnt/sda/wuwei/topas_carbon_proton_100threads_20260930/`。
- 修正后的作业21230 COMPLETED/exit0；24GiB申请，实际峰值约12GiB。
- 当前GPU源码、优化配置与已接受剂量均未修改；没有commit/push。

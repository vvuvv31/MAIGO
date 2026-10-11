# RT07575 carbon minibeam 优化（2026-10-10，吞吐初筛）

在RTX A6000（v@172.17.152.19）上测试当前代码、原RT07575头部crossfiring全计划。原400个有效spot、90°/270°场、CT世界网格、源位置/方向/发射度/能量及场270°准直器偏移1.5mm一并保留；旧555条campaign记录按original_spot_id合并为原400个spot，没有删除照射场或改变权重。

仅用隔离的数字源驱动与源码快照，正式runtime和默认八预设未改动。所有源/CT输入SHA256校验通过；同一A6000 sm86构建设置、私有运行库，禁用PTX JIT。核中性产物仍为未追踪能量sink，没有局部沉积补偿。

## 对照与步骤

三组初始设置全部关闭显式电子/电磁光子输运，保持continuous26离子EM、核反应、全部带电碎片、Urban、FP32计分，以及患者主粒子/铜中主粒子的涨落：

1. secondary_on：患者次级及铜碎片连续涨落开启，空气slit cap0.05mm。
2. secondary_off：患者次级涨落关闭，铜碎片受限连续涨落关闭，空气仍0.05mm。
3. air_020：在secondary_off上将空气slit cap改为0.2mm；铜0.05mm、患者0.1mm不变。

铜原生g4_ion_delta的碎片路径原来强制开启涨落，且未读取开关。隔离实现新增受限涨落选择，关闭时保留原restricted mean和硬δ的Poisson/能谱/形状因子抽样，不删除这部分能损；开启时224万组件输入的能损/RNG与旧实现完全一致，关闭时同数检查均有限、零无效。患者开关使用原continuous26行为，也关闭原聚合δ矩涨落。这两种输运模型的开关含义不能混淆。

患者主粒子候选（point_reuse目录）：同一步内复用完整continuous26 point（保留所有导数和涨落），并将主粒子网格内/外审计账目按history汇总后提交；实际体素剂量原子计分仍为FP32。262万实际continuous26+δ矩组件输入的能损与RNG完全一致。该组件验证不替代全计划GPU审计。

碎片候选（fragment_region目录）：在已有物种/能量分组上增加铜出生/其他出生的索引分区，保持所有物种、粒子状态、parent history和rng_stream，不抽掉粒子，也不改核级联代数或步长。与主粒子候选分别测量，不混合未验证的收益。

## 配对方法

原计划每组四个独立3M full-plan campaigns，以及代码候选ABBA+BABA长测。用户指出当前目的只是吞吐优化、该安排反馈过慢后，已停止原初始测试及自动长测后续进程，排除被中止的campaign2部分输出。保留初始三组各两个完整campaign（各6M剂量粒子、4M暖态计时粒子），正反顺序交错。代码候选改为各一个3M初筛，复用此前同campaign的secondary_off对照；没有同时期交错复测，不宣称稳定最终收益，也不作新剂量验收。

每campaign连续3批×1M，先对整个campaign用最大余数分配原400spot权重，再均衡分批。保留每spot种子，配对候选/对照的CSV逐字节相同。每批包含两个照射场，复用同一个GPU context；首批只从计时排除，不从剂量统计排除。

各field-batch dose_scale为原4Gy入射数13,872,495,687.640352除以3M；每campaign求和三个批次的两个场，再按实际已完成的独立campaign数平均，不乘批数。按原角度变换回患者坐标。

所有field-batch必须accepted、零overflow、能量残差<1e-4，保留EM/铜过程/完成审计、质量、能量账目、misses和unsupported结果。性能记录包含batch墙钟及上游/患者primary/secondary分阶段时间；GPU未锁频，不从短测宣称通用硬件上限。

## 剂量边界

比较配对GPU剂量、ROI积分、GTV平均、局部RMS变化与四campaign的配对统计误差。历史TOPAS full-plan缺少GenericIon核弹性；其local3%/0.2mm、search0.02mm结果仅作描述，不能作为匹配物理的新接受门槛。碳minibeam独立剂量验收仍待完成，broadbeam的local1%/1mm>99%门槛不能代替这里的验证。

## 证据与当前状态

本地`out/carbon_minibeam_opt_20261010/`保存驱动、源码快照、三个候选的sm86二进制、component结果、manifest和输入散列；远端目录为`/home/v/MAIGO_clean_a6000_20261010/carbon_minibeam_opt/`。原run.py/followups.py长测进程已停止，不应从头重跑覆盖既有输出。`screen_schedule.json`记录改用初筛的原因，`quick_screen.py`依次运行两个代码候选；`initial_screen.json`记录已完成初始对照。

前两轮初始对照平均暖态吞吐：secondary_on 9,536/s，secondary_off 10,300/s（+8.01%），air_020 10,844/s（相对secondary_off +5.28%，相对secondary_on +13.71%）。全部36个已完成field-batch审计通过、零溢出。它们是当前A6000/原RT07575头部全计划的初筛证据，不是缺少匹配TOPAS验证的物理近似验收。

## 当前吞吐门槛

初始两轮、每组4M暖态计时粒子的阶段预算（秒/百万原始入射粒子）：

| 配置 | 总墙钟 | Cu/空气上游 | 患者主粒子 | 患者次级 kernel | 队列压缩 | 其他 |
|---|---:|---:|---:|---:|---:|---:|
| secondary_on | 104.861 | 31.722 | 65.650 | 4.737 | 0.023 | 2.730 |
| secondary_off | 97.084 | 24.650 | 65.589 | 4.361 | 0.023 | 2.461 |
| air_020 | 92.216 | 18.877 | 66.584 | 4.282 | 0.022 | 2.449 |

Cu原生碎片处理计入上游，不应将上述患者次级kernel当成所有碎片总成本。关闭次级涨落的主要时间收益来自上游（约7.07s/M），患者主粒子时间几乎不变。air_020中患者主粒子占72.2%；仅优化/消除队列压缩几乎无收益，即使患者次级kernel全部免费，理论总吞吐提升也只有约4.9%。即使整个上游全部免费，提升上限约25.7%（并非现实可实现的优化）。

患者主粒子有效能损调用约362.9/原始入射粒子，主粒子核rate queries约362.9/原始入射粒子；每百万入射对应约3.63亿次患者主粒子能损评估。患者次级约219.7步/原始入射粒子。这是全计划归一工作计数，不是单个进入患者的粒子的轨迹步数。主粒子保留0.1mm cap、Urban边界/真实路径/位移、完整continuous26能损与涨落。当前完整prepare还评估相邻能量的δ矩导数；形状因子活跃时可能走八点数值积分。没有动态内部计时/硬件counter，不能据此宣称数值积分、SFU、访存或spill是哪一个硬件主瓶颈；A6000部署未安装ncu/nsys。

患者主粒子候选一个3M初筛：暖态10,115/s，对照同源同种子10,241/s，未显示提速，不推广。六个field-pair的离散事件/步数/EM计数一致，剂量relative L1最大2.17e-8、最大单体素差异3.30e-7倍对照峰值；FP32累加能量的末位差异不作为离散计数不一致。完整模型point视图复用及审计账目汇总并未在此次初筛解决主粒子成本，下一步应优先分离步数限制与单步EM/Urban成本，避免继续押注低占比队列工作。

碎片区域分组候选一个3M初筛：暖态10,241/s，与同源对照10,241/s基本一致，不显示有效收益，不推广。两项代码候选共只运行6M，原24M/项的交错长测试已取消。

另一个应优先排查的并行度因素：原RT07575 carbon minibeam保留`history_chunk_size=10000`，每个primary chunk独立完成上游和患者完整轨迹；Cu挡掉的线程不会提供患者输运并行度。患者次级首代队列则约12万至22万条，使用分段存活压缩。主粒子与次级的批规模、涨落设置和调度不同，不能把二者每步耗时差全部解释成碳EM计算本身。尚未测量各chunk有效入射数、活跃warp或步长限制命中率，故小chunk/长尾只列为优先验证的假设；硬件/模型/工况与旧失败实验不同，但不据此宣布增大chunk必然加速。

## 患者1mm上限短测

用户授权仅将患者`maximum_step_mm`从0.1改为1.0，Cu保留0.05mm，空气沿用已测试的0.2mm。关闭电子与次级涨落，患者/Cu主粒子涨落继续开启；原400spot全计划及种子逐字节一致。使用同一baseline二进制（SHA256 c12d22d6…），没有混入此前未提速的两个代码候选。

各3M完整campaign，首批冷启动、后两批共2M暖态：

| 指标 | 患者0.1mm | 患者1mm |
|---|---:|---:|
| 冷启动吞吐，primary/s | 9,555 | 9,985 |
| 暖态吞吐，primary/s | 10,995 | 11,250 |
| 患者primary，s/M | 65.245 | 64.740 |
| 上游，s/M | 18.894 | 18.808 |
| 患者secondary kernel，s/M | 4.337 | 2.884 |
| primary loss calls/原始入射 | 362.980 | 329.904 |
| secondary loss calls/原始入射 | 207.958 | 76.482 |

暖态吞吐初筛+2.32%，primary调用-9.11%但患者primary时间仅-0.77%，secondary调用-63.22%。本轮收益更多来自患者次级成本下降。仅放宽最大步长并不保证实际步长达到1mm；能损、δ候选距离、Urban及体素边界仍竞争。尚未量出各限制的命中比例，不能断言其中某一项主导。

全部六个候选field-batch质量/能量/EM/完成审计通过、零overflow，最大能量残差7.70e-7。证据`out/carbon_minibeam_opt_20261010/remote_results/patient_step_1mm/screen_summary.json`及各field原始剂量/审计。单campaign复用较早对照，无交错重复，不作精确小收益或新TOPAS剂量接受结论。正式默认八预设未改，1mm仅保留为本次授权研究设置。

进一步检查步长竞争：Urban主粒子即使关闭电子仍执行`UnifiedEmClock`，以`tau / proposal_rate`限制路径，再叠加range-step、combined mean-loss、Urban、体素面及核事件。读取当前continuous26/delta-moments包的CPU数值评估（非CPU输运、非GPU limiter频率测量）保存于`limiter_samples.cpp/csv`。CT组成10在有效密度域对数中点ρ=1.16553g/cm³时，C12 200/100MeV/u的名义电离候选平均距离1/rate分别为0.07156/0.04830mm；20MeV/u的候选rate为0；实际微束1% combined mean-loss guard仅0.02247mm。Water_75eVρ1对应候选距离0.07475/0.04975mm、20MeV/u能损guard0.02574mm。此前样表按YAML的0.5%评估，需要纠正：该值没有进入当前continuous26微束步长路径，实际内核固定为1%。候选距离服从随机光学深度，并非固定步长；缓存proposal rate还可保守大于当前名义rate。数值表说明高能候选距离/低能能损上限足以远小于1mm，但未统计这些状态在RT07575中的耗时权重，不能宣称其精确全计划限制占比。

同beta代表状态的C12电离rate约为质子的36倍（含当前模型质量/修正差异），解释主碳离子比轻碎片更难从放宽最大cap获益；不是把碳质量12直接当作12倍耗时。两组primary阶段时间/能损调用数之比升高约9.2%，但阶段时间包括固定开销、Urban/几何/核查表以及并行长尾，不能据此证明每个能损函数调用本身变贵。更少的循环不一定同比降低完整轨迹kernel墙钟。后续优先量proposal、mean-loss、Urban、face限制频率及primary活跃线程；不直接删除原calibrated segmentation或宣称移除clock为无偏优化。

## 患者电离切步时钟1/4初筛

用户随后授权尝试放宽CT内部步长。隔离候选只在`CarbonMinibeam && JointResponse==0 && kernel_stage==4 && in_ct`将电离**切步时钟**使用的rate乘0.25，distance及最终accepted-path consume均使用同一缩放值。物理包hard rate、loss prepare的名义rate/均值/方差/导数及原条件能损校准不改，主粒子涨落仍开；这不等于保持原模型的轨迹分布，较长Urban/loss步是明确的物理近似。CT外时钟、Cu0.05mm和空气0.2mm原样，患者仍1mm；没有混入此前未提速的point/ledger或碎片分区候选。全部带电碎片、弹性/非弹性核反应及CT边界保留。

纠正之前对能损限制的说明：当前`sycl_pipeline_setup.inc`微束combined mean-loss guard固定为0.01（1%），YAML的`maximum_relative_energy_loss=0.005`没有进入该步长路径；本候选不改变原1%。包range-step和Urban自身限制仍在。

一次3M原400spot全计划，source/seed CSV与1mm对照逐字节一致，首批计冷、后两批共2M计暖：

| 指标 | 原切步时钟、患者1mm | 切步时钟1/4、患者1mm |
|---|---:|---:|
| 冷启动吞吐，primary/s | 9,985 | 18,670 |
| 暖态吞吐，primary/s | 11,250 | 23,422 |
| 患者primary，s/M | 64.740 | 18.684 |
| 上游，s/M | 18.808 | 18.638 |
| 患者secondary kernel，s/M | 2.884 | 2.931 |
| primary loss calls/原始入射 | 329.904 | 112.964 |

暖态初筛+108.2%（2.08倍），患者primary时间减少71.1%、调用减少65.8%；上游/患者primary现各约18.7s/M，进一步只优化primary的收益开始受上游限制。原对照复用较早同源运行，未交错重复或锁频；大幅收益成立为本次初筛结果，不作通用精确倍数承诺。

候选六个field-batch全部accepted、零overflow，EM/能量/完成审计通过，最大能量残差8.67e-7。3M配对剂量：GTV平均4.02133→4.02466Gy（+0.0828%，三个均衡full-plan批的粗略配对SE0.0732%）；BODY平均-0.0127%（粗略SE0.0154%）。BODY∪GTV、对照>=10%患者峰值mask上局部剂量变化RMS5.803%，粗略配对voxel SE RMS5.844%。这些低统计数值不能排除局部系统误差，也不是TOPAS gamma验收；对照阈值mask本身还会引入选择噪声。保留研究候选，不更改八个正式默认。

代码/构建/输入与散列：`out/carbon_minibeam_opt_20261010/ct_clock_quarter/`；结果：`remote_results/ct_clock_quarter/screen_summary.json`，分析脚本`analyze_clock_screen.py`。二进制SHA256 `63d34b440f99f69ac5ad8e79b21e53cd2f8ad5570c78680ed64e540271e96857`，原始分场剂量和所有审计已传回本地。

## 继续稀疏到1/16

用户授权继续优化后，将同一CT主粒子切步时钟从0.25调整到0.0625。除这一常数外，其余源码/输入设置与1/4候选相同。低能区不是另加未经测试的截止或局部沉积：原1% mean-loss、package range、Urban、CT faces和核事件继续限制步骤；主粒子涨落与Cu0.05/air0.2不变。此候选仍是有待剂量验证的步分割近似。

一次3M全计划，首批冷启动、后两批2M暖态，复用同源/种子1/4对照：

| 指标 | 时钟1/4 | 时钟1/16 |
|---|---:|---:|
| 冷启动，primary/s | 18,670 | 23,655 |
| 暖态，primary/s | 23,422 | 32,496 |
| 患者primary，s/M | 18.684 | 7.143 |
| 上游，s/M | 18.638 | 18.569 |
| 患者secondary kernel，s/M | 2.931 | 2.834 |
| primary loss calls/原始入射 | 112.964 | 61.019 |

暖态初筛再提升38.74%，相对原1mm/未稀疏时钟11,250/s约2.89倍。两个暖态批次分别30.748/30.798s/M。上游占总时间约60.3%，患者primary约23.2%；下一步吞吐优化应更多关注保持Cu0.05mm物理行为的上游计算/调度，不能把继续减少患者步骤的收益外推成无限增长。

全部六个候选field-batch审计/能量/EM/完成检查通过、零overflow，最大能量残差8.59e-7。GTV平均4.02466→4.02643Gy（+0.0441%，粗略3batch配对SE0.1028%），BODY平均-0.0294%（粗略SE0.0205%）。对照>=10%患者峰值BODY∪GTV mask局部差异RMS5.957%，粗略配对SE RMS5.687%。单campaign不能排除局部系统偏差；mean小变化/energy闭合不构成TOPAS验收，未修改正式八预设。

保存于`out/carbon_minibeam_opt_20261010/ct_clock_sixteenth/`，完整原始结果与摘要位于`remote_results/ct_clock_sixteenth/`。分析命令：`python3 out/carbon_minibeam_opt_20261010/analyze_clock_screen.py ct_clock_sixteenth ct_clock_quarter 0.0625`。二进制SHA256 `3363d1d7ade01f2a6272808be7506a16cd26b10de0ca323ec0b627037a4a692e`。无交错复测/锁频，所列增幅仅指这轮A6000初筛。

已排除：旧场YAML未显式指定pipeline导致generic未编译的启动失败；铜碎片关闭被原校验拒绝的启动失败；未修改候选的链接及组件编译拼写错误。失败日志和此前已完成但不属于最终相同构建的3M对照保留，均不混入最终配对统计。


## Cu/air upstream screening: larger continuous launch chunks

Original RT07575 **head** C12 minibeam full plan, 400 spots, A6000; each candidate is one 3M campaign (three continuous 1M batches, first cold, last two warm). Existing research physics remains: patient cap1mm/CT primary segmentation clock1/16, Cu0.05mm, air slit0.2mm, primary fluctuationsON, patient-secondary/Cu-fragment restricted fluctuationsOFF, Cu fragment hard deltaON, explicit electronsOFF. No production/default changes.

Two exact-code candidates were screened separately: skip the two CSV stopping interpolations whose mean is overwritten by native Cu ion fluctuations; aggregate upstream audit counts per history while keeping failures immediate. Both match discrete/EM/Cu audits and dose (L1~2e-8), but small sequential changes do not establish a useful stable benefit. Restore `source/src/detail/sycl_primary_beamline.inc` from `cu_unused_csv/baseline_beamline.inc` and rebuild `carbon_core`; neither candidate remains in active source. Carbon baseline logs contain no AIR_IONIZATION_CLOCK line; do not infer an air-clock workload from the proposed aggregate counters. Both candidates and logs remain as evidence.

The retained change is **only research history_chunk_size**, 10000→262144. All chunk candidates use the byte-identical 1/16 binary, SHA256 `3363d1d7ade01f2a6272808be7506a16cd26b10de0ca323ec0b627037a4a692e`. Source CSVs/spot seeds/weights/placement/normalization unchanged; continuous context reuse remains, sum three normalized batch doses once.

| Experiment | Cold primary/s | Warm primary/s | Upstream s/M | Patient primary s/M |
|---|---:|---:|---:|---:|
| Initial 10000 | 23,655 | 32,496 | 18.569 | 7.143 |
| Skip unused CSV | 23,825 | 32,701 | 18.373 | 7.139 |
| Audit aggregation | 23,132 | 32,045 | 18.971 | 7.190 |
| Chunk32768 | 32,030 | 50,774 | 11.518 | 3.098 |
| Chunk98304 | 36,673 | 62,146 | 9.675 | 1.332 |
| Chunk262144 | 38,107 | 65,885 | 9.085 | 1.043 |
| Final 10000 repeat | 23,237 | 31,938 | 18.934 | 7.228 |

Compared with the final repeated 10000 control, chunk262144 warm throughput rises **106.29% (2.063×)**; Cu/air falls 18.934→9.085s/M. Patient primary also falls 7.228→1.043s/M, so the total gain must not be attributed exclusively to Cu. Incremental warm gain98304→262144 is 6.02%; stop expanding chunks in this screen. GPU tracked peak 2.929GiB excludes driver/runtime allocations; this is not total GPU memory.

All6field pairs of each candidate have exact discrete/native-EM/step/live-ion/Cu counters versus the initial baseline, zero overflow, completed child queues, accepted coverage/energy reports. Chunk262144 dose max field relative L1 8.02e-08, maximum absolute difference/control peak8.84e-7; normal FP32 scoring order differences. Max energy residual 8.59e-07. These establish matched implementation behavior against the **existing research approximation**, not TOPAS dose acceptance. Carbon minibeam dose validation remains deferred.

Measured large chunk dependence supports larger launches, not a proven occupancy/register/warp-tail causal attribution (no hardware profiling). Sequential short screens and the final original-size control repeat resolve a large effect; no locked-clock ABBA or general preset/device throughput claim. Release eight defaults stay unchanged.

Current research baseline: `out/carbon_minibeam_opt_20261010/current_research_baseline.json`; executable/configs: `upstream_chunk_262144/run_gpu_candidate` and `upstream_chunk_262144/campaign0/candidate/`. Results: `remote_results/<stage>/`, paired checks: `matched_audit.json`, summary: `upstream_summary.json`. Reproduce summary using `python3 out/carbon_minibeam_opt_20261010/analyze_upstream_screen.py` (writes `remote_results/upstream_summary.json`). No jobs left running after screens.

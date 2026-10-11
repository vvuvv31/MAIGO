# RT07575 head proton CT minibeam patient step screening (2026-10-10)

原头部487spot全计划，A6000 sm86；独立out源码快照、仅编译ProtonMinibeam。Production src/include/config和八默认值均未改动。主/次级离子、铜主/碎片涨落全部ON；铜0.05mm，空气slit0.2mm（已有RT07575验证），Urban和核反应保留，电子OFF，history_chunk262144。未把carbon secondary-OFF方案复制给proton。默认lung与RT07575 head是不同工况。

## Short screens and repeat

每项3M、三个连续1M批次，原20source参数/487spot/per-spot batch seeds逐字节匹配；最大余数跨全campaign分配，再平衡成三个批次。W=236086432317.40372，dose_scale=W/3M，求和三批一次，不乘3。初始顺序control→patient1mm→clock1/4→patient1mm-repeat→control-repeat。重复用于时序性能验证，不是独立剂量样本。

初筛warm189768→270637/s (+42.61%);反向repeat187925→259057/s (+37.85%)；两个时序screen合并warm提升40.18%。原步长primary loss calls163.375/incident，1mm44.934/incident。重复相同配置的全部离散/live-ion/EM/air clock/Cu loss计数精确相同，只有FP32剂量累加顺序差异。证据repeat_summary.json。

Clock1/4仅限ProtonMinibeam、Joint0、CT stage4/in_ct的distance与consume；物理rate、moments、conditional terminal与核模型不变。患者primary0.605→0.481s/M，但总warm270637→269614/s，未显示整体收益；3M历史TOPASgamma同时下降1.459pp。恢复私有source_primary.inc并重建carbon_core，保留冻结候选与结果。不测1/16、不推广额外分段近似。

3M历史TOPAS local3%/0.2mm/search0.02mm由74.5289→73.3627%，下降1.166pp，因此不能将1mm初筛直接保留。

## 12M paired-source confirmation

原计划真实权重对12M重新一次largest remainder分配，平衡成三批各4M；保持source浮点字符串和原spot/batch seeds，两臂CSV逐字节一致。12M是独立统计量扩展，包含与3M相同seed的history前缀，不能将3M+12M当15M独立剂量样本。dose_scale=W/12M，sum3batch once。确认驱动只有离线输入history-count校验变为4M，transport库为恢复后的baseline；四项cap使用同一binary。

Gamma：原历史AllChargedUrbanPhysics加权DIJ，patient BODY∪GTV且reference≥10%max，185559voxels；local3%/0.2mm，search0.02mm，spacing2/0.5/0.5mm，不改变原CT grid。一次relative≤1pp只作为step研究筛选门槛，并非把电子包replacement的验收自动当物理步进正式验收。

| Patient cap | Cold primary/s (first4M) | Warm primary/s (last8M) | Patient primary s/M | Historical TOPAS gamma | Change pp |
|---|---:|---:|---:|---:|---:|
| 0.1mm | 199,564 | 219,677 | 1.899 | 87.2262% | +0.0000 |
| 1mm | 268,044 | 302,607 | 0.626 | 86.1122% | -1.1139 |
| 0.5mm | 261,966 | 305,375 | 0.741 | 85.8880% | -1.3381 |
| 0.2mm | 238,685 | 272,719 | 1.079 | 86.9621% | -0.2641 |

保留**0.2mm研究候选**：warm+24.15%，gamma变化-0.2641pp。1/0.5mm观察下降1.114/1.338pp，均不保留；步长与低统计gamma并非单调，不能据此给出确定性模型偏差或‘0.5mm比1mm更差’结论。

0.2mm：GTV mean变化+0.00853%（粗略三平衡批pairedSE0.13977%），BODY mean变化+0.00156%。患者mask localRMS变化6.581% versus粗略pairedSERMS6.559%。

所有候选quality/energy/EM/coverage/completion accepted、零overflow、无electron队列；12M三组candidate upstream AIR_IONIZATION_CLOCK和COPPER_PROTON_LOSS_QUALITY精确匹配control。source/weights/placement/collimator/normalization保持；actual minibeam mean-loss guard固定1%，YAML maximum_relative_energy_loss0.005不控制该路径。放宽cap会改变患者primary和已有secondary的分段/Urban/涨落抽样，因此是物理近似，不是不变物理重构。

当前12M gamma绝对值仍约87%，不可宣称TOPAS99%已通过；三批SE仅粗略描述，不证明近似无偏。需要独立更高统计量确认后再考虑正式配置。历史TOPAS逐体素方差不可用，无新TOPAS模拟。未锁频顺序运行，GPU温度/clocks在原report保留（12M control210→1860MHz、54→81°C；1mm1860→1800MHz、81→85°C；0.2mm0→1860MHz、61→83°C）；不同温度会影响小差异，不能宣称0.5mm物理计算本身快于1mm。吞吐含输出，4M/batch优于1M/batch含输出速率不能全归因于运输。

## Artifacts

out/proton_minibeam_opt_20261010/下：manifest.json、confirmation_manifest/source_audit/upstream_audit.json、screen_summary.json、repeat_summary.json、confirmation_summary.json、current_research_baseline.json、active_source_hashes.json。选中设置confirm_0p2mm/case/gpu.yaml和run_gpu（要求3×4M源）；remote_results下保存全部dose/energy/quality/log。remote /home/v/MAIGO_clean_a6000_20261010/proton_minibeam_opt/。

重算分析：python3 out/proton_minibeam_opt_20261010/analyze.py confirm_control confirm_1mm confirm_0p5mm confirm_0p2mm。首批冷启动从subprocess到首dose写完，文件系统缓存未清；warm为后两批transport/output墙钟，连续复用GPU context/packages。

12M driver SHA256 84a4c58f19f9f7e312c364ec0654d32c56f98ae0ca965ef0035c6fd04a90e5a2。正式minibeam仍要求电子包，私有config.cpp仅使用此前限定proton/CT/research/all-electron-path-empty的例外，未修改production验证。


## Requested236M continuation

Userrequested236086432histories at0.2mm completed; historicalTOPASlocal3%/0.2mm/search0.02mm 98.993851% (183692/185559), belowstrict99%. GTVmeanchange+0.073385%; GPU_SE_RMS1.100728%. Warm275318/s,24auditspass/zerooverflow. No matched236M0.1mm reference/newTOPAS/defaultpromotion. See docs/proton_minibeam_236m_step0p2_20261010.md and out/proton_minibeam_236m_step0p2_20261010/results/report.json.

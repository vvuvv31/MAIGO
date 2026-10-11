# 碳 CT broadbeam 次级碎片涨落消融（2026-10-10）

原 RT07575 head full plan，电子输运关闭，最大离子步长1 mm。只将 `enable_secondary_energy_straggling` 从true改为false，primary涨落、平均能损、CT精确边界、散射和核反应保持一致。

关闭该开关同时将受限连续能损涨落与delta矩抽样替换为各自的均值；没有停止带电碎片输运，没有将核中性产物能量改为局部沉积。`unified_em_explicit_loss()`现有路径直接支持此research实验，无需修改输运代码。

每组1600万粒子、1200万计时粒子；ABBA+BABA四配对。每个campaign第一批仅排除计时，全部剂量参与归一化；源分配、每spot seeds成对匹配，跨campaign平均剂量。历史TOPAS患者内reference≥10%max，local1%/1mm、搜索0.1mm，不拟合或重标定剂量。

|指标|次级涨落开启|次级涨落关闭|
|---|---:|---:|
|吞吐 /s|136,707|173,913|
|患者gamma|99.559277%|99.556428%|
|患者评价区域剂量和相对TOPAS偏移|-0.613011%|-0.613408%|

吞吐提升 **27.22%**；四个配对比值为[1.2488538935733011, 1.2984413879432735, 1.2793182560823393, 1.262156686940045]，均为正收益。

primary有效能损调用变化+0.0000%，secondary变化-0.0095%；收益主要体现为每次调用更便宜，不能推断为已测硬件瓶颈。患者评价区域剂量和相对对照变化-0.000400%，两组剂量相对L1差0.000727142，包含有限蒙卡统计误差。

32个保存批次全部通过accepted/energy/completion/coverage/zero-overflow检查；实际cap标记全部1 mm。同一GPU executable始终未改变。GPU未锁频，绝对吞吐仅适用于本机工况。历史TOPAS体素方差不可用；本轮1600万/组是候选消融验证，不是8000万最终验收或其他病例的精度保证。细网格、LET及minibeam不在本轮验收范围。

结果支持将其作为碳CT broadbeam的快配置候选。实验配置保留在 `out/carbon_ct_secondary_fluctuations_20261010/candidate.yaml`；八个默认预设仍保留primary/secondary涨落，不修改AGENTS的默认物理约束。原始summary、audit_summary、manifest和逐批结果同目录。

# 保留primary涨落的CT broadbeam后续优化（2026-10-10）

两类实验都保留primary完整连续/delta涨落、1 mm最大步长和原RT07575 head full plan；电子输运关闭。射程反解代码对照同时保留secondary涨落。质子次级消融仅修改 `enable_secondary_energy_straggling`。

每种粒子每组1600万histories，1200万计时histories；ABBA+BABA四配对，每campaign首批仅排除计时。保持原CT/source/placement/weight/normalization；全campaign分配源权重、per-spot seeds成对一致；跨campaign平均归一化剂量。历史TOPAS患者内reference≥10%max，local1%/1mm、搜索0.1mm，无剂量拟合。

|候选|粒子|对照 /s|候选 /s|增益|候选gamma|处理|
|---|---|---:|---:|---:|---:|---|
|射程反解提前结束|proton|424,170|425,620|+0.342%|99.954485%|回退，配对结果混合|
|射程反解提前结束|carbon|133,319|133,673|+0.265%|99.488909%|回退，配对结果混合|
|仅关闭次级涨落|proton|411,861|418,599|+1.636%|99.954485%|保留实验配置，默认不变|

Newton候选在精确FP32固定点时提前结束，其余情况仍执行最多5轮，保留原校准射程积分和端点算法。实际包18粒子/26材料/多密度能量838656状态与旧5轮逐位一致；GPU matched checks中源、完整EM/discrete/live-ion账目匹配，剂量L1<1e-5（FP32原子计分顺序误差），TOPAS gamma一致。组件收敛频率不是runtime inverse调用率。实际测速增益不足且配对混合，已回退header和两个selective executable/libcarbon_core.a；full executable未改。记录于failed.md，候选与证据保留在 `out/broadbeam_inverse_fixedpoint_20261010/`。

质子仅关闭次级涨落：四配对比值[1.0118934900484327, 1.023292888638156, 1.0112154259346997, 1.0190842506627151]，均提速，但总增益仅1.636%。对照gamma99.954796%，候选99.954485%。32批accepted/energy/coverage/completion/zero-overflow检查通过，源与spot seeds匹配。保留research候选于 `out/proton_ct_secondary_fluctuations_20261010/candidate.yaml`，未改默认配置。

两轮模拟均已完成，无新增输运源码优化保留。默认八预设primary/secondary涨落仍开启，CT broadbeam最大步长1 mm，water/minibeam设置不变。此前碳仅关闭secondary涨落的候选约+27.2%仍是当前更大的快配置收益；本轮不重报为新增收益。

本机RTX2080Ti未锁频，历史TOPAS体素方差不可用。16M/arm是配对筛选结果，质子小幅收益尚未经过80M/arm确认；不外推其他患者/LET/minibeam。所有实验实际开关以归档配置及manifest为准。

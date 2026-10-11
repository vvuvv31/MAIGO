# RT07575 CT broadbeam：0.5 mm → 1 mm 步长对照

仅改变 `maximum_step_mm`，使用同一已验证 GPU executable。质子、碳电子输运均关闭；原计划、CT、源、权重、归一化、散射、primary/secondary 涨落和核输运保持一致。八个默认配置未修改。

每种粒子每组累计 1600 万 histories；ABBA+BABA 四个配对，各 campaign 4×100 万 histories。每个 campaign 第一批仅从计时中排除；剂量使用全部 histories 并按原计划入射数一次归一化。每组计时 1200 万 histories。

Gamma：历史 TOPAS 患者剂量，local 1%/1 mm，搜索 0.1 mm；患者内 reference ≥10%max，未拟合剂量或改变评价掩膜。

|粒子|0.5 mm 吞吐 /s|1 mm 吞吐 /s|增益|0.5 mm gamma|1 mm gamma|
|---|---:|---:|---:|---:|---:|
|proton|245,180|421,331|+71.85%|99.940456%|99.937650%|
|carbon|84,040|133,669|+59.05%|99.408001%|99.437059%|

## 工作量与检查

- proton：primary 有效能损调用 -40.66%，charged secondary 有效能损调用 -12.68%；四配对吞吐比 [1.7101419866701726, 1.7265633172833608, 1.7195376099899007, 1.717487228568232]。患者剂量相对 L1 差 0.0147527，包含有限统计误差。
- carbon：primary 有效能损调用 -28.62%，charged secondary 有效能损调用 -26.48%；四配对吞吐比 [1.5960026090787616, 1.5779641055917981, 1.6072868211523645, 1.5811854743234308]。患者剂量相对 L1 差 0.00855591，包含有限统计误差。

所有保存批次通过 runtime acceptance、能量闭合与零溢出检查；源分配和每 spot seeds 成对一致。每组16个实际 step-cap 标记核验为0.5或1.0 mm。1 mm 是上限，实际步长仍受能损、CT几何和核反应竞争限制。

这是1600万粒子/组的步长近似实验，不是8000万粒子的最终验收，也不是八预设通用精度结论。历史TOPAS的体素方差不可用，RTX2080Ti未锁频。

原始结果：`out/broadbeam_step_1mm_20261010/summary.json`，配置和复现命令位于同目录。

## 默认配置采用

2026-10-10 按用户要求将 `config/proton_ct_broadbeam.yaml` 和 `config/carbon_ct_broadbeam.yaml` 的 `maximum_step_mm` 改为 `1.0`。电子输运继续关闭，其他物理设置不变；water broadbeam 仍为0.5 mm，四个minibeam仍为0.1 mm。上文“八个默认配置未修改”描述实验执行时的状态。

默认配置关键物理参数与已测1 mm实验配置匹配；release identity/mass/TPS/geometry检查及173项资产、8预设manifest检查通过。本次配置采用未重新运行蒙卡，剂量和吞吐证据为上面的1600万粒子/组对照。

# 26成分电子包默认切换（2026-10-06）

八个默认预设已切换到独立26成分电子包。旧82/83节点包归档至 `data/archive/electron_density_nodes_20261006/`，仅用于显式离线参考，无默认回退。空间joint继续保留。

共用包 `data/electrons/shared/` 登记载荷1,746,666,256字节；`proton/parameters.bin` 和 `carbon/parameters.bin` 各44字节，三包独立载荷合计1,746,666,344字节。shared中的proton26/carbon26是参数文件的硬链接别名。charged/photon索引各26组成，无运行时密度节点。

## 默认预设回归

当前FP32构建，每场景连续两批；下表为smoke，不是稳定吞吐或water剂量验收。

| 预设 | Primary | 运行审计 | 最大相对能量残差 |
|---|---:|---|---:|
| proton_ct_broadbeam | 2×10,000 | PASS | 6.99e-06 |
| proton_ct_minibeam | 2×10,000 | PASS | 3.91e-07 |
| proton_water_broadbeam | 2×10,000 | PASS | 1.63e-07 |
| proton_water_minibeam | 2×10,000 | PASS | 5.02e-08 |
| carbon_ct_broadbeam | 2×10,000 | PASS | 4.6e-06 |
| carbon_ct_minibeam | 2×10,000 | PASS | 1.34e-06 |
| carbon_water_broadbeam | 2×10,000 | PASS | 7.27e-06 |
| carbon_water_minibeam | 2×10,000 | PASS | 1.66e-06 |

所有批次质量、能量、EM、覆盖及完成数检查通过，队列溢出为零。均匀水broadbeam选区为空，不加载电子worker。文件openat跟踪确认默认运行旧密度节点包打开次数为零。包结构校验、FP32与体素缓存生成一致性检查通过。

首次water minibeam smoke发现电子Water_75eV section25未转换为离子EM section−1，导致出生查询失败。现已修复并完整重跑八场景，失败证据保留在failed.md和首次smoke目录；未使用旧包或放宽覆盖审计。

## 剂量与验证范围

冻结的CT剂量验证见[完整报告](electron_complete26_20261006.md)：proton/carbon broadbeam local1%/1mm分别99.86565492%/99.79050826%，均>99%，搜索0.1mm。RT07575是头部病例，其0.1% full plan共236,086,480个primary，新包local3%/0.2mm为98.90008030%，旧包98.90331377%，下降0.00323347个百分点，符合用户≤1个百分点退化门槛；搜索0.02mm。它尚未达到绝对>99%标准。默认CT minibeam为独立肺CT场景。

仍保留actual-cut硬率质量缩放、组成标定Penelope振子律和photon条件终态近似。当前gamma门槛不要求继续补齐这些近似。Carbon minibeam准确性及water剂量验收仍待独立验证；本次切换不宣称吞吐提升。

证据：`out/electron_complete26_20261006/default_promotion_smoke_v2/summary.json`、`default_promotion_open_v2.log`；结构化切换记录见同名JSON。正式CT剂量使用冻结v2构建；当前构建另包含源字节诊断修正及water编号修复，后者不改变Schneider CT分支。

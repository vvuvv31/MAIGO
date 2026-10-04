# RT07575 proton minibeam：当前FP32版本0.1% full plan

完成 236,086,432 primary，487 spots，119批单进程，group32。按4 Gy计划入射粒子数的0.1%取样，输出按粒子数缩放回4 Gy计划；无剂量拟合。

执行文件SHA256：`ab913f824ddd13968f56a47acd897c9283f0a872a4d2af338c6e2cf9ab47d19a`。使用全FP32独立电子birth/joint/charged/photon路径、参数复用、速率认证和体素缓存；未启用已否决的elastic-batch候选。

蒙卡进程墙钟 2178.596 s（36.31 min）；含加载吞吐 108366.3 primary/s；排除首批后的常驻吞吐 108716.1 primary/s。电子阶段累计 956.427 s。后处理耗时另计。

119/119批research质量检查通过；主输运/EM审计与历史同源同种子计划一致，电子全部完成，缺失覆盖、invalid和队列溢出均为0，零旧电子包加载/回退。最大相对能量残差 1.104e-07。质量检查通过不等同正式患者物理验收。

| Gamma vs TOPAS | Local | Global |
|---|---:|---:|
| 1% / 0.2 mm | 92.1098% | 99.6513% |
| 2% / 0.2 mm | 97.1923% | 100.0000% |
| 3% / 0.2 mm | 99.1006% | 100.0000% |
| 3% / 0 mm | 92.2806% | 100.0000% |

参考为既有TOPAS加权DIJ；BODY∪GTV、10%全参考最大剂量阈值、185,559体素。0.2 mm结果采用三线性插值和0.02 mm搜索格，不能解释为参考原始网格具有0.02 mm分辨率。

与上一次完整Schneider joint＋旧电子包混合路径相比，BODY剂量相对L1变化 **0.07261%**，BODY总剂量变化 **+0.00104%**。在旧GPU剂量≥1%BODY最大值的体素中，局部差异99/99.9/max为0.6258%/1.2641%/8.1479%；随机采样与模型差异尚未分离，不把平均吻合解释为所有局部体素1%一致。详见[历史GPU对比](results/previous_comparison.json)。

| GTV Gy | TOPAS | 当前GPU |
|---|---:|---:|
| mean_Gy | 4.918585 | 4.917729 |
| D95_Gy | 4.000110 | 3.991061 |
| D98_Gy | 3.661170 | 3.662124 |
| D50_Gy | 4.982727 | 4.981149 |
| D2_Gy | 5.704635 | 5.721620 |

[4 Gy患者坐标剂量](results/dose_4Gy.mhd) · [剂量平面图](results/dose_comparison.png) · [DVH与横向剖面](results/dvh_profile.png) · [完整运行检查](batch_checks.json) · [耗时](results/timing.json) · [Gamma](results/gamma.json)

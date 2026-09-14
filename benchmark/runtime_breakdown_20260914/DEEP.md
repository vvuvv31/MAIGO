# RT07575 preparation breakdown — deep probe

沿用 runtime_current_1m_baseline 的冻结配置、粒子数、随机种子和物理数据。正式源码未插桩。

诊断 Elapsed=52.402 s；相对基准增加 2.63%。最大体素剂量差=0.00002174% of peak；步数、EM audit、核反应计数相同，零 overflow。

以下为每约 2048 步采样 lane elapsed cycles 的分布，含等待/分歧/插桩效应，不是各过程独立 wall-time；不能直接换算可消除秒数。

## Primary

| Region | Sampled cycle share | Intervals |
|---|---:|---:|
| CT/material and initial state | 4.44% | 1075545 |
| EM material selection/coverage | 3.17% | 1075545 |
| EM stopping/range/fluctuation parameter preparation | 13.28% | 1075545 |
| Delta rate/clock/EM step | 6.39% | 1075545 |
| Geometry clamps | 3.90% | 1075545 |
| Nuclear rate/hazard | 8.22% | 1075545 |
| Pre-loss mean/audit/legacy branches | 20.41% | 1075545 |
| Unified EM loss | 29.66% | 1075545 |
| Scoring branches | 2.26% | 1075545 |
| MCS | 4.35% | 1075545 |
| Advance/face handling | 2.06% | 1075545 |
| Nuclear resolution/bookkeeping | 1.85% | 1075545 |

## Secondary

| Region | Sampled cycle share | Intervals |
|---|---:|---:|
| CT/material/stopping and initial state | 4.64% | 330793 |
| EM material selection/coverage | 5.23% | 330667 |
| EM stopping/range/fluctuation parameter preparation | 19.21% | 330667 |
| Delta rate/clock/EM step | 9.77% | 330667 |
| Geometry clamps | 3.96% | 330667 |
| Nuclear rate/hazard | 9.38% | 330667 |
| Other pre-loss/recoil/legacy branches | 2.55% | 330667 |
| Unified EM loss | 32.89% | 330667 |
| Post-EM geometry/scoring | 5.06% | 661196 |
| MCS | 4.35% | 330529 |
| Inelastic replay/scoring | 2.50% | 330667 |
| Elastic/bookkeeping | 0.47% | 330529 |


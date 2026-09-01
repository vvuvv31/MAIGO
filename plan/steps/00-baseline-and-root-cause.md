# 00：基线与根因证据（COMPLETED）

## 目标

在修改物理模型前冻结可复现基线，并区分 rate、birth、transport、scorer semantics 和 package coverage 问题。

## 已完成

- [x] 建立 validation build，RTX 2080Ti/sm_75 可运行。
- [x] TOPAS/GPU rate-only 100k：rate、CDF、survival、H/O 均在 1% 内。
- [x] G0 birth-only：born/queued Z1--Z6 closure，overflow=0。
- [x] 200 MeV/u G0/G1/G2；G1 明显优于 G0，G2 增益有限。
- [x] GPU Z1 改为 p+d+t，与 TOPAS AtomicNumber=1 对齐。
- [x] 四能量无过滤 total + 7 species TOPAS/GPU 100k 3D IDD。
- [x] G1/G2 和 double-Gaussian core/halo 对比。
- [x] 确认 TOPAS particle-filtered species 之和不等于 total，主要来自 electron/positron attribution。
- [x] 确认当前 package 未统计认证，nearest-energy-node 事件池接近单事件。

## 结论

碰撞率不是首要根因。顺序必须是 scorer semantics → package coverage/sampling → reaction ledger → birth/cascade → neutral/transport。

## 证据

- `plan/artifacts/rate-only-comparison.json`
- `plan/artifacts/species-idd-100k-e{100,200,300,400}-g1-total/`
- `plan/artifacts/species-idd-100k-e{100,200,300,400}-g2-total/`
- `plan/artifacts/double-gaussian-sigma-g2-100k/`

## 验收

- [x] 基线数据和配置可追溯。
- [x] 后续依赖顺序明确。

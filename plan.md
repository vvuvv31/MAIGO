# FRED GPU 碳碎裂复刻 — 工作计划

审查后只勾选代码存在且本轮验证过的项。

## 已完成（本轮验证）

- [x] Newton 与 GPU 使用同一套碎片生成规则：弹核最多 8 个碎片；O-16 靶抽到第一个重核 (Z≥3 或 A≥6) 即停
- [x] `products` 扩到 `kMaxInelasticProducts=12`，与 `num_frags` 上限一致；容量截断单独计数。400 MeV/u 100k：`product_capacity_overflow = 0`
- [x] 重残核射程用 17 核素阻止本领 LUT 在当前 MeV/u 插值估计（`E/SP`），短于 0.05 mm 或低于 cutoff 才局域
- [x] 非弹相关 `malloc_device` 失败时释放已分配 USM 再抛 `bad_alloc`
- [x] 四套反解 `max_abs_fraction_error` 写入 ledger：`fred_invert_error_proj_h/o`、`tgt_h/o`。本轮 400 MeV/u：0.122 / 0.137 / 0.622 / 0.182（**未达 1%**）
- [x] 容量溢出、重采样失败、Q/中子/残核/residual 分项可观测
- [x] `carbon_tests` 通过；400 MeV/u 100k overflow=0、capacity overflow=0、scaled=0

## 未完成（禁止标 x）

- [ ] Newton 主要核素绝对误差 &lt;1%；未收敛禁止 GPU（tgt_h 误差 0.62，proj 约 0.12–0.14）
- [ ] CSDA 积分射程 LUT（当前是单点 SP 插值，不是沿能量积分）
- [ ] 全能量 TOPAS 验收（400 峰值/ROI 此前失败；本轮未重跑 `validate_metrics.py` 四能量）
- [ ] 本轮物理的 1M 复测
- [ ] 400 Bragg 峰空间分物种剂量
- [ ] 纯 EM 400 对照
- [ ] model residual 物理解释：400 MeV/u residual/E_in ≈ **9.71%**（仍回填 untracked；会计残差 ~6e-7）

## 本轮 400 MeV/u 100k

| 量 | 值 |
|---|---|
| inelastic | 71774 |
| resample_failed | 700 |
| capacity overflow | 0 |
| residual / E_in | 9.71% |
| invert err proj_h/o, tgt_h/o | 0.122, 0.137, 0.622, 0.182 |

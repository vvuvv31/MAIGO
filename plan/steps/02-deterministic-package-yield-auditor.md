# 02：Deterministic Package Yield Auditor

## 目标

离线精确复现 `cinel02_find_event_device()` 的 `±0.51 MeV/u` support window，不依赖 full-cascade 稀疏 birth 自然收敛。

## 计算

1. 对每个 `(projectile,target,E)` 枚举 tolerance window 中所有完整 correlated events。
2. 输出 candidate count、event reuse weight、parent outcome、isotope yield、mean birth E/A、angle 和 total product KE。
3. 用 Step 01 的 GPU parent-energy occupancy、H/O occupancy 和 generation occupancy 加权，得到 package exact expectation。
4. 两级比较：
   - actual GPU births vs package exact expectation：判定 runtime index/RNG/sampler bug。
   - package exact expectation vs TOPAS source campaign：判定 capture/package construction 问题。
5. 按 source-initial-energy 拆分 campaign，检查 duplicate event、boundary、merge weight、O16 identity 和 hybrid-grid weight。

## Be9 当前门槛

- C12+O16、50--100 MeV/u：package 2,982 births，TOPAS 100k 仅43 births，差异约2σ。
- source mixing 已排除：98.3% 反应来自 200 MeV/u campaign，各 source yield 一致。
- TOPAS 1M job 485 用于趋势判定；deterministic expectation 用于决定是否修 package。

## 退出条件

- [ ] host auditor 与 device fixed-seed candidate/event sequence 一致。
- [ ] GPU actual births 落在 package exact expectation 的统计 CI 内。
- [ ] job 485 完成并与 package/source campaign 做同 cell 比较。
- [ ] 每个结论报告 event count、CI/ESS，不对低统计 ratio 强制 2%。

## 禁止项

job 485 与 auditor 不同时指向 package 前，禁止修 sampler/yield，禁止 `Be9_scale`。

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

## 200 MeV/u G1 100k 运行记录

- 代码基线：`c4ceee4` 之后的 10-MeV/u replay diagnostic refinement（GPU physics 未改变）。
- Config：`config/beam_200MeVu_cinel02_e200light107_g1_transitiondiag4_100k_xy04.yaml`；seed `2026095100`；100,000 histories；3D scorer `200×200×800`，0.4×0.4×0.5 mm，80×80 mm FOV。
- Package SHA256：`8a54b8544aea484fa3ff649fc372c22d4b48deff4c2fe37dbd31b2f7f6adb25d`；rate SHA256：`aa811ff18684a6a8f81f79fa38f1130b4103555ea2932abb77f7163e1c160ed7`；config SHA256：`c29cbe1a4957035294e4fdd1abc3798740cdf723b8aa0b2dc3cc7281e4cf9299`。
- Auditor：`startup/package_tools/audit_cinel02_runtime_package.py`；JSON/CSV/summary：`plan/artifacts/cinel02-package-audit-e200-g1/audit_10mev.json`、`audit_10mev.csv`、`RESULTS.md`。
- Package：3,663,101 interactions；30,051,236 products；3,573,494 global energy nodes；UUID `00000000-0000-4000-8000-000000910000`。
- GPU generated/queued transitions：`379,379/379,379`；package expected `378,905.516 ± 594.083` package-event sampling σ；total z=`0.80`。
- Be/Li isotope rows：6Li z=`0.60`、7Li z=`−0.27`、7Be z=`0.37`、9Be z=`0.28`、10Be z=`0.04`、6Be z=`0.66`；未发现需要修改 sampler/yield/package 的一致偏差。
- Replay status：candidate/valid/lookup-miss/invalid/cutoff=`71524/71318/192/0/14`；true runtime lookup miss=`192/71524=0.268%`。
- 10 MeV/u refinement 将旧 50-MeV/u cell-mean aggregate discrepancy（−2217 products，约 −3.7σ）降为 +473 products（+0.80σ）。

## 结论

- [x] CINPKG03 header、global index、interaction/product stream 完整扫描。
- [x] 以 runtime ±0.51 MeV/u window 和实际 isotope/target/generation occupancy 计算 deterministic expected product yield、parent outcome 与 kinetic energy。
- [x] GPU actual generated/queued transition 与 package expectation 完成比较；Be/Li 未显示 package-wide 或 sampler-wide bias。
- [x] TOPAS job 485 同 cell source evidence 已绑定：Be9 50–100 MeV/u TOPAS `547/46962=0.0116477`，package `2982/242173=0.0123135`，约 1.2σ。
- [x] 结果和限制已归档到 `plan/artifacts/cinel02-package-audit-e200-g1/RESULTS.md`。
- [ ] fixed-seed host/device 逐事件 sequence equality 不能由当前 aggregate ledger 证明；这是 Step 03 replay-support consistency 的输入限制，不作为 package 修改理由。

Step 02 完成非物理 package gate；sampler、yield、package 与 physics 参数保持冻结，进入 Step 03。

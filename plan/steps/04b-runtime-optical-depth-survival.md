# 04B：Runtime optical depth 与 empirical survival self-audit

## 目标

验证 secondary CINEL02 hazard sampler 是否与自身 runtime rate 一致，并把
generation gate 的潜在反事实 optical depth 与实际 candidate 分开记账。本步骤只增加
diagnostics，不修改 rate、target selection、generation gate 或 transport physics。

## 实现

- exposure ledger 扩展 `hazard_blocked_h/o/total`，记录 generation-blocked step 在同一
  H/O coverage 下的 counterfactual optical depth。
- 汇总器输出 isotope 与 isotope×transport-generation 两种视图，并计算
  `candidate_minus_tau_runtime`、`candidate_to_tau_runtime`、Poisson z-score、
  `blocked_hazard_fraction`。
- package auditor 输出 isotope×target×reaction-generation 的 expected/actual parent
  continue/kill outcome。

## Canonical 运行

- 200 MeV/u、100k histories、seed `2026095100`、G1、compatibility=true。
- 本机 RTX 2080 Ti / sm_75；3D scorer，未修改 package/rate/stopping/MCS。
- Ledger：`out/beam_200MeVu_cinel02_e200light107_g1_topascompat_100k_xy04/energy_ledger.json`。
- Summary：`plan/artifacts/cinel02-survival-e200-g1-topascompat/exposure-summary.json`；package
  audit：`plan/artifacts/cinel02-survival-e200-g1-topascompat/package-audit.json`（均为本地
  运行产物，不纳入提交）。

## 结果

主要 transportable isotope：

| isotope | τ runtime | blocked τ | blocked-hazard fraction | candidates | candidate z |
|---|---:|---:|---:|---:|---:|
| 6Li | 267.175 | 12.555 | 4.49% | 279 | +0.72 |
| 7Li | 279.265 | 5.835 | 2.05% | 263 | −0.97 |
| 7Be | 229.547 | 4.101 | 1.76% | 204 | −1.69 |
| 9Be | 36.270 | 1.868 | 4.90% | 31 | −0.88 |
| 10Be | 40.106 | 4.988 | 11.06% | 49 | +1.40 |

- 04B-1：主要 isotope 的 candidate-vs-τ runtime 均在约 ±2σ 内；未发现 hazard
  sampler identity 证据。100k 对稀有 Be isotope 只适合 self-consistency，不作 2% 物理判断。
- 04B-2：generation-blocked path 对应的 hazard fraction 明显低于 path fraction，
  但 10Be 仍约 11.06%，因此暂不修改 G1 gate，留给 04C/因果分析。
- 04B-3：package expected parent outcome 对当前 occupied cells 均为 100% kill；GPU
  sampled valid secondary outcomes 33,306 kill、0 continue，与 package expectation
  一致，无 continuation selection bias。

## 退出结论

- [x] candidate-vs-τ runtime closure 完成。
- [x] blocked counterfactual hazard 完成；未静默抑制 generation gate。
- [x] package-vs-runtime parent outcome 完成。
- [x] 不修改任何 physics；下一步进入 04C stopping residence 与 continuous optical depth，
  并保留 G1 gate 作为待验证假设。

## 遗留问题

- `τ_runtime` 仍是 step-start rate 的 compensator；尚未与独立
  `τ_continuous = ∫λ(E(s))ds` 或 TOPAS survival reference 比较。
- 04B 不能单独证明 GPU secondary survival 与 TOPAS 一致；04C 需要补 stopping
  residence/continuous integration，并继续区分 generation-blocked、rate-uncovered 和
  actual reaction outcome。

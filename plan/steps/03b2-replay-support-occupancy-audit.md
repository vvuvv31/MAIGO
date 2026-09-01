# 03B-2A：Occupancy-aware replay-support audit

## 目标

在修改 runtime rate、target CDF 或 step segmentation 之前，使用已记录的 10-MeV/u
replay status 与 rate/package census，确定实际 miss 的归属。6Be 已由 TOPAS compatibility
policy 标记为 non-transportable，不参与本 audit。

## 实现

- 新增 `startup/package_tools/audit_cinel02_replay_support_occupancy.py`。
- 输入仅为只读 `census.json` 与 runtime `energy_ledger.json`；不改 package、rate、tolerance、CDF 或 target mix。
- 按 isotope×target×reaction_generation×energy-bin 汇总 candidate/valid/miss/cutoff、E_rate、E_replay、missed replay KE 和 continuous loss。
- 使用 occupied-cell mean 做 provisional 分类：
  `raw_or_compiler_support_gap`、`post_em_support_boundary`、
  `index_or_lookup_anomaly`、`rate_occupancy_or_interpolation_gap`。
- 输出 count-weighted 与 replay-energy-weighted miss fraction，并明确 `-log(1-f)` 只是 hazard proxy；compact ledger 没有 per-collision path length，因此不宣称 physical optical depth。
- 6Be 输出为 `non_transportable_prompt_decay` / rate=N/A / replay=N/A。

## 200 MeV/u canonical evidence

- Config：`config/beam_200MeVu_cinel02_e200light107_g1_topascompat_100k_xy04.yaml`；compatibility=true。
- Ledger：`out/beam_200MeVu_cinel02_e200light107_g1_topascompat_100k_xy04/energy_ledger.json`。
- Census：`plan/artifacts/cinel02-rate-package-census-e200-g1/census.json`。
- Audit：`plan/artifacts/cinel02-replay-support-occupancy-e200-g1-topascompat/audit.json`。
- 固定条件：200 MeV/u、100k、seed `2026095100`、G1、3D scorer 200×200×800、80×80 mm FOV。
- candidate=71524，lookup miss=192，count fraction=0.268441%；其中 provisional `raw_or_compiler_support_gap` 为 174，`index_or_lookup_anomaly` 为 18；`post_em_support_boundary` 与 `rate_occupancy_or_interpolation_gap` 未在实际 miss cell 中出现。
- Li6/Li7、Be7/Be9/Be10 的 candidate miss 均为 0；因此 03B-2A 不支持 lookup miss 是当前 Be/Li 积分剂量偏差主因。

## 退出条件

- [x] 6Be 排除并显式标记 non-transportable。
- [x] 实际 occupied miss 已按 energy/target/generation/isotope 分类并给出 energy-weighted proxy。
- [x] 未修改 runtime physics；补充 Python synthetic tests，手工运行 2/2 passed。
- [ ] 逐 collision path length/optical depth 仍未记录，留给 reaction-survival/stopping 阶段。
- [ ] 03B-2B 需先修 source/compiler support consistency；只有 raw source 真 gap 才允许考虑 suppress hazard，并记录 suppressed hazard。

## 遗留问题

- 由于分类基于 cell mean，`raw_or_compiler_support_gap` 与 boundary 的最终归属需要逐 collision energy 或 source/compiler manifest 证据确认。
- 03B-2A 不是 physics fix；sampler/yield/package、stopping、straggling、MCS 均保持冻结。

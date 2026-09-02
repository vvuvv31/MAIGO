# 04C：Stopping residence 与 continuous optical depth

## 目标

把 04B 的离散 runtime compensator

```text
tau_runtime = Σ lambda(E_step_start) * ds_actual
```

与沿实际 stopping 能量轨迹的独立连续近似分开记录，并量化
`generation-blocked` 反事实路径。该步骤只增加诊断，不改变 rate、target selection、
step size、generation gate、stopping、straggling 或 MCS。

## 实现

- exposure schema 从 13 个扩展到 18 个 sum metrics。
- 每个 secondary step 在 `E_start`、`E_start - dE/2` 和 `E_end` 查询 H/O rate，
  对实际（含 collision truncation）的 `ds` 使用 Simpson quadrature：

  ```text
  tau_continuous = ds / 6 * (lambda_start + 4 lambda_mid + lambda_end)
  ```

- 只有三个能量点的 H/O lookup 全部 covered 时才累计 continuous hazard；否则累计
  `path_mm_continuous_rate_uncovered`，避免把缺失 support 静默当成零率。
- `hazard_continuous` 只包含 generation-eligible path；
  `hazard_blocked_continuous` 是 generation-blocked path 的连续反事实 hazard。
- `stopping_loss_MeV` 记录实际每步 EM loss，汇总器进一步输出
  `stopping_residence_mm_per_MeV`。
- 新增 `cinel02_simpson_hazard()` host/device 共用辅助函数及 synthetic regression。

## Canonical 运行

- 200 MeV/u、100,000 histories、seed `2026095100`、G1、
  `cinel02_topas_compatibility_mode: true`。
- 本机 RTX 2080 Ti / sm_75，CUDA SYCL，3D scorer。
- 配置：`config/beam_200MeVu_cinel02_e200light107_g1_topascompat_100k_xy04.yaml`。
- 输出 ledger：`out/beam_200MeVu_cinel02_e200light107_g1_topascompat_100k_xy04/energy_ledger.json`。
- 汇总为本地临时产物，未提交 raw/3D dose。

## 结果

| isotope | tau_runtime | tau_continuous | continuous/runtime | blocked tau (runtime) | blocked tau (continuous) | candidates | candidate z |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6Li | 267.176 | 266.481 | 0.997402 | 12.555 | 12.181 | 279 | +0.72 |
| 7Li | 279.265 | 278.843 | 0.998488 | 5.835 | 5.657 | 263 | -0.97 |
| 7Be | 229.547 | 229.098 | 0.998046 | 4.101 | 4.013 | 204 | -1.69 |
| 9Be | 36.270 | 36.154 | 0.996792 | 1.868 | 1.795 | 31 | -0.88 |
| 10Be | 40.106 | 40.044 | 0.998442 | 4.988 | 4.954 | 49 | +1.40 |

- `tau_continuous` 比 `tau_runtime` 低约 0.15--0.32%，说明当前 step-start rate
  近似在本配置下不是百分之几级别的主因。
- generation-blocked continuous hazard 与 04B 的 runtime 反事实量级一致；
  `10Be` 仍是最高的 blocked-hazard 贡献，但本步骤不据此直接修改 G1 gate。
- 04B 的 candidate-vs-runtime-tau 结论保持不变。

## 验证

- NVIDIA SYCL release build：成功。
- CTest：`2/2` 通过。
- Python synthetic tests：`3/3` 通过（pytest 未安装，采用直接调用）。
- GPU canonical run：成功，131,082,460 steps，耗时约 5.36 s。

## 退出结论

- [x] runtime tau 与 continuous tau 分离并记录。
- [x] continuous rate coverage 与 stopping loss 可审计。
- [x] isotope 与 isotope×generation 汇总保留。
- [x] 不修改任何 physics；进入 TOPAS lineage/survival 与 Be/Li 因果分析。

## 遗留问题

- 该 continuous tau 是每步三点 Simpson 近似，不是独立的更细步长积分；
  如后续发现 `tau_runtime - tau_continuous` 达到物理显著量级，再增加 finer integration。
- 尚未建立 TOPAS 的 isotope lineage survival reference，因此目前只能证明 GPU
  runtime 与自身 rate/stopping exposure 的一致性，不能单独证明 GPU-TOPAS survival 一致。
- p/d/He4 residual 和 aggregate dose mismatch 仍未解决。

# 04A：Generation eligibility 与 H/O rate-coverage exposure audit

## 目标

在 reaction-survival optical-depth audit 前，量化 transportable secondary isotope 在每个
transport generation 和 10 MeV/u step-start energy bin 的实际路径暴露。generation-blocked
path 与 H/O rate 未覆盖 path 必须独立记账；本步骤不修改 rate、target selection、step size、
package、stopping、MCS 或 sampler。6Be 遵循 `TopasCompatKill`，因为不进入 secondary queue，
不产生 transport exposure。

## 实现

- `Cinel02ExposureLedgerSchema`：18 isotope × 3 transport generation × 40 个 10 MeV/u bin。
- path sums：`path_mm_total`、`path_mm_generation_eligible`、`path_mm_generation_blocked`、
  `path_mm_rate_covered`、`path_mm_rate_uncovered`、`path_mm_h_uncovered`、
  `path_mm_o_uncovered`。
- hazard sums：分别累计 covered H/O 的 `rate(E)·ds` 和两者同时 covered 时的
  `hazard_total`。
- event counts：`collision_candidates`、`replay_valid`、`parent_killed`、
  `parent_continued`。
- device → host → JSON → multi-batch accumulator 全链路接入；新增只读汇总器
  `startup/package_tools/summarize_cinel02_exposure.py` 与 synthetic tests。
- 所有 path 统计使用实际 secondary step 长度；发生 hazard 时使用被 collision point 截断后的
  `ds`，coverage 使用同一 step-start `E_rate`，因此不改变当前 runtime hazard。

## Canonical 运行证据

- Config：`config/beam_200MeVu_cinel02_e200light107_g1_topascompat_100k_xy04.yaml`。
- 200 MeV/u、100,000 histories、seed `2026095100`、G1；本机 RTX 2080 Ti/sm_75，CUDA SYCL。
- 3D scorer：200×200×800，0.4×0.4×0.5 mm，80×80 mm FOV；仅横向求和形成 IDD。
- Package/rate 与 c2e25b8 compatibility baseline 完全冻结。
- Ledger：`out/beam_200MeVu_cinel02_e200light107_g1_topascompat_100k_xy04/energy_ledger.json`。
- Summary command：

```bash
python3 startup/package_tools/summarize_cinel02_exposure.py \
  out/beam_200MeVu_cinel02_e200light107_g1_topascompat_100k_xy04/energy_ledger.json \
  -o plan/artifacts/cinel02-exposure-e200-g1-topascompat/summary.json
```

### 关键总量

| isotope | total path (mm) | eligible path (mm) | generation-blocked (mm) | rate-covered / eligible | candidates | valid |
|---|---:|---:|---:|---:|---:|---:|
| 6Li | 61,545.5 | 59,034.9 | 2,510.6 | 99.942% | 279 | 278 |
| 7Li | 59,712.0 | 58,664.6 | 1,047.4 | 99.962% | 263 | 263 |
| 7Be | 47,971.2 | 47,273.4 | 697.8 | 99.954% | 204 | 203 |
| 9Be | 6,707.8 | 6,393.6 | 314.2 | 99.796% | 31 | 31 |
| 10Be | 7,567.9 | 6,766.7 | 801.2 | 99.939% | 49 | 49 |

全体 secondary exposure 为 16,994,005.1 mm；eligible 为 13,921,212.1 mm，
generation-blocked 为 3,072,793.0 mm（18.08%）。eligible path 中 rate-covered 为
13,916,848.8 mm、rate-uncovered 为 4,316.7 mm（约 0.031%）；H/O uncovered path 在本次
水 phantom 运行中相同，说明这些 occupied segments 同时缺少 H/O pair coverage。

全局 event counts：candidate=33,512，valid=33,306，parent-killed=33,306，
parent-continued=0；这些与已有 replay status 的 secondary generation-1 candidate/valid
partition 一致。

## 结论与退出条件

- [x] generation-blocked 与 eligible path 已分开；不会把 generation gate 当成 rate=0。
- [x] H/O 独立 coverage、pair coverage、`rate(E)·ds` 和 actual reaction counts 已记录。
- [x] 04A 不改变任何 transport physics；CTest 2/2 通过，04A Python synthetic tests 通过。
- [x] 结果显示 Be/Li 的 H/O rate coverage 接近完整，但 G1 generation-blocked path 不可忽略：
  7Be 1.45%、9Be 4.68%、10Be 10.58%、6Li 4.08%、7Li 1.75%。
- [ ] 尚未比较 `tau_runtime` 与独立连续 `tau_continuous`，也尚未做 per-episode censoring；
  这些留给 04B/04C。

## 遗留问题

04A 发现的首要待验证项不是 replay miss，而是 cascade generation eligibility：在当前 G1
配置下，部分 Be/Li path 被 gate 阻断。下一步必须先在 04B 中比较 empirical reaction survival
与 `tau_runtime`，并在 04C 中检查 stopping residence；不能据此直接修改
`cinel02_max_secondary_inelastic_generations`，也不能用 dose 改善作为 gate。

# 03B-1：Be-6 rate/package coverage root-cause gate

## 目标

解释为什么 200 MeV/u G1 GPU 诊断中 `6Be` 的 replay candidate 为零，并区分：

1. rate/package compiler 过滤了已有的 Be-6 projectile exposure；
2. raw campaign 从未以 Be-6 为 projectile 运行；
3. Be-6 仅作为 primary C-12 fragmentation 的 direct child 生成。

本阶段不把 Be-7 rate alias 给 Be-6，也不改变 runtime rate、sampler、stopping、MCS 或 dose normalization。

## 证据

### 1. 生产 campaign scope

当前 `research_hybrid_e200light107` 的 `summary.csv` 有 102 个 source-energy campaign row，17 个 projectile identity；列表包含：

`1H, 2H, 3H, 3He, 4He, 6He, 6Li, 7Li, 7Be, 9Be, 10Be, 8B, 10B, 11B, 10C, 11C, 12C`。

没有 `(Z=4,A=6)`。原始 pilot/production 脚本也明确排除了 Be-6：

- `scripts/run_cinel02_hybrid_pilot.slurm` 的 17-species数组不含 `4 6`；
- `scripts/run_cinel02_hybrid_pilot_worker_v4.sh` 的 17-species数组不含 `4 6`；
- `scripts/run_cinel02_lightion_production_5m.slurm` 只运行 `1H/2H/3H/4He`。

### 2. raw 全量流式审计

使用 `startup/package_tools/audit_cinel02_projectile_campaign.py` 扫描：

`/mnt/sda/wuwei/cinel02-lightion-production-5m-v2/research107/research_hybrid_e200light107.cinel02`

该文件约 3.8 GB，审计通过 raw record CRC、interaction/product count 校验。结果：

- interactions：`3,663,101`；
- products：`30,051,236`；
- raw projectile interactions `Z4A6`：`0`；
- raw products `Z4A6`：`27,186`；
- `Z4A6` products 全部 role `0`（direct secondary）。

### 3. compiler 语义

`startup/package_tools/cinel02.py::compile_package()` 按 raw record 的 projectile/target/energy 分组，不包含 Be-6 专属的 projectile 删除分支；`startup/package_tools/compile_cinel02.py` 直接调用该 compiler。当前 package 因此没有 6Be projectile interaction/node，是输入 campaign scope 的结果，而不是已有 Be-6 projectile event 被 compiler 丢弃。

### 4. 与 package/rate census 的交叉验证

`plan/artifacts/cinel02-rate-package-census-e200-g1/census.json` 显示：

- `6Be+H1`：rate group 存在但 445 个 sample 全为零；package nodes `0`；
- `6Be+O16`：rate group 存在但 445 个 sample 全为零；package nodes `0`；
- `7Be/9Be/10Be`、`6Li/7Li` 均有正 rate 和 package nodes。

## 结论

[x] 03B-1 root-cause gate 完成。Be-6 的 secondary rate/package coverage 缺失来自 **source campaign 未运行 Be-6 projectile exposure**；Be-6 只在 C-12 event 中作为 direct child 出现。

这足以解释 `6Be` secondary hazard/candidate 为零，但不能仅凭 coverage 直接估算 +10.39% dose residual 的全部幅度。要验证 reaction-survival 影响，必须补充独立 `6Be+H1/O16` TOPAS exposure/event campaign。

## 下一项（03B-1R，数据补充）

1. 添加只针对 `GenericIon(4,6)`、200 MeV/u 的本地 TOPAS extraction smoke（先 50k histories）；
2. 检查 contract/raw CRC、H/O target coverage 和 Be-6 projectile interaction count；
3. 通过 smoke 后再提交 5M production campaign，并以 package/rate compiler 原有统计门禁编译；
4. 只把新 package/rate 放到独立输出目录，冻结当前 `research107` 基线不覆盖；
5. 新数据通过 deterministic auditor 和 200 MeV/u 100k GPU A/B 后，才决定是否将其纳入 physics baseline。

在 03B-1R 完成前，不修改 Be-6 rate 数值、不做 isotope alias、不做 dose scale。

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

[x] 03B-1 campaign-provenance gate 完成。Be-6 的 secondary rate/package coverage 缺失来自
**source campaign 未运行 Be-6 projectile exposure**；Be-6 只在 C-12 event 中作为 direct child 出现。

[x] 该结论随后被 TOPAS prompt-decay smoke 修正为不完整：普通 `GenericIon(4,6)` 的基态寿命为
`4.95333e-21 s`，200 MeV/u 下衰变长度约 `1e-9 mm`。开启 `g4decay/g4radioactivedecay` 时 50k
histories 得到 `Scored Entries: 0`，所以“没有 Be-6 projectile interaction”不仅是 campaign scope，
也是因为可输运的 Be-6 track 在产生后立即衰变。关闭 decay 的 50k 诊断得到 H/O interactions，
证明 INCL++/scorer 可以处理 Be-6，但该配置产生 energy-conservation/resampling warnings，
不能作为物理 package 输入。

因此 `6Be` candidate=0 目前只能说明冻结 package 没有稳定 Be-6 projectile replay coverage，不能直接
推出“应补一张 Be-6 secondary rate 表”。GPU 当前却把 C-12 event 中的 Be-6 child 当作稳定离子输运，
这使 Be-6 prompt-decay semantics 成为比 generic rate supplementation 更优先的物理 gate。

## 03B-1R 状态

详见 [03B-1R prompt-decay semantics gate](03b1r-be6-prompt-decay-semantics.md)。在该 gate 完成前：

- 不将 no-decay Be-6 raw 数据编译为生产 rate/package；
- 不把 Be-7/Be-9/Be-10 alias 给 Be-6；
- 不修改 Be-6 dose/rate scale 或 sampler；
- 冻结 `research_hybrid_e200light107` package 作为可复现基线。

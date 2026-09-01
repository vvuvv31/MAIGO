# 03B-1R：Be-6 TOPAS reference compatibility policy gate

## 目标

将 6Be 从“可输运次级离子”与“TOPAS reference 的 production observable”分离。
当前 TOPAS 4.2.p3 / Geant4 11.3.2 没有可用的 z4.a6 RadioactiveDecay table；
reference 对 GenericIon(4,6) 的行为是立即 StopAndKill，无 daughter、无 local
deposit。此阶段不实现猜测性的 alpha+p+p，也不补稳定 Be-6 的 rate/package。

## 已确认 reference 语义

- Geant4 RadioactiveDecay 空 decay table 路径：SetNumberOfSecondaries(0)、
  fStopAndKill、ProposeLocalEnergyDeposit(0.0)。
- Job 495：200 MeV/u、50,000 histories、seed 2026099609；50,000/50,000
  Be-6 parent 均由 RadioactiveDecay 终止，pre/post KE=1200 MeV，step deposit
  为数值零，daughter=0。
- RadioactiveDecay6.1.2 没有 z4.a6 scheme；PhotonEvaporation6.1/z4.a6
  不是可直接用于 RDM 三体衰变的 daughter table。

结论：这是当前 validation reference 的兼容性语义，不是已验证的 6Be 物理衰变模型。
因此不生成 alpha+p+p kernel，不把 Be-6 rate/package coverage 当作缺失物理数据补齐。

## 实现策略

使用 table-driven Cinel02UnstableIonPolicy：

- StableForTransport
- RejectUnsupported（预留）
- TopasCompatKill
- PromptDecayKernel（预留）

当前 (Z,A)=(4,6) 的 policy 为 TopasCompatKill，其他 17 个显式 isotope
保持 StableForTransport。配置项：

    cinel02_topas_compatibility_mode: true

开启时，CINEL02 role-0 Be-6 product 在 queue 之前：

1. 记录 generated transition / production observable；
2. 记录 topas_compat_discarded_count 与 topas_compat_discarded_kinetic_MeV；
3. 不进入 secondary queue，不走 stopping、straggling、MCS 或 H/O rate lookup；
4. 不生成 daughter、不 local deposit。

sink 名称明确表示 reference model defect，不得解释为 dose、nuclear local deposit 或
reaction export。

## 已完成的代码与回归

- policy、compatibility mode 配置解析与 nuclear_model: cinel02 validation；
- primary/secondary CINEL02 product queue 前拦截；
- per-isotope discarded count/kinetic sink，支持多批次累加；
- TransportResult 双 closure：
  - physical_relative_energy_balance_error() 不含 reference sink；
  - relative_energy_balance_error() 为含 sink 的 accounting closure；
- energy ledger JSON 输出 compatibility policy、sink 数组、physical/accounting closure；
- policy、17 isotope 默认行为、配置 gate、sink accumulator 和双 closure synthetic tests；
- 沙盒外 SYCL build 与本机 runtime CTest：2/2 通过。

## 200 MeV/u GPU A/B（待执行）

冻结：G1、100,000 histories、seed 2026095100、3D scorer 200×200×800
（0.4×0.4×0.5 mm，80×80 mm FOV）、package/rate/stopping/MCS/straggling 全部不变。

- A：cinel02_topas_compatibility_mode: false；
- B：cinel02_topas_compatibility_mode: true。

比较 produced Be6、queued Be6、Be-6 stopping/dose、discarded KE、Be-6 hazards、
He/Z1 和 Total。B 预期：produced 不变、queued/hazard/stopping 为 0、discarded KE
等于 produced Be-6 KE、He/Z1 不因 Be-6 增加；accounting closure（含 sink）<0.1%。

## 200 MeV/u GPU A/B 证据（2026-09-01）

运行均在本机 RTX 2080Ti/sm_75、最终 SYCL 二进制、100,000 histories、seed
2026095100、G1、同一 3D scorer 200×200×800（0.4×0.4×0.5 mm，80×80 mm FOV）、
同一 package/rate/stopping/MCS/straggling 完成。A/B 只改变
cinel02_topas_compatibility_mode。

输出：
- A ledger：out/maigo_be6_A/energy_ledger.json
- B ledger：out/maigo_be6_B/energy_ledger.json
- A config SHA256：cad4cdbba5055ec8e94e1ec56448b37b938fa76ebac044ae849297ffcc8c1b84
- B config SHA256：ad29b5d421cce3819c0314d76900e12247db422a0abc9d4f3ab7e5903e4c86b0
- package SHA256：8a54b8544aea484fa3ff649fc372c22d4b48deff4c2fe37dbd31b2f7f6adb25d
- rate SHA256：aa811ff18684a6a8f81f79fa38f1130b4103555ea2932abb77f7163e1c160ed7

| 指标 | A native transport | B TOPAS compatibility |
|---|---:|---:|
| generated Be6 count | 281 | 281 |
| generated Be6 kinetic (MeV) | 130184.256791 | 130184.256791 |
| queued Be6 count | 281 | 0 |
| queued Be6 kinetic (MeV) | 130184.256791 | 0 |
| discarded Be6 count | 0 | 281 |
| discarded kinetic (MeV) | 0 | 130184.242188 |
| nuclear interactions | 71524 | 71524 |
| total deposited energy (MeV) | 228097665.594 | 227967481.539 |
| physical closure residual | 0.0153638865 | 0.0159063200 |
| accounting closure residual | 0.0153638865 | 0.0153638857 |

Be6 count closure 为 281 - 0 - 281 = 0；kinetic closure 为
130184.256791 - 0 - 130184.242188 = 0.014603 MeV（1.1e-7 relative，
来自 device float atomic 累加）。B 的 deposited-energy 差为 -130184.055 MeV，
与显式 sink 的差为 0.187 MeV；该差远小于 Be6 sink，来自并发 FP32 scorer/
ledger 累加顺序。

A/B 的 generated transition count 与 nuclear interaction count 完全相同。
B 没有 Be6 stopping step、hazard、queue 或 daughter；He/Z1 没有因 Be6
增加 daughter。其他 species 的亚千分比波动不构成 daughter 注入证据。

全局 accounting closure 仍约 1.536%，与 A native baseline 相同，来源是既有
CINEL02 model residual（quality report 以 research/non-production approximation
记录），不是 compatibility sink；compatibility sink 已被单独纳入 accounting
closure。physical closure 则显式保留该 reference sink，因此不能用它作为
守恒物理模型的通过判据。

## 验收与边界

- [x] A/B 固定 seed 运行完成并保存输出 JSON、config hash、package/rate hash；临时 config 未提交。
- [x] produced Be6 = discarded Be6 + queued Be6；count closure=0，sink KE 与 generated
  KE 相对残差约 1.1e-7。
- [x] B 的 compatibility accounting closure 相对 A 无新增 residual；sink 已显式报告。
  全局约 1.536% 的既有 CINEL02 residual 仍保留为后续问题，physical closure 不伪装成守恒沉积。
- [x] generated transition 与 nuclear interaction count 完全一致；其他 isotope 仅有
  亚千分比并发 FP32 累加波动，无 daughter 注入。
- [x] A/B 已证明 compatibility sink 能复现当前 TOPAS reference；进入 Step 03B-2 replay-support，
  Be-6 在 coverage auditor 中标记 non_transportable_prompt_decay。

## 禁止事项

- 不实现 6Be -> alpha+p+p；
- 不补 6Be+H/O secondary exposure/rate/package；
- 不给 Be-6 alias 到 Be-7/Be-9/Be-10；
- 不改 sampler、event selection、yield 或 dose scale；
- 不静默丢弃 Be-6 动能；必须写入显式 compatibility sink。

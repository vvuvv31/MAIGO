# 03B-1R：Be-6 prompt-decay semantics gate

## 目标

确定 GPU 对 `6Be` 的处理应模拟 radioactive decay，而不是通过普通 TOPAS
`GenericIon(4,6)` campaign 补一张稳定 Be-6 projectile rate/package 表。该步骤只做
语义审计和最小隔离验证；在退出条件满足前不改变生产 sampler、rate、stopping 或 dose scale。

## 已知证据

- Geant4 data file：`z4.a6`；基态寿命 `4.95333e-21 s`。
- 200 MeV/u 下 Be-6 的 boost 后 decay length 约 `1e-9 mm`，远小于水中核反应长度。
- TOPAS Job 489：普通 decay 配置、`GenericIon(4,6)`、200 MeV/u、50k，`Scored Entries: 0`。
- TOPAS Job 490：关闭 `g4decay/g4radioactivedecay` 的 no-decay 隔离、200 MeV/u、50k，
  interactions `16,001`（H1 `4,624`、O16 `11,377`），Be-6 projectile products `599`；
  同时出现 energy-conservation/resampling warnings。

## 执行项

- [x] 读取并记录 Be-6 的 Geant4 lifetime 和 TOPAS decay/no-decay smoke 结果。
- [x] 审计 TOPAS full-cascade raw 中 Be-6 direct child 的后续 decay products、decay energy 和 scorer 归属：未发现 `fDecay` interaction record 或后续 daughter；direct child 计数为 27,186，projectile interaction 为 0。
- [ ] 审计 GPU CINEL02 replay 后 Be-6 child 的 queue、stopping、dose scorer 和 cascade eligibility。
- [ ] 定义 decay conversion 的四动量/能量账本：`6Be → 4He + p + p`（若 Geant4 raw 给出不同终态，
      以 raw decay record 为准），并明确 daughter 的 generation/origin/scorer 语义。
- [ ] 实现最小 device-side conversion，或给出有证据的“不转换、只在 scorer 中排除 Be-6”的结论；
      不得静默保留 prompt-unstable Be-6 为稳定 transport species。
- [ ] 200 MeV/u、100k、G1 GPU A/B：冻结当前代码 vs conversion，比较 Be6/He/Z1/Total 及 energy ledger。

## Job 495 增强 scorer 证据（2026-09-01）

本机 TOPAS `4.2.p3` / Geant4 `11.3.2` 已重新编译 `CarbonDecayNtuple`，并在本地通过
`sbatch` 提交 Job 495（200 MeV/u，`GenericIon(4,6)`，50,000 histories，seed
`2026099609`，诊断盒为真空 `TsBox`，不参与生产 package）。输出目录为
`/mnt/sda/wuwei/cinel02-be6-decay-diagnostic/z4a6_e200_h50000_v2/`。

- scorer header：`Number of Original Histories = 50000`，`Number of Scored Entries = 50000`；
- 50,000 条记录全部是 `decay_parent`，粒子 `(Z,A)=(4,6)`，post-step process 为
  `RadioactiveDecay`，track status 为 `2`；
- 每条记录 `pre KE = post KE = 1200 MeV`，step length 为约 `10^-9 mm`，step deposit 为
  数值零（约 `10^-33 MeV`）；
- `decay_daughter` 记录数为 **0**，未观测到 `He4+p+p` 或其他 daughter；
- Geant4 `RadioactiveDecay6.1.2` 目录没有 `z4.a6` 文件，只有
  `PhotonEvaporation6.1/z4.a6`（核级寿命/能级数据），因此当前 TOPAS RDM 没有可用于
  生成 ^6Be 三体 daughter 的显式 decay scheme。

这组结果确认了“Be-6 在当前 TOPAS 配置中被立即终止且无可记录 daughter”的**参考行为**，但
不能证明 Geant4/RDM 实现了物理上预期的 `^6Be -> alpha+p+p` 三体衰变。Job 490 的 no-decay
数据仍包含 energy-conservation/resampling warnings，继续禁止用于生产 rate/package。

因此本 gate 暂不实现 GPU daughter conversion，也不把 Be-6 作为稳定 projectile 补充 rate/package。
在决定 GPU 如何处理“无 daughter 但 parent KE 未沉积”的 reference 语义前，必须先明确 TOPAS
能量归属（或确认 reference physics/data 缺失）并设计显式的 non-transportable/prompt-decay policy；
不得静默丢弃 Be-6 动能。

## 禁止事项

- 不使用 Job 490 no-decay raw 直接编译生产 package。
- 不使用 Be-7/9/10 的 rate 或 stopping table alias 代替 Be-6。
- 不用 Be-6 global dose/yield scale 拟合 TOPAS。
- 不因该步骤而扩大 tolerance 或使用 nearest-event fallback。

## 退出条件

- [ ] TOPAS full-cascade 中 Be-6 的 decay daughter 数、能量和 dose 归属有明确证据。
- [ ] GPU 对 Be-6 的 fate（decay conversion 或明确排除策略）在代码和 JSON/schema 中可追踪。
- [ ] conversion 前后 signed energy ledger closure `<0.1%`，无未记录 daughter energy。
- [ ] Be-6 不再作为稳定 secondary projectile 参与不具物理意义的 rate lookup；其他 isotope 的结果无
      可测回归（100–300 MeV/u 后续冻结回归）。
- [ ] 本步骤完成后再继续 Step 03B-2 support-aware rate consistency。

## 当前问题

GPU 当前 `get_charged_species_idx(4,6)==17`，使用显式 Be-6 stopping table，并将 `product.role==0`
的 Be-6 直接写入 `SecondaryParticle` queue；secondary transport 没有 radioactive-decay hook。
因此 Be-6 dose residual 可能是 decay semantics 缺失，而非单纯 rate/package coverage。需要先审计
TOPAS raw decay 语义再决定 conversion 入口，避免凭借猜测硬编码终态。

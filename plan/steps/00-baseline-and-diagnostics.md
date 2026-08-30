# 步骤 00：冻结基线与建立诊断框架

## 目标

以 `38b5150` 对应行为为可复现实验基线，增加可切换的 ablation 配置和统一诊断，使后续每个偏差都能归因到 attenuation、projectile fragments、target fragments、remnant、neutron 或 MCS。

## 前置条件

- 确认当前工作分支和工作树；不得覆盖用户未提交修改。
- 记录编译器、SYCL 实现、GPU 型号、sm_75 构建参数和默认配置。
- 如不能直接从 `38b5150` 建分支，则记录当前 commit 与该提交的差异，不做破坏性回退。

## 实施任务

1. 增加或确认以下独立开关：

```yaml
inelastic:
  enable_attenuation: true
  enable_projectile_fragments: true
  enable_target_fragments: true
  enable_neutron_kerma: false
  enable_energy_dependent_yields: false
  strict_energy_conservation: true
  strict_az_conservation: true
  secondary_mcs_mode: unscaled
```

2. 为每次 run 输出并持久化：primary C12 fluence/reaction count 随深度、H/O reaction count、projectile/target/neutron KE、local remnant deposit、kerma deposit、signed energy residual、A/Z residual、charged multiplicity、species yield、secondary termination reason。
3. 将 dose 分量拆为 primary C12、projectile fragments、target fragments、species Z/A、remnant、neutron-associated 和 escaped energy。
4. 建立固定命名规则：`energy_model_histories_commit_seed`，保存配置副本与日志。
5. 增加小规模 deterministic/smoke test，确认关闭某分量后其计数和剂量严格为零。

## 基线运行

- 本地 GPU：100/200/300/400 MeV/u，各 100k histories。
- 保存现有 full-model 结果，但它只作“修复前”参照，不能作为正确性标准。
- TOPAS 比较必须使用 3D dose scorer，并沿两个横向轴求和得到 IDD。

## 验收

- 所有开关可独立生效，日志能区分各剂量来源。
- 诊断量具备单位、定义和总和关系说明。
- smoke test 无 queue overflow、step-limit、NaN。
- README 进度表中记录基线输出路径和 commit。

## 建议提交

```text
chore(fred): add staged ablation controls and diagnostics
```


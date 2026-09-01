# 01B.1：Replay semantics cleanup

## 目标

在不改变 CINEL02 rate、event sampler、stopping 或 transport physics 的前提下，
把 compact replay ledger 中的状态和能量语义变成可用于 deterministic auditor 的可信输入。

## 实施内容

1. 将 replay status 从四类扩为五类：
   `collision_candidate`、`replay_valid`、`replay_lookup_miss`、
   `replay_invalid_event`、`post_em_below_cutoff`。
   对每个 collision candidate 保持一条 candidate 记录，随后恰好记录一个 terminal replay
   status，满足 `candidate = valid + lookup_miss + invalid + below_cutoff`。
2. 同一 status cell 累加三种能量：
   - `rate_query_energy_MeV`：step 开始、hazard/target lookup 使用的 pre-EM 能量；
   - `replay_query_energy_MeV`：碰撞点、event window lookup 使用的 post-EM 能量；
   - `continuous_loss_to_collision_MeV`：两者之间的连续损失。
   记录并检查 `E_rate - E_replay ≈ dE_to_collision`。
3. primary 与 secondary replay 使用相同参数顺序和状态分类；generation 字段在 JSON 中明确命名为
   `reaction_generation`，避免与 G0/G1 产生 off-by-one 误读。
4. host accumulator 和 JSON 同步扩展至五状态及三类 energy array。
5. synthetic regression 覆盖 status slot count、multi-batch energy accumulation 和能量恒等式。

## 验收

- [x] 沙盒外 CMake 编译成功。
- [x] 沙盒外 CTest 全部通过。
- [x] 200 MeV/u、G1、100k、seed `2026095100` GPU 重跑完成。
- [x] status partition 满足 candidate = valid + lookup_miss + invalid + cutoff。
- [x] 新 JSON 包含五状态 layout 与三类 collision energy array。
- [x] 10 MeV/u replay occupancy refinement 已编译、测试并完成同配置 GPU 重跑，为 Step 02 提供可分辨的 support 诊断。
- [x] 记录 lookup miss 与 post-EM cutoff 的分项结果；不以 p/d/He4 residual 阻塞后续 auditor。

## 运行记录

- Commit（代码基线）：`c4ceee4`（replay semantics cleanup）；后续 10-MeV diagnostic refinement 工作树中。
- Config：`config/beam_200MeVu_cinel02_e200light107_g1_transitiondiag4_100k_xy04.yaml`。
- Seed/histories：`2026095100` / `100000`；设备：RTX 2080Ti/sm_75，CUDA SYCL。
- Package SHA256：`8a54b8544aea484fa3ff649fc372c22d4b48deff4c2fe37dbd31b2f7f6adb25d`。
- Rate SHA256：`aa811ff18684a6a8f81f79fa38f1130b4103555ea2932abb77f7163e1c160ed7`。
- Config SHA256：`c29cbe1a4957035294e4fdd1abc3798740cdf723b8aa0b2dc3cc7281e4cf9299`。
- Output：`out/beam_200MeVu_cinel02_e200light107_g1_transitiondiag4_100k_xy04/energy_ledger.json`。
- Build：沙盒外 `cmake --build build -j2` 成功。
- Tests：沙盒外 CTest `2/2` 通过。
- Status：`71524/71318/192/0/14`（candidate/valid/lookup_miss/invalid/cutoff），逐 cell 守恒；当前 replay status 使用 40 个 10-MeV/u bins（旧 50-MeV/u coarse layout 已由本次 auditor 前的诊断细化取代）。
- Energy invariant：全局相对残差 `1.98e-7`，最大 cell 相对残差 `1.20e-6`。
- 物理 residual：本步骤不改变；p/d/He4 仍留给后续 reaction-survival / stopping optical-depth 分析。

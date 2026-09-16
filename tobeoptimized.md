# 当前 GPU 性能工作进度与待优化项

日期：2026-09-16。本文汇总本工作树当前的输运性能改动、已验证结果、三卡吞吐/occupancy
实测，以及尚未完成、仍值得继续的方向。所有加速数据来自本地 RTX 2080 Ti（sm_75）的配对
A/B，除非另注明确为其他卡。

相关提交：

- `2c93c84` Skip and shard per-step unified EM audit atomics
- `a7e6075` Preflight concurrent plan manifests and make audit shard count tunable

未提交的工作树里仍有与本目标无关的实验改动（统一 EM dense/shared-search 研究选项、
`maximum_step_mm`/`maximum_relative_energy_loss` 放宽、`secondary_tail_threshold` 等），
本文不把它们计入已验证结论，也不在本次提交中包含。

---

## 1. 已完成的性能修复：每步诊断原子

### 1.1 问题

统一 EM 的原发与次级每步都通过两个计数 helper 向同一组 8 个全局 `uint64_t`
计数器做 `fetch_add`。其中：

- `draw.proposed` / `draw.accepted` 在当前实现中恒为 0，属于无效的原子写；
- 其余计数在整卡范围内对同一地址竞争。

### 1.2 改动（`src/transport_sycl.cpp`）

1. **跳过零增量**：两个计数 helper 开头加 `if (count == 0) return;`，去掉恒零的原子。
2. **64 路分片**：8 个计数器按 `counter*64 + shard` 布局摊到 64 个逻辑分片，
   shard 取 `global_linear_id % 64`（次级取 `item_id[0] % 64`）；输运结束把所有分片
   求和还原为原来的 8 个计数，审计数值不变。
3. **续跑状态裁剪**：`CARBON_EM_LOCAL_AUDIT=0` 时从 `SecondaryResumeState`
   移除 `unified_secondary_audit`，相应保存/恢复/提交都统一用编译条件控制。

### 1.3 实测（RT07575 shard，3,240,963 原发，2080 Ti，5 次配对）

| 实验 | 内容 | elapsed | primary | secondary |
|---|---|---:|---:|---:|
| A | 跳过零增量 | **−13.67%** | −7.47% | **−20.43%** |
| B | 64 路分片 | −1.03% | −0.20% | −2.87% |
| C | 续跑状态裁剪（wrapper 205→186 寄存器） | −1.04% | −0.16% | −2.16% |
| **A+B+C** | 合并 vs 基线 | **−15.30%** | **−6.82%** | **−24.62%** |

墙钟合并 −14.44%；对应吞吐提升约 +17%。

### 1.4 被否决的实验

- **D 显式次级工作组**：强制 32/128/256 线程分别使次级 −14.4%/−11.4%/−8.5%，
  后端默认更快，已完全回退。
- **He-4 续跑状态裁剪**：把 `he4_audit` 改为按分段 flush、不进续跑状态后，
  launch wrapper 寄存器 186→**197**，反而变差，已回退。
- **压紧并行化**：`compaction_s` 中位 0.026 s，仅占 elapsed **0.10%**，不值得重构。

### 1.5 分片数扫描

`CARBON_EM_AUDIT_SHARDS`（默认 64）做成构建选项后测试：

| 对比 | elapsed |
|---|---:|
| 256 vs 64 | −0.11% |
| 1024 vs 64 | +0.04% |

**64 已达平台**，更多分片只增缓存占用，默认保持 64。

### 1.6 正确性验证

- `[unified-em-audit]` 在所有分片数、`LOCAL_AUDIT` 开/关、次级分段开/关下逐项相同：
  `0 1036320939 1429988519 0 0 6897079514135703 528848467657861 0`。
- quality 全 pass、零 overflow、能量平衡误差一致（3.6833e-6）。
- 剂量差 ≤5.7e-5% of peak，处于基线自身原子累加波动（~4.3e-5%）量级。
- 故意向**非零分片**注入失败计数 → `audit[0]=30632`、`runtime-rejected-kernel`、退出码 1，
  确认归并包含所有分片且仍拒绝输出。

---

## 2. 已完成的并发入口修复

并发工具把完整 manifest 拆成多个进程后，每个进程只能看到自己那组，跨进程的输出覆盖
无法被 C++ 的 per-process 校验发现。

改动：

- 新增 `--plan-preflight`：对**完整** manifest 逐片套用 CLI 覆盖并校验设备一致性与
  输出冲突，然后直接退出，不创建 context、不跑输运。
- `tools/plan_concurrent.py` 在拆组前先对未拆分的 manifest 预检；结束后按配置集合校验
  完成记录（每个预期配置恰好一条 `quality=pass` 的 `[plan-shard-done]`，无缺失/重复），
  histories 只统计实际完成记录。

验证：重复配置 manifest 在启动任何 worker 前被拒；假 worker（预检返回 0、其余什么都不做）
被判为 `incomplete plan`；正常两片串行运行 `histories=6481926`、`quality=pass`、退出码 0。

---

## 3. 三卡吞吐对比（同一工作树源码）

同一 RT07575 shard、同一物理配置（仅路径不同）、各 5 次取中位数。本地 sm_75 二进制与
发往 Titan 的 sm_75 二进制 SHA256 完全相同；A6000 为同一源码的 sm_86 构建。

| GPU | 架构 | wall (s) | transport_loop (s) | primary (s) | secondary (s) | 端到端吞吐 (h/s) | 纯 kernel 吞吐 (h/s) |
|---|---|---:|---:|---:|---:|---:|---:|
| RTX 2080 Ti（本地） | sm_75 | 24.60 | 19.10 | 8.38 | 10.64 | **131,750** | 169,684 |
| RTX A6000 | sm_86 | 26.88 | 12.21 | 4.88 | 7.38 | **120,550** | **265,436** |
| TITAN RTX | sm_75 | 43.00 | 21.65 | 9.26 | 12.11 | **75,375** | 149,699 |

所有运行 `histories=3240963`、`audit[0]=0`、审计逐项一致。

结论：

- **A6000 的 kernel 明显更快**（primary 1.72×、secondary 1.44×、纯 kernel 吞吐 ~1.56×
  2080Ti），但端到端反而比 2080Ti 慢 9%；
- **Titan kernel 本身比 2080Ti 慢 ~12%**（0.88–0.90×），加上最老的主机，端到端只有 57%。

### 3.1 主机端开销是 A6000/Titan 的瓶颈

`wall − transport_loop`：2080Ti 4.9 s、A6000 14.3 s、Titan 21.6 s。单进程主要项：

| 阶段 | 2080Ti | A6000 | Titan |
|---|---:|---:|---:|
| unified_em_package_load_verify | 1.32 | 4.35 | 6.60 |
| transport_setup_including_upload | 3.90 | 12.12 | 17.93 |
| quality_and_output | 0.46 | 1.55 | 2.12 |

统一 EM 包 1.387 GB（三机字节一致），load+校验+建索引偏 CPU/磁盘：本地 EPYC 9965
（~1.05 GB/s）、A6000 Xeon Gold 6148（~0.32 GB/s）、Titan Xeon E5-2680 v3（~0.21 GB/s）。

**A6000 每轮 kernel 省 ~6.8 s，但主机端多花 ~9.4 s，净亏。** 要兑现 A6000 的算力，
优先级在主机端 setup/包加载与并发摊销，而不是继续压 kernel。

---

## 4. 当前 eligible warp / warps per SM

ncu 2024.3.2，`--clock-control none --cache-control none`，取代表性单次启动。
Titan 的 ncu 在 WSL 上返回 `==ERROR== Unknown Error on device 0`，无法采集；其 device
image 与本地 sm_75 相同（寄存器/block 一致），每 SM 行为可参考 2080Ti。

| GPU | kernel | regs | block | warps/SM | occupancy | eligible warp/scheduler | SM throughput |
|---|---|---:|---:|---:|---:|---:|---:|
| RTX 2080 Ti (32 warp/SM) | primary | 172 | 128 | 7.85 | 24.5% | 0.09 | 8.73% |
| | secondary | 186 | 32 | 7.90 | 24.7% | 0.08 | 7.94% |
| RTX A6000 (48 warp/SM) | primary | 182 | 128 | 7.50 | 15.6% | 0.14 | 13.28% |
| | secondary | 180 | 32 | 7.93 | 16.5% | 0.09 | 8.79% |

`launch__occupancy_limit_registers`：primary = 2 block，secondary = 8 block，两者都受
寄存器限制（primary 2×128=8 warp，secondary 8×32=8 warp）。

与优化前（`docs/fix_review_20260916.md` 的历史构建）对比：

| kernel | 寄存器 | warps/SM | eligible/scheduler |
|---|---:|---:|---:|
| primary 旧 → 新 | 255 → **172** | 7.88 → 7.85 | 0.086 → 0.09 |
| secondary 旧 → 新 | 203 → **186** | 7.95 → 7.90 | 0.077 → 0.08 |

**eligible warp 仍然极低（0.08–0.14），warps/SM 仍钉在 ~7.5–7.9，基本没变。**
上一轮 ~15% 的加速来自减少串行化的全局原子指令，而不是提升延迟隐藏。当前瓶颈依旧是
「驻留 warp 太少 + 长 scoreboard 等待」。

---

## 5. 待优化项（按建议顺序）

### 5.1 抬高驻留 warp（最高优先级，但门槛明确）

**门槛修正为每线程 ≤168 寄存器**（原稿的 ≤170 是忽略了分配粒度与子分区的粗算）。
Turing/Ampere 每个 SM 四个子分区，各 16 384 个寄存器；寄存器按 warp 粒度 256 个分配，
等价于每线程按 8 个向上取整：`R_alloc = 8*ceil(R/8)`，于是

```
W_SM = 4 * floor(16384 / (32 * R_alloc))
```

关键断层在 168：

```
R=170 → R_alloc=176 → 176*32*3 = 16 896 > 16 384 → 每子分区仍只能 2 warp
R=168 → R_alloc=168 → 168*32*3 = 16 128 ≤ 16 384 → 每子分区可放 3 warp
```

代入当前值（不足 168 的差额）：

| GPU / kernel | 当前寄存器 | 分配后 | 寄存器限制的 warp/SM | 到 ≤168 还需减少 |
|---|---:|---:|---:|---:|
| 2080Ti 原发 | 172 | 176 | 8 | 4 |
| 2080Ti 次级 | 186 | 192 | 8 | 18 |
| A6000 原发 | 182 | 184 | 8 | 14 |
| A6000 次级 | 180 | 184 | 8 | 12 |

达到 ≤168 后，理论驻留上限为：原发 3×128 线程 = **12 warp/SM**，次级 12×32 线程 =
**12 warp/SM**。这是上限，实际均值仍受启动/结束/负载不均影响。

- 单纯设置寄存器上限会把值挤到 local memory，反而增加当前最怕的访存等待，必须先做
  「消除不必要状态/缩短存活区间」的实验，再看门槛。He-4 裁剪已试并否决（198 寄存器）。
- 下一步应从实际 SASS/资源报告出发，定位仍占用寄存器但可缩短存活区间的量，
  而不是机械降低上限。
- 即使到 12 warp，也不会自动解决 eligible 极低（见下），必须同时减少关键加载依赖。

### 5.2 主机端 setup / 包加载（A6000/Titan 的净值瓶颈）

- 量化 `unified_em_package_load_verify`（1.387 GB 的 SHA + 解析 + 精确索引主机构建）
  与 `transport_setup_including_upload` 各自的占比，区分磁盘、单线程 CPU、H2D。
- 已有工作树实现但**未验收**的只读设备常驻缓存可直接针对此项；需按仓库约定先做
  内容签名、预算与失败清理，再评估端到端收益。
- 并发入口（`tools/plan_concurrent.py`，现已带全局预检与完成校验）可摊掉主机段，
  A6000 显存允许约 6 组常驻；这是当前最确定的 A6000 收益来源。

### 5.3 已否决、不要重启

- EM 搜索键分离、更大的精确索引（含指数+2 位尾数）、RNG 替换、强制寄存器上限、
  每线程双轨迹/大规模拆核、次级压紧前缀和并行化。
- 每步诊断原子已处理；剩余三个非零计数（步数、连续能损、delta 能损）如需继续减少，
  只在更大分片仍受限时才试「每分段累计三个热计数」，且必须覆盖所有退出路径、
  保持每步先量化再累加。

### 5.4 采样与验收口径

- 所有候选以端到端无 profiler 配对耗时为准，不能只看查表速度、寄存器数或 L2 命中率。
- 记录绑定同一提交、编译选项、实际 kernel wrapper 与启动尺寸；两张卡分别判定。
- 保留 Schneider v2.1 最低核数据、统一 EM 与 delta moments、原生受限涨落 scale=1、
  精确 CT 边界；GPU 自对照不替代患者 BODY Gamma，低密度阈值与患者精度验收仍未完成。

---

## 6. fix.md 第二轮实验记录（2026-09-16 续）

第一目标是让 2080Ti 原发 172→≤168。本轮先做代价最小的两个源码实验，再做加载归因。

### 6.1 实验 A：delta 确定性参数提前合成 —— 否决

把 `unified_em_explicit_loss()` 中 `delta0/delta_slope/delta_mean/delta_variance` 的计算
挪到 restricted fluctuation 抽样之前（有效性检查与 RNG 调用顺序不变）。

结果（sm_75 反汇编，`transport_sycl_impl<1>` 的 `nd_item` lambda）：

| 内核 | 改前 | 改后 |
|---|---:|---:|
| 原发 `transport_sycl_impl<1>` | 172 | 172 |
| 次级 wrapper | 186 | 186 |

寄存器无变化（编译器已自行重排），按「SASS 未变即结束」否决并回退。

### 6.2 实验 B：masked rate total-only 接口 —— 否决

新增 `schneider_masked_total_rate_device` / `secondary_masked_total_rate_device` 及
`primary_total`/`secondary_total`，只在纯总量查询点（原发 hazard、原发 miss 日志、次级
hazard、He-4 诊断）使用，保留逐通道能区屏蔽、插值与 `1e-12` 阈值。

结果：原发仍 172、次级 wrapper 仍 186 —— 纯总量点上的 13 个 partials 已被编译器 DCE。
按同口径否决并回退。

### 6.3 实验 C：纯 rate 查询与 EM prepare 错开 —— 未实施

`mass_rate`/`el_rates` 的输入（`section_id`、`cur_e_u`、密度）在 `prepare()` 之前就可用，
理论上可前移；但它嵌在 `if(enable_inelastic…)` 内，且相邻的 hadronic cache 更新、
光学深度消耗、弹性分支、RNG 都有副作用，必须原序保留。鉴于 A/B 已显示编译器会自行
调度这些标量的存活区间，手动大范围重排的风险/收益比不佳，本轮不实施，留待 A/B 类
小改动确实产生寄存器收益后再评估。

### 6.4 实验 D：加载归因 —— 本地内存是 long-scoreboard 的重要来源

2080Ti，ncu 2024.3.2，代表性单次启动：

| 指标 | 原发 | 次级 |
|---|---:|---:|
| global load sectors | 1 400 166 271 | 2 146 120 782 |
| **local load sectors** | **899 040 026** | **1 082 838 732** |
| local store sectors | 147 794 098 | 430 968 867 |
| long scoreboard（占 warp-active） | 43.31% | 57.06% |
| short scoreboard | 0.84% | 0.84% |
| warps/SM | 7.87 | 7.90 |
| eligible warp/scheduler | 0.09 | 0.08 |

local load 占（global+local）load sector 的比例：原发约 **39%**、次级约 **33%**。
也就是说 long-scoreboard 有相当一部分来自线程私有 local memory，而不是 EM 表。

SASS 静态统计（sm_75，`cuobjdump -sass`）：

| 内核 | LDL | STL | stack frame |
|---|---:|---:|---:|
| 原发 `transport_sycl_impl<1>` | 88 | 1326 | 2816 B |
| 原发 `transport_sycl_impl<0>` | 1985 | 4217 | 6544 B |
| 次级 `CarbonSecondaryTransportKernel<1>` | 505 | 859 | 2080 B |

原发 STL 遍布整个函数体（代码地址 0xd0–0x625e0），local 偏移覆盖 0x8–0xafc，
即存在贯穿始终、约 2.8 KB 的线程私有帧；次级约 2.0 KB。

**结论：** 当前 long-scoreboard 主要由「EM 表 + 核反应率 + 约 2–2.8 KB 线程私有 local
帧」共同构成，单纯继续改查表顺序不足以提高 eligible。下一轮的定位目标是这份本地帧的
来源（spill 还是编译器放置的聚合/动态索引对象），并用 ncu 的 local/global sector 与
实际耗时验证；只有确认收益后才考虑 §5.1 的驻留门槛。

---

## 7. 复现位置

- 本轮实验产物：`scratch/eligible_warp_20260916/`（`ab_final/`、`ab_shards*/`、
  `combo/`、`failtest/`、`ncu_*`）。
- 三卡部署与逐次日志：本地 `/tmp/opencode/gpu_compare/`，
  远端 `~/MAIGO/scratch/gpu_compare_20260916/`（A6000 用 `a6000.yaml`+sm_86，
  Titan 用 `titan.yaml`+sm_75，均通过自带 `runtime/` 加载器运行）。
- Titan 为 WSL，运行需让 `/usr/lib/wsl/lib` 先于系统 `/usr/lib/x86_64-linux-gnu`
  （否则会加载过期的 535 `libcuda.so.1` 并报无 GPU）。

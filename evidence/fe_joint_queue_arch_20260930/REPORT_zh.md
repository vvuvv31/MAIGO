# FE 架构优化：主输运与电子 joint 回放分离

## 已接入的结果

第一轮架构优化已接入 `minibeam` 工作树和 `build/fe-joint/carbon_mc`，未提交 git。

最终二进制内，两对相反运行顺序、同种子完整 RT07575 carbon CT 计划比较：

| Seed | inline wall，s | queued wall，s | inline k/s | queued k/s | 提升 | 两端 local 1%/1mm |
|---|---:|---:|---:|---:|---:|---:|
| 929203412，queued→inline | 48.013 | 37.899 | 135.004 | 171.032 | 26.69% | 99.8847% |
| 929203512，inline→queued | 49.216 | 39.640 | 131.702 | 163.520 | 24.16% | 99.8449% |
| 总历史数／总 wall | — | — | **133.333** | **167.191** | **25.39%** | 无下降 |

每个 arm 6,481,909 histories。端到端计时包含新进程输入、校验、加载、GPU 输运和写盘；没有用 kernel-only 吞吐或最快单次代替结果。200k/s 目标仍未达到。

两对整数输运／EM／电子审计全部一致，剂量相对 L1 为 1.1076e-7 / 1.1074e-7，属于 FP32 计分累加顺序差异。现有能量门限通过，relative residual 最大约 3.522e-6，queue overflow、domain miss、invalid march、drain mismatch 均为 0。

原有 `unvalidated_electron_joint_response` failure、accepted=false 和 exit=1 保留。用户 local11>98% 门槛通过，不表示该电子响应包已取得生产质量准入。

## 架构变化

```mermaid
flowchart LR
    A[主粒子批次] --> B[EM / FE / 核反应主输运]
    B --> C[每步响应域检查与原 roulette]
    C --> D[48 字节已抽中响应任务]
    D --> E[独立 CDF 选择与 ordered-path 回放核]
    E --> F[FP32 端点计分与能量账本]
    F --> G[排空检查后进入下一批]
```

每批 primary 结束后消费该批任务，完成后再进入下一批；带电次级仍在原有阶段输运。这里没有同时运行两个 GPU 输运任务，也没有修改核反应或次级调度。

主核只保留每个原始响应 query 的域检查、fraction、原 roulette 和任务生成。独立核执行原 CDF 二分选择、路径旋转、同体素 bounds 快路径及 FP32 mass polyline march。可变长度回放循环不再和主核的几千行物理循环一起编译。

48-byte 任务包含：

- FP32 birth position、direction；这些与旧回放实际消费的值相同。
- 原 CDF 随机数、原 FP32 path azimuth、原 packet energy。
- channel、source voxel 和 source depth bin。

任务不携带父粒子的输运状态、可变 RNG 状态或 continuation packet cursor。CDF、birth、azimuth 和 roulette 使用原 RNG identity／dimension 的值，没有为队列引入新的抽样流。每步的本地能量仍按原 `deposited - packet` 舍入；回放核添加原 packet。逃出 scorer 的能量转入 outside-grid 账本；invalid march 恢复源体素计分并保留质量失败。队列溢出或排空计数不一致时拒绝整批结果。

编译模板中 response execution 为 0=OFF、1=inline、2=queued，运行前按 YAML 选择对应实例。主循环没有动态 inline/queued 分支。主核与次级核的名字分别区分模式，避免不同闭包使用同一个 SYCL kernel name。响应关闭的实例仍在编译时删除回放。

### 与以前失败路线的区别

此前 open-CT material-packet group1 队列携带完整出生／continuation 状态并执行显式多次 packet advance；它不是本次的轻量 joint endpoint/path 任务。此前仅延后 CDF 和仅将路径转 FP32 的候选也没有把完整回放循环移出主核。

本次重试依据是 `../em_dense_cause_20260930/REPORT_zh.md` 的新硬件证据：主核参与线程少、电子路径读取特征及取指等待明显。没有原样重试失败队列、noinline、split keys 或 cooperative EM loads。

## 物理与参考保持一致

- `config/carbon_rt07575_fe_joint_fast_segment1.yaml` 的既有研究物理：Gaussian FE、内部 cap1mm、mean-loss cap5%、primary/secondary straggling OFF，核反应和带电次级 ON。
- 原 r3 电子 response、ordered path 和 SHA pins；weighted packet probability floor0.05。
- FP32 dose ON、FP64 dose OFF。双精度 grid in/out 累加器是原有能量账本，不是 dose scorer。
- Water_75eV，材料、stopping scale、包数据及系数不变。
- TOPAS 参考：`/mnt/sda/wuwei/topas10x_threecase_20260905_r2/RT07575/topas_sum.raw`，129,638,170 histories。
- GPU ×20、原三轴变换、10% reference max mask，共459,684 voxels；3D local1%/1mm 使用 trilinear 搜索，步距 DTA/10=0.1mm。

只有计分累加顺序和代码执行布局改变，本轮没有增加物理近似。既有 fast-profile 近似和包的研究限制仍存在。

## 阶段计时与静态资源

最终同 binary 两对的事件时间：

| Seed | inline primary，s | queued primary，s | queued replay，s | inline secondary，s | queued secondary，s |
|---|---:|---:|---:|---:|---:|
| 929203412 | 23.246 | 11.171 | 1.951 | 22.213 | 22.075 |
| 929203512 | 23.584 | 11.613 | 2.068 | 23.057 | 23.374 |

primary + response 平均约 23.415→13.402s，减少约43%；secondary 没有稳定变化。此时最大的测量阶段是带电次级，约22–23s。独立响应约2s，其剩余成本不足以独自实现200k/s。

CUDA driver 读取实际编译 fatbin 的资源：

| Kernel | registers | stack/local bytes per thread | block128 理论 occupancy 上限 |
|---|---:|---:|---:|
| inline primary | 246 | 4624 | 25% |
| queued primary | 226 | 4384 | 25% |
| standalone replay | 82 | 32 | 62.5% |
| secondary（两模式） | 220 | 2336 | 25% |
| secondary wrapper（两模式） | 211 | 2144 | 25% |

主核没有跨越占用率档位，不能将收益写成“主核 occupancy 提高”。独立核62.5%是资源允许的上限；默认17,408 lanes/block128只有136 blocks，68 SM平均两个block，实际grid也限制驻留。本轮没有用新 NCU active-warps 测量宣称它实际达到62.5%。stack分配不是实测spill traffic。

默认队列预算256MiB，capacity5,592,405，按批复用。完整计划每次约2.307亿任务，峰值约140万任务（约64MiB），无溢出。队列在 primary 阶段结束后释放；此配置整体 tracked peak 仍约8.36GiB，后续 secondary peak 大于带队列的 primary阶段。tracked memory 不包含驱动/runtime 内存。

## 筛选、确认及集成验证

1. 50k 同分布 spot pilot：整数审计一致，doseL1=4.902e-8；任务全部消费，无 overflow。小粒子数仅检查重构，没有 TOPAS gamma 验收结论。
2. R1 完整计划 old→queue→old：前后基线135.330k/s，queue169.853k/s，+25.51%；doseL1=1.106e-7，local11同为99.7901%。
3. R1 两个新种子反向配对：132.630→167.707k/s，+26.45%；整数审计一致、local11同为99.8747% /99.8294%。
4. YAML接口集成后的同 binary full pair（seed929203312）：136.753→169.766k/s，+24.14%，local11同为99.8690%，整数审计一致。
5. 最终工作树 binary 两个新种子反向配对：见首表；133.333→167.191k/s，+25.39%。
6. 最终 inline 对旧冻结 binary 的50k同种子检查：整数审计一致，doseL1=2.236e-8。
7. overflow（人为预算1MiB）、非法模式、关闭包后仍请求queued均按预期报错，没有dose输出。

GPU串行直接运行；隔离Slurm构建21081、最终接口构建21082（4CPU/24GiB，实测MaxRSS约2.6GiB）。不同运行有热/频率和负载波动，因此使用反向顺序并报告汇总，不能把171k/s当作固定吞吐或精确提速承诺。没有新 TOPAS 作业：物理模型与参考未变，复用已冻结的高统计参考计算gamma。

初次编译21080引用了不存在的配置成员，已修正；初次guard脚本误以为异常退出码为2，实际main的EXIT_FAILURE为1。原日志保留，已记录failed.md，溢出原运行仅重新分析未重跑；没有修改异常或质量退出规则。

## 使用

新配置：`config/carbon_rt07575_fe_joint_fast_queued.yaml`，在既有1mm研究配置上添加：

```yaml
ct_electron_joint_execution_mode: queued
```

默认 `inline`。最终 `build/fe-joint/carbon_mc` 同时包含两模式，改此键即可 A/B。queue mode仅支持active research broadbeam CT voxel dose、ordered path包、无LET/component scoring。现有parser和包的C12限制保留；没有将该响应包改成proton包。

重建需要 `CARBON_FE_JOINT_QUEUE=ON`，并遵守FP32 ON/FP64 OFF。CMake队列选项默认OFF；未编译支持时请求queued会明确报错，不会静默忽略配置。

研究调度覆盖可用环境变量 `CARBON_FE_JOINT_QUEUE_BUDGET_MIB`（默认256）和 `CARBON_FE_JOINT_REPLAY_LANES`（默认17408）。本轮没有把其他lane设置当作优化结果。

最终二进制SHA256：`109768dd6a61854a4cfe92257c9eba6d38f1d8b31f1315b2bc1431f4c52fcc98`。

旧基线SHA256：`e0b36616559d7a3224d95933d29b2343b4fe2271e2ec7bd976a31da616e8d5b0`；备份`/mnt/sda/wuwei/fe_joint_queue_arch_20260930/baseline_carbon_mc`。所有本轮源文件在复制前核对原始hash，保留工作树原有改动；只接入列出的8个源文件和新YAML、文档。

## 范围与下一步

本轮性能／gamma验收限定于RT07575 carbon Gaussian FE CT研究配置。core+tail queued变体已编译，但没有独立计时／剂量验证。proton、其他CT、minibeam没有从这组数字取得性能或剂量认证。新的queue没有代替详细material electron/minibeam路径。

下一阶段优先分析带电次级EM的实际查询与均值路径状态，仍需要避免大表/native混合回退带来的warp重复执行；不能直接重做failed.md中的split keys、cooperative loads或普通noinline候选。

## 文件

- `candidate.patch` / `candidate_r2.patch`：初始隔离架构和最终YAML集成增量。
- `before_manifest.json` / `r1_manifest.json` / `final_manifest.json`：源码、构建和binary校验。
- `pilot_pair.json` / `screen_summary.json` / `confirmation.json` / `integration.json` / `same_binary_confirmation.json`：完整原始结果的汇总。
- `guards.json` / `integrated_inline_equivalence.json`：拒绝路径及inline回归。
- `resources_r1` / `resources/kernels.json`：实际fatbin资源和occupancy API。
- `run.py` / `build.sbatch` / `resources.py`：GPU基准、Slurm构建和资源提取脚本。
- `/mnt/sda/wuwei/fe_joint_queue_arch_20260930/{source,source_r1,runs}`：完整源码、冻结R1和各次配置／日志／剂量／质量报告。

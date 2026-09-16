# 次级输运 kernel closure ABI 实验（2026-09-16）

## 结论

`CarbonSecondaryTransportKernel<1>` 的大 `[=]` closure 确实是次级线程私有 local-memory
热流量的主要来源。把不随续跑轮次变化的参数上传为一个只读 device context，并让实际
SYCL kernel 只显式捕获 context 指针和续跑热指针后，RT07575 固定分片的 Elapsed 中位数
改善 **8.64%**，次级 kernel 时间改善 **19.97%**。早段 local sectors 降 **39.90%**，
long-scoreboard 降 **44.30%**，eligible warp/scheduler 升 **19.49%**。

候选由 `CARBON_SECONDARY_CONTEXT_POINTER` 控制，当前默认 **OFF**。它没有跨过168寄存器
门槛：实际 wrapper 从186升到187 registers/thread，次级各匹配阶段的active warp没有提高，
晚段还因后端自动选择128线程block而明显下降。因此它通过了正确性、吞吐和依赖等待的
结构门槛，但没有通过最终组合要求的驻留门槛，不能切为默认。

机器可读结果见
`benchmark/secondary_context_abi_20260916/analysis.json`。未提交的原始profile、运行日志、
SASS和剂量保存在 `scratch/secondary_context_abi_20260916/`。

提交时把候选独立移植到干净的 `d4e579a` 基线，避免带入工作树中其他默认关闭的EM查表
研究改动。提交版 `src/transport_sycl.cpp` SHA256为
`be3d714da4b9579422199255559518d1b8e2dafb02e675c93784a95c975aac35`；ON/OFF均重新完成sm_75
构建和完整固定分片，审计、步数、核相互作用、零overflow及quality与正式campaign一致。
提交版单次热机确认中，ON/OFF Elapsed为20.969/23.048 s、次级为9.032/11.040 s；正式收益
仍以上述五次交错配对为准。

## 实现

候选在每个次级generation开始时构造并上传一次 `SecondaryTransportContext`。该POD只包含
本generation不变的物理表视图、CT几何、计分指针和常量。下列每轮可能变化或直接参与
续跑关联的值继续作为显式capture：

- `resume_states`、`resume_ready`、`active_order`、`keep`；
- `segment_secondaries`、`finish_tail`；
- `secondary_transport_ctx` 指针。

kernel body通过 `CARBON_SECONDARY_CONTEXT_FIELD(name)` 在候选和原始回退路径之间共用同一份
物理代码。候选closure具有两个编译期约束：必须trivially-copyable，大小不得超过64 B。
默认OFF时仍使用原 `[=]` 路径，并完整复现186 registers、1840 B stack和1984 B constant0
的基线wrapper。

没有在kernel入口执行 `const auto context = *device_context`；各字段直接经指针读取，避免把
大POD重新物化到每线程local stack。续跑热指针也不属于只读context，避免跨round间接访问
产生错误关联。

## ABI stop gate

在改生产kernel前，隔离probe让两种kernel消费相同的1984 B数据并保留checksum，结果如下：

|传参方式|constant0|每线程stack|registers/thread|
|---|---:|---:|---:|
|大对象按值capture|2368 B|2016 B|55|
|device POD，仅capture一个指针|376 B|0 B|47|

这证明当前oneAPI/NVIDIA后端会把大closure物化到每线程stack，而device context pointer能消除
该复制。生产候选实际wrapper的资源变化为：

|入口|constant0|每线程stack|registers/thread|
|---|---:|---:|---:|
|默认OFF基线wrapper|1984 B|1840 B|186|
|context pointer wrapper|408 B|264 B|187|
|候选具名kernel|394 B|464 B|180|

constant0和stack达到结构stop gate，寄存器没有改善。

## 正确性

使用与campaign相同的RT07575分片、spots、随机种子、Schneider v2.1、统一EM包、δ矩包、
`straggling_scale=1`、精确CT边界、16步续跑和8192尾阈值。最终工作树二进制结果为：

- 3,240,963 histories，1,034,976,717 steps，1,344,312次核相互作用；
- 所有续跑active count逐轮匹配基线；
- 8项统一EM整数审计完全一致：
  `0 1036320939 1429988519 0 0 6897079514135703 528848467657861 0`；
- production quality通过，overflow为0，能量记账误差 `3.6833317e-06`；
- 五次候选与基线的3D剂量均值最大绝对差为峰值的 `0.00001989%`；五次基线重复的全局
  波动包络为 `0.00004263%`，候选均值差在包络内。

Schneider v2.1校验通过。统一EM SHA256为
`8c5d970b3b639bfca2f448730271bed4fc04721aba73100e2efbe09dffe44855`，δ矩包SHA256为
`c551bc52fa30ff7e3ea229c8b89792e8ecd2b0fb12f18fcad50d1c6f206bc7cc`。

## 无profiler配对性能

RTX 2080 Ti/sm_75上先预热2次，再做5次正式配对，正反交错。正式阶段GPU为80–82°C、
SM 1815 MHz。改善定义为 `baseline / candidate - 1`。

|指标|中位改善|五对范围|
|---|---:|---:|
|进程wall|+8.04%|+7.84% – +8.42%|
|Elapsed|+8.64%|+8.52% – +9.07%|
|原发kernel|+0.01%|−0.63% – +0.35%|
|次级kernel|+19.97%|+19.25% – +20.33%|

五对Elapsed改善为 `8.52%, 9.07%, 8.52%, 8.64%, 8.78%`，超过5%推广吞吐门槛且远大于
重复波动。原发未被此候选改写，实测保持不变。

复现时分别以 `-DCARBON_SECONDARY_CONTEXT_POINTER=OFF/ON` 构建sm_75二进制，再用
`tools/benchmark_single_gpu.py` 指定同一个冻结config执行2次预热和至少5次交错配对。计时结束
后单独运行 `tools/profile_transport_stages.py`；次级实际wrapper使用mangled名称匹配，脚本已
支持用 `--data-dir` 指定冻结数据目录，避免把具名device函数或空匹配误当成实际启动入口。

## 匹配阶段NCU

每个阶段使用3次独立应用运行。次级按代次、续跑轮次和active count匹配；表中均为中位数。
ready fraction定义为 `eligible / active`。

|阶段|block 基线→候选|active warp/sched|eligible warp/sched|ready fraction|long scoreboard|local sectors|
|---|---:|---:|---:|---:|---:|---:|
|原发稳态|128→128|1.976→1.978|0.0970→0.0964|4.91%→4.87%|9.31→9.48|1.110B→1.110B|
|次级早段 gen1/r0|32→32|1.950→1.939|0.0862→0.1030|4.42%→5.31%|10.29→5.73|10.179B→6.117B|
|次级中段 gen1/r17|32→64|1.994→1.952|0.0853→0.0985|4.28%→5.07%|14.66→10.80|2.530B→1.478B|
|次级晚段 gen1/r34|64→128|1.296→0.995|0.0678→0.0814|5.23%→8.17%|11.39→3.69|0.014B→0.041B|
|第二代 r0|256→64|1.943→1.934|0.0881→0.1028|4.54%→5.31%|15.03→9.77|3.055B→1.246B|

候选在所有次级阶段都提高eligible和ready fraction，并降低long-scoreboard。早段、中段和
第二代local sectors分别下降39.90%、41.59%和59.22%。晚段样本的local sectors反而增加，
同时自动block变化并使active warp下降23.22%；不能用其他阶段的改善掩盖这一退化。

原发的寄存器、block、active、eligible和local sectors均处于基线重复范围附近，说明本次
机制归因仅适用于次级closure，不能机械扩展到原发。

## 两个已否决的中间实现

1. 把编译器生成的lambda对象整体复制到device memory，再调用其 `operator()`。wrapper结构
   指标明显改善到约376 B constant0、232 B stack和182 registers，但次级数量、整数审计和
   能量记账分叉，quality失败。编译器生成closure类型不能充当稳定的显式ABI。
2. 显式POD context中同时保存 `resume_states`、`resume_ready`、`active_order` 和 `keep`。
   第0轮及首次压紧匹配，第一次恢复后分叉：基线round 2 active为5,841,290，候选为
   3,929,615，quality失败。续跑可变指针必须保留为每轮显式capture。

这两个方案已追加到 `docs/eligible_warp_failed_attempts.md`，没有进入生产路径。

## 默认与后续工作

`CARBON_SECONDARY_CONTEXT_POINTER=OFF` 保持默认。结构候选已经证明closure ABI是次级
local-memory和ready fraction的重要杠杆，但当前187 registers仍只能提供约8 warp/SM，且
次级active warp未满足最终门槛。由于硬件推广门槛已经失败，本轮没有继续消耗资源做
RT06423和20022516各5次的最终推广回归；50k固定分片的正确性和5对性能已完成。

下一步应在这个小closure入口上做production context类型专门化，使关闭的计分、诊断和研究
字段在形成kernel ABI前不存在，目标是把实际wrapper从187降到不高于168。达到该结构门槛后
再扫描32/64/128线程block，并补齐RT06423、20022516和50k水模推广回归。

患者BODY Gamma和低密度production-cut阈值精度验收仍未完成；本轮quality通过不代表这两项
精度已经验证。

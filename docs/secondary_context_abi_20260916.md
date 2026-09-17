# 次级输运 kernel closure ABI 实验（2026-09-16）

## 结论

`CarbonSecondaryTransportKernel<1>` 的大 `[=]` closure 确实是次级线程私有 local-memory
热流量的主要来源。把不随续跑轮次变化的参数上传为一个只读 device context，并让实际
SYCL kernel 只显式捕获 context 指针和续跑热指针后，RT07575 固定分片的 Elapsed 中位数
改善 **8.64%**，次级 kernel 时间改善 **19.97%**。早段 local sectors 降 **39.90%**，
long-scoreboard 降 **44.30%**，eligible warp/scheduler 升 **19.49%**。

候选由 `CARBON_SECONDARY_CONTEXT_POINTER` 控制，当前默认 **OFF**。它没有跨过168寄存器
门槛：实际 wrapper 从186升到187 registers/thread，次级各匹配阶段的active warp没有提高，
晚段还因后端自动选择128线程block而明显下降。这些结果说明驻留量尚未改善，但所有采样
次级阶段的eligible和ready fraction均提高，完整RT07575吞吐也超过5%门槛。因此168寄存器
是后续提高驻留量的研究目标，不再作为否决这个已获得吞吐收益候选的必要条件。RTX 2080 Ti
上的跨病例推广现已通过；候选当前保持默认关闭的原因仅剩A6000推广验证尚未完成。

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

次级 `+19.97%` 使用 `baseline/candidate - 1` 定义，即速度约为基线的1.20倍，对应次级耗时
减少约16.64%。它不表示次级耗时减少19.97%。

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

`CARBON_SECONDARY_CONTEXT_POINTER=OFF` 暂时保持默认。结构候选已经证明closure ABI是次级
local-memory和ready fraction的重要杠杆；当前187 registers仍只能提供约8 warp/SM，但这
不抵消RT07575上已验证的8.64%完整吞吐收益。当前状态是“RTX 2080 Ti上的RT07575、
RT06423、20022516和50k水模通过；A6000推广待完成；occupancy目标未达到”。晚段active warp
下降需要结合所有尾部启动的总耗时占比判断，不能单独作为吞吐否决条件。

后续隔离生产路径特化已把次级wrapper从177降到164 registers/thread，并在匹配早段观察到
active warp/SM从7.76升到11.51；RT07575次级中位再改善9.69%、Elapsed改善3.92%。该项通过
结构门槛但仍默认关闭，详见`docs/secondary_production_specialization_20260916.md`。

后续已经在这个小closure上完成两项独立实验：显式`nd_range`使端到端吞吐下降
6.50%–7.59%；不变projectile registry索引复用使wrapper增至228寄存器，五对吞吐变化处于
波动内。两者均保持默认关闭并已写入失败台账。下一项结构研究应是少量host-dispatch生产
路径特化。结构指标用于解释机制；是否推广由正确性、完整吞吐和代表性配置回归决定。

## 编译配置兼容修复

后续检查发现原始候选把剂量context字段固定写成`float*`，且`CARBON_EM_SPECIALIZE=OFF`的
`transport_sycl_impl<-1>`没有把运行时`unified_em`值带入小closure。修复后：

- 深度计分使用`DepthAtomicT*`，3D及来源分解计分使用`DoseAtomicT*`，没有指针强制转换；
- `EmMode < 0`从device context读取运行时EM模式，`EmMode >= 0`仍由模板常量折叠；
- context经`DeviceMemoryTracker`分配和释放，copy、提交或等待抛异常时由tracker清理。

sm_75/NVPTX的四种组合均完成设备代码编译和链接：EM特化ON/OFF × dose FP32/FP64。最终
FP32特化构建又完成同一固定分片：3,240,963 histories、1,034,976,717 steps、整数审计完全
一致、quality通过、overflow为0。通用EM FP32、特化FP64和通用EM FP64分别以同一物理配置
的缩小history scale完成GPU smoke，均为3,532 histories、1,110,124 steps、quality通过且
overflow为0。该smoke只验证公开开关组合可运行，不替代FP64性能或剂量推广验收。

## RTX 2080 Ti推广回归

兼容修复后用同一组context OFF/ON固定二进制补齐代表性配置。患者分片各2次预热、5次
交错正式配对；50k水模初次5对剂量均值差略超小样本基线包络，因此扩为10对后再判定。

|配置|正式配对|Elapsed变化|次级速度变化|3D剂量均值差/基线包络|
|---|---:|---:|---:|---:|
|RT06423固定分片|5|+7.73%|+19.74%|0.00001819% / 0.00002577%|
|20022516固定分片|5|+6.50%|+19.11%|0.00001695% / 0.00002712%|
|50k水模|10|+1.35%|+16.94%|0.00002991% / 0.00020365%|

三组整数审计、步数、核相互作用和quality均通过，overflow为0。两例患者端到端收益超过5%；
50k水模端到端变化范围−1.15%至+3.96%、中位+1.35%，满足回退不超过2%的门槛。A6000尚无
本轮二进制的运行结果，因此仍不切换默认。

患者BODY Gamma和低密度production-cut阈值精度验收仍未完成；本轮quality通过不代表这两项
精度已经验证。

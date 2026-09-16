# 输运内核线程私有 local-memory 复核（2026-09-16）

## 结论

RTX 2080 Ti/sm_75 上的低并发包含两个独立门槛：172/186寄存器把稳态驻留限制在约
8 warp/SM，而驻留warp的ready fraction只有约4%–5%。线程私有local-memory流量确实是
下一轮应优先处理的对象，但本次反汇编和SourceCounters表明，次级的首要可见来源不是
`SecondaryResumeState saved`，而是SYCL包装内核在入口把约2 KiB lambda捕获参数物化到
约1.8 KiB线程local frame。

因此，直接删除pause/resume处的外层整结构体临时量没有收益。后续候选应缩小实际包装
内核的捕获对象，或通过生产配置编译期专门化让未使用的捕获字段和分支在进入lambda前消失。
AoS改SoA只有在恢复状态访问被动态计数证明是热点后再做。

## 基线复现和ready fraction

实验使用 `scratch/transport_concurrency_20260916/source` 的冻结快照。独立重建的基线二进制
SHA256为 `1202cd92c2ae54f129a53e6207ade7018729ecb130848a70df7641575189dfca`，与campaign
基线逐字节一致。Schneider v2.1、统一EM和delta moments校验通过。

|阶段|active warp/scheduler|eligible warp/scheduler|ready fraction|
|---|---:|---:|---:|
|原发稳态|1.9757|0.0970|4.91%|
|次级早段|1.9497|0.0862|4.42%|
|次级中段|1.9937|0.0853|4.28%|
|次级晚段|1.2957|0.0678|5.23%|
|第二代|1.9427|0.0881|4.54%|

晚段只有10,533个活动粒子，包含tail underfill；其active warp不能作为稳态驻留上限。
其余不同block尺寸均接近8 warp/SM，与寄存器分配档位一致。

## pause/resume整结构体临时量实验

候选 `resume_fieldwise_reference` 只做两处改动：

- `const auto saved = resume_states[state_idx]` 改为对该元素的常量引用；
- 局部 `SecondaryResumeState saved` 加最终整结构体赋值改为直接引用目标元素并逐字段写入。

恢复 `UnifiedEmState` 后仍执行 `unified_secondary_state.tables = &unified_device`。候选二进制
SHA256为 `182ee286bd65f4b7d26bbdb7ee64a75ec7e49076b05476dad1f93faa81503c6c`。

|实际入口/函数|指标|基线|候选|
|---|---|---:|---:|
|实际 `__pf_kernel_wrapper<CarbonSecondaryTransportKernel<1>>`|寄存器/线程|186|186|
||stack frame|1840 B|1840 B|
||LDL / STL|122 / 136|122 / 136|
|具名 `CarbonSecondaryTransportKernel<1>`|寄存器/线程|178|178|
||stack frame|2080 B|2080 B|
||LDL / STL|505 / 859|506 / 858|

包装入口SASS完全没有资源改善；具名函数只是1条LDL/STL的对换，指令数反而增加136。
按结构门槛停止，没有运行计时或候选NCU。结果说明编译器原来已经消除了外层聚合临时量，
2 KiB线程栈不能归因于这两行源码。

## 次级local-memory的主要可见来源

对campaign已有NCU报告重新导出 `--page source --print-source sass --csv`。次级实际入口的
开头存在循环：从constant kernel parameters执行逐字节 `LDC.U8`，再以 `STL.U8` 写入线程
local frame。循环在恢复状态判断和物理热循环之前，包装入口资源同时报告1840 B stack和
1984 B constant参数区。这与SYCL包装层物化lambda捕获对象一致。

|阶段|SourceCounters可归因local sectors|其中 `STL.U8` 比例|
|---|---:|---:|
|原发稳态|64,540,415|10.20%|
|次级早段|2,374,841,847|80.49%|
|次级中段|460,974,106|78.31%|
|次级晚段|2,759,840|78.44%|
|第二代|933,841,055|84.34%|

次级各阶段结果一致，说明入口捕获物化是次级local流量的主要可见组成。原发没有同样的
包装入口特征，其local流量更多来自函数体内较宽的64位栈访问，需要独立归因。

这里的SourceCounters逐指令合计不能与MemoryWorkloadAnalysis的raw总数互换；两者采集和
聚合口径不同。long-scoreboard样本也落在等待依赖的消费指令上，不能把上述80%直接解释为
80%的停顿时间或端到端耗时。

## 下一步

下一项隔离实验应针对实际包装层捕获大小：为已验证生产配置提供一个专门化入口，优先让
关闭的诊断、计分和研究路径在lambda形成前不再被引用。首先检查包装入口的constant参数区、
stack、入口 `LDC.U8/STL.U8` 动态sectors和186寄存器；只有这些结构指标改善后再做matched
NCU和端到端配对。若专门化不能缩小捕获对象，应停止并评估把只读参数打包成显式设备上下文
指针的方案，同时计入新增global依赖加载。

机器可读结果和精确补丁位于
`benchmark/local_memory_attribution_20260916/analysis.json` 与
`benchmark/local_memory_attribution_20260916/resume_fieldwise.patch`。患者BODY Gamma和
低密度阈值精度验收仍未完成。

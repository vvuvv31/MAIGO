# 次级显式 `nd_range` 入口实验（2026-09-16）

## 结论

在已经缩小closure的context-pointer候选上，显式`nd_range`没有进一步提高吞吐。32、64、
128线程三种入口均保持固定分片正确性，但次级kernel耗时增加约17%，端到端吞吐下降
6.50%–7.59%。候选保持默认关闭，不进入正式五对计时或NCU全阶段采样。

这个结果隔离了新的小closure入口，与历史大closure上的block-size扫描不是同一个编译产物。
显式入口把静态寄存器从实际rounded wrapper的187降到具名入口的170，但每线程stack从264 B
增加到480 B，仍未跨过168寄存器驻留档位。资源数字与单次完整筛选方向一致：更低的寄存器
数字没有转化为更快的输运。

## 实现与边界

实验由默认关闭的`CARBON_SECONDARY_EXPLICIT_ND_RANGE`控制，只允许与
`CARBON_SECONDARY_CONTEXT_POINTER=ON`组合。`CARBON_SECONDARY_ND_RANGE_SIZE`只接受32、64、
128。global range向上补齐；补齐线程在读取`active_order`、`resume_ready`、`keep`或粒子队列
之前返回，因此物理粒子编号、RNG身份和续跑压紧顺序不变。

原range入口保持原代码路径和默认行为。profiler新增`--secondary-entry`，可明确匹配rounded
wrapper或具名入口；同时显式记录`--replay-mode`与`--cache-control`，避免不同缓存协议交叉
比较。

## RT07575完整筛选

使用同一固定配置、RTX 2080 Ti/sm_75、Schneider v2.1、统一EM与δ矩包、原生涨落
`scale=1`、精确CT边界、16步续跑和8192尾阈值。改善定义为`baseline/candidate - 1`。

|入口|线程块|Elapsed (s)|次级 (s)|端到端吞吐变化|次级速度变化|
|---|---:|---:|---:|---:|---:|
|range control|runtime|20.229883|8.631476|基线|基线|
|显式 `nd_range`|32|21.636933|10.102255|−6.50%|−14.56%|
|显式 `nd_range`|64|21.832124|10.129329|−7.34%|−14.79%|
|显式 `nd_range`|128|21.891748|10.128049|−7.59%|−14.78%|

四次运行均为3,240,963 histories、1,034,976,717 steps、1,344,312次核相互作用，8项
统一EM整数审计一致，quality通过且overflow为0。三种显式入口都朝同一方向退化，幅度远大于
已知重复波动，因此按筛选stop gate结束，不用正式五对重复消耗GPU时间。

尾部两个直接完成轮次在range control中合计0.027502 s，只占次级kernel的0.32%；显式64
线程时合计0.029967 s。晚段单个NCU样本的active warp下降不能等权否决整个context-pointer
候选，完整尾部时间权重很小。

## 编译资源

从运行时dump的sm_75设备镜像读取：

|入口|registers/thread|stack/thread|constant0|
|---|---:|---:|---:|
|range rounded wrapper|187|264 B|408 B|
|显式 `nd_range` 具名入口|170|480 B|400 B|

三种显式block尺寸生成相同静态资源。NCU动态LaunchStats未纳入结论；当前主机普通用户没有
performance-counter权限。筛选结论来自无profiler完整计时，避免profiler改变端到端结果。

机器可读结果见`benchmark/secondary_ndrange_entry_20260916/analysis.json`。患者BODY Gamma与
低密度production-cut阈值精度验收仍未完成；本次固定分片quality通过不代表这两项精度已验证。

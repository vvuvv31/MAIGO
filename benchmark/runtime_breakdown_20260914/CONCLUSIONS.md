# RT07575 runtime breakdown：结论

本次冻结工作树、100 万原发、chunk=34816，本地 RTX 2080 Ti / sm_75。
完整次级统一 EM、原生涨落和种类分组开启，生产配置不含独立核弹性。
这些结果不能与此前含弹性的 14.7k/s 实验混为同一配置。

## 实测阶段

内部 Elapsed 51.057 s，吞吐 19585.9 histories/s；完整进程 52.775 s。
原发 kernel 25.595 s、次级 21.640 s，合计占完整进程 89.5%。
输运初始化（含上传）4.506 s，输入/源准备 0.441 s，质量/输出 0.442 s，读回/后处理 0.056 s。
Nsight 的 H→D 0.1386 s、D→H 0.00442 s 是上述阶段的内部活动，不能再次求和。

即使删除全部 kernel 外开销，按两个 kernel 合计 47.23 s，吞吐也只有约 21.2k/s。
50k/s 必须显著降低输运 kernel 的工作量。

## 内部采样定位

细分诊断构建耗时增加约 2.7%，步数与 EM audit 一致，运行接受且零 overflow。
采样得到原发 EM 准备/能损约 52.3% 周期，另有 32.5% 几何/核率准备；
次级 EM 准备/能损约 66.9%。MCS 约 4–5%，事件处理相关区约 2–3%。
EM 能损内部，涨落约占 50–56%，直接 δ 时钟/接受/能谱区约占 24–26%。

这些是诊断构建的 sampled lane elapsed cycles，不是独立 GPU wall-time。
它们包含等待、warp 分歧、编译调度及插桩影响，外层 EM 还包含内部探针开销。
全程序扰动较小不保证每个采样区间同比例小；不将这些比例乘以原 kernel 秒数冒充独立耗时。

本轮原发统一 EM 步约 22.02 亿、次级约 6.76 亿。δ 候选不仅消耗抽样成本，还缩短步长，
让几何、查表和涨落重复执行。因此不能用直接 δ 区间比例推算删除/合并 δ 的总加速上限。

## 资源限制与后续方向

Nsight 元数据：原发 255 registers/thread，次级 178；两者 block=128、shared memory=0。
按 sm_75 寄存器容量和分配粒度，理论约最多 2 blocks/SM，即 8/32 warps=25% occupancy。
这是资源上限，不是实际 occupancy 或 memory-stall 测量；不能据此直接断言 memory-bound。
不要盲目压低寄存器上限而引入 spill。

下一步拆开原发几何/核率准备，以及次级 EM 准备中的材料选择、range/stopping 准备与 δ 率更新。
目前没有它们各自的独立数据，不把混合区全部归因于某一项。
优先减少高频准备及 EM 微步重复工作，MCS/事件库重放/输出不是当前数倍提速的主要入口。
当前电子已经局部沉积，没有可以再关闭的完整电子 tracking kernel。

原始 JSON、图、配置和源码 manifest 见本目录；探针仅存在于 scratch 诊断副本中。
本结论不改变生产物理，不推广此前失败的 RNG、macro-tick 或 delta-only quantile 候选。

## 后续细分结果

[DEEP.md](DEEP.md) 已拆开上述混合区：原发几何 3.90%、核率 8.22%、外层平均能损/审计区 20.41%。次级 stopping/range/涨落参数准备 19.21%，δ 率/时钟/步长 9.77%。该结果使外层重复平均能损成为首先验证的候选。百分比仍为采样周期，非独立墙钟。

## 重复计算 A/B 已完成

外层平均能损在诊断关闭时可避免重复计算。两轮百万粒子吞吐均值 19.41k → 21.45k/s，约 +10.5%；诊断开关打开时 14 桶审计仍完全一致。见 [MEAN_GUARD.md](MEAN_GUARD.md)。这仍是隔离候选，正式源码未接入，未 commit/push。

## 后续核率缓存查询实验

两轮 1M 对照未观察到收益：相对 mean_guard 基线分别 -1.92%、-0.36%，不接入生产。审计与计数一致、零 overflow；详见 [RATE_GUARD.md](RATE_GUARD.md)。

## 后续次级 stopping 区间缓存实验

保留全部检查的表单元缓存两轮 1M 吞吐变化为 -1.59%、-0.27%，次级 kernel 第二轮基本无变化，不接入生产。1003467 次主机逐位查表比较通过；GPU 审计一致、零 overflow。详见 [SEC_CELL.md](SEC_CELL.md)。

## mean_guard 后的模块再定位

新增细分采样表明：次级旧 stopping 步首/中点区合计约 4.76%，统一 EM 参数准备约 19.04%、EM 能损抽样约 33.21%；EM 内部涨落占约 56.18%。主要剩余热点在统一 EM 准备与抽样，见 [MODULE_FINDINGS.md](MODULE_FINDINGS.md)。采样占比不是可消除墙钟。

## 涨落循环诊断

原发/次级的 universal compound 分支占采样涨落调用 93.71%/83.81%，平均约 20.7/20.5 draw；Gamma 尝试均值约 1.02、高斯约 1.0001。应优先关注频繁 compound/Poisson/显式能损累加，而非稀有拒绝长尾。详见 [FLUCT_FINDINGS.md](FLUCT_FINDINGS.md)。

## Compound 四路展开实验

保持 RNG 和累加顺序的 `#pragma unroll 4` 两轮 1M 吞吐分别下降 11.72%、10.52%，不接入。审计和计数一致、零 overflow；详见 [COMPOUND_UNROLL.md](COMPOUND_UNROLL.md)。

## 专用 sampler kernel 可行性

真实采集 1,405,166 条输入的独立 kernel 测试：34,816 条队列按分支分组约 +9.59%，小队列吞吐明显降低。未包含轨迹暂停恢复、GPU 分桶和队列成本，不是端到端加速证据，详见 [FLUCT_QUEUE.md](FLUCT_QUEUE.md)。

## 轨迹长度/虚拟队列重建

次级warp有效EM步槽位代理指标23.67%，原发76.06%；不是硬件occupancy。虚拟队列低于8192时仅承载约1%的EM工作，却占大量轮次。下一步应优先评估次级分段推进/压紧，避免逐步无条件拆分sampler。详见 [QUEUE_LIFE.md](QUEUE_LIFE.md)。

## 分段续跑可行性

64步一块、队列<8192时跑完的离线模型需40次输运启动/38次压紧。真实队列长度驱动的稳定压紧微基准：256字节状态约0.193秒，512字节约0.425秒；均不含完整物理续跑。建议进入64步受控原型，详见 [SEGMENT_COMPACTION.md](SEGMENT_COMPACTION.md)。

## 64步完整续跑原型

50k门槛通过后，两轮RT07575 1M吞吐提升36.58%/39.66%，平均21403→29558 histories/s。次级kernel约22秒降至8.85秒，审计/步数/整数账本一致、零overflow。仍为隔离候选，独立弹性及正式集成未验收。详见 [SEGMENTED_PROTOTYPE.md](SEGMENTED_PROTOTYPE.md)。

## 跨场景与逐轨迹验证

五组含all-ion elastic的50k筛选通过；无弹性197422条、含弹性208981条记录轨迹终止前状态逐位一致。弹性1M两轮吞吐+34.67%/+39.16%，但同一体素约0.00095%峰值的稳定差高于自重复波动，仍待定位。详见 [SEGMENT_VALIDATION.md](SEGMENT_VALIDATION.md)。

## 默认生产接入（用户接受精度）

水/RT07575 两个生产预设开启 `secondary_step_chunking`，正式代码保留关闭路径且关闭时不分配续跑池；同时接入 mean_guard。正式源码 1M RT07575 开关对照为 21559→29535 histories/s（+36.99%），次级 21.852→8.859 s，最大差 0.00009662% 峰值，审计/步数/整数账本一致、零 overflow。先前原型阶段“未接入/差异阻断”状态由本次用户授权和验证更新，差异原因未证明的说明保留。详见 [SEGMENT_PRODUCTION.md](SEGMENT_PRODUCTION.md)。

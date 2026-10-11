# Carbon CT broadbeam 后续受控实验 — 2026-10-09

本轮四项候选均未测出稳定的端到端加速，全部回退。保留上一轮同一步EM视图复用和精简birth查询（正式测量+3.52%，约75,254原初粒子/s）。当前carbon-only可执行文件和静态库均恢复到已验证版本，SHA256 `26726a83f86eacdb90c185eb9497381736d15a2b79688c3a355657afd04339aa`。

## 实验范围

RTX2080Ti/sm_75、相同carbon-only CMake开关、原RT07575**头部**carbon CT broadbeam计划。continuous26、strict26电子/光子、涨落、核反应、所有带电碎片、0.5mm ion cap、FP32计分、source/spot seeds和campaign归一化保持原样。没有改变任何其他预设默认配置。CUDA_DISABLE_PTX_JIT=1执行成功。

每项独立seed、ABBA，每进程4×1M粒子，GPU context/packages连续复用，首批不计性能。每项16M粒子，四项共64M；每臂正式暖批6M。四项均两对方向相反，因此不采用这些百分比作为可靠加速或减速常数，也不追加无意义的重复扫描。

| 候选 | 总吞吐变化 | 两个配对比 | secondary吞吐比变化 | 结果 |
|---|---:|---|---:|---|
| 核物种索引复用 | -0.202% | 0.980259 / 1.015421 | -0.048% | 回退 |
| 分段内材料视图复用 | -0.201% | 0.987832 / 1.007923 | -0.078% | 回退 |
| CT归一化坐标复用 | +0.152% | 0.988419 / 1.014379 | +0.677% | 回退 |
| 分段成功EM计数合并 | -0.156% | 0.985961 / 1.010701 | +0.097% | 回退 |

secondary列来自完整批次已有 `[segmented-secondary] kernel_s`，与总wall分开。GPU不能锁频，各组存在热漂移。

## 做了哪些改动

1. 核物种索引：已有track registry结果复用于hazard、He4 audit和post-EM replay，无新增state。
2. 分段材料视图：只在当前kernel segment内缓存，材料/密度变化失效，resume重建；与此前扩大continuation的缓存不同。
3. CT坐标：ct_sample已经计算的normalized坐标送入未改数学的DDA/face clamp，省3次除法。CPU2,097,152条件，1,634,251 inside，697,272 face hits，sample、全部DDA fields和clamp结果逐位一致。包括随机、平移、精确和相邻FP32边界。
4. 成功EM审计：每段累计valid count、continuous/delta microMeV整数值，再原子提交；每步整数截断不变，错误立即计数，无续跑字段。区别于此前primary整条history的累加实验。

## 验证

全部source/primary_batches、离散事件、live ion species ledger、整数EM/charged/photon/joint计数一致；energy、coverage、completion、zero-overflow通过。所有配对剂量relative L1≤4.2698191e-08，10%peak以上同位置local1%通过率100%，因此local1%/1mm、search0.1mm的匹配通过率下界100%。这是GPU实现一致性验证，没有新的绝对TOPAS验收。

原始证据分别在表中候选对应的 `out/carbon_ct_*_20261009/`，含冻结control/candidate源码、命令、日志、逐批剂量及quality/energy ledger、ABBA与summary。新的失败路线均写入failed.md，禁止原样重复试验。

## 判断

本轮没有证据说明这四处源代码层面的重复工作是主要动态瓶颈；也不能从“减少原子/除法/表搜索”推断可测吞吐收益。上一轮+3.52%仍保留。碳束与质子约两倍的差距尚未缩小，需要进一步针对碎片输运本身及电子链路开展更大的工作量/状态布局改动；带权碎片抽样若实施，必须贯穿后代和计分，并按相同统计误差耗时验收。

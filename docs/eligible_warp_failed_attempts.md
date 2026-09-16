# Eligible warp / warp/SM 失败方案台账

本文件汇总仓库中所有以提高 `eligible warp`、提高实际/理论 `warp/SM`、跨越寄存器驻留
门槛、改善block共驻，或通过缩短依赖链降低long-scoreboard为直接目标，但最终没有通过
推广门槛的方案。

这里的“失败”只描述上述并发目标。某些改动减少了端到端耗时，仍可能没有提高
eligible或warp/SM；这类结果单独列出，不能把吞吐收益改写成occupancy收益。

## 当前参照与判定口径

2026-09-16冻结的 RTX 2080 Ti/sm_75 生产一致入口如下。每项为3次独立 NCU 应用运行的
中位数；次级按代次、续跑轮次和活动粒子数匹配。

|阶段|block|寄存器/线程|寄存器限制 block/SM|active warp/scheduler|eligible warp/scheduler|long-scoreboard cycles/issue|
|---|---:|---:|---:|---:|---:|---:|
|原发稳态|128|172|2|1.976|0.0970|9.31|
|次级早段，gen1/r0|32|186|8|1.950|0.0862|10.29|
|次级中段，gen1/r17|32|186|8|1.994|0.0853|14.66|
|次级晚段，gen1/r34|64|186|4|1.296|0.0678|11.39|
|第二代r0|256|186|1|1.943|0.0881|15.03|

Turing寄存器按warp粒度分配。当前原发172和次级186都落在每SM约8个驻留warp的档位；
要进入约12 warp/SM的下一档，需要实际启动入口达到不高于168寄存器/线程。具名函数的
寄存器下降不算通过，必须检查实际SYCL包装内核。

判定始终使用无profiler的端到端配对运行。栈帧大小不直接等同于spill，long-scoreboard
比例也不直接等同于某模块耗时。NCU和SASS只用于归因。

以下历史条目由现有文档、提交和实验目录重建。条目没有列出的硬件计数器、重复次数或
正确性字段一律视为“未测”，不能从相邻实验或结构大小推断。F节的完整字段要求适用于
本台账建立后的新增实验。

## A. 直接提高驻留warp或block共驻的失败方案

### A1. 强制寄存器上限 `maxrregcount=128`

- 目标：用spill换取更低寄存器数和更高理论occupancy。
- 结果：吞吐变化约0%；新增local-memory访问抵消了驻留提升。
- 结论：完全回退。没有新的live-range证据时不要重试机械寄存器限额。
- 证据：`docs/rejected_approaches.md` §6。

### A2. 直接插值字段，避免完整 `UnifiedEmNode` 临时对象

- 目标：缩短EM准备阶段临时对象的活跃范围，降低寄存器。
- 结果：审计一致，但RT07575 Elapsed约慢3%，次级更慢；实际资源仍处于约8 warp/SM
  的同一档位，没有跨过168寄存器门槛。
- 结论：回退。
- 证据：`docs/rejected_approaches.md` §6。

### A3. 拆分次级非弹性末态冷路径

- 目标：把低频核反应末态从热输运kernel移走，降低热kernel寄存器和栈。
- 诊断：直接删除整段事件代码只把次级具名内核178→174寄存器、栈2096→1984 B；原发
  仍为168。32线程入口仍未跨驻留门槛，128线程入口仍只能驻留2个block。该路径约
  1.3e6次调用，而次级步数超过1e9。
- 结果：收益上限不足，完整拆核在实现前停止。
- 结论：生产未改。只有热路径占比或实际包装内核资源发生实质变化才可重开。
- 证据：`docs/rejected_approaches.md` §6，`scratch/opt_gpu_20260915/binary_inelastic_stub`。

### A4. 编译期关闭生产未用计分分支

- 目标：专门化 `enable_charged_origin_voxel_scoring=false`，删除分支和状态。
- 结果：次级具名内核178→174寄存器，但实际启动包装内核保持191；Elapsed
  27.93→28.24 s，没有提速或驻留档位变化。
- 结论：回退。以后必须报告实际包装入口，不能用具名内核资源代替。
- 证据：`docs/rejected_approaches.md` §6。

### A5. He-4续跑审计状态裁剪

- 目标：不在续跑状态中携带 `he4_audit`，减少状态传递和寄存器。
- 结果：实际包装内核从186升到197寄存器，方向相反。
- 结论：回退。
- 证据：`tobeoptimized.md` §1.4。

### A6. delta确定性参数提前合成

- 目标：把 `delta0/delta_slope/delta_mean/delta_variance` 提前计算，缩短 `pre` 和相关
  插值量的live range，使原发172降到不高于168。
- 结果：原发172→172，次级包装186→186；编译器已完成等价调度。
- 结论：SASS资源不变，回退。
- 证据：`tobeoptimized.md` §6.1，commit `ad9cfc8`。

### A7. 核反应率 total-only 查询接口

- 目标：纯总率调用不再携带13个partial rate，缩短局部数组/标量寿命。
- 结果：原发仍172、次级包装仍186；未用partials原本已被编译器DCE。
- 结论：回退。
- 证据：`tobeoptimized.md` §6.2，commit `ad9cfc8`。

### A8. 原发spot定位与初始采样拆到初始化kernel

- 目标：将spot二分、初始能量/发射度采样和入射几何移出原发热循环，只跨kernel保存
  输运初态与RNG身份。
- 结果：质量、审计和3D剂量通过；探索性RT07575配对中wall −0.53%、Elapsed −0.48%、
  原发 −0.65%。额外状态读写和launch没有被热kernel缩短状态的收益覆盖。
- 结论：候选仅在隔离源码中，未进入生产。
- 证据：`docs/transport_concurrency_optimization_20260916.md`，
  `benchmark/transport_concurrency_20260916/candidate_screen_summary.json`。

### A9. 固定显式工作组/线程块扫描

- 目标：用32/64/128/256线程块改变寄存器分配、实际驻留和尾段填充率。
- 结果：在16步续跑生产一致构建中，各候选整体约慢1.1%–1.8%；更早一轮显式次级
  工作组扫描也显示后端默认入口更快。资源变化前重新扫block没有收益。
- 结论：保持后端/运行时选择；只在实际包装内核寄存器档位变化后重扫。
- 证据：`docs/transport_concurrency_optimization_20260916.md`，`tobeoptimized.md` §1.4。

### A10. A6000 MPS与多进程共驻

- 目标：用MPS让多个分片的block同时驻留，填补单kernel延迟槽。
- 结果：1/3/6并发组中MPS开关差异小于0.3%；当时实际包装入口约191寄存器，无法形成
  有效SM block共驻。多进程可摊薄主机初始化，但不是eligible/warp/SM修复。
- 结论：MPS作为内核occupancy方案停止。
- 证据：`docs/rejected_approaches.md` §6。

### A11. Philox四输出缓冲

- 目标：摊薄RNG计算，并希望用更多独立值增加ILP。
- 结果：寄存器达到255并发生local-memory spill，warp补充路径分歧；吞吐约慢5%。按步
  缓冲还会破坏随机流。
- 结论：完全回退；RNG/ALU不是当前主要杠杆。
- 证据：`docs/em_macro_step.md`，`docs/rejected_approaches.md` §6。

### A12. 次级压紧前缀和并行化

- 目标：缩短续跑轮次之间的存活索引压紧，减少低填充阶段的调度开销。
- 结果：生产16步续跑中 `compaction_s` 中位数仅0.026 s，约占Elapsed的0.10%；即使把
  压紧成本全部消除，也不足以改变eligible或warp/SM。候选因此在实现前停止，硬件指标未测。
- 结论：未实施。只有压紧占比出现数量级增长时才重新评估。
- 证据：`tobeoptimized.md` §1.4。

## B. 降低long-scoreboard、提高eligible比例的失败方案

### B1. EM节点/分段搜索键分离

- 目标：把52 B节点和24 B分段记录中的搜索键抽成连续float数组，提高每cache line键密度。
- 结果：搜索比较约占次级long-scoreboard样本的11%，但候选次级15.36 vs 14.73 s，约慢4%；
  依赖加载次数不变，并增加第二条全局数据流，原记录已大多驻留L2。
- 结论：`CARBON_EM_SPLIT_SEARCH_KEYS`保持关闭。
- 证据：`docs/rejected_approaches.md` §6。

### B2. 更细的精确索引

- 目标：指数目录再加入6位尾数，缩短节点/分段二分区间。
- 结果：真实包逐位一致，但RT07575 wall −4.84%、Elapsed −5.19%、原发 −5.43%、次级
  −5.94%；额外索引访问比省下的比较更贵。
- 结论：默认关闭，未推广。
- 证据：`docs/transport_concurrency_optimization_20260916.md`，
  `benchmark/transport_concurrency_20260916/candidate_screen_summary.json`。

早期只增加2位尾数的分析也显示：现有指数索引的候选区间中位数已为1，平均比较1.31次；
扩大约4倍索引仅降到1.13次，因此未实施。它属于实现前停止，不算独立运行候选。

### B3. 两条密度曲线共享精确搜索

- 目标：共享已验证相同网格的node/cubic区间定位，同时保留两套系数和原运算顺序。
- 结果：真实包节点/边界/随机采样逐位一致。5次正式RT07575配对的Elapsed平均仅+0.59%，
  95%区间−0.90%–2.09%；次级−0.01%，远低于5%门槛。
- 结论：研究开关保持关闭，未做硬件推广采样。
- 证据：`docs/transport_concurrency_optimization_20260916.md`，
  `benchmark/transport_concurrency_20260916/shared_search_analysis.json`。

### B4. 跨步sticky区间缓存

- 目标：优先复用上一步node/cubic索引，减少相邻能量步的搜索。
- 结果：审计一致，但续跑状态272→304 B；RT07575 Elapsed约慢9%，原发和次级均退化。
  hint状态和命中/回退分支超过已有窄索引搜索成本。
- 结论：回退，不重启同类跨步索引缓存。
- 证据：`docs/rejected_approaches.md` §6。

### B5. 缓存EM覆盖能区端点

- 目标：材料/密度选择时缓存 `covers()` 所需能区交集，避免每步读取端点。
- 结果：1,294,650次真实包查询等价；状态结构仍64 B、续跑状态仍272 B，但这不证明寄存器
  不变。两轮候选Elapsed约改善0.57%/0.60%，原发约改善3%，次级反而慢约0.8%；收益太小
  且候选轮次不足。
- 结论：未推广。
- 证据：`docs/em_cost_evaluation_20260916.md` 第五轮，
  `docs/rejected_approaches.md` 末节。

### B6. 旧式区间缩窄

- 目标：用类似 `CARBON_EM_EXACT_INDEX` 的窄区间减少依赖比较。
- 结果：端到端约−1.1%。
- 结论：该具体实现回退；后来被不同、完整验收的精确索引实现取代，不能混为同一候选。
- 证据：`docs/rejected_approaches.md` §6。

### B7. 次级重分组加入出生CT section

- 目标：在 `(species, energy)` 之外增加section，提高同warp材料表局部性和有效lane。
- 结果：质量通过、零overflow、剂量处于原子顺序噪声内，但次级约慢3%。粒子沿程跨材料
  削弱出生section相关性，同时增加CT材料读取和26倍bucket原子操作。
- 结论：回退，保留已接受的 `(species, 16 energy buckets)`。
- 证据：`docs/rejected_approaches.md` §6。

### B8. 独立/查表化受限涨落采样

- 目标：用专用sampler、分支分组或compound-Poisson表减少分歧和依赖等待。
- 结果：隔离sampler可快约8%–13%，但完整次级候选Elapsed 27.645→28.060 s、次级
  14.731→15.075 s；额外表、状态和缓存竞争抵消局部收益。更完整的请求/采样/继续队列
  还会在16步续跑中增加多轮状态写回和launch；真实输入140万次采样本身仅约1.2 ms，
  不足以形成5%端到端收益。
- 结论：停止，不扩展到原发。
- 证据：`docs/em_cost_evaluation_20260916.md`，
  `benchmark/runtime_breakdown_20260914/FLUCT_QUEUE.md`，
  `docs/transport_concurrency_optimization_20260916.md`。

### B9. macro/grouped micro共享尾部

- 目标：减少每tick重复准备和尾部逻辑，从而缩短依赖链。
- 结果：精确的grouped micro在水中与K1近似同速、RT07575更慢；codegen/寄存器压力曾使
  K1路径额外慢约5%。非精确macro虽快，但引入不可接受的Bragg峰偏差。
- 结论：共享tick路径关闭；只保留逐tick精确执行。
- 证据：`docs/em_macro_step.md`，`docs/rejected_approaches.md` §3–5。

### B10. 降低次级直接跑完尾部阈值

- 目标：把 `secondary_tail_threshold` 从8192降到2048或512，让较小活动队列继续经过
  续跑和压紧，尝试提高尾段有效lane及设备利用率。
- 结果：轮次归因显示直接跑完轮次和全部低填充轮次合计不足次级kernel的1%；2048使
  Elapsed慢约0.6%，512慢约1.5%，剂量不变。matched eligible、active warp和warp/SM未测。
- 结论：保持8192；不再用增加小队列launch的方式修复eligible。
- 证据：`docs/rejected_approaches.md` §6。

## C. 性能有效但没有修复eligible/warp/SM的改动

以下不是失败的性能改动，但不得引用为“eligible或occupancy已经改善”的证据。

### C1. 跳过零审计原子、64路审计分片、续跑审计状态裁剪

- 合并后相对旧基线Elapsed约减少15.3%，次级约减少24.6%；次级包装寄存器205→186。
- 但2080 Ti实际warp/SM仍约7.9，eligible仍约0.08–0.09，没有跨过168寄存器档位。
- 后续1/32/128/256/1024路扫描表明64路已在平台区；改变分片数不再修复eligible。
- 证据：`tobeoptimized.md` §1、§4，`docs/transport_concurrency_optimization_20260916.md`。

### C2. 16步次级续跑与稳定压紧

- RT07575端到端约提高4.6%–4.7%，机制是更频繁压紧存活索引、改善后续启动中的线程
  填充；它不改变物理步长。
- 当时没有候选matched NCU证明eligible提高；最新硬件采样仍显示各阶段eligible很低。
- 证据：`docs/secondary_scheduling_latest.md`、`tobeoptimized.md` §1和§4。

### C3. `(species, energy)` 次级重分组

- 该方案保留在生产中，改善了warp内物种/能区一致性；历史active lane变化只能作为支持性
  证据。采样构建、block和寄存器布局不同，不能据此单独归因eligible或warp/SM提升。
- 证据：commit `b69ac98`，`tobeoptimized.md`。

## D. 未进入实验的提案

仓库还提到“每线程双轨迹”和不限定边界的“大规模拆核”。现有记录没有对应隔离实现、
二进制或A/B结果，因此它们是未实施提案，不计作失败实验。次级非弹性冷路径的具体拆分
诊断已单列为A3。若将来实现这些提案，应作为新候选记录，不能沿用本台账中的失败结论。

## E. 已定位但尚未解决的根因

- 当前内核受寄存器驻留限制，原发172、次级186尚未跨过168门槛。
- NCU报告真实local-memory流量：历史代表采样中，local load sectors约占原发总
  `(global+local)` load的39%、次级33%。
- SASS静态统计曾记录原发统一EM入口88条LDL/1326条STL、约2816 B栈；次级505条LDL/
  859条STL、约2080 B栈。这说明线程私有local memory参与等待，但不能只凭栈大小判断spill。
- long-scoreboard由EM表、核反应率、CT/剂量访问和线程私有local memory共同造成；现有证据
  不支持把某个比例直接当作模块耗时。

## F. 后续追加规则

从本文件建立后，任何以以下任一指标为目标、但未通过推广门槛的实验，必须在任务结束前
追加到本文件：寄存器/线程、理论或实际warp/SM、active/eligible warp、occupancy、
long-scoreboard、local-memory事务、block共驻或以此为目的的状态/live-range重构。

每条新记录至少包含：

1. 日期、候选ID、源码commit/工作树哈希、GPU和架构；
2. 目标机制与实际启动内核名称，不能只写具名device函数；
3. baseline/candidate的寄存器、block、理论驻留、实际active/eligible warp、long-scoreboard和
   local-memory指标；没有测量的字段写“未测”，不得用推断补数；
4. 无profiler端到端、原发和次级配对结果，以及重复次数/波动范围；
5. 审计、能量、overflow和3D剂量结果；
6. 最终状态：完全回退、默认关闭、实现前停止或被其他方案取代；
7. 允许重试所需的新证据。没有新证据时不要重复同一方案。

本文件采用追加历史。发现旧记录错误时新增“更正”条目并指向原记录，不静默删除失败史。

## G. 建档后的追加记录

### 2026-09-16 `resume_fieldwise_reference`：删除pause/resume外层整结构体临时量

- 基线：campaign冻结源码，基线二进制SHA256
  `1202cd92c2ae54f129a53e6207ade7018729ecb130848a70df7641575189dfca`；RTX 2080 Ti/sm_75。
- 目标：把 `const auto saved = resume_states[state_idx]` 和局部
  `SecondaryResumeState saved` 改为直接引用目标元素并逐字段读写，尝试减少线程local frame
  及local-memory流量。恢复后继续重绑 `unified_secondary_state.tables`。
- 实际入口：`__pf_kernel_wrapper<CarbonSecondaryTransportKernel<1>>`。
- 资源结果：包装入口保持186寄存器、1840 B栈、122 LDL/136 STL；具名函数保持178寄存器、
  2080 B栈，505/859变为506/858 LDL/STL。active/eligible warp、long-scoreboard及动态local
  sectors未测，因为结构门槛已失败。
- 性能和正确性：未测；按“SASS和资源不改善即停止”结束，没有进入计时或推广验证。
- 结论：完全回退，生产源码未改。编译器原本已消除外层聚合临时量；没有新codegen证据时
  不重试同一改写。
- 新证据：已有SourceCounters显示次级各阶段约78%–84%的逐指令可归因local sectors来自
  包装入口的 `LDC.U8 → STL.U8` lambda捕获物化，下一候选应缩小实际包装层捕获对象。
- 证据：`docs/transport_local_memory_attribution_20260916.md`，
  `benchmark/local_memory_attribution_20260916/analysis.json`。

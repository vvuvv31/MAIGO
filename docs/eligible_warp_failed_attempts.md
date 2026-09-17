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

### 2026-09-16 `opaque_lambda_device_closure`：把编译器lambda对象复制到device

- 基线：commit `d4e579a`后的隔离工作树，RTX 2080 Ti/sm_75；实际入口为
  `__pf_kernel_wrapper<CarbonSecondaryTransportKernel<1>>`。
- 目标：把约2 KB的lambda closure整体上传到device，wrapper只capture该指针并调用原
  `operator()`，验证能否绕开constant参数到每线程local stack的逐字节物化。
- 资源结果：wrapper从186 registers、1840 B stack、1984 B constant0变为182 registers、
  232 B stack、约376 B constant0。active/eligible、long-scoreboard和动态local sectors未测，
  因为正确性门槛先失败。
- 正确性：次级粒子数量、统一EM整数审计和能量记账分叉，production quality失败；端到端
  性能和3D剂量未进入正式测量。
- 结论：完全回退。编译器生成的lambda closure类型不能作为跨host/device复制的稳定显式ABI。
- 允许重试所需证据：只有后端明确保证该closure对象表示和device copy/call语义时才可重开；
  当前替代方案是手写trivially-copyable POD context。
- 证据：`docs/secondary_context_abi_20260916.md`，
  `benchmark/secondary_context_abi_20260916/analysis.json`，
  `scratch/secondary_context_abi_20260916/dump_candidate/resources.txt`。

### 2026-09-16 `immutable_context_with_round_pointers`：续跑热指针放入只读context

- 基线和入口同上一条。
- 目标：用显式POD替代opaque lambda closure，但把 `resume_states`、`resume_ready`、
  `active_order` 和 `keep` 一并保存在每generation上传一次的context中。
- 资源结果：大closure物化消失，隔离wrapper约180 registers、232 B stack、376 B
  constant0；active/eligible、long-scoreboard和动态local sectors未测。
- 正确性：第0轮和第一次压紧完全匹配；第一次恢复后active count分叉，基线round 2为
  5,841,290，候选为3,929,615，随后审计与能量记账失败。性能和3D剂量未进入正式测量。
- 结论：完全回退。续跑和压紧指针必须留在per-round显式capture中，不能经一次上传的
  immutable transport context访问。
- 允许重试所需证据：仅当context在每次指针交换后重新上传并证明其成本低于显式capture时
  才可重开；现有小closure实现已无此需要。
- 证据：同上一条，以及 `scratch/secondary_context_abi_20260916/diag_pod/run.log`。

### 2026-09-16 `secondary_context_pointer`：结构成功但未通过warp/SM推广门槛

- 候选：`CARBON_SECONDARY_CONTEXT_POINTER=ON`；正式campaign源码工作树SHA256
  `0d45cd79a23a2e55375cd3e7d78f928144259c590f965f931c67be1640a6e776`，干净 `d4e579a`
  提交版源码SHA256 `be3d714da4b9579422199255559518d1b8e2dafb02e675c93784a95c975aac35`；
  RTX 2080 Ti/sm_75。
  实际入口为 `__pf_kernel_wrapper<CarbonSecondaryTransportKernel<1>>`。
- 机制：immutable物理/几何/计分参数经一个device POD指针访问；续跑热指针和round标量使用
  显式capture；closure具有trivially-copyable和不大于64 B的编译期契约。
- 结构结果：wrapper constant0 1984→408 B、每线程stack 1840→264 B；但registers
  186→187，32线程时寄存器限制仍为8 blocks/SM，没有跨过168门槛。
- 三次matched NCU中，次级早段eligible 0.0862→0.1030、ready fraction 4.42%→5.31%、
  long-scoreboard 10.29→5.73、local sectors降低39.90%；中段和第二代也提高eligible并降低
  long-scoreboard/local sectors。晚段eligible提高20.08%，但自动block 64→128，active warp
  1.296→0.995，local sectors增加186.02%。次级各阶段active warp均未提高。
- 五次无profiler正式配对：wall中位改善8.04%，Elapsed改善8.64%，次级改善19.97%，原发
  改善0.01%。五对Elapsed改善范围8.52%–9.07%。
- 正确性：3,240,963 histories的续跑active count和8项整数审计一致，quality通过，
  overflow=0，能量误差 `3.6833317e-06`；五次3D剂量均值差在五次基线全局波动包络内。
- 结论：候选保留但默认关闭。它成功修复主要的closure local-memory与readiness问题，却未
  满足最终组合要求的active warp/驻留提升；硬件门槛失败后未继续RT06423和20022516推广回归。
- 允许推广所需证据：在此小closure入口上继续删除body live state，使实际wrapper达到不高于
  168 registers，并在匹配阶段观察到active warp增加；然后重新扫描block并完成全部回归门槛。
- 证据：`docs/secondary_context_abi_20260916.md`，
  `benchmark/secondary_context_abi_20260916/analysis.json`。

### 2026-09-16 更正：`secondary_context_pointer`不是吞吐失败候选

- 更正上一条的采用解释：168 registers/thread和active warp提高仍是occupancy研究目标，但
  不再是接受吞吐优化的必要条件。候选在所有匹配次级阶段提高eligible和ready fraction，
  RT07575五对Elapsed中位改善8.64%、次级速度改善19.97%，已经修复原问题中的readiness和
  local-memory部分。
- 当前状态：候选保留、默认关闭；RTX 2080 Ti上的RT07575、RT06423、20022516和50k水模
  正确性及吞吐回归通过，occupancy目标未达到，A6000推广待完成。晚段active下降属于需以
  尾部总耗时加权调查的硬件现象，实测两个直接完成轮次仅占次级时间约0.32%。
- 编译兼容修复：context计分指针恢复为`DepthAtomicT*`/`DoseAtomicT*`；通用EM实例从context
  读取运行时模式；context纳入`DeviceMemoryTracker`。sm_75上EM特化ON/OFF × FP32/FP64四种
  组合均完成设备编译链接，FP32固定分片和其余三种GPU smoke均quality通过、overflow为0。
- RTX 2080 Ti推广：RT06423/20022516五对Elapsed分别改善7.73%/6.50%，次级改善
  19.74%/19.11%；50k水模10对Elapsed中位改善1.35%，最差−1.15%。三组3D剂量均值差均在
  各自基线包络内，审计一致且overflow=0。
- 允许默认推广所需证据：A6000结果需在可用主机上补测。是否跨过168档位作为机制结果单独
  报告，不作为唯一否决条件。
- 证据：`docs/secondary_context_abi_20260916.md`，
  `benchmark/secondary_context_abi_20260916/analysis.json`，
  `scratch/context_compat_validation_fp32_specialized_20260916/run.log`。

### 2026-09-16 `secondary_explicit_nd_range`：小closure具名入口

- 基线：修复编译兼容后的`CARBON_SECONDARY_CONTEXT_POINTER=ON` range入口；RTX 2080 Ti/
  sm_75。候选用显式`nd_range`和越界早退，分别固定32、64、128线程；物理循环、有效粒子
  range和RNG身份不变。
- 实际入口：基线为
  `__pf_kernel_wrapper<CarbonSecondaryTransportKernel<1>>`；候选为具名
  `CarbonSecondaryTransportKernel<1>`。补齐线程在任何续跑数组访问前返回。
- 资源结果：基线wrapper为187 registers、264 B stack、408 B constant0；候选具名入口为
  170 registers、480 B stack、400 B constant0，三种block相同。仍未跨过168档位。动态
  active/eligible、long-scoreboard和local sectors未测，因为无profiler吞吐stop gate已失败。
- 单次完整筛选：range control Elapsed/secondary为20.229883/8.631476 s；32、64、128候选
  分别为21.636933/10.102255、21.832124/10.129329、21.891748/10.128049 s。按
  `baseline/candidate-1`，端到端退化6.50%–7.59%，次级速度退化14.56%–14.79%。三个尺寸
  方向一致且远超重复波动，未进入五对正式计时。
- 正确性：四版均为3,240,963 histories、1,034,976,717 steps、1,344,312次核相互作用；
  整数审计一致，quality通过，overflow=0。3D剂量未做五次统计，因为性能门槛先失败。
- 尾部归因：range control两个直接完成轮次合计0.027502 s，占次级时间0.32%；不能把单个
  晚段occupancy样本与稳态阶段等权作为context-pointer候选的否决依据。
- 结论：候选保留默认关闭，不进入组合。只有后端版本变化、具名入口stack显著下降或出现
  与当前完整计时相反的新codegen证据时才重试。
- 证据：`docs/secondary_ndrange_entry_20260916.md`，
  `benchmark/secondary_ndrange_entry_20260916/analysis.json`。

### 2026-09-16 `secondary_projectile_index_reuse`：延长物种索引live range

- 基线：兼容修复后的context-pointer range入口，RTX 2080 Ti/sm_75；实际入口为
  `__pf_kernel_wrapper<CarbonSecondaryTransportKernel<1>>`。
- 目标：复用continuation入口已计算的`(Z,A) → registry index`，删除步内hazard、He-4审计和
  步后碰撞处的重复身份映射；能量、材料、密度、rates和partials仍逐步精确查询。
- 资源结果：wrapper从187增至228 registers/thread；stack保持264 B、constant0保持408 B。
  active/eligible、long-scoreboard和动态local sectors未测，因为正式配对没有吞吐收益。
- 五次交错配对：Elapsed中位变化−0.065%，范围−0.478%至+0.536%；次级速度中位变化
  +0.232%，范围−0.659%至+0.897%。改善定义为`baseline/candidate-1`，结果处于重复波动内。
- 正确性：3,240,963 histories、1,034,976,717 steps、1,344,312次核相互作用逐次匹配；
  整数审计一致、quality通过、overflow=0。五次3D剂量均值差为峰值0.00001705%，在本轮
  五次基线0.00002558%全局波动包络内。
- 结论：默认关闭，不进入组合。除非编译器codegen变化后索引复用不再增加寄存器，或SASS
  证明身份映射成为新的动态热点，否则不重试。
- 证据：`docs/secondary_projectile_index_reuse_20260916.md`，
  `benchmark/secondary_projectile_index_reuse_20260916/analysis.json`。

### 2026-09-16 更正：`secondary_production_specialize`不是失败候选

- 隔离`b3b906d`源码排除dense/shared/fine EM和步长研究改动后，实际生产wrapper从context通用
  实例的177降到164 registers/thread，stack均为352 B，首次跨过sm_75的168寄存器档位。
- RT07575五次正式配对：Elapsed中位改善3.92%（3.41%–4.59%），次级中位改善9.69%
  （8.71%–10.53%）；审计一致、quality通过、overflow=0，3D剂量差不超过基线重复包络。
- 一次匹配NCU中，次级早/中/第二代active warp/SM分别从7.76/7.83/7.74提高到
  11.51/10.65/10.61，eligible/scheduler分别从0.0993/0.0950/0.0978提高到
  0.1209/0.1049/0.1099。晚段仍受tail underfill限制。
- ready fraction在稳态阶段下降，long-scoreboard上升；该候选通过增加驻留warp隐藏延迟，
  没有减少单warp等待。动态结果目前每阶段仅一次采样，推广前仍需重复。
- 结论：从失败列表更正为“结构和次级吞吐成功、默认推广待完成”。候选保持默认关闭，因为
  单项端到端收益不足5%，且A6000、跨病例组合回归和重复硬件采样尚未完成。
- 证据：`docs/secondary_production_specialization_20260916.md`，
  `benchmark/secondary_production_specialization_20260916/analysis.json`。

### 2026-09-16 `primary_production_specialize`：资源与吞吐成功，严格剂量包络失败

- 目标：将固定统一EM + Schneider CT生产配置在host-dispatch层特化，删除通用原发kernel中
  不可达的诊断、all-elastic、额外voxel clamp和旧物理路径。
- 资源结果：实际非offset入口从170降到162 registers/thread，stack从2864降到2848 B，跨过
  sm_75的168寄存器驻留档位。
- 十次交错配对：wall中位改善5.20%，Elapsed改善5.70%，原发kernel改善13.83%；十对Elapsed
  改善范围5.00%–6.65%。次级中位变化+0.48%，处于波动范围。
- 输运审计：每次3,240,963 histories、1,034,976,717 steps、1,344,312次核反应；八项统一EM
  整数审计一致、quality通过、overflow=0。
- 失败门槛：候选最大3D dose差为峰值`0.00005684%`，十次控制重复包络为`0.00004618%`；扩大
  到十次后仍未覆盖。差异量级约一个FP32原子ULP，但既定严格包络不允许据此豁免。
- 结论：默认关闭，不进入组合、不做跨病例推广。若将来改变确定性dose归并方法，可在新基线
  重新测量；不得沿用本轮性能结果直接推广。
- 证据：`docs/transport_occupancy_followup_20260916.md`，
  `benchmark/transport_occupancy_followup_20260916/analysis.json`。

### 2026-09-16 `secondary_known_species/non_he4`：compile-only上界，尚无安全调度

- known-species删除generic recoil后，实际164-register wrapper降到155；再删除non-He4路径中的
  `he4_audit[6]`、三点rate查询和续跑保存/恢复后降到145。stack均为352 B。
- 145达到继续研究路径级拆分的结构门槛，但仍未到约128的下一明显驻留档。
- 当前species×energy grouping只生成一个全局`secondary_order`；continuation state和compaction
  覆盖整个generation，没有按bucket子范围隔离。直接启用probe会让He-4审计语义缺失，也不能
  安全容纳未来all-elastic generic recoil配置。
- 结论：两个probe默认关闭，不进行正式计时、不进入组合。只有完成known non-He4、He-4和
  unknown/recoil的独立bucket范围调度，并逐粒子验证RNG/续跑/审计后，才可转成运行候选。
- 证据：`docs/transport_occupancy_followup_20260916.md`，
  `benchmark/transport_occupancy_followup_20260916/analysis.json`。

### 2026-09-17 `em_empty_bucket_fast_path`：真实输运零命中

- 目标：当exact exponent index的`begin==end`时直接确定`cursor-1`区间，跳过dependent node/segment key load。
- 正确性：82,741,572个GPU节点、边界及相邻浮点查询逐位失败为0。
- 动态结果：完整RT07575中node/raw0/raw1/raw2分别执行7.163B/4.933B/4.933B/0.531B次bounds查询，空桶命中全部为0。
- 结论：默认关闭，不计时、不进入组合。除非EM网格或指数索引构建改变并出现非零真实命中率，否则不重试。
- 证据：`docs/transport_exact_species_followup_20260917.md`，`benchmark/transport_exact_species_followup_20260917/analysis.json`。

### 2026-09-17 `secondary_hot_species_specialize`：低寄存器slice被多launch成本抵消

- 结构结果：exact proton/deuteron实际wrapper为124/124 registers，跨过约128档；He-4为130，未进入专用调度。stable compaction后按有序state index恢复两个热点物种边界，粒子/RNG/resume身份不变。
- 五次配对：secondary中位退化4.47%，Elapsed退化1.86%；三路launch和slice尾部成本超过低寄存器收益。
- 正确性：3,240,963 histories、1,034,976,717 steps、1,344,312次核反应和八项整数审计一致，quality通过、overflow=0；候选3D剂量差也略超过本轮五次控制包络。
- 结论：默认关闭，不进入组合。只有后端支持无额外launch的单kernel物种静态分派，或专用slice覆盖更高且launch成本显著下降时才可重试。
- 证据：同上。

### 2026-09-17 `subgroup_cooperative_em_search`：地址共享诊断未达实现门槛

- 诊断：4,000,000条实际comparison地址中，node/raw0/raw1每warp查询组平均约18.4个unique address；约42k组中仅4–6组不超过4个unique。raw2平均7.72 unique，但动态占比较低。
- 结论：没有实现leader load+broadcast。当前地址重复度不足以覆盖ballot、shuffle、分歧和额外live state成本。
- 允许重试所需证据：新的排序使热点node/raw查询大多数warp稳定低至1–4个unique address。
- 证据：同上。

### 2026-09-17 `em_pair_prepare`：lo/hi提前物化筛选退化

- 候选：在`UnifiedEmState::prepare`中先物化两个density endpoint，再调用两个独立prepare，尝试增加memory-level parallelism。
- 资源：实际wrapper保持150 registers/thread、352 B stack。
- 单次完整筛选：secondary 7.9114→8.1555 s，Elapsed 19.7554→20.0897 s，方向明显退化，未进入五次正式配对。
- 结论：默认关闭。除非SASS证明后端能同时发出两个endpoint load且不增加等待/live state，否则不重试。
- 证据：同上。

### 2026-09-17 `primary_production_specialize`同一二进制复核

- 同一binary内通过诊断环境变量交错选择generic与specialized，排除了不同链接产物作为剂量差来源。
- 五次配对再次得到primary中位+13.76%、Elapsed +5.97%，但FP32 3D剂量最大差0.00005684%峰值，超过generic五次包络0.00004263%。
- FP64 scorer机制诊断的generic/specialized输出逐位相同，说明FP32差异来自并发原子累加顺序；既定FP32包络门槛不豁免。
- 结论：继续默认关闭，除非引入并验收确定性剂量归并或重新定义正式剂量统计门槛。
- 证据：同上。

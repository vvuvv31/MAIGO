# MAIGO 性能优化失败路线台账

本文件是性能候选的去重入口。开始任何新实验前，必须先搜索本文件及同义关键词。
已经列出的路线不得原样重试；只有满足“允许重试条件”时，才能建立新的隔离候选。

统一约束：GPU 性能运行固定使用 FP32 dose scorer。FP32 原子累加顺序造成的微小剂量差异
不再单独否决调度候选，但 histories/steps、Unified EM 整数审计、quality、能量记账和 queue
overflow 仍必须通过。FP64 性能路线由 `AGENTS.md` 明确禁止。

## Intel Arc B580（2026-09-17）

| 路线 | 结果/失败原因 | 状态与允许重试条件 |
|---|---|---|
| Secondary segment 8 | Arc暖运行 rounds 76/59，secondary 20.20 s、Elapsed 46.15 s、140.5k/s，明显慢于16步基线 | 已回退；除非 kernel 状态/寄存器布局改变后重新全扫 |
| Secondary segment 32 | Arc暖运行 rounds 20/16，secondary 19.35 s、Elapsed 45.28 s，落在16步基线波动内且无稳定收益；历史配对16步更优 | 已回退；除非 kernel 状态/寄存器布局改变后重新全扫 |
| Secondary segment 64 | 比16步更慢：暖运行约45.47–45.49 s，secondary约19.17 s；减少轮次不足以抵消存活lane分歧 | 已回退；除非 kernel 状态/寄存器布局改变后重新全扫 |
| Primary work-group 512 | 相对默认256没有独立稳定收益 | 不重试；只有寄存器档位或SIMD宽度改变后重扫 |
| Primary超大chunk（2,228,224或整分片单次launch） | Arc完整RT07575扫描中，2,228,224仅3次launch但primary约12.71 s、吞吐约155.3k/s；整分片单次launch约12.40 s、156.3k/s，均不如1,114,112的约11.70–11.89 s、吞吐中位160.9k/s。减少launch后设备内尾部/负载不均重新占优 | 已回退；保持Arc默认1,114,112。只有primary kernel资源、设备CU数或任务规模显著改变后才重扫 |
| EM empty-bucket fast path | 与primary specialization组合后没有可测收益；真实输运空桶命中为0 | 永久停止，除非EM索引/数据包构建方式改变并产生非零空桶 |
| Secondary production specialization（Arc复测） | 放宽FP32差异后重新交错A/B；generic secondary 19.28/19.12 s，specialized 19.62/19.61 s，稳定慢约0.34–0.48 s；同时约119/2.86B EM draws发生代码生成漂移 | 已从Intel preset回退；只有kernel主体或编译器发生实质变化后可重试。CUDA历史正收益不外推到Arc |
| 每4轮按当前 `(species, energy)` 动态重分组 | 两对交错结果：Elapsed改善约0.32%和0.02%，没有稳定收益；排序kernel约0.145 s | 已回退；除非分组可融合进已有compaction且不增加独立histogram/scatter |
| 每8轮动态重分组 | 排序成本较低，但能量顺序退化，Elapsed 45.14 s、secondary 19.26 s，不优于4轮或基线 | 已回退；同上 |
| `noinline` 拆出EM loss | Primary spill 5312→8384 B，kernel变慢，审计也变化 | 永久停止此具体拆法；只有新的live-range证据才可拆分其他边界 |
| 强制256 GRF（环境变量） | spill降为0、primary约快5%，但每次运行重新JIT 225–257 s | 禁止生产使用；只有AOT且不触发重复JIT时可建立新候选 |
| kernel `grf_size<256>` 属性 | 与环境变量相同：每次重新JIT且审计变化 | 同上 |
| FP64 dose scorer | 用户明确禁止；会极大拖慢性能开发 | 不得构建/运行/建议，除非用户在当次请求明确覆盖 |

## 调度、并发与occupancy失败路线

| 路线 | 结果/失败原因 | 允许重试条件 |
|---|---|---|
| 显式 `nd_range` 32/64/128/256 | RTX 2080 Ti整体慢约1.1%–1.8%；Arc固定WG扫描也无稳定收益 | kernel寄存器档位或目标GPU架构改变后重新扫描 |
| Hot-species三路slice（p/d/fallback） | p/d降至约124 registers，但三次launch和slice尾部使secondary退化4.47% | 单次launch内能静态分派不同物种资源预算时 |
| Hot-species slice去同步 | 去掉中间wait只追回约0.7个百分点，候选仍退化3.79% | 同上；不要把通用in-order异步清理与此失败候选混为一谈 |
| p/d continuation 32/64、fallback 16 | rounds仅36→30，fallback决定总轮数；低填充尾部使secondary退化7.54% | 能同时减少fallback轮数并维持填充率时 |
| H(p+d) class kernel | 139 registers，高于清理后的fallback 136，未跨过≤128档 | H class实际wrapper降到≤128且覆盖率足够高时 |
| Tail threshold 2048/512 | 低填充尾部合计不足secondary的1%；分别使Elapsed慢约0.6%/1.5% | 工作负载尾部占比显著变化时 |
| Tail threshold 32768（Arc） | rounds 39/30→34/26，但secondary均约19.56 s；更长直接跑完尾部抵消launch节省 | 已回退到8192；工作负载尾部占比显著变化时 |
| Secondary compaction prefix并行化 | 单线程prefix仅约0.04 ms；额外kernel/同步成本更高 | resume group数量或prefix占比显著增长时 |
| Primary初始化独立kernel | Elapsed约退化0.48%，新增状态流量和launch抵消收益 | 初始化计算量显著增加且可复用时 |
| A6000 MPS/多进程共驻 | 单进程已受寄存器和访存等待限制，多进程未改善单任务吞吐 | 目标变为多任务总吞吐而非单任务时 |
| 出生CT section加入次级分组 | 次级约慢3%；粒子沿程跨材料削弱相关性，bucket原子增加 | 能按当前材料低成本重排且有地址复用证据时 |
| He-4 sparse audit relocation作为提速 | 寄存器150→136、He-4 130→127，但secondary变化仅+0.008% | 仅保留为资源清理；不得宣称吞吐优化 |
| Known-species / non-He4 probes直接生产使用 | 只是compile-only资源上界，没有安全运行时分流 | 有低launch、安全保持粒子/RNG身份的调度后 |

## 寄存器与live-range失败路线

| 路线 | 结果/失败原因 | 允许重试条件 |
|---|---|---|
| `maxrregcount=128` | spill/local-memory流量增加，未改善eligible | 热状态显著缩小后重新评估，不直接强压寄存器 |
| 直接插值字段代替完整 `UnifiedEmNode` 临时量 | 没有跨过寄存器驻留档，吞吐无稳定收益 | 编译器代码生成或节点结构改变后 |
| 拆分次级非弹性冷路径 | 调用边界和状态保存增加，未改善实际hot wrapper | profiler证明冷路径仍长时间live时 |
| 编译期关闭生产未用计分分支 | 有资源下降但未跨档/收益不足 | 新增可删除分支足以跨寄存器档时 |
| Delta确定性参数提前合成 | live range延长或收益为零 | 有新的SASS/IGC live-range证据时 |
| 核反应率total-only接口 | 没有形成端到端收益 | 查询结构或表布局改变后 |
| Philox四输出缓冲 | 增加持久状态，吞吐约零或负收益 | RNG成为独立硬件热点时 |
| Fieldwise resume复制 | 删除整结构临时量未改善实际wrapper资源/吞吐 | 编译器重新出现整结构复制时 |
| Opaque lambda closure搬到device | 额外全局加载，未解决hot状态 | closure再次成为主要local-memory来源时 |
| Immutable context加入round pointers | 扩大context访问，收益不足 | 指针payload重新成为ABI或寄存器瓶颈时 |
| Projectile index reuse | 延长物种索引live range，没有稳定收益 | species查询成本显著上升时 |

## EM查表与memory-latency失败路线

| 路线 | 结果/失败原因 | 允许重试条件 |
|---|---|---|
| EM search key分离 | 新数组访存抵消结构体字段节省 | 数据布局/缓存行为改变后 |
| 更细exact index | Elapsed约退化5.19%，额外索引访存超过比较节省 | 每bucket候选宽度显著增大时 |
| 共享两条密度曲线搜索 | 正式配对仅约+0.59%，低于推广门槛 | 能删除更多dependent loads时 |
| Sticky跨步区间缓存 | resume state 272→304 B，Elapsed约慢9% | hint无需进入续跑状态时 |
| 缓存EM覆盖能区端点 | Elapsed仅约+0.6%，secondary反而慢约0.8% | covers读取成为新热点时 |
| 旧式区间缩窄 | 端到端约退化1.1% | 不重试；已被现有exact index取代 |
| Empty-bucket直达 | node/raw0/raw1/raw2真实命中全部为0 | 数据包产生显著空桶后 |
| Packed-pivot/single-knot bucket | 真实查询全部宽度≥4，没有单knot桶 | 索引构建改变后 |
| Subgroup cooperative EM load | warp内node/raw地址平均约18.4个unique，广播成本无法摊薄 | 热点查询稳定降至每warp 1–4个unique地址 |
| `em_pair_prepare` | wrapper寄存器不降，secondary 7.91→8.16 s | SASS/IGC证明lo/hi加载能并行且不延长live state时 |

## EM采样与物理步合并失败路线

| 路线 | 结果/失败原因 | 允许重试条件 |
|---|---|---|
| Delta batching（Ts split） | 批处理/状态/launch成本抵消采样收益 | 采样成本成为端到端显著占比时 |
| Delta quantile tables | 内存和插值成本高，精度/性能未达到推广条件 | 新表格式同时降低带宽与误差时 |
| K-tick macro step | 虽快但破坏Bragg峰/精度 | 不重试非精确macro |
| Adaptive macro | 高低能切换仍有不可接受物理偏差 | 新的正式物理验收模型出现时 |
| Grouped micro/shared tail | 水中近似持平，RT07575更慢，codegen/寄存器压力增加 | 能减少状态且保持逐tick精确时 |
| 独立或查表化受限涨落采样 | 隔离sampler快8%–13%，完整secondary却变慢；采样本身约1.2 ms | 完整请求/恢复可零额外状态与launch时 |
| Universal conditional-sum lookup | 未达到端到端与精度推广门槛 | 数据和正式误差界重新设计后 |
| Joint Universal compound-Poisson lookup | 集成成本/精度不满足要求 | 新模型完成正式物理验证后 |
| 通用RNG/ALU微优化 | 多数约零或负收益，当前为访存等待主导 | profiler证明ALU/RNG成为主瓶颈时 |

## 已成功或仍可继续的路线（不得误记为失败）

- Primary production specialization：吞吐和资源均为正收益；此前仅因严格FP32 dose包络被停止，
  当前项目规则已明确不再以该差异单独否决。
- Secondary context pointer：解决closure/ABI并有吞吐收益，不是失败候选。
- Secondary production specialization在CUDA历史实验中有正收益；只有当前Arc组合因EM审计变化被隔离。
- 16步secondary continuation和稳定compaction：已验证正收益，是当前基线。
- `(species, 16 energy buckets)`出生分组：当前生产基线，不等同于已失败的动态重分组。
- 64路EM audit shard与跳过零审计原子：已合入的有效优化。

## 追加规则

每个新失败候选必须在任务结束前追加以下信息：日期、平台、基线、唯一改变量、至少一次完整
运行结果、正确性门槛、回退状态和允许重试条件。不得只写“变慢”或只保存临时日志。
详细历史证据仍保存在：

- `docs/eligible_warp_failed_attempts.md`
- `docs/rejected_approaches.md`
- `docs/transport_concurrency_optimization_20260916.md`
- `docs/transport_occupancy_followup_20260916.md`
- `docs/transport_exact_species_followup_20260917.md`
- `docs/transport_scheduling_audit_followup_20260917.md`

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

### NVIDIA RTX 2080 Ti / sm_75（2026-09-17）

| 路线 | 结果/失败原因 | 允许重试条件 |
|---|---|---|
| 增大CUDA primary chunk（139,264 / 278,528 / 1,114,112 / 整分片6,481,909） | 当前`352fa95`、FP32 dose、完整RT07575 6,481,909-history shard单次扫描；默认34,816的两次正式夹测为150.92/149.83k histories/s、primary 17.27/17.31 s。四个候选依次为148.94/146.79/146.41/145.07k/s，primary 17.66/18.10/18.48/18.65 s；均为quality pass、统一EM整数审计一致、能量记账通过、queue overflow=0。减少187次primary launch未抵消sm_75设备内尾部/负载不均，已恢复`history_chunk_size: 0`（CUDA选择34,816） | 已回退；只有primary kernel资源/代码生成、CUDA驱动或目标GPU架构显著改变后才重扫。单次候选用于拒绝，不作为精确排序认证 |

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

## Copper minibeam 精度候选（2026-09-17）

| 路线 | 结果/失败原因 | 状态与允许重试条件 |
|---|---|---|
| Copper 步长从 0.25 mm 缩到 TOPAS 的 0.05 mm | RTX 2080 Ti、250 MeV/u、256k histories；water-primary 仅从 58,344 降到 51,609，仍远高于 TOPAS 相空间的 30,878，touched-primary 均值 756.6 MeV 仍远低于 2,083.7 MeV；Elapsed 由 15.79 s 增到 11.89 s 的跨构建值不可作严格性能排序，但 primary kernel 工作量约增至 5 倍，且精度改善不足 | 已恢复 0.25 mm；只有 MCS 改成步长不变的模型后才能重测步长收敛 |
| 连续 Copper 段末端一次性 Highland 散射 | RTX 2080 Ti、250 MeV/u、256k histories；water-primary 26,032 接近但低于 TOPAS 30,878，然而总剂量比降至 0.879、IDD L1 升至 13.1%、lateral L1 升至 12.8%，入口 valley 仅为 TOPAS 的 0.748 | 已回退；除非同时有经过相空间验证的 lateral-displacement/非高斯 MCS 模型，不再采用段末单次散射 |
| Copper INCLXX ±0.25 MeV/u local-event window | RTX 2080 Ti、150/250/300 MeV/u 各 256k；相对双节点基线，IDD L1 分别 1.154→1.139%、1.497→1.521%、2.534→2.580%，Bragg valley 比 1.208→1.206、1.213→1.221、1.079→1.080，没有联合改善；吞吐约从 74/52/44k 降到 50/37/32k histories/s | 已回退；只有 package 在编译期构建 O(1) 固定能量桶、且更高统计证明 final-state 方差主导时再试 |
| 2026-09-18 水中 primary MCS 高频窄尾（strength/width=1.0/0.5、2.0/0.35） | RTX 2080 Ti、250/300 MeV/u 各 1.024M；保持约 25% 投影方差转入尾部，但相对稀有宽尾 0.5/0.7，2-D L1、lateral L1 和 Bragg PVDR 均更差；250 MeV/u Bragg PVDR 分别仅 0.818/0.807，300 为 0.891/0.892 | 已回退；证据指向更稀有、更宽的非高斯尾。只有直接 Geant4 角分布显示更高尾事件率时才重试 |
| 2026-09-18 水中 primary MCS 强稀有宽尾无条件用于 150 MeV/u（strength/width=0.5/0.85） | RTX 2080 Ti、1.024M；150 MeV/u 原基线 Bragg peak/valley/PVDR 约 1.032/1.022/1.010，候选变为 1.009/1.208/0.835，明显破坏已对齐能量；250/300 同参数则有稳定改善 | 已从 150 MeV/u 配置回退、仅保留 250/300；除非独立 150 MeV/u primary-origin 高统计数据推翻当前结果，不得跨能量无条件启用 |
| 2026-09-18 水中 primary MCS 过强稀有宽尾（strength/width=0.5/1.0） | RTX 2080 Ti、250/300 MeV/u 各 1.024M；整体 2-D/lateral L1 继续下降，但 250 MeV/u Bragg peak/valley/PVDR 过冲到 1.094/0.931/1.175，跨能量联合验收失败 | 已回退到插值后并经 10M 确认的 0.5/0.85；只有目标改为单能量或新的多能量数据支持时重试 |
| 2026-09-18 水中 primary CINEL03 H/O 全统计固定 1 MeV/u 桶 | RTX 2080 Ti、FP32 dose、250/300 MeV/u 各 1.024M；H/O 从稀疏精确节点改为 431/429 个固定节点、每节点中位 122/111 事件。charged-fragment 总量比仅 1.042→1.040、1.029→1.031，distal 比仅 1.192→1.167、1.170→1.164；secondary-C 反而 1.021→1.133、1.090→1.194，跨物种无联合改善。Unified EM missing=0、queue overflow=0、能量记账通过 | 诊断包隔离保留，正式配置已回退；证明 primary 单事件节点方差不是主因。只有新的逐反应 final-state 基准显示 H/O primary 采样偏差时才重试 |
| 2026-09-18 `cinel02_max_secondary_inelastic_generations=1` 用作 minibeam cascade | RTX 2080 Ti、FP32 dose、250/300 MeV/u 各 1.024M；数值 1 允许 generation-0 碎片反应，却因正式模式关闭 terminal-generation EM 而把其 charged products 原地沉积。2-D L1 从 10.28%/9.72% 恶化到 14.25%/16.65%，虽 distal total 从 1.146/1.131 过修正到 0.935/0.933 | 已改为物理文档规定的 generation=2；只有正式模式支持 terminal-generation EM transport 后，数值 1 才有重试意义 |
| 2026-09-18 Unified-EM primary delta aggregate 横向 Gaussian relocation（sigma=0.5 mm） | RTX 2080 Ti、FP32 dose、完整性检查关闭、generation=2，150/250/300 MeV/u 各 256k，同 seed 基线仅改 delta aggregate 的横向计分位置。入口 PVDR 分别从 1.435/1.346/1.155 改善到 1.289/1.198/1.081，但 2-D L1 从 18.061/18.626/17.510% 恶化到 18.541/18.744/17.565%，lateral L1 从 6.883/5.869/4.452% 恶化到 7.371/6.038/4.461%；能量守恒与 queue-overflow 检查通过，但跨能量剂量门槛失败 | 已回退代码和配置入口；只有取得逐电子能谱/横向响应数据，或建立经独立数据约束的深度依赖 delta 响应后才重试 |
| 2026-09-18 Copper General Ion Elastic rate x0.05 + delta lateral sigma=0.5 mm | RTX 2080 Ti、FP32 dose、完整性检查关闭、generation=2，150/250/300 MeV/u 各 256k；相空间提示离散 elastic 尾过量后，唯一组合改变量为 elastic rate x0.05 与上述 delta relocation。300 MeV/u IDD L1 从 1.355% 改善到 1.017%，但 150/250 MeV/u IDD 从 1.145/0.752% 恶化到 1.300/0.865%，Bragg valley 比从 1.041/1.131 恶化到 1.151/1.146，三能量 2-D L1 均恶化至 18.544/18.756/17.594%；审计、能量记账和 queue overflow 门槛通过，联合剂量门槛失败 | 已回退代码和配置入口；只有独立 Geant4 process-resolved elastic rate/angle benchmark，并能与 delta-electron 响应解耦时才重试 |
| 2026-09-18 OpenTOPAS creator-process/ancestor-process filter 用于 MT delta-dose 分解 | 本地 Geant4 11.3.2、TOPAS 4.2.p3、192 threads、250 MeV/u、2,560 histories；electron/non-electron carrier 分割以 L1 `2.4e-8` 闭合且电子占总剂量 7.12%，但 `ionIoni/hIoni` ancestor filter 仅得 0.0256%，direct `ionIoni` 也仅 0.0255%，与独立电子分量及 GPU Unified-EM delta 能量比例均不相容。过滤器按 `G4VProcess*` 匹配，MT worker process 实例不能作为可靠 process-resolved 标签 | 已从正式诊断配置移除；只有改用按 process name 记录、并显式传播 track lineage 的自定义 scorer，或单线程验证指针身份后才重试 |
| 2026-09-18 以 water-entry slit residual 的 `sin²` 权重增加 surviving-primary Copper 能损 | RTX 2080 Ti、FP32 dose、完整性检查关闭、generation=2、discrete elastic 关闭，150/250/300 MeV/u 各 256k，同 seed 基线只增加位置条件能损。第一档按 valley 平均能量标定 (`delta=0.1479/0.0826/0.0260`) 后 valley energy 比达到 1.013/1.016/1.000，但 stopping proxy 过冲到 1.253/1.098/0.956，150 中深度 valley 比从 1.041 恶化到 0.863。第二档按 stopping proxy 标定 (`0.0428/0.0521/0.0382`) 后 proxy 为 0.962/1.014/0.957；相对基线 2-D L1 仅 18.084/18.620/17.527%→18.047/18.631/17.503%，IDD 为 1.320/0.901/1.029%→1.270/0.757/1.269%，300 明显退化，Bragg PVDR 150/300 也下降。Unified-EM missing=0、queue overflow=0，三能量完整运行完成 | 代码和正式配置已回退，256k 诊断结果保留在 `out/minibeam_valley_loss_*` 与 `out/minibeam_valley_stopping_*`；单一位置能损标量不能同时描述入口和深度演化。只有取得 TOPAS primary track 的 path length/exit angle/energy 联合条件分布并建立多变量 transport 模型后才重试 |
| 2026-09-18 Copper primary Highland 250 MeV 动能 floor + 高能 core scale | RTX 2080 Ti、FP32 dose、完整性检查关闭、generation=2、discrete elastic 关闭；相对严格 elastic-off 10,000,128-history 基线，唯一组合改动为 primary Highland 计算能量下限 250 MeV，且 250/300 MeV/u 的 scale 从 0.785 调到 0.810/0.830（150 保持 0.785）。修正分析网格为 lateral/depth `0.1/0.25 mm` 后，150/250/300 的 2-D L1 从 3.6368/3.9508/3.9196% 变为 3.6322/3.8996/3.8128%，IDD 从 1.0925/0.7156/0.7099% 变为 1.0831/0.5694/0.6045%；但入口 valley 仍仅 0.823/0.861/0.902，Bragg PVDR 仍为 0.911/0.946/0.938。审计 missing=0、queue overflow=0、能量残差 1.6e-5–2.7e-5 | 有小幅统计稳定收益但未推广，代码和正式配置已回退；它同时裁低能尾并调 core，不能修复 Copper 边界、步内位移—角度相关和出口能量回补造成的联合相空间不自洽。只有结构性 Copper 模型完成并以独立 foil/slit-edge 数据重新约束时才重试 |
| 2026-09-18 250 MeV/u 水中 primary 低能 Highland core 缩放关闭、保留稀有宽尾 | 新增同一份 TOPAS 水入口 parent-0 C12 相空间回放后，RTX 2080 Ti、30,878 个 surviving-primary histories 的 Bragg peak/valley/PVDR 比由生产水模型的 1.017/0.808/1.258 改善为 0.965/0.917/1.052，distal PVDR 由 1.020 改善为 1.005；但把唯一改动带回完整 Copper+water 256k 运行后，Bragg valley 从 1.141 恶化到 1.232、PVDR 从 0.894 恶化到 0.811，2-D L1 18.620%→18.625%，说明 Copper 输入误差与水中低能修正正在相互补偿。两次 GPU 运行均为 FP32 dose、完整性检查关闭、generation=2、discrete elastic 关闭，审计/能量记账/queue overflow 通过 | 正式 250 MeV/u 配置已恢复 transition=180 MeV/u、low-energy scale=0.20；不能在修复 Copper 联合相空间前单独推广水中关闭方案。Copper 入口相空间匹配后，必须用同源回放重新验收并允许重试此项 |
| 2026-09-18 水中 primary 最后约 10 mm 极低能 Highland core 恢复（60 MeV/u 以下线性回到 scale=1） | RTX 2080 Ti、FP32 dose、完整性检查关闭、generation=2、discrete elastic 关闭。250 MeV/u 同源 water-entry replay 的 Bragg valley 比从 0.8083 降到 0.8011、PVDR 从 1.2576 升到 1.2678；完整 Copper+water 256k 的 Bragg valley 从 1.1413 升到 1.1715、PVDR 从 0.8939 降到 0.8756。两种隔离测试都未改善目标，审计、能量记账和 queue overflow 均通过 | 已回退。极低能角修正发生得太晚，且缺少同一步内的相关横向位移；只有实现并独立验证 Fermi–Eyges/Geant4 displacement-angle correlated transport 后才重试 |
| 2026-09-18 水中 primary 每步 Fermi–Eyges 相关横向位移（保留现有角分布） | RTX 2080 Ti、FP32 dose、250 MeV/u 同一份 30,878-primary water-entry replay；按 `Var(y)=L²Var(theta)/3`、`Cov(y,theta)=LVar(theta)/2` 为每个 0.1 mm 步增加位移。坪区 PVDR 比仅 1.00055→1.00002，Bragg valley 反而 0.80835→0.80403、PVDR 1.25758→1.26801，2-D L1 17.5888%→17.5914%；能量残差 5.16e-6、Unified-EM missing=0、无 queue overflow 报告 | 已回退，不再做完整 Copper 256k。0.1 mm 步内遗漏位移不是主导误差；只有改成经过独立 Geant4 验证且步长不变的累计散射功率模型（而非保留逐步 Highland 角方差）后才重试 |
| 2026-09-18 Copper primary 精确截断到矩形 slit 横向边界 | RTX 2080 Ti、FP32 dose、完整性检查关闭、generation=2、discrete elastic 关闭，250 MeV/u 256k。相对同 seed 基线，Bragg valley 比 1.161→1.133、PVDR 0.856→0.863，入口也改善；但坪区 valley 0.920→0.869、PVDR 1.053→1.140，2-D L1 18.480%→18.626%、IDD L1 1.196%→1.255%，跨深度联合门槛失败。能量残差 1.71e-5、Unified-EM missing=0、无 queue overflow 报告 | 已回退。只有在 Copper charged-fragment 产额/相空间和水中累计散射功率分别匹配后，才将边界截断与重新标定的 Copper 模型一起重试 |
| 2026-09-18 水中 primary 累计辐射长度 Highland 方差增量 | RTX 2080 Ti、FP32 dose、250 MeV/u 同一份 30,878-primary water-entry replay；每步运动学取当前能量，但 Highland 对数项按水入口以来累计 `x/X0` 的方差差分，消除独立 0.1 mm 薄层的对数项步长依赖。Bragg valley 0.808→1.102，但 peak 1.017→0.892、PVDR 1.258→0.809；坪区 peak 0.986→0.925、PVDR 1.001→0.931，2-D/lateral L1 17.59/4.64%→18.00/5.51%，明显过度展宽。审计、能量记账和 queue overflow 通过 | 已回退。不能直接把标准 Highland 的全程对数修正差分成局部散射功率；只有取得 Geant4 水中逐深度 beam moments/散射功率数据并约束窄核与非高斯尾后，才以数据驱动形式重试 |

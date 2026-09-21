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
| 2026-09-18 仅用直线首交点拆分圆柱/slit Copper 材料段 | RTX 2080 Ti、FP32 dose、完整性检查关闭、250 MeV/u、256k；已使用 slab 验证的 primary Fermi--Eyges/tail，并拆分 fragment MCS scale。boundary off/on 的总剂量比为 0.99399/0.99422、IDD L1 0.967/0.930%、2-D L1 18.611/18.612%，没有整体改善；局部 PVDR 有大幅但互相矛盾的低统计变化，且分段改变 RNG 序列，不能归因于几何精度。两者 charged survivors 48828/48776，仅差 0.11% | helper 作为可选诊断保留，正式配置暂启用以继续 slit-edge 隔离；不得把本次结果宣称为完整联合输运。只有实现/验证随机位移诱导跨界处理并匹配 fragment 出口产额后，才用多 seed slit-edge 数据重试剂量结论 |
| 2026-09-18 水中 primary 累计辐射长度 Highland 方差增量 | RTX 2080 Ti、FP32 dose、250 MeV/u 同一份 30,878-primary water-entry replay；每步运动学取当前能量，但 Highland 对数项按水入口以来累计 `x/X0` 的方差差分，消除独立 0.1 mm 薄层的对数项步长依赖。Bragg valley 0.808→1.102，但 peak 1.017→0.892、PVDR 1.258→0.809；坪区 peak 0.986→0.925、PVDR 1.001→0.931，2-D/lateral L1 17.59/4.64%→18.00/5.51%，明显过度展宽。审计、能量记账和 queue overflow 通过 | 已回退。不能直接把标准 Highland 的全程对数修正差分成局部散射功率；只有取得 Geant4 水中逐深度 beam moments/散射功率数据并约束窄核与非高斯尾后，才以数据驱动形式重试 |
| 2026-09-18 将旧 Copper survivor 出口能损系数直接改作逐步局部 stopping scale | RTX 2080 Ti、FP32 dose、真实圆柱外体和 primary optical-depth 碰撞点候选，150/250/300 MeV/u 各 256k。把旧 `0.8663/0.9430/0.9647` 按局部能量插值后，会复活原先被未缩放 stopping 淘汰的低能 primary；Copper-touched survivor 数量比 TOPAS 变为 1.030/1.101/1.095，平均能量比仍只有 0.961/0.964/0.956。改回 Geant4 表本身的局部 scale=1 后，三能量 touched 数量比改善为 0.955/1.009/0.988，250 MeV/u 的角 RMS、valley fluence 和 stopping proxy 分别为 1.015/0.983/1.024。独立 1/10/60 mm homogeneous slab 随后确认未缩放表的 CSDA 出口能量与 TOPAS primary mean 在 0.013--0.341 MeV 内一致 | 已否决旧三点直接迁移，结构性圆柱/真实碰撞点保留，正式配置删除该 calibration。当前 Geant4 11.3.2 Copper 表不需要局部 stopping scale；只有更换 Geant4/material table 后，独立 slab 显示统计显著偏差时才能重试 |
| 2026-09-18 仅放大现有 Copper Highland 标量以匹配 slab 角 RMS | RTX 2080 Ti、FP32 dose、完整性检查关闭、250 MeV/u、256k histories 的 1/10 mm solid-Cu 隔离基线。primary survival GPU/TOPAS 为 0.99970/0.99933、mean E 为 1.00006/1.00130，证明核衰减和 stopping 已匹配；但 `(A0,A1,A2)` 比分别为 `(0.491,0.555,0.730)` 与 `(0.700,0.706,0.724)`，q99.9 仅 0.538/0.688。任何单一角宽 scale 即使把 A2 拉到 1，仍不能同时修复薄层 A0/A1 和随 quantile 增大的尾部缺口 | 未修改正式 scale，解析否决该单参数路线。只有实现由独立 slab A0/A1/A2 约束的步长不变 scattering power、相关位移以及单独 tail kernel 后，才可重新标定 core；不得从完整剂量/PVDR 反推该 scale |
| 2026-09-18 新 Copper Fermi--Eyges/tail 后关闭 250 MeV/u 水中 low-energy core correction | RTX 2080 Ti、FP32 dose、完整性检查关闭、256k histories；Copper 已换为 slab 验证的步长不变模型，唯一水中改动为 transition/scale `180/0.20 -> 0/1.0`，保留 tail `0.5/0.90`。相对同 seed 新 Copper 基线，2-D L1 `18.628% -> 18.662%`，plateau PVDR `0.864 -> 0.899`，但 Bragg valley `1.149 -> 1.237`、Bragg PVDR `0.891 -> 0.818`，distal PVDR `1.098 -> 1.107`；能量残差 `1.76e-5`、Unified-EM missing=0、queue overflow=0 | 已恢复 transition=180 MeV/u、scale=0.20。即使 Copper bulk slab 与入口总体相空间改善，完全关闭水中低能 core correction 仍不能联合匹配 plateau/Bragg；只有以 TOPAS 同源 water-entry 多平面 A0/A1/A2 和高统计 dose 同时约束新的水中 scattering-power 模型后再试，不再用开/关标量补偿 |
| 2026-09-18 Copper Fermi--Eyges 加单一 Gaussian Poisson tail event | RTX 2080 Ti、FP32 dose、完整性检查关闭、250 MeV/u、256k histories；1 mm 的 A0/A1/A2 与 q99.9 已改善到 `1.092/1.069/1.028/1.036`，10 mm 的 A0/A1/A2 为 `1.023/1.015/1.004`，但 10 mm q99.9 仍只有 0.879，无法同时描述薄/厚 slab 的尾部中心极限定标 | 已由同一路径长度 Poisson 过程中的窄/宽两分量 event 取代；新模型的 1/10 mm q99.9 为 0.997/1.000，并通过 0.05 mm 步长检查。除非新的 Geant4 数据否定双分量厚度演化，不再退回单一 Gaussian event |
| 2026-09-18 将 Copper slab 宽角尾缺口归因于 TOPAS/GPU 的 General Ion Elastic 开关不同 | 过程隔离诊断，不是新 GPU 性能候选。本地 TOPAS 4.2.p3/Geant4 11.3.2、Slurm 6432/6433、各 16 CPU/8 GB、250 MeV/u、256k histories、1/10 mm；相对 full-list 仅移除 CarbonIonElasticPhysics，两项完整运行成功。角方差比为 0.9918/0.9936，q99.9 比为 1.0025/0.9905，远不足以解释已有 FP32 GPU 的 q99.9 比 0.538/0.688。已统一到物理 Copper 出口平面，排除 0.01 mm 真空 drift 的影响 | 已排除此项作为主要误差解释；保留诊断开关，未修改正式物理配置。该结果不否决更准确的核弹性模型，仅在新的过程分解数据证明其显著贡献时重新评估主因。完整数据与模型建议见 docs/minibeam_accuracy_audit_20260918.md |
| 2026-09-18 用单一高能 p/d/t+Cu pilot 包覆盖 fragment 二次反应，或对低能查询做端点钳制 | RTX 2080 Ti、FP32 dose、完整性检查关闭、250 MeV/u、256k。初始 pilot 的 p/d/t 最低节点仅 337/426/378 MeV/u，46,648 次 fragment 反应中 32,621 次低于能区，严格命中率仅 29.8%；虽然匹配计分面产额从 0.815 改善到 0.920，但该包不能作为完整模型。没有启用 nearest/endpoint fallback | 已用 Slurm 6438 的低能矩阵将 p/d/t 覆盖降至 4.07/2.52/1.82 MeV/u，严格命中率达到 97.8%、产额比 0.979。不得恢复端点钳制；剩余 118 个缺物种、165 个低能和 728 个 >5 MeV/u gap 查询只能通过新增过程一致数据补齐 |
| 2026-09-18 将 Copper cascade generation 直接带入水中，并在水核反应上限原地沉积 charged final state | RTX 2080 Ti、FP32 dose、完整性检查关闭、250 MeV/u、256k。Copper 子代以 generation=1 进入水，使 cascade 开关同时少一代水核反应；generation cap 处的带电产物又被原地计分。原 terminal/cascade 的 2-D L1 `18.60/18.68%` 和 Bragg PVDR `0.872/0.871` A/B 因此不能归因于 Copper | 已分离 Copper/水 generation namespace，Copper survivor 从 water generation 0 开始；水核上限现在只关闭后续核 hazard，所有高于 cutoff 的带电末态继续 EM。重跑 terminal/cascade 后 2-D L1 为 `18.323/18.290%`、Bragg PVDR 为 `0.879/0.877`；旧 A/B 结论已撤回 |
| 2026-09-18 将单代 Copper fragment cascade 视为已收敛并据此进入最终剂量验收 | RTX 2080 Ti、FP32 dose、250 MeV/u、同 seed 256k。cap=1 的末代 `sum(tau)=4517.52`，逐轨迹 `sum(1-exp(-tau))=4133.68`；cap=2 降到 `144.23/135.84`，cap=3 降到 `2.02/1.94`。cap=2/3 的总剂量比 `1.002109/1.002090`、IDD L1 `0.5808/0.5807%`、2-D L1 `18.28669/18.28668%`、Bragg PVDR 均 `0.876834`，队列无溢出、能量残差约 `1.73e-5` | 单代收敛结论已否决；诊断实现允许最多三代。后续三能量 10M 验证确认 per-energy package 的 generation 3 均可正式启用；只有过程包/截面变化，或末代期望反应数不再可忽略时，才提高代数上限 |
| 2026-09-19 将 192-thread TOPAS phase-space replay 的 EventID 当作入口相空间行号 | 12.8M 纯 EM 同源 water replay；文件开头存在偶然对齐，但 worker block 后记录顺序重排。按该映射约 1.84% 的 40 mm C12 会出现不可能的出口能量高于映射入口能量，且位置产生大幅不连续，因此入口能量/slit 分组和所谓 identity-matched `A0/A1/A2` 均无效 | 分析脚本已移除该 join，仅保留当前平面总体角度、pitch-folded 位置、协方差、Fourier 调制度和固定区域通量/条件能谱。只有入口与所有下游 scorer 在同一次原始源运行中输出稳定 `(RunID,EventID,TrackID)`，或 TOPAS 显式传播唯一 source ID 时才恢复逐轨迹分析 |
| 2026-09-19 用 `em_primary_step_scale=1.5` 代替水 MCS 分步收敛测试 | 1,622,795-history 同源纯 EM replay；Unified-EM research extension 在当前能量/节点限制下未改变实际步数，baseline/candidate 均为 875,052,612 steps，因此不能作为散射分步实验 | 改用水散射模型内部独立 `max_segment_mm=0.10/0.05`；区间矩和固定区域剂量通过。只有 Unified-EM 日后允许显式缩短物理步时，才用全 transport step 再做交叉检查 |
| 2026-09-19 将水中相关散射整步终态线性插值到诊断平面 | 对候选 `fermi_eyges_tail`，路径比例 `f` 的线性角插值产生 `f²TL` 而非正确的 `fTL` 方差；整步位移线性分配也会把晚于平面的 Poisson tail event 搬到上游。plane scoring on/off 的 dose L1 `5.90e-8` 只证明诊断不扰动输运，不能证明记录无偏 | 已替换为条件于不变终态的 integrated-Brownian Gaussian bridge，并按真实 tail event 位置决定是否进入交点状态；不得再用起末状态线性插值反推区间散射。固定终态 bridge sampler 的三个半步矩为解析值的 `0.9991/0.9980/0.9979` |
| 2026-09-19 将 250 MeV/u 区间约束的水中 `9.9/0.0025/2.4` 候选直接推广到三能量正式配置 | 冻结参数、相同 Copper/cascade/seed 的 256k full-chain A/B。150/250/300 MeV/u 的 2-D L1 分别为 legacy `17.672/18.197/17.224%`、candidate `17.923/18.009/17.227%`；lateral L1 为 `7.287/5.598/5.327%`、`7.078/5.296/5.383%`。250 有小幅整体收益，150 的 lateral 收益伴随 2-D 退化，300 持平略退化；Bragg PVDR 比也不是统一改善。随后独立 150/300 纯 EM 同源水回放已满足原重试条件：候选区间矩和 fixed-region dose 明显优于 legacy，证明结构可跨能量，但 26k/35k 统计不足以约束 q99.9，且并未改变 full-chain 不统一改善的事实。三能量 Unified-EM missing/overflow=0，energy residual `1.6e-5--2.7e-5` | 当时不推广且不继续从剂量调参。2026-09-19 后续 12.8M 纯 EM 和 10M 全物理入口均证明 Copper/slit 联合 shape TV 约 `0.64%`，原上游补偿疑虑已消除；冻结参数的三能量 10M full-chain Bragg PVDR 比为 `1.032/1.004/0.956`，故 retry condition 已满足并转为接受，正式配置现启用该结构。仍禁止重新从 PVDR 调 `9.9/0.0025/2.4` |
| 2026-09-19 在步长稳定 Copper FE/tail 模型下把外层 Copper 最大步长从 `0.25` 缩到 `0.05 mm` | 该重试满足旧 Highland 候选的 retry condition。RTX 2080 Ti、FP32 dose、完整性检查关闭、250 MeV/u、同 seed 256k；water-entry valley fluence/stopping proxy 从 `0.914/0.848` 改善到 `1.004/0.939`，但 touched 三维联合 shape TV 仅 `5.629% -> 5.591%`，2-D L1 `18.323% -> 18.314%`、IDD L1 `0.934% -> 0.956%`、lateral L1 `5.745% -> 5.884%`，且 80--100 mm 固定 valley 反而恶化。审计、quality、能量记账和 queue overflow 均通过 | 保持正式步长 `0.25 mm`。缩步长改变了入口局部指标，却没有形成跨深度的稳定剂量收益，并增加约 5 倍 Copper 分段工作。只有实现随机位移诱导的精确材料跨界、并有高统计 slit-edge 条件数据表明当前边界离散主导误差时才重试 |
| 2026-09-19 用同一份含 250 MeV/u C12 节点的 fragment-cascade 包直接覆盖 150/250/300 MeV/u | 10,000,128 histories、FP32、冻结水 FE/tail。150 和 250 分别保持/改善，但 300 的 primary Copper products 从 `58.57M` 降到 `40.26M`，总剂量比从 `0.9819` 恶化到 `0.9195`，IDD L1 从 `1.827%` 恶化到 `8.049%`。原因是该包不仅添加 fragment 节点，也把能量专用 C12+Cu 末态替换成 250 MeV/u campaign | 已否决共享单包。离线重编 per-energy package：保留各自原 C12 campaign，只合入共同 fragment sources；300 修复到 total `0.9955`、IDD L1 `0.614%`、2-D L1 `2.975%`。正式三能量配置只使用各自 package，不得交叉复用 |
| 2026-09-19 将 p/d/t 中间能区 package gap 视为 300 MeV/u Bragg peak/PVDR 的主要修复 | RTX 2080 Ti、FP32 dose、冻结 water primary MCS/Copper EM/slit、10,000,128 histories。按参考 physics list 提取 p 的 Binary Cascade 和 d/t 的 INCL++，严格填满 `299.967--337.355`、`299.989--426.112`、`300.000--377.541 MeV/u` 空档；300 MeV/u lookup gap 从 `91,175` 降为 0、总 miss 从 `108,740` 降为 `17,604`。总剂量比 `0.99546 -> 0.99603`、IDD L1 `0.614% -> 0.578%`、2-D L1 `2.975% -> 2.960%`，但局部 Bragg peak `0.96307 -> 0.96305`、PVDR `0.95546 -> 0.95519`，基本不变。150 回归中剂量指标不变；250 用相同 binary、seed `2026092510` 严格重跑后，2-D/IDD/lateral L1 为 `3.15535/0.65507/1.07266% -> 3.14970/0.64191/1.05884%`，但局部 peak/PVDR 为 `1.004775/1.001911 -> 1.004781/1.001784`，峰形不变 | gap-filled package 作为消除非物理吸收的正确性修复保留，不能再作为剩余 Bragg 局部误差的解释或调参目标。只有分物种水中剂量/峰形诊断证明剩余 miss 或某一 fragment 分量主导 Bragg 后，才继续扩大 package；下一步应查峰位/峰形和 fragment 水中输运 |
| 2026-09-19 仅切换 secondary C12 到 FE 即可恢复 primary/secondary 全部水输运等价 | 1,622,795 条完全相同的 250 MeV/u 水入口 C12、FP32、纯 EM。FE 将四个 20 mm 区间的角方差/位移方差/q99 比从 legacy 的 `1.356--1.477/1.379--1.445/1.163--1.204` 改善到 `0.984--1.013/1.015--1.018/0.992--1.007`，散射部分通过；但 IDD L1 仅从 `1.898%` 变为 `1.892%`，Bragg 仍为 `123.875 mm`，比 TOPAS/primary FE 的 `124.625 mm` 提前 0.75 mm。原因是 primary minibeam 路径额外应用冻结的 `0.9958` stopping scale，secondary 路径没有；150/300 筛查也分别提前 `0.25/1.25 mm` | 不否决 secondary-C12 FE 散射模型，否决用它单独宣称完整路径等价或用 MCS 参数补偿纵向差异。正式开关保持关闭；只有统一并独立验证 C12 primary/secondary stopping 与 straggling 语义、再以真实 secondary-C12 出生谱闭合后，才重试完整剂量推广 |
| 2026-09-19 使用未启用 Unified EM 的 legacy secondary replay 验证 straggling/能损语义 | 旧 secondary 配置的 `enable_secondary_unified_em=false`，因此 secondary straggling 开关没有进入与 primary 相同的采样路径；由此得到的 Bragg/IDD 差异混合了两套 stopping 实现，不能用于判定 `0.9958` 的作用 | 已改为同一入口、双方 Unified EM 的受控对照。无 straggling scale=1、straggling scale=1、straggling scale=0.9958 的 primary/secondary IDD L1 分别为 `0.00827/0.02993/0.03137%`。不得复用旧配置做能损闭合；只有明确比较不同 EM 实现时才允许 |
| 2026-09-19 将 Unified-EM secondary FE 的周期性深度剂量尖峰解释为 FE 0.10/0.05 mm 分段不稳定 | secondary z 边界逻辑忽略 `0 < dz_step <= 1e-5 mm`，轨迹在边界前残留后会用下一长步跨越多个 bin；连续能损计分还曾错误复用终点/核碰撞 bin。修正为接受所有正边界距离并使用步首 bin 后，FE010/FE005 IDD L1 从约 `2.65%` 降至 `0.03095%`，固定 ROI 和区间矩稳定 | 已否决该归因，不能据旧尖峰调 FE 参数。只有精确边界截断下仍能在多 seed/固定 ROI/联合矩中复现步长依赖时，才重新审查 FE 分段模型 |
| 2026-09-19 用“644 条反向真实出生粒子成功完成回放”代替反向边界计分验证 | 反向粒子恰好位于深度边界时，旧 `floor(z/dz)` 仍把步首能损归到下游 bin；legacy 路径还会用 `fmax(step,1e-5)` 放大真实微边界步。能量账本仍可闭合，因此完成数和残差都无法发现空间错位 | 已实现按方向的边界归属，并保护真实微边界步。边界前/上/后、正反方向、legacy/Unified 共 12 个确定性逐-bin 案例全部通过。不得再用“轨迹完成”替代空间计分测试；只有改变边界/步长代码时重跑该固定回归 |
| 2026-09-19 根据同源 C12 相空间改善直接在正式 full-chain 启用 secondary FE | 300 MeV/u、FP32、C12-only Unified 和 `0.9958` 固定后，对 Highland/FE 做两个 seed、各 10,000,128 histories。secondary-C12 分量体素 L1 改变约 31%，但只占总剂量约 0.55%；D/C 总剂量比为 `0.9999945/1.0000064`，2-D、IDD、lateral、固定 ROI 没有跨 seed 一致联合收益。150/250 的 256k 回归也仅有小量变化 | 正式 secondary FE、C12-only Unified 和 secondary `0.9958` 均保持关闭。保留 FE 作为已通过同源散射验证的研究模型；只有 ancestry-resolved 高统计结果显示 secondary C12/后代成为可辨识误差主因，或其剂量占比显著提高时，才重试正式推广 |
| 2026-09-19 对所有 CT C12 强制 `0.10 mm` 真实 FE 材料刷新步 | RT07575 同一 6,481,909-history shard、FP32、相同 seed/package/Unified EM；把原本仅在 MCS 内部使用的 `fermi_eyges_max_segment_mm=0.1` 同时作为 primary/secondary C12 外层 transport 上限。质量、能量和 overflow 检查通过，但总 steps 从 `2.069B` 增到 `7.709B`、Elapsed 从 `54.22 s` 增到 `113.73 s`。相对既有 TOPAS shard，抽样 Global/Local 1%/1 mm 从 `94.754/75.646%` 变为 `94.814/75.710%`，Local 3%/0 mm 从 `69.754%` 变为 `70.278%`，但 Global 3%/3 mm 从 `99.968%` 降到 `99.942%`；收益混合且接近单-shard统计波动，不足以支持约 2.1 倍 wall-time | 已回退无条件真实细分；保留内部 FE 分段和原有精确直线 CT face clamp。只有新增诊断证明随机横向位移确实跨越不同 Schneider material，并能只对这些 segment 做事件位置/边界联合拆分，或多 shard 显示严格 Gamma 有稳定显著收益时才重试；不得用全 CT 缩步代替边界算法 |
| 2026-09-19 用单一 Gaussian FE core + 单类 Poisson Gaussian tail 同时拟合 p/d/t/He-4 的 1/10 mm 水片全部分位数 | 24 个纯 EM TOPAS case、每例 250k histories、FP32 GPU 同源回放。10 mm 累积输运较好：角/位移方差比中位数 `1.0079/1.0077`，q99.9 `0.9928`；但 1 mm 的 q68/q99 中位数同时为 `1.0749/0.9158`，显示 UrbanMsc 肩部结构不能由当前三参数形状同时表达。继续调 core/rate/width 只能在核心、肩部与远尾之间制造补偿 | 保留当前分物种参数作为已接入候选，不再用同一三参数形式追逐薄层全部 quantile。只有引入额外 shoulder/tail 形状分量，并用未参与拟合的厚度/能量验证后，才重试薄层联合匹配 |
| 2026-09-19 使用 TOPAS 默认 600 MeV EM table 上限生成 300 MeV/u He-4 水片参考 | 300 MeV/u He-4 总动能为 1200 MeV，旧日志明确显示 `Max kinetic energy for tables = 600 MeV`。旧参考 10 mm 平均能损 6.4043 MeV/u，造成 GPU 仅为其 55%的假象。设置 `Ph/Default/EMRangeMax = 10 GeV` 后，TOPAS/GPU 平均能损为 `3.50323/3.52421 MeV/u`，出口谱宽为 `0.174162/0.174193 MeV/u` | job 7024 与旧目录不再作为标定输入；权威参考为 job 7026。任何高总动能离子 slab 必须显式检查 Geant4 EM table 能区，不能用经验 stopping scale 补偿超表参考 |
| 2026-09-21 Copper 三段解析尾（窄/宽/极端）拟合 60mm 准直器出射宽角分布 | 新数据：单 spot 250 MeV/u、2M，TOPAS Water_75eV 准直器出射 C12 角分布 vs GPU 相空间。基线 GPU 尾不足：>40mrad 0.950、>80mrad 0.789、>160mrad 0.518。候选 `copper_fermi_eyges_tail_step` 加极端分量（core 0.8076→0.79，fw 0.0156→0.02，wr 4.52→6.0，fe 0.0015，er 10）。相空间 >160mrad 0.52→0.99，但 >80mrad 仍 0.80、>40mrad 略降；10M EM-only 剂量 IDD L1 0.300%→0.178%，入口对比度 1.055→1.067（略差），深处不变 | 已回退（`extreme_tail_fraction=0`）。三段解析尾在 >40/>80/>160 间互相补偿，无法同时拟合；与 #146/#164 的厚度演化限制一致。只有改用表格化角分布（数据驱动采样）或 Geant4 Urban 单散射分布后才重试 |
| 2026-09-21 Copper process-level 单次散射模型（screened Rutherford）替换凝聚 kick | 用 `/tmp/copper_single_scatter.py`：每步 N~Poisson(Σ0·h)，事件角从 screened Rutherford P(θ)∝θ/(θ²+θs²)² 抽样，θs/θmax~1/(βp)，拟合 4 参数到 TOPAS 1/10/15/20mm。最优 rms **0.299**，q50/90/99/99.9 G/T 1mm=0.55/0.54/0.82/0.90、20mm=1.43/1.44/1.41/1.17，严重不拟合 | 已否决该简化形式。要真正 process-level 必须逐项复刻 Geant4 `G4UrbanMscModel`/`G4IonSingleScatteringModel`：screening 角、步长限制、skin/range factor、lateral displacement、几何步进耦合。这是完整模型移植工程，超出参数/公式修正范围。只有以 Geant4 源码为规格逐函数移植并以 slab 分位数验收后才可重试 |
| 2026-09-21 Copper Urban MSC 低能散射功率修正（process-resolved T(E) 拟合） | 新数据：`CarbonMscStepNtuple` 在真实准直器逐步记录 31.5M 步（50k histories、250 MeV/u、EM-only）。`T_topas/T_model` 高能≈1，<600 MeV 升至 1.14–1.34（Highland 对数）；拟合 `(1+0.05422 ln(2006/E))²`。kernel 加该修正后准直器出射 >40/80/160mrad G/T 0.950/0.789/0.518→0.985/0.935/0.734，q99.9 0.93→0.97（明显改善）；但 10M EM-only 剂量对比度几乎不变（入口 1.055→1.053，2-D L1 2.029%→2.030%） | 已回退。该修正物理正确、改善相空间尾，但不改变剂量对比度：入口误差来自 peak/valley 核心（+2.8%/−2.3%，与 direct/touched 透射比或 slit 边缘相关），不是极端尾。下一步应隔离 slit 透射/源几何，而非继续改散射尾 |
| 2026-09-21 Copper 表格化 per-step kick（TOPAS 0.25mm 单步片 CDF）替换解析尾 | 新数据 EM-only 铜片 0.25mm、50–300 MeV/u、1M/例；reduced kick 形状跨能量近似普适（q50/rms≈0.71、q99.9/rms≈5.3）。但把该 CDF 作为每 0.25mm 步的独立 kick 累积后，1mm q99/99.9 G/T=6.5/9.7、20mm=5.5/5.3，严重过尾 | 已否决；Geant4 Urban MSC 不是固定 kick 的 Markov 过程（步长/分布随路径变化），不能把单步片 kick 直接累积。只有 hook `G4VMultipleScattering::AlongStepDoIt` 取真实 process-level 步进数据，或直接移植 Urban 模型才可重试 |
| 2026-09-21 Copper FE/tail 加入 Highland 对数项（step/cumulative）修正厚度演化 | 新数据 EM-only 铜片 1/10/15/20mm。baseline 无对数：q50/99/99.9 G/T=1.026/1.011/0.977(1mm)、0.967/0.945/0.868(20mm)。cumulative 版过冲（1.05–1.16）；step 版需把 scattering_energy 从 13.0 降到 12.5，最优 rms 0.038，但 20mm q99.9 仍 0.859（与 baseline 0.868 相当），只是把核心从 0.967 抬到 0.971 | 已否决；对数项与 Es 重标定互相抵消，不能修复 20mm 极端尾。Copper 大厚度（低能）尾需 process-resolved 数据 |
| 2026-09-21 Copper 单次散射幂律尾（P(θ)∝θ⁻³, [a0,amax]·scattering_ratio）拟合厚度演化 | 新数据：EM-only C12 铜片 250 MeV/u、1/10/15/20 mm。baseline FE/tail 的 q99.9 G/T = 1.03/1.01/0.971/0.877（能量演化已验证与 TOPAS 一致：2914/2052/1455 vs 2914/2055/1462）。把 per-event 高斯宽度混合换成幂律单次散射事件后，4 参数最小二乘 rms 0.057，q99.9 G/T 变成 0.886/1.178/1.193/0.969，不能同时拟合薄/厚 | 已否决该解析形式；连同极端分量、三段混合，三种解析尾都不能表达 Urban MSC 的厚度演化。只有提取 process-resolved 单次散射角分布（按能量/材料）并表格化采样后才重试 |
| 2026-09-21 C12 water 5 参数（core + 窄/宽双分量 Poisson 尾）重拟合 Water_75eV slab | 新数据：C12 @100/150/200/250/300 MeV/u × 1/10/40 mm，Water_75eV，250k/例。当前候选 9.9/0.0025/2.4 全例 rms 0.0169、留出 40mm 0.0269；5 参数全例 0.0178（更差）、留出 40mm 0.0240（仅边际改善，宽尾权重常塌缩为 0） | 保留当前候选。解析 per-event 窄/宽混合不能表达薄片肩部；只有表格化角分布或 Urban 分布才可能同时拟合 q68 与 q99 |

# 当前 proton minibeam 吞吐分析（2026-10-01）

本文保留修改前的分析快照。后续已修复构建宏并合并公共铜/空气路径；
候选实验、回退和交付验证见 [共享路径报告](minibeam_shared_20261001.md)。

结论：水模体应优先评估水电子响应审计分片、铜短步几何裁剪和铜终止核事件拆核。
CT 则先修复构建选项未传入编译器的问题，再建立当前二进制的有效基线。
本轮检查了当前源码、`failed.md`、来源工作树的历史报告，执行原配置基线，未修改输运或构建配置，也没有运行新优化候选。

## 当前构建与实测

RTX 2080 Ti，二进制 SHA256：
`16eaeedead2cb2ba7fb475644d067d2195c3a157f42b3b598ef521d65357a8e3`。
FP32 dose ON、FP64 dose OFF。水参考保持 Water_75eV，stopping scale=1。

水模体原预设：170 MeV、4M histories、30×30 mm 源、60 mm 铜、0.5 mm 狭缝、3.6 mm 间距，
铜最大步长0.05 mm，水最大步长0.25 mm，Urban、核反应、电子响应、带电次级与研究中子输运。

| 部分 | 秒数 | 外部墙钟占比 |
|---|---:|---:|
| 铜/空气主核 | 11.6978 | 41.3% |
| 水中主核（含电子响应和评分） | 10.2348 | 36.1% |
| 其余初始化、次级、中子、归并和输出 | 6.3934 | 22.6% |
| 完整进程 | 28.3260 | 100% |

完整吞吐 **141,213 histories/s**。日志的 `Elapsed=27.1691s`、`Throughput=147,226/s`
不是完整进程口径，不能替代上述数字。单次基线不证明相比旧版本加速，也不是稳定性能分布。

研究质量 `accepted=true, status=non_production, failures=[]`；能量相对残差7.8585e-8；
quality报告queue overflow=0，铜带电子任务1,879,962条全部完成且overflow=0。
原物理研究限制保留；本轮没有新增TOPAS精度认证。

两个主核合计77.4%。假设它们各减少10%耗时、其余固定，端到端吞吐仅提高约8.4%；
不能把局部加速比例直接当成整体收益。铜带电子任务核只有0.1652s，继续优化其直接运行成本空间很小。

## 首先处理：构建脚本中的四个选项实际未生效

当前 `scripts/build_options.json` 和 `build/CMakeCache.txt` 虽然都设置ON，
但 `CMakeLists.txt` 缺少以下选项的声明/目标编译定义，`carbon_core.dir/flags.make` 也没有对应宏：

- `CARBON_CT_ELECTRON_QUEUE`
- `CARBON_FE_JOINT_QUEUE`
- `CARBON_ELECTRON_REPLAY_REUSE`
- `CARBON_ELECTRON_CROSSING_ROWS32`

实际执行 `./scripts/run.sh proton_ct_minibeam` 在初始化阶段退出1：
`Queue execution requires CARBON_CT_ELECTRON_QUEUE build`。
没有输运历史，没有有效CT吞吐。7配置preflight通过和编译成功不能覆盖这个运行时问题。
不要把请求的500k histories除以失败进程1.7569秒报告成吞吐。

来源工作树的 `CMakeLists.txt:612–629` 有完整定义，可作为最小修复参照：前三项为ON时定义宏1，
FE项定义布尔0/1。源码前三项使用 `#if defined(...)`，不能在OFF时仍统一定义成0。
修复后检查实际编译命令，执行CT队列和FE队列小样本运行，再建立完整基线。
本报告记录问题，尚未实施该修复；当前水模体数字仅对应上面固定SHA的二进制。

## 优先级1：水电子响应审计分片，保留逐步整数语义

`src/transport_sycl.cpp:3434,6185–6267,13791`：水响应使用单份17槽 `uint64_t` 数组。
每个有效能损响应调用至少更新steps、loss-sum、delta-sum三个槽；响应有效且packet为正时还更新能量和ROI槽。
本次响应调用665,951,172次，源码语义对应至少1,997,853,516次审计atomic add调用。
这是实际调用数结合源码的工作量估算，不是SASS指令或硬件争用测量。
现有 `CARBON_EM_AUDIT_SHARDS=64` 针对另一套Unified EM审计，没有覆盖这里。

建议独立候选：把这17槽复制为64个设备分片，按lane/warp映射选择分片，读回时逐槽整数相加；
分片可补齐为32槽，64份仅16KiB。保留原先每步 `uint64_t(value*1e6)` 的转换位置，
不能先累计浮点能量再取整；保留invalid计数、全部ROI项和拒绝条件。
零增量跳过可以另作单一候选，不与首次分片实验混合。

它与失败台账中的Cu valid-loss、CT Urban-step逐历史累加不同：目标是独立的水响应审计，
实施方式是改变全局地址分布，不给每条history新增17个长期存活计数器。
此前那些失败说明“原子多”不等于“原子是瓶颈”；这里也只值得低成本筛查，不能预报收益。

## 优先级2：精确减少铜每步的几何和查表工作

`include/carbon/minibeam_collimator.hpp:270` 的 `minibeam_path_to_surface_interval`
对每次调用扫描全部15条狭缝；主核 `src/transport_sycl.cpp:4817` 每步调用它。
0.05mm铜短步远小于3.6mm pitch，可用横向可达区间保守裁剪候选slit，再执行原slab求交。
保留圆柱求交、有限狭缝端面、旋转和原FP32边界规则；长空气步走原完整扫描。
先做针对掠射、face±ULP、近零方向、圆柱边缘和旋转的原实现对照，再做GPU整批审计。
这不同于已失败的公共除法hoist，也不同于旧的固定“三个邻近slit+前后探针”导航。

`src/transport_sycl.cpp:5192` 的步后elastic nearest和inelastic rate源码位于collision分支之前，
只在该步碰撞发生时用于分类。可先检查编译代码是否已下沉；若未下沉，再建立显式分支候选。
必须保留步前hazard与步后能量查询，不能用步前rate替代步后rate。
这两个改动分别计时，不用组合结果认定各自收益。

## 优先级3：把铜终止核事件生成搬离热主核

现有队列仅搬走带电子产物的后续输运，`src/transport_sycl.cpp:5262–5326` 仍在主核中执行
CINEL03查找、事件旋转、遍历产物和中子出生记录。本次有681,782次Cu初级核反应，
每条初级至多一个终止事件，适合小型事件任务。

保留原碰撞判定和主粒子终止点；队列保存history/RNG history、spot seed、beamline step、
碰撞能量、坐标和方向。独立核按原dims63/64/65采样，产生现有带电子任务及原中子出生记录。
不能使用队列槽号替代RNG身份，不能遗漏lookup miss分类、removed能量账本、容量和排空检查。
可优先按现有history chunk排空，避免新增全批次大容量任务数组。

当前二进制静态资源（从内嵌CUDA images提取）：

| 主核 | registers/thread | stack bytes/thread |
|---|---:|---:|
| 水模体铜/空气stage9 | 250 | 1400 |
| 水模体水stage4 | 255 | 3888 |
| CT铜/空气stage9 | 254 | 1400 |
| CT患者stage4 | 255 | 7456 |

CT行是缺少电子队列宏的当前构建，修复后必须重新提取，不能外推为有效group32队列资源。
这些是静态资源，不是运行时spill、occupancy或stall。驱动 `RmProfilingAdminOnly=1`，
本轮未重试台账中受限的NCU计数器，也未改变驱动权限。
拆核目标是降低热路径代码和状态负担；寄存器小幅下降本身不构成速度证据。
水核也可进一步隔离冷路径，但要先确认编译器实际保留内容。
不能把6.66亿次水响应调用直接逐次变成大任务队列。

## CT的后续方向与已经完成的优化

当前YAML已选择compact、group32、tiered 2GiB hot budget、host_mapped和retain_bank。
源码已包含CT材料电子目录索引。水模体已有铜/快lane分组、铜带电子任务队列、16档水入口能量排序。
这些都不应被重复当成新优化。

来源工作树历史数据显示CT500k冷启动约20秒；10M在单进程复用包后完整吞吐约23.18k/s。
这是旧有效构建的证据，不能移植为当前构建实测。
修复构建后，优先用已有 `--plan-manifest` 和retain_bank摊销重复初始化，分别报告完整墙钟与暖批次。
需要进一步减少冷启动时，可考察离线缓存原始轨迹生成的links/crossings/birth索引，
以源SHA、格式/构建版本和布局参数验证缓存，仍校验原数据完整性；不能跳过SHA来提速。
在取得队列阶段计时、访问分布之前，不重复已失败的indices-first摆放或盲目增加hot budget。
group32仍有严格local gamma限制，不能把扩大group size作为无需精度验证的调度优化。

## 不重复的路线和验收方法

`failed.md` 已排除或限制：原样FE替换水Urban、铜内部删粒子、原样自适应大步长、
距离独立分组、常数factor端点压缩、强压寄存器、仅改查表索引、原样局部审计/能量账本累加。
条件铜响应历史有约13%收益，但局部谷剂量预算未证明通过；保留独立研究路线。
只需总剂量时，关闭分量输出历史约有1.7%收益，主要节省输出处理，不代表主核提速。

每个精确候选先检查编译资源和小样本，再做同种子4M的至少三组成对交替顺序计时，
固定同一构建/物理输入并记录GPU温度、时钟，报告完整墙钟、两主核及新增队列成本。
保持FP32 dose，逐项比较transport/EM整数审计、水响应整数项、中子计数、quality、能量账本和所有队列溢出/排空。
FP32剂量累加顺序差异按项目规则处理。剂量模型未改的候选不应通过放宽物理门槛来接受。
失败候选保留证据并追加台账。

## 证据位置

- 本轮：`out/throughput_review_20261001/summary.json`、`water_execution.json`、`water.log`、
  `water_quality_report.json`、`water_energy_ledger.json`、`ct_execution.json`、`ct.log`。
- 静态资源：同目录 `resources/primary.json`、`extract_resources.py`、`baseline_flags.make`。
- 原始剂量：`out/proton_water_minibeam/`。
- 历史报告：`/mnt/sdb/wuwei/MAIGO_review_fix_20260923/docs/` 下的
  `proton_minibeam_four_routes_20260930.md`、`proton_minibeam_next_arch_20261001.md`、
  `proton_minibeam_further_speed_20261001.md`、`proton_ct_compact_group32_20261001.md`。

本轮唯一新失败是CT基线启动的构建缺项，已追加 `failed.md`；没有新候选的实测加速或性能拒绝结论。

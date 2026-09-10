# 材料与方法

修订日期：2026-09-10。当前科研方法稿，不声明临床准入或通用 Geant4 等价。
[English](mm.md)。[物理规格](docs/TOPAS_GPU_Physics_Model.md) 描述 9 月 5 日基线；
后续实现变化在下文单列，并与源码核对。
旧稿的质子/多离子探索完整保留在 [归档](docs/archive/README.md)，不提升为已验证 CT 能力。

本次核对源码与构建图、配置及数据加载、TOPAS 提取工具、测试、benchmark 产物和研究计划，
源码检查点为 `f6245c5`（2026-09-09）。已有未提交输运修改不属于冻结发布版本。
除非另行标注，下文 CT 数值和 benchmark 配置均指 Git 中的 9 月 5 日数据集，
不代表当前全部代码路径。

## 1. 计算框架与参考

MAIGO 使用 C++20 / SYCL 实现带电离子凝聚历史 Monte Carlo。
患者后端为 [transport_sycl.cpp](src/transport_sycl.cpp)，串行后端仅支持子集。
本批使用本地 NVIDIA RTX 2080 Ti（sm_75）。
GPU 插值输运表并重放相关核末态，不在运行时调用 Geant4 或核级联生成器。

参考为 TOPAS 4.2.p3 / Geant4 11.3.2，模块：
g4em-standard_opt4、g4h-phy_QGSP_BIC_HP、g4decay、g4ion-inclxx、
g4h-elastic_HP、g4stopping、g4radioactivedecay。
模块名不能证明每种粒子的实际模型；参考 C12 已观察到调用 INCLXX，
不能因 list 含 BIC 就解释成 C12 BIC 与 INCLXX 的差异。

随机流按历史和过程通道索引。复现绑定 executable、输入、seed 和分配，
不保证跨设备/构建逐位一致。

## 2. 材料、源与几何

HU 映射为连续密度和 25 个 Schneider section，组成含 13 元素。
组成用共享 LUT，不复制到每体素；不能用材料探针代表密度替代 CT 实际密度。
CCTG/DICOM 与记分几何共同核对。

当前源采用 TPS spots CSV histories、beam-model emittance、能散、
虚拟扫描磁铁和显式 TOPAS placement。匹配每个 replica 的逐 spot 整数 histories，
再分配 GPU shards。TOPAS 被动 RotZ 的 world→patient 组件局部坐标为
R(+RotZ) × (world − Trans)，方向只旋转不平移。
source、placement、CT packing 是三个操作；按冻结映射恢复患者剂量，
不做剂量拟合配准。见 [planning](docs/planning.md)。

当前均匀水模与 CT 共用 CINEL03 核输运框架，使用 SHA 固定的原生 G4_WATER 组成，
并从材料 partial-rate 表恢复 H/O 元素反应率；不伪造 Schneider 水分区。
水的 stopping、密度、辐射长度和涨落输入仍按自身材料定义，不能静默换成 CT 表。
统一水路由拒绝旧 CINEL02 event/rate 配置键。允许水模 `run_mode: production`
不等于 TOPAS 精度已验收，质量报告仍明确记录验证未完成。
见 [统一框架记录](plan2/water_ct_unification.md) 与
[材料反应率接口](include/carbon/material_nuclear_rates.hpp)。

## 3. 电磁输运

步长受最大长度、相对能损、几何、核光学深度与终止条件约束。
三病例冻结上限为 0.5 mm 和 0.005。
Primary C12 使用 Schneider mass stopping/range 及局部密度；
次级使用 water-ion stopping、CT 材料因子与局部密度，
不是直接提取的完整 isotope×section stopping 数据集。
Upstream air 能损独立处理。

本批 primary/secondary 都用 section X0 与局部密度的 Highland MCS，
不是 Geant4 msc 算法复刻。
Primary straggling 开启、冻结 scale=1.2；secondary straggling 默认关闭。
记录这些参数用于复现，不授权按剂量拟合调参。

## 4. 核重放与次级范围

最低数据栈为 Schneider v2.1：SCHNRATE/SCHN2RAT v3 rates、
CINPKG04 v4 primary 和 14-projectile secondary 包、SCHNSTOP v1。
哈希由 [AGENTS](AGENTS.md) 和 [manifest](data/schneider/v2_1_data_manifest.json) 固定。

Hazard 为局部密度乘有效元素 partial-rate 总和，查询时按通道域 mask。
Lookup 精确匹配 projectile Z/A 和 target Z。
有效 bracket 上节点概率为 (Eq−E0)/(E1−E0)，非精确节点 gap >5 MeV/u 拒绝。
节点内抽取完整相关事件、保留能量，方向旋转到入射系；
无近邻靶 alias，无整体产物 KE 缩放。

9 月 7 日起，primary 和 secondary CINEL03 replay 对每个事件抽取一个绕入射轴的
均匀方位角（Philox dimension 60），对所有产物施加同一个旋转，保留相对角及事件关联。
不能逐产物独立随机化方位角。这一改动晚于 9 月 5 日冻结 executable。
见 [旋转函数](include/carbon/cinel03_event_rotation.hpp)。

Post-EM null 保留能量继续，不 replay、不局部倾倒；null 和失败分开报告。
有限步能量差与域屏蔽仍是声明近似，不证明无限制物理覆盖。

带电产物入队进行 EM 与已支持核输运。
冻结 generation=2，不是无限 cascade。
He6/B8/C10 采用 scope 外 EM-only 核策略，Be6 单列。
Overflow 使运行无效；本批独立 nuclear elastic 关闭，
非弹性末态不意味着隐含包含 elastic。通用 neutral/衰变输运未验证。

## 5. 电子响应

严格剂量栈含 TOPAS 派生的 section-0 primary C12 部分能损横向重分配，
提取点为 150/200/225 MeV/u。
它搬运已记账能量，不额外加能；source eligibility、目的材料和 scorer 逃逸限定范围，
不是通用电子输运。

9 月 5 日三病例 executable 含 entrance-mask candidate。后续纵向、联合响应及完整电子家族
有序路径实现属于独立实验，不受该 benchmark 验证。有序重放保留记录的路径和谱系，
不把轨迹简化为起终点之间的直线。

进一步的材料响应候选为 water 或 Schneider CT 加载 SHA 固定的出生分布、原始电子段和
续接索引。其输运对象是具有能量单位的统计包 W，与抽样电子的动能不同。
每个包只能有一个最终能量归属：沉积、患者/记分网格逃逸或明确的未覆盖能量。
有限源路径耗尽需要续接，不等于患者逃逸；缺失的光子续接单独记账，不局部沉积。
同 section 密度插值混合实测条件分布，完整密度域和界面精度仍未验证。
电子出生材料与边界遍历必须采用一致的体素归属；响应库支持显式预算的设备内存或
host-mapped 内存模式。

材料候选与旧 delta-tail、joint 和显式电子模式互斥，要求 3D 记分并关闭 LET。
质量报告仍拒收为 `unvalidated_material_electron_response`，闭合通过不能消除此项。
见 [能量包输运](include/carbon/electron_packet_transport.hpp)、
[密度采样](include/carbon/material_electron_density.hpp) 和 [质量门禁](src/run_quality.cpp)。
跨密度、几何、出生条件和正式升级门禁仍由 [plan2](plan2/README.md) 管理。

## 6. 记分与质量

Dose(Gy) = Edep(MeV) × 1.602176634e−13 / mass(kg)，
mass = density(g/cm3) × volume(mm3) × 1e−6。
输出为累计 3D DoseToMedium；IDD 由横向求和得到，不使用 1D scorer。
等中心 profile 为插值取线，不是 IDD。

可选 LET_d = sum(L × dE)/sum(dE)，多片先合两类矩再相除。
本批 dose benchmark 关闭 LET，不提供新 LET 验证。
Origin dose 只是诊断，不是独立 TOPAS 分种类参考。

验收包含有限数值、严格粒子/candidate 计数、零 overflow、数据 provenance 与能量闭合。
显式 sinks 检查 deposited = in-grid + outside-grid、voxel sum = in-grid，
当前实现相对容差为 1e−3。
全局闭合不证明每个核顶点的精确质量/Q 闭合。
字段与兼容条件详见 [记分契约](docs/scoring_validation.md)。

## 7. 评价与局限

2026-09-05 数据集为 60 个 accepted shards、457,898,870 histories、零 overflow。
数值维护于 [当前结果索引](docs/results.md) 与冻结证据。

采用名义累计 Gy，无 LS scale、无拟合配准。
Mask 为全 reference 体积 Dref ≥10% 全体积 Dmax，无 BODY mask。
Global 容差基于 Dmax，Local 基于当前参考体素剂量。
判据为 3%/3mm、2%/2mm、1%/1mm、3%/0mm。
非零距离为 0.5 mm 球内格点加三线性插值，不是解析连续最小化；
3%/0mm 比较同体素。

结果仅绑定冻结输入/executable。严格 Local、低密度和界面残差仍存在，
未证明唯一电子原因或全物理等价。
本批不代表 plan2 完成、包升级或临床准入。

## 8. 后续评价与性能研究

本地未纳入 Git 的 `benchmark/topas10x/gpu_current_20260909.md` 记录后续三病例
60-shard 运行、Gamma 加密重评及 RT07575 实验性电子 full20 A/B。
本次仅更新两个方法文件，不发布这些本地产物，也不以其替换已发布的 9 月 5 日证据。
其中电子实验不提升最低物理数据版本；复现需对应 executable、配置、原始剂量 SHA
和质量报告。

后续工具 `tools/evaluate_gamma_adaptive.py`（源码检查点 `f6245c5`）先复现冻结的
0.5 mm pass mask，仅对失败点使用同一三线性插值体上的 0.25 mm 格点重搜。
保留粗筛通过点，3%/0mm 不变。工具核对剂量 SHA 和 histories，独立写入
`gamma_refined.json`，不覆盖冻结结果。粗细结果必须分别标注：搜索加密改变数值评价，
不改变输运剂量；两种有限格点都不是精确连续最小值。该工具属于后续本地提交，
不包含在本次仅文档发布中。

性能报告区分 primary+secondary GPU kernel 时间、程序内部 transport elapsed，
以及包含响应库加载和 I/O 的完整进程墙钟。Histories/s 必须说明时间分母、统计量、
chunk、记分选项和硬件；小样本 smoke 不能证明相对 TOPAS 的全计划加速。

[已入库的 64-history 短程对比](benchmark/benchmark20260909/short_range_64_comparison/README.md)
使用患者 20022516，阈值为 0、0.1、0.25 mm。三组 raw dose 逐位相同、overflow 为零，
但均因材料响应未验证而拒收。Primary kernel 分别为 61.2625、61.5884、61.5946 s，
没有证明加速。短路要求完整、无子代的终止尾段位于当前体素内，阈值默认零。
后续缓存尾长实现仍是候选，不构成已验收性能结果。

## 9. 复现与验收流程

构建由 [CMake](CMakeLists.txt) 和 [presets](CMakePresets.json) 定义：C++20、开启 SYCL、
NVIDIA target `nvptx64-nvidia-cuda`、本地架构 `sm_75`。
冻结 executable SHA、解析后配置、source/CT 变换、全部数据 pins、spot 分配、seed、
记分选项和 shard manifest。每批 Schneider CT 运行前执行
`python3 tools/verify_schneider_v2_1_data.py`；缺文件、SHA/大小/schema 不符或
14-projectile 覆盖缺失均为硬失败。新 clone 不保证包含大二进制包和外部原始响应。

GPU 仅在本地 RTX 2080 Ti 运行。TOPAS 提取使用本地 `sbatch`，数据位于
`/mnt/sda/wuwei`，源码、构建与 extension 位于 `/home/wuwei/topas`。
所有任务合计最多 192 CPU 线程、160 GB 内存，并按计算量分配。
大量 GPU histories 必须分片；任何次级 overflow 都需要缩小分片重跑。
仅合并逐片验收通过的结果，再检查聚合能量和 histories 记账。

数据升级要求明确 TOPAS/Geant4 provenance、manifest 与 host/device lookup 检查、
50k 闭合、配对单 shard Gamma 和零溢出完整验证；电子候选另需独立 phantom、域及几何门禁。
历史 fixtures 和旧文档不证明当前测试已通过；本次方法稿更新没有新增输运或物理精度验证。

## 参考与追溯

- [FRED 论文解读与源文入口](docs/FRED_Carbon_Fragmentation_Model.md)
- [物理规格](docs/TOPAS_GPU_Physics_Model.md) 与 [代码地图](docs/structure.md)
- [冻结 benchmark 与 provenance](benchmark/topas10x/gpu_current_20260905.md)

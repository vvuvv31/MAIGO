# Scoring 与验证契约

更新日期：2026-09-05。替代根目录 scorer_benchmark 的建议稿；
描述现行输出和验证方法，不把建议字段假称为已实现。
源码入口：[io.cpp](../src/io.cpp)、[run_quality.cpp](../src/run_quality.cpp)、
[transport.hpp](../include/carbon/transport.hpp)。

## 1. 主输出：3D DoseToMedium

GPU 累积体素 Edep（MeV），按实际 CT 连续密度和体积转为累计 Gy：

```text
mass_kg = density_g_cm3 × voxel_volume_mm3 × 1e−6
dose_Gy = edep_MeV × 1.602176634e−13 / mass_kg
```

不拿 water 或 section 代表密度代替逐 voxel 密度。
MHD 描述尺寸、spacing、origin、ElementType 和 raw 文件；
读 raw 前必须看 header，不能假设 TOPAS float64 Sum 是 float32。
仅有 raw 数组不足以定义物理剂量。

对齐的同网格累计 Gy shards 直接求和，不除 histories 后又乘 scale。
若换算 MeV/primary，则先做逐体素 `dose × mass / MeV_to_J` 求和，
再除实际 histories。`sum(Gy)` 不是总能量，不能直接用它检验能量守恒。

只使用 3D dose scorer。IDD 由横向求和获得，必须声明求和的是 Gy 还是
先换算成 MeV 的沉积；过等中心 XYZ profile 则是插值取线，两者不同。

## 2. Origin / species 诊断

`enable_charged_origin_voxel_scoring` 与
`charged_origin_voxel_mhd_output_prefix` 控制带电 origin 体素输出。
MHD writer 目前包含 8 类：

`primary`、`secondary_carbon`、`secondary_boron`、
`secondary_beryllium`、`secondary_lithium`、
`secondary_helium`、`secondary_proton`、`secondary_other_charged`。

这是 origin 类别，不等于所有 isotope 的独立物理 scorer。
nuclear-local 沉积的归类以调用点为准，不能因为 8 类闭合就声称有独立
nuclear-local reference；也不能把一级 p/He 的类别解释成原发电子剂量。
Be isotope 和 OOS isotope 还有专门诊断，字段与 null 语义以 writer 为准。

必须核对：类别和与总 dose 闭合；开/关 scorer 不改变 total dose；
几何 header 与 total 一致。缺失/未分配字段记“未测”或 null，不当成物理零。
CT origin 诊断优先使用已经验证的 MHD 路径；稀疏 CSV 有不同的数组/类别检查，
旧配置默认输出键可能触发不兼容 writer，不能假设 MHD 通过就代表 CSV 通过。

计数诊断包括 survival、核 candidate/replay/null/失败、born/queued/terminal、
generation 和 overflow。仅有“secondary steps 很多”不足以证明末态覆盖。

## 3. LET

LET_d 使用 `sum(L_i × dE_i) / sum(dE_i)`。保留 numerator/denominator，
多 shard 先分别求和再相除，不能对 LET 图取算术平均。
需声明 primary 与 all-hadron 的范围、restricted stopping/cut 定义及终止处理。

LET delta-fraction 表调整 LET 定义；section-0 dose delta-tail 表调整剂量空间分布；
这不是同一功能。当前三病例 `LET: false`，其 dose Gamma 不构成 LET 验证。

## 4. Energy ledger 与质量

全局 deposited、escaped、beamline、untracked 等字段用于能量账本；
neutral/unsupported/Q 的信息性拆分不能重复加入已含这些项的总账。
完整方程和 residual 以该版 ledger/schema 为准，不用差额冒充独立测量。

显式 grid sinks 还检查：

```text
deposited_total ≈ in_grid_deposited + outside_grid_deposited
sum(voxel_Edep) ≈ in_grid_deposited
```

当前实现的 split/voxel-in 相对容差为 1e−3；另保留 voxel 超 total 2% 的粗检查。
不能只检查后者就称闭合。旧/CPU 无 split 数据时门禁有兼容条件，
因此还要确认当前运行确实分配并记录非零 sinks。
delta-tail 搬到 scorer 外的能量有专用项；scorer 逃逸不等同于 primary 逃逸。

每片运行至少检查：

- accepted 与 failures，NaN/Inf、队列/步数等 overflow；
- primary/secondary lookup 的 missing、domain、gap、empty 分类；
- candidate = replay + post-EM null + stopped + 各互斥失败；
- born 与各终态严格等量，Be6 特例按同一口径单列；
- scope 外 EM-only 的 tracks/birth/沉积/逃逸及未测字段；
- energy / grid 闭合及输入、executable、配置的 provenance。

上述检查通过只支持声明 scope 内的 research run，不是全物理认证。
overflow 后不可保留降级剂量进入正式合并。

## 5. 当前 Gamma 定义

正式数值入口：[2026-09-05 benchmark](../benchmark/topas10x/gpu_current_20260905.md)。
reference 为 TOPAS 3D DoseToMedium Sum replicas，evaluation 为完整 GPU shards 之和。

| 项目 | 当前冻结定义 |
|---|---|
| Mask | 全 reference 体积中 Dref ≥10% 全体积 Dmax；无 BODY mask |
| Histories / dose | 实际逐 spot 整数匹配；累计绝对 Gy；无 LS scale |
| 坐标 | 输入决定的患者映射；无剂量拟合配准 |
| Global | 剂量差容差以 reference 全体积 Dmax 为分母 |
| Local | 以当前 reference 体素剂量为分母 |
| 33 / 22 / 11 | 3%/3mm、2%/2mm、1%/1mm |
| 非零距离搜索 | 0.5 mm 三维球内格点 + 三线性插值；不是解析连续最小化 |
| 30 | 3%/0mm，同体素剂量差，无空间搜索 |
| 主报告统计 | 全阈值 mask；50k seed42 抽样结果仅作附加 |

旧 BODY∩10% BODY peak、LS 拟合、不同插值/采样数的 Gamma 不能直接作本批增益。
3%/0mm 对局部空间差异敏感；1%/1mm 通过不代表同体素差已经满足 3%。
mask 外 halo 不进入该 Gamma，须单独按空气/组织、界面、剂量带分析。

## 6. 证据冻结与重复运行

至少保留 executable SHA、源码 commit/dirty 状态、配置及输入 SHA、实际 histories、
各 shard quality/ledger/log/dose SHA、reference headers、mapping 和 Gamma 脚本。
失败或取消的运行单列，不混入 accepted 聚合；新结果用新目录。
只读比较不改冻结证据。不得将“测试通过”“数据 verifier 通过”“GPU 运行通过”
混写成同一个结论。

工具：
[evaluate_topas10x_gpu_gamma.py](../tools/evaluate_topas10x_gpu_gamma.py)、
[plot_topas10x_isocenter_profiles.py](../tools/plot_topas10x_isocenter_profiles.py)。
编译期 validation scorers 与运行期输出开关各有职责，见 [structure](structure.md)。

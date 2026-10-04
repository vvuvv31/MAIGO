# Broadbeam 共用 FP32 电子输运接入（2026-10-04）

后续更新：用户已明确要求将四个broadbeam默认切换到低密度＋5mm tracking策略，实施见[默认开关报告](broadbeam_tracking_defaults_20261004.md)。下文“保留可选、不替换默认”记录的是验收完成时的历史部署决定，已由该指令更新。

本轮实现为显式研究入口，不替换已验收的默认物理配置。Proton CT 默认去掉 FE tail 的先前用户选择仍保留。两种 CT broadbeam 可以复用 minibeam 的电子/光子输运，以独立的 proton/C12 出生分布提供初始能量、方向和电子能损占比。

## 实现与范围

- `broadbeam_independent_electrons: true` 选择 `broadbeam_fe_independent` 特化；原来的 local / C12 joint 特化仍保留。
- 原 `src/detail/sycl_independent_electron_fp32.inc` 直接复用，没有第二套 broadbeam 电子物理代码。所有 GPU 剂量评分和原子加法保持 FP32。
- `broadbeam_electron_density_threshold: 0.1`（g/cm³）、`broadbeam_electron_halo_mm: 5` 是首轮研究参数。按 CT 体素密度选区，用三轴盒状扩张纳入周边体素；按实际随机出生位置判断是否进入电子队列。
- 选区外主粒子电离能损局部沉积；选区内按出生包的电子能量占比分配能损，group32 带权抽样沿用原队列。已进入队列的电子/光子可以离开选区，继续使用完整共享输运，直到沉积或逃逸；不在选区边界截断。
- 周边宽度是新近似参数，不是用 CSDA 平均射程证明绝对包含。当前只接管主 proton/C12 的电离电子；带电核次级自身产生的电子仍局部沉积。电子产生后的电子/光子级联使用共享模块。
- 默认均匀 water broadbeam 没有低密度 CT 区域，保持已有局部路径和零电子包加载。新显式研究入口要求 CT，不能直接给非 CT water 配置开启此开关。

## 三类数据复用

1. Proton：`data/ct_electron/proton_independent_v1/birth_bank.bin`，原包不变。
2. Carbon：研究扩展 `data/ct_electron/carbon_independent_v2/birth_bank.bin`，1,029,661,608 bytes。原 v1 的 63 节点 payload 逐字节保留，新增 19 节点、39,657,600 bytes；共 82 个节点。原 carbon minibeam 默认和软组织试验的包未替换。
3. Shared：原 `data/ct_electron/shared_independent_v1/` 的 `schneider_joint.bin`、charged 和 photon 索引及 payload，不重新提取。

原 carbon 出生包对 ≤0.1g/cm³ 区域本身已有覆盖，但扩张 2mm 就包含缺少出生节点的骨材料；5mm 扩张涉及 section15–24。新预检逐一检查选区中的出生材料/密度；不能用局部沉积、密度截断或 proton 出生包掩盖缺口。共同输运材料覆盖仍检查整个 CT。

新增19节点使用原生 C12 255MeV/u（总能量3060MeV）、每节点64历史、0.05mm production cut、主步长≤0.25mm 的独立出生提取。记录器不抽随机数、不杀粒子、不改物理过程。先以节点63的4历史完整输运记录逐条核查：6,484个出生电子的能量完全一致，方向余弦最大差1.11e-16，所有主粒子能量bin闭合。新增节点也逐个检查能量账；有限出生统计不等于CT剂量验证。提取主步长与当前broadbeam主步长0.5mm不同，需由剂量对照检验使用该出生分布的近似。

Carbon扩展包 SHA256：`982728e3b78c9561e926fdb014204d93286a84e2e1781c5ebc9ab4e23310c91f`。

## 运行方式与内存

入口：`config/proton_ct_broadbeam_independent_trial.yaml`、`config/carbon_ct_broadbeam_independent_trial.yaml`。

RTX2080Ti 上不要用新 carbon 路径直接跑单个 6,481,909-history shard：主粒子和电子已跑完，但后续带电次级的续跑状态分配失败，程序无有效剂量输出。未关闭或丢弃次级；改用已有的单进程连续批次，按整个计划的整数 spot 配额平衡分配，包只加载一次：

- Carbon：7 ×925,987 =6,481,909 primary。
- Proton：10 ×1,253,150 =12,531,500 primary。

两组所有批次通过运行质量、EM、能量、覆盖与零队列溢出检查；分别有6/9次包缓存命中。完整记录位于 `out/broadbeam_shared_electron_20261004/`。GPU批剂量为FP32；分析阶段汇总各批剂量使用FP64再存储合并FP32，这不是FP64 GPU scorer。

运行审计和 TOPAS 剂量验收分别记录如下；本病例通过不改变数据包的通用验证范围。

## 用户更新的验收标准与5mm结果

按用户本轮明确要求，主验收改为 **local γ 1%/1mm >99%**。3%/0mm保留诊断，不作为否决项。下表均为参考最大剂量10%阈值、全3D评估区、绝对剂量归一化、不做剂量比例拟合或配准。

| 场景 | 局部沉积对照 local11 | 共用电子5mm local11 | 结论 |
|---|---:|---:|---|
| Proton RT07575 | 99.4458% | **99.9120%** | 通过 |
| Carbon RT07575 | 96.2842% | **99.8677%** | 通过 |

Proton 的历史 BODY/GTV 评估区另为99.9486% →99.9489%。Carbon 将搜索步长从0.1mm细化至0.05mm后，共用电子为99.8782%。这是本病例、束流谱、选区参数和统计量的验收，不是所有病例、任意密度阈值或carbon minibeam的通用验收。

Proton对照来自已固定SHA的优化计划TOPAS剂量影响矩阵重建，按原计划 `optimized_protons_per_fraction / actual_histories` 归一化；Carbon来自五个原始TOPAS replica（129,638,170 primary），核查原始剂量/几何/源SHA后按各自primary数归一化。不同粒子的参考生成方式不同，未新增TOPAS计划模拟。

同粒子数的Carbon local3%/0mm为局部72.2003%、共用72.5155%；Proton全评估区为79.3058%、77.8789%，BODY/GTV为84.2701%、84.1959%。这些0mm结果受蒙卡噪声影响，未单独拟合噪声上限，也不作为此次否决条件。不能拿先前19.45M三种子合并的joint结果与当前6.48M结果直接解释为模型精度下降。

5mm两种新路径均已通过全部17个连续批次的运行门禁：proton最大能量相对残差4.42e-7、carbon3.79e-6；零电子/光子覆盖错误、零队列溢出，包完成数分别38,994,104和20,089,776。缺少骨出生节点、错用proton出生包、非法周边参数三项负例均明确拒绝，且不输出剂量。

## 5mm →10mm 邻域收敛检查

保持每批 spot 整数配额、种子、总历史数、FP32 评分和其余物理配置相同，仅扩展出生选区。两组所有批次的源文件 SHA 完全一致。

| 场景 | 5mm local11 | 10mm local11 | 评估区剂量总和变化 | 剂量差绝对值之和 / 5mm剂量和 | 10mm持续 primary/s |
|---|---:|---:|---:|---:|---:|
| Proton | 99.9120% | 99.9165% | −0.000456% | 0.07330% | 79,807 |
| Carbon | 99.8677% | 99.8699% | +0.000583% | 0.15101% | 50,256 |

Carbon 低密度区（≤0.1g/cm³）local11 从局部沉积的81.8990%提高到5mm的99.9813%，10mm为99.9824%；界面区分别97.6798%、99.9033%、99.9077%。改进主要覆盖了原局部沉积失配的区域。

两种邻域均通过用户 >99% 门槛。保留5mm作为这两个可选预设的参数：10mm没有显著改善病例gamma，持续吞吐分别下降约11.3% /7.2%。这是当前病例的收敛检查，不是任意几何的射程保证，也不能由剂量总和稳定推断每个体素都在1%以内。5→10mm的局部绝对变化P95（以TOPAS局部剂量为分母）为proton0.5248%、carbon1.0787%，包含抽样和路径改变的影响；未独立分离蒙卡噪声。

10mm全部17批通过能量、EM、电子/光子覆盖、队列完成和零溢出检查，仍只加载一次包。最终局部对照、5mm、10mm共51批的审计，以及正式二进制和源码SHA核对，记录在 `out/broadbeam_shared_electron_20261004/acceptance.json`；收敛和分区明细分别为 `halo_comparison.json`、`carbon_regions.json`。

## 吞吐与部署决定

以下是相同二进制、相同整数spot分配和批次种子的对照，持续吞吐排除首批加载，包含后续每批准备与输出。两组的 `primary_batches.csv` 已逐批SHA比对一致。

| 场景 | 局部沉积持续 primary/s | 共用电子5mm持续 primary/s | 新路径电子kernel总耗时 |
|---|---:|---:|---:|
| Proton | 403,892 | **89,983** | 84.50s |
| Carbon | 192,083 | **54,128** | 75.26s |

共用输运当前没有吞吐优势，主要成本来自显式电子输运。它完成了物理模块复用和病例剂量门槛，不是性能优化胜出。

Carbon旧joint预设仍带唯一历史research quality失败，原生连续批次会在首批后停止；没有关闭该门禁来凑配对。上表Carbon基线是局部沉积，不是旧joint。此前旧joint的独立高统计验收保留在 `docs/carbon_joint_acceptance_20261002.md`，不能当作本次同批次测速。

两个water broadbeam各50k primary、原proton CT minibeam20k的前后回归通过：EM审计一致、quality通过，剂量relativeL1分别5.00e-8、2.22e-7、2.59e-9。无需重新编译的已测二进制已安装到 `build/carbon_mc`，SHA256 `834e98c493e487e7838989d88d3c5dfcb6c12e4723db20922e7f9ee697de1198`；原二进制保存在证据目录 `before/carbon_mc`。

保留新路径为显式可选入口，八个默认物理场景不因此次通过剂量门槛自动改用更慢的电子处理。运行示例：

```bash
./scripts/run.sh proton_ct_broadbeam_independent_trial --batch-count 10 --histories-per-batch 1253150 --batch-output out/proton_broadbeam_shared_example
./scripts/run.sh carbon_ct_broadbeam_independent_trial --batch-count 7 --histories-per-batch 925987 --batch-output out/carbon_broadbeam_shared_example
```

输出目录必须新建或为空。代码、数据、原生提取审计、烟雾/回归/保护检查、连续批次原始剂量、参考审计和gamma脚本均留档在本报告所列路径。

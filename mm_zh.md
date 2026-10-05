# 材料与方法

更新：2026-10-05。[English](mm.md)。本文描述八预设清理及 CT 调度优化版，详见[清理验证](docs/repository_cleanup_20261005.md)与[CT 吞吐和验收记录](docs/ct_throughput_20261005.md)。新增[CT broadbeam 电子优化](docs/broadbeam_300k_20261005.md)。两个 CT broadbeam 已通过指定门槛；肺 CT proton minibeam 仍未通过。历史测量只对应其记录的程序、束流、几何及统计量。

## 八个预设

仅保留以下八个 `config/` 入口。粒子种类和几何各自选择，共用电子传播实现。

| 预设 | 离子多重散射 | 电子出生选区 | 当前验证状态 |
|---|---|---|---|
| `proton_ct_broadbeam.yaml` | Gaussian Fermi–Eyges | CT 低密度区及周围5 mm | 当前三维 local 1%/1 mm 99.8935%，通过 |
| `carbon_ct_broadbeam.yaml` | Gaussian Fermi–Eyges | CT 低密度区及周围5 mm | 当前三维 local 1%/1 mm 99.8699%，通过 |
| `proton_ct_minibeam.yaml` | Urban | CT 内全部 primary 电离损失 | 共用 Urban/FP32；肺 CT 尚未通过新 >99% 门槛 |
| `carbon_ct_minibeam.yaml` | Urban | CT 内全部 primary 电离损失 | 调度提速1.946倍；输运审计通过；剂量待验收 |
| `proton_water_broadbeam.yaml` | Gaussian Fermi–Eyges | 单位密度均匀水中选区为空 | 两批 smoke 通过；water 专项验证推迟 |
| `carbon_water_broadbeam.yaml` | Gaussian Fermi–Eyges | 单位密度均匀水中选区为空 | 两批 smoke 通过；water 专项验证推迟 |
| `proton_water_minibeam.yaml` | Urban | 水内全部 primary 电离损失 | 共用 worker smoke 通过；剂量待验收 |
| `carbon_water_minibeam.yaml` | Urban | 水内全部 primary 电离损失 | 共用 worker smoke 通过；剂量待验收 |

两个 CT broadbeam 预设使用 RT07575 计划输入。两个 CT minibeam 预设使用肺 CT，分别为170 MeV 质子和250 MeV/u C12；它们不是 RT07575 minibeam full plan。计算后者需要使用该计划的实际 CT、spots、归一化和参考剂量。Carbon water minibeam 保留250 MeV/u spots 源及1.2%相对能散。

## 物理和几何

MAIGO 是 C++20/SYCL 蒙特卡罗程序，使用 TOPAS/Geant4 提取的材料表、条件终态和核反应事件样本。提取/参考环境为 TOPAS4.2.p3、Geant4 11.3.2。使用这些数据及通过特定病例验证，不等于与任意 Geant4 设置完全等价。

统一 EM 表覆盖水、Schneider 材料及密度节点和输运的带电粒子。凝聚电离损失使用提取的阻止本领和 delta 损失矩。Primary、secondary 核反应保留粒子特定的反应率及 CINEL03 事件；质子和 C12 的电子出生分布也不能互换。铜准直器保留自身的阻止本领、弹性散射、核反应和空气数据。

水的阻止本领参考是 **Water_75eV，平均激发能75 eV**。`data/water_unified/g4_water_material.json` 提供组成/辐射长度信息，文件中的 G4_WATER 标识不能把阻止本领参考改成78 eV。本次没有引入经验阻止本领缩放。

CT 电子直接使用患者材料/密度网格。均匀水的电子使用覆盖物理水箱的单个 Water_75eV 体素，交给同一个 worker；物理输运边界与剂量评分网格分开，离开评分 ROI 不等于逃逸出水箱。构造电子几何不会把水中离子切换为 Schneider CT 物理。

Minibeam primary 最大步长不超过0.25 mm；carbon water 当前保留更小的0.005 mm 上限。离子的 Urban 多重散射、独立电子传播和 FP32 剂量累加是不同层次。“Urban＋FP32 电子”不表示用离子的 Urban 散射公式替代电子弹性散射律。

## 共用电子输运

Primary 电离损失按粒子、材料及密度对应的能量加权电子出生比例分配。剩余部分局部沉积，电子部分成为带权任务。当前 minibeam 路径尚未加入 secondary ion 的显式电子出生或铜/空气中的上游电子 cascade。核反冲沉积不混入 primary 电子出生响应。

四个 minibeam 预设使用同一个独立 FP32 队列 worker：抽取电子出生能量和方向，使用提取的弹性及硬碰撞数据、Urban 能损涨落推进带电包，使用共用 photon operator 处理光子分支，并保留能量账目。Schneider 中有覆盖的状态可使用短 joint-response 推进；Water_75eV 没有 joint operator，完整使用同一个独立输运实现。不会回退到已删除的旧水/材料空间轨迹库。

当前选用 FP32 输运/几何及 FP32 剂量原子累加；CDF 保留整数抽样。出生队列只保留能量、材料节点和原始 CDF 随机位，不再携带旧轨迹引用。Minibeam 默认 group32，以加权 reservoir 抽样把多个 primary 出生贡献合成代表性电子任务。分组会影响方差，不能只凭任务数降低判断效率和精度。

Broadbeam 使用 YAML 开关：

```yaml
broadbeam_electron_tracking: true
broadbeam_electron_density_threshold: 0.1
broadbeam_electron_halo_mm: 5.0
```

CT 默认对密度≤0.1 g/cm³ 区域做5 mm 盒形扩张，并按实际抽到的出生位置决定是否入队。入队后的输运不局限于选区。单位密度均匀水的选区为空，因此在该策略下保持局部沉积。关闭开关会选择局部沉积，并不为该次运行加载独立电子包。无需质子专用 tail 或碳专用 joint replay 路径。

当前 CT broadbeam YAML 还保留低密度包存活概率0.5及相应权重补偿，必须检查带符号权重账目、任务与能量审计。这是方差缩减，不能据此无权重补偿地丢弃体外空气电子或免除剂量验证。

两个 CT broadbeam 默认开启 `broadbeam_electron_birth_order: true`，按出生能量和密度对电子任务索引分组；`broadbeam_electron_tissue_survival: 0.5` 对选区内密度高于0.1 g/cm³的出生任务做 W/p 轮盘抽样，幸存电子及光子后代仍完整输运。设为1关闭组织抽样；tracking OFF 同时复原这些选项。它们不应用到 minibeam 或 water。

## 数据包

电子包注册表见 [docs/electron_packages_manifest.json](docs/electron_packages_manifest.json)，共三个包：

| 包 | 运行内容 | Payload 字节数 |
|---|---|---:|
| `data/electrons/proton/` | 质子 `birth_bank.bin` | 328,620,496 |
| `data/electrons/carbon/` | C12 `birth_bank.bin` | 1,033,490,328 |
| `data/electrons/shared/` | `joint.bin`、charged/photon operators 与索引 | 3,837,722,550 |

去重后的 payload 共5,199,833,374字节，不含 JSON 清单。每次只加载所选粒子的出生包和共用包。83节点加载器报告的设备数据约为质子3.127 GB、碳3.597 GB，不包括其他物理表、队列、评分器，也不代表进程总占用或加载峰值。

Birth、charged、photon 各含82个 Schneider 密度节点和1个 Water_75eV 节点。电子材料 section25 表示水，Schneider 仍为0–24。水中电子/光子覆盖至0.8 MeV；Schneider 保持0.65 MeV。水中 primary 出生数据覆盖质子250 MeV、碳300 MeV/u；Schneider 分别为250、255 MeV/u。不能把这些上限解释为任意源均已验证。实际材料/能量覆盖缺失必须报错，不能替换成水或局部沉积。

现用格式为 MAIBIR02 birth、MAIJNB02 joint、MAIECB02 charged index、MAIPHB02 photon index。索引为材料编号、密度、相对文件路径三列。检查内容包括尺寸、字节数、材料/粒子身份、有限值、抽样支持及碰撞率上界。运行时和当前提取工具均不再做文件摘要校验。旧配置键直接报错，不提供摘要字段或旧包格式的兼容入口。

其他必要包包括统一 EM/delta moments、粒子特定核反应率/事件、所选弹性/反冲数据、minibeam Urban 表、铜/空气和可选 neutron 数据。活跃依赖见 [docs/package_manifest.json](docs/package_manifest.json)。大文件和患者输入保存在被 Git 忽略的 `data/`、`inputs/` 中，仅克隆仓库不足以运行全部预设。

## 编译和运行

需要支持 CUDA 的 SYCL 编译器及插件。RTX2080Ti 示例：

```bash
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=/home/wuwei/sycl_workspace/llvm/build/install/bin/icpx \
  -DCMAKE_BUILD_TYPE=Release -DCARBON_ENABLE_SYCL=ON \
  -DCARBON_ENABLE_MINIBEAM=ON \
  -DCARBON_SYCL_TARGETS=nvptx64-nvidia-cuda -DCARBON_CUDA_ARCH=sm_75 \
  -DCARBON_DOSE_FP32=ON -DCARBON_DOSE_FP64=OFF
cmake --build build -j 6
python3 scripts/check_electron_packages.py
```

其他 GPU 应选择相应架构。`build/carbon_mc` 已包含次级粒子弹性检查修复（`5cc096d`），冻结副本为 `out/ct_throughput_20261005/guard_fixed_carbon_mc`。清理前的程序保存在 `out/repository_cleanup_20261004/before/carbon_mc`。

单进程连续批次：

```bash
build/carbon_mc --config config/proton_ct_minibeam.yaml \
  --batch-count 10 --histories-per-batch 1250000 \
  --batch-output out/proton_ct_minibeam_run
```

输出目录应为新目录。每批独立种子、剂量和质量报告，同一进程复用数据包及只读几何。计划权重按请求的总历史数分配，已归一化的计划剂量不能再次乘以批次数。不同配置依次运行可用 `--plan-manifest FILE`，并用 `--plan-preflight` 预检查。

## 验收和吞吐优化

四个 CT 预设采用以下验收标准：

| CT 预设 | 剂量门槛 | Gamma 搜索步长 |
|---|---|---|
| Proton / carbon broadbeam | Local 1%/1 mm **>99%** | 0.1 mm |
| Proton minibeam | Local 3%/0.2 mm **>99%** | 0.02 mm |
| Carbon minibeam | 暂不以剂量准确性为门槛 | 若计算则单独报告 |

对比前固定参考、绝对剂量归一化、坐标映射、剂量阈值和评价 mask，并随结果报告。当前 RT07575 broadbeam 使用参考最大剂量10%以上的全三维评价区，未拟合剂量比例或配准。搜索步长变化不允许伴随评价区或分母改变。零 DTA gamma 保留为受蒙卡噪声影响的诊断。

RTX2080Ti 最新 broadbeam 结果采用两个独立种子的连续吞吐及合并剂量 gamma；minibeam 行保留此前较大统计量研究：

| 预设 | 历史数 | 连续批次 primary/s | TOPAS local gamma |
|---|---:|---:|---|
| Proton CT broadbeam | 2 × 12,531,500 | 203,021 | 1%/1 mm：**99.8935%**，通过 |
| Carbon CT broadbeam | 2 × 6,481,909 | 138,019 | 1%/1 mm：**99.8699%**，通过 |
| Proton CT minibeam | 20,000,000 | 51,821 | 二维投影 3%/0.2 mm：**98.3183%**，未通过 |
| Carbon CT minibeam | 两个种子，每个对照/优化组各3 × 200,000 | **10,048**，对照5,165 | 剂量准确性推迟验收 |

最新同轮对照中，质子201,356→203,021/s（+0.8%，变化较小），碳离子129,244→138,019/s（+6.8%）。两种粒子的合并 gamma 均与同种子对照完全一致。共用 EM 模块复用同一步中相邻密度节点的搜索下标，保留原插值、能损、随机数及输运；网格一致性只在不可变数据包加载时检查一次。详见[EM 搜索复用报告](docs/broadbeam_em_search_20261005.md)及[数据](docs/broadbeam_em_search_20261005.json)。

[上一轮电子优化](docs/broadbeam_300k_20261005.md)已接入出生排序及组织存活率0.5，得到+22.8%/+22.0%的连续吞吐改善及 BODY 统计效率改善，本轮保留其估计量。两轮的种子和计时条件不同，不能直接比较绝对吞吐或 gamma 推断额外收益。300k/s仍未达到。表中 minibeam 保留此前较大统计量研究；最新三批一致性测试为质子49,147→50,524/s、碳离子9,965→10,429/s，不构成新的 gamma 验收。

Proton minibeam 沿用历史 PyMedPhys 0.41.0 径向搜索（`interp_fraction=10`），比较肺 CT 的 BODY 投影剂量：|y|<15 mm、|x|<15 mm、参考最大值3%以上。它不是三维体素或 RT07575 full plan 验收。笛卡尔格点交叉检查在同样0.02 mm步距下为97.1013%，两种算法均未通过 >99%。PyMedPhys 峰/谷条带分别为99.9605%和99.0052%，不能替代全场门槛。共用 EM 搜索优化没有改变该剂量结论；本轮 minibeam 检查为实现一致性回归。

Carbon CT minibeam 的 `history_chunk_size`、`material_electron_queue_chunk` 改为131,072，并开启已有的按粒子分组及16步次级粒子续跑。Group32、16,384个电子 worker、576 MiB队列预算、物理步长及每历史随机数保持不变。相同程序、反向顺序的独立种子确认得到 **1.946倍**连续吞吐；每20万粒子批次的 primary kernel 从22.320降至7.565 s，secondary从6.917降至3.498 s。所比较的输运整数、EM、电子计数完全一致，剂量相对 L1 最大5.01e-8。该场景队列峰值3,798,118 / 5,807,497个任务。这是调度一致性验收，不代表 carbon minibeam 剂量已验收。

增加共用 worker 数量、缩小错误诊断状态、C12主输运特化及缩小 secondary tail 的候选均未获得明确有效的收益，没有合入。实际 profiler 仍显示大量 local-memory 请求，每条已发射 warp 指令约11个活跃线程；仅降低静态寄存器数未提高运行速度。详见[完整报告](docs/ct_throughput_20261005.md)及[原始数值汇总](docs/ct_throughput_20261005.json)。

本轮 EM 搜索优化完成68个 broadbeam 完整批次，另有12个 CT minibeam 批次和其余六预设各两批回归。离子、EM、电子计数均保持一致，broadbeam 剂量相对 L1 最大4.823e-8。质量、能量、完成数、覆盖及零溢出检查通过。小粒子 smoke 不是 water gamma 或稳定吞吐验证。Water birth 提取每种粒子使用64个原生历史，仍需独立剂量/统计验证。先前清理引入的分支错误及较大统计量修复验证见[清理修正记录](docs/repository_cleanup_20261005.md#larger-statistics-correction)。

计时应分开冷启动、首批及常驻批次，报告 primary/s、kernel 时间、任务量和峰值内存，并记录 GPU、编译器、束流、种子及粒子数。比较相同入射 primary 口径，不能用电子任务或穿过准直器的 primary 替换。`scripts/benchmark_ct_devices.py` 已为四个 CT 预设统一使用原生连续批次，逐批检查，支持 `--smoke-only`；命令见 CT 报告。

每个候选前先查 [failed.md](failed.md)，只有满足记录中的新证据条件才能重试。FP32 原子累加顺序差异本身不否决调度改动，但输运/EM、能量、完成数和队列溢出审计仍必须通过。保留 TOPAS 参考和历史实验记录。Water 专项吞吐与剂量验收仍按要求推迟至 CT 验证之后。

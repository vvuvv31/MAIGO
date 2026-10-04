# 材料与方法

更新：2026-10-05。[English](mm.md)。本文描述八预设清理版；[清理验证记录](docs/repository_cleanup_20261005.md)与吞吐/剂量验收分别记录。后续 CT 吞吐优化尚未完成。历史测量只对应其记录的程序、束流、几何及统计量。

## 八个预设

仅保留以下八个 `config/` 入口。粒子种类和几何各自选择，共用电子传播实现。

| 预设 | 离子多重散射 | 电子出生选区 | 当前验证状态 |
|---|---|---|---|
| `proton_ct_broadbeam.yaml` | Gaussian Fermi–Eyges | CT 低密度区及周围5 mm | 历史剂量验收通过；清理回归通过 |
| `carbon_ct_broadbeam.yaml` | Gaussian Fermi–Eyges | CT 低密度区及周围5 mm | 历史剂量验收通过；清理回归通过 |
| `proton_ct_minibeam.yaml` | Urban | CT 内全部 primary 电离损失 | 沿用已验证质子逻辑；清理回归通过 |
| `carbon_ct_minibeam.yaml` | Urban | CT 内全部 primary 电离损失 | 输运审计通过；剂量尚未验收 |
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

其他 GPU 应选择相应架构。清理版已在 `out/repository_cleanup_20261004/build/carbon_mc` 完成测试，并安装到 `build/carbon_mc`。清理前的程序保存在 `out/repository_cleanup_20261004/before/carbon_mc`。

单进程连续批次：

```bash
build/carbon_mc --config config/proton_ct_minibeam.yaml \
  --batch-count 10 --histories-per-batch 1250000 \
  --batch-output out/proton_ct_minibeam_run
```

输出目录应为新目录。每批独立种子、剂量和质量报告，同一进程复用数据包及只读几何。计划权重按请求的总历史数分配，已归一化的计划剂量不能再次乘以批次数。不同配置依次运行可用 `--plan-manifest FILE`，并用 `--plan-preflight` 预检查。

## 验收和吞吐优化

后续四个 CT 预设采用以下验收标准：

| CT 预设 | 剂量门槛 | Gamma 搜索步长 |
|---|---|---|
| Proton / carbon broadbeam | Local1%/1 mm **>99%** | 0.1 mm |
| Proton minibeam | Local3%/0.2 mm **>99%** | 0.02 mm |
| Carbon minibeam | 暂不以剂量准确性为门槛 | 若计算则单独报告 |

对比前固定参考、绝对剂量归一化、坐标映射、剂量阈值和评价 mask，并随结果报告。历史 RT07575 broadbeam 使用参考最大剂量10%以上的全3D评价区，未拟合剂量比例或配准。搜索步长变化不允许伴随评价区或分母改变。零 DTA gamma 保留为受蒙卡噪声影响的诊断。

历史共用电子 RT07575 local1%/1 mm 为质子99.9120%、碳99.8677%，见[验收报告](docs/broadbeam_shared_electrons_20261004.md)。这些数值不自动验收后续每项调度或方差缩减改动。

最新清理检查已覆盖八个预设、每个两批。八种场景的输运整数计数、EM 和电子审计完全一致，剂量相对 L1 差异最高1.45e-7。全部通过质量、能量、任务完成、覆盖及零溢出检查。小粒子 smoke 不是 gamma 或稳定吞吐测量。Water birth 提取每种粒子使用64个原生历史，后续仍需独立剂量/统计验证。

计时应分开冷启动、首批及常驻批次，报告 primary/s、kernel 时间、任务量和峰值内存，并记录 GPU、编译器、束流、种子及粒子数。比较相同入射 primary 口径，不能用电子任务或穿过准直器的 primary 替换。每个候选前先查 [failed.md](failed.md)，只有满足记录中的新证据条件才能重试。FP32 原子累加顺序差异本身不否决调度改动，但输运/EM、能量、完成数和队列溢出审计仍必须通过。

保留 TOPAS 参考和历史实验记录。下一轮吞吐优化以清理提交为基线；water 专项吞吐和剂量验证按要求推迟至 CT 之后。

较大统计量的 proton broadbeam 测试发现并修复了清理引入的次级粒子弹性散射检查分支错误。同一失败种子现已通过3,759,450个历史及全部运行审计；见[清理修正记录](docs/repository_cleanup_20261005.md#larger-statistics-correction)。

# MAIGO minimal runtime worktree

面向 proton / carbon 的水模体、Schneider CT、铜多缝准直器输运。
此目录从 minibeam 的 bce9806 创建，并复制原工作树当时未提交的最新
src/、include/、failed.md。初始整理保留原输运源码；后续水/CT公共路径
合并与运行验证见 [minibeam共享路径报告](docs/minibeam_shared_20261001.md)。

## 构建与运行

本机需 SYCL CUDA 编译器、CUDA 驱动/工具链、CMake、OpenSSL/libzstd 开发库和 Python 3。
默认编译器为 `/home/wuwei/sycl_workspace/llvm/build/install/bin/icpx`；
换机器时设置 `CXX=/path/to/icpx`。运行依赖同一 SYCL 安装的 lib/。

```bash
cd /mnt/sdb/wuwei/MAIGO_minibeam
./scripts/build.sh
./scripts/run.sh                             # 列出配置
./scripts/run.sh proton_water_minibeam
# Proton CT minibeam 使用 proton_independent_v1 + shared_independent_v1。
# 三包清单及 SHA：docs/electron_packages_manifest.json；不再生成旧 packed 包。
./scripts/run.sh proton_ct_minibeam
./scripts/run.sh proton_water_broadbeam
./scripts/run.sh proton_ct_plan_fe --batch-count 10 --histories-per-batch 1253150 \
  --batch-output out/proton_ct_broadbeam_default
./scripts/run.sh carbon_water_broadbeam
./scripts/run.sh carbon_water_minibeam
# C12 CT 的派生格式独立于 proton；只需制包一次。
python3 scripts/prepare_material_electron.py --index data/ct_electron/carbon/index.json \
  --cache data/ct_electron/carbon_compact --carbon-compact
./scripts/run.sh carbon_ct_minibeam --histories-per-batch 100000 --batch-count 4 \
  --batch-output out/carbon_ct_four_batches
./scripts/run.sh carbon_ct_plan_fe_joint --batch-count 7 --histories-per-batch 925987 \
  --batch-output out/carbon_ct_broadbeam_default
```

scripts/run.sh 自动切换到工作树根目录，以便所有运行输入使用相对路径。
每个配置的剂量输出为 out/<配置名>/dose.mhd 和关联 raw 文件。
配置中的原始历史数/spot 权重保留，启动前可在 YAML 中修改。
TPS 计划请同时检查 tps_spot_weight_mode 和 spot 表；不要把
--histories 直接理解为所有 TPS 模式下都有效的总粒子数覆盖。

构建使用 FP32 剂量原子累加：CARBON_DOSE_FP32=ON、CARBON_DOSE_FP64=OFF。
启用 minibeam、CT 电子队列、FE 联合电子响应队列；
完整编译参数在 scripts/build_options.json。

原生多批次运行无需生成多份 YAML。例如每批 2M incident primaries，连续 10 批：

```bash
./scripts/run.sh proton_ct_minibeam \
  --histories-per-batch 2000000 --batch-count 10 \
  --batch-output out/proton_ct_10_batches
```

同样适用于本仓库所有 proton/carbon、CT/water、broadbeam/minibeam 预设，以及 TPS 和 TOPAS spot 计划。指定的粒子数是**每批整个计划的总数**；
计划权重按全部批次的总粒子数分配，再均衡拆分。每批独立随机种子、剂量和审计报告写入
`batch000000/` 等目录；`batches.csv` 和各批 `primary_batches.csv` 记录实际粒子数与源参数。
只读物理包和适用的 CT 电子响应库跨批复用；输运状态清零，任一批质量失败即停止。
EM、Urban 和 air 的不变设备数组默认也跨批保留，省去重复上传和 EM 索引构建；保留字节仍计入每批显存预算。
可用 `CARBON_RETAIN_PHYSICS_DEVICE=0` 关闭以作对照。RT07575 四批交替实测驻留吞吐提升约 1.9%，详见[逐项优化记录](docs/proton_sequential_optimization_20261004.md)。
输出根目录须不存在或为空；暂不支持断点续跑。可追加 `--plan-preflight` 先检查而不运行输运。
不要同时使用 `--histories`、`--plan-manifest` 或 `--sequential-spots`。
该功能降低进程启动和重复读取开销，不会自动提高单批队列容量；电子 group size 和 queue chunk 仍由 YAML 控制。
详见 [原生多批次说明与验证](docs/native_batches_20261002.md)。

所有预设使用相同入口；把上例的 `proton_ct_minibeam` 换成 `proton_ct_plan_fe`、
`carbon_ct_plan_fe_joint`、`proton_water_broadbeam`、`carbon_water_broadbeam`、
`proton_water_minibeam`、`carbon_water_minibeam` 或 `carbon_ct_minibeam` 即可，输出目录也应换成新目录。
普通 `--plan-manifest` 和逐 spot GPU 运行也持有进程级物理缓存。
CT 几何复用不再依赖 minibeam 电子库开关；联合电子响应、水响应和中子包共用只读输入缓存。
Carbon CT 默认已切换为共用电子路径，支持连续批次；历史 joint 配置原有质量门禁仍保留。
详见 [全预设共用缓存与批次验证](docs/shared_batches_20261002.md)。

连续执行 proton CT minibeam 时，可在同一进程内复用已验证的 CT 和电子响应表。
以下脚本需要 PyYAML；每批历史数和种子独立，输出目录必须不存在：

```bash
python3 scripts/run_ct_batch.py --histories 2000000 \
  --seeds 100115501 100115502 100115503 --output out/ct_batch_example
```

脚本保存配置、日志、二进制校验值、各批剂量和质量报告；完整运行吞吐见输出目录的
`execution.json`。CT 专用化及性能验证见 [CT 优化报告](docs/ct_minibeam_optimization_20261001.md)。
当前 proton CT minibeam 的 group32 批次为 262,144：RTX 2080 Ti 两组 2M 配对，
相对 65,536 批次的完整吞吐提高 7.48%，输运吞吐提高 12.55%；
详见 [GPU 批次调度优化](docs/gpu_schedule_20261001.md)。
以下为此前旧电子路径的性能记录，不代表当前独立电子路径的吞吐。
在上述 262,144 基线上，旧预设使用无损电子记录预解码 `tiered_decoded`：
两组反向顺序 2M 配对的完整吞吐进一步提高 12.95%，输运吞吐提高 26.07%。
FP32 计分、原始响应文件和全部回放物理保持不变；
详见 [CT 电子响应优化](docs/ct_response_20261002.md)。
在每事务及时补充任务版本上，旧 CT 预设将次级队列预留设为 8M、响应热预算设为 4 GiB：
RTX 2080 Ti 两组反向 2M 配对，完整吞吐进一步提高 9.95%，电子吞吐提高 73.63%，
采样显存峰值降低 554 MiB。该容量配置验证至每次 2M 历史，电子任务队列容量不变；
详见 [CT 电子包显存分配优化](docs/electron_memory_20261002.md)。
[Proton 离线紧凑电子包](docs/proton_electron_model_trials_20261002.md)将图压缩和预校验移到一次性制包：
RTX 2080 Ti、肺 CT、group32，1M 新进程筛选完整吞吐 +61.0%；
两组反向原生 2×2M 确认完整吞吐 +25.2%，已驻留第二批 +3.3%。
运行时 bank 为 6.09 GB，采样显存峰值减少 782 MiB；全部配对输运/剂量审计通过。
`proton_ct_minibeam.yaml` 现默认启用**独立 birth/joint/charged/photon** 研究路径，
电子响应拆分为 `data/ct_electron/` 下的三个包：`proton_independent_v1/`（质子出生谱）、
`carbon_independent_v1/`（碳离子出生谱）和 `shared_independent_v1/`（joint / charged / photon）。
Proton 只依赖 proton + shared，两者四组件磁盘文件合计 **3.85 GiB**；carbon + shared
目前用于独立软组织试验，未替换原 carbon 默认路径。详见[三包布局与验证](docs/electron_packages_20261004.md)。
当前默认电子计算精度为 FP32（`material_electron_precision: 4`），转换后的运行视图约 **2.89 GiB**，跨批次复用。
直接二进制、脚本及连续批次均从配置启用；缺包、SHA 或支持范围错误会明确失败，不回退旧库。
RT07575 已验证旧电子库零读取，19.84M primary 的 BODY 相对 L1 差0.148%、汇总峰谷差≤0.046%；
局部仍有异常，尚未正式 TOPAS 验收。初始独立路径约64.5k primary/s；[此前电子内核优化](docs/electron_throughput_20261003.md)
在直接配对中将驻留吞吐从64.95k提高到72.07k primary/s（+10.96%），仍低于旧路径约180k/s目标。
后续[全 FP32 基线](docs/electron_precision_20261003.md)在 RT07575/group32 的交替对照中达到 **86.86k primary/s**，
相对同时测量的原精度73.90k/s提高17.54%；剂量计分始终FP32，仍为研究结果。
当前又启用[同一步参数复用](docs/electron_parameter_reuse_20261003.md)，独立交替对照的驻留吞吐
从86.65k提高到 **93.80k primary/s（+8.26%）**，电子阶段耗时降低14.27%，整数工作审计一致。
进一步将硬速率上界检查移到[加载时区间验证](docs/electron_rate_certificate_20261003.md)，
新的独立配对为95.01k→**100.28k primary/s（+5.55%）**，电子阶段约9.75秒；默认仍为全FP32。
不能套用上面的旧路径加速数字。
默认肺 CT/170MeV 的连续2×500k批次亦通过运行审计并复用新包缓存。
详见 [独立路径与剂量结果](docs/electron_charged_runtime_20261003.md)、
[运行包来源与 SHA](docs/proton_independent_runtime_manifest.json)。旧包仍保留作对照。

## 配置范围

| 模式 | Proton | Carbon |
|---|---|---|
| Water broadbeam | 已有 | 已有 |
| CT broadbeam | 已有 | **已验收（RT07575，共用电子 local 1%/1 mm >99%）** |
| Water minibeam | 已有 | 已有 |
| CT minibeam | 已有 | [已可运行，剂量验收中](docs/carbon_electron_compact_20261002.md) |

“已验收”对应已记录病例与标准；不代表其他病例或束型自动通过。

| 配置 | 场景 | 说明 |
|---|---|---|
| proton_water_broadbeam | 170 MeV 水开放野 | FE 专用核；低密度 tracking 开启，均匀水选区为空，电子局部沉积 |
| proton_water_minibeam | 170 MeV，3×3 cm 入射，铜多缝＋水 | Urban、电子响应、研究用中子输运 |
| proton_ct_minibeam | 170 MeV，铜多缝＋肺 CT | Urban 密度缩放、独立电子路径 group32（研究）；中子关闭 |
| carbon_ct_minibeam | 250 MeV/u，铜多缝＋肺 CT | C12 材料电子响应 group32；首轮剂量验收中 |
| proton_ct_plan_fe | RT07575 proton 优化计划 | Gaussian FE＋低密度区域共用电子；原计划第一分片，12,531,500 histories，不是完整计划 |
| carbon_water_broadbeam | 200 MeV/u 水开放野 | FE 专用核；低密度 tracking 开启，均匀水选区为空，电子局部沉积 |
| carbon_water_minibeam | 250 MeV/u 单 spot，铜多缝＋水 | 碳原生弹性、Urban 主粒子、电子响应；保留较细的旧步长 |
| carbon_ct_plan_fe_joint | RT07575 carbon 6,481,909 histories | FE 专用核＋低密度区域共用电子；旧 joint 文件名保留，已通过 RT07575 >99% 验收 |

非 minibeam 配置使用 FE。Minibeam 保留各原配置的散射、电子响应和核反应选择。
水参考为 Water_75eV，停止本领比例保持 1.0。

这些配置继承原来的研究限制：CT group32 的严格 local gamma 验收尚未全部完成；
当前 proton/carbon CT broadbeam 共用电子路径已通过[RT07575 剂量验收](docs/broadbeam_shared_electrons_20261004.md)
（local 1%/1 mm分别99.9120% /99.8677%，标准 >99%），通用包验证仍不由单病例替代；
中子包的 runtime_validated: false 状态原样保留。
carbon CT minibeam 已支持独立的 C12 紧凑材料电子响应包，TOPAS 剂量验收仍在进行；
不把 proton 材料电子响应包用于 carbon。现有源代码与碳 CT/铜/水所需模块均保留。
当前采用[位图索引、32位关联与无损端点共享](docs/carbon_endpoint_20261003.md)，
完整保留电子/光子状态、轨迹关联、出生分布与横向输运。
运行库93.56→45.92→35.03→29.36 GiB，完整肺CT连续group32已通过同种子剂量、能量及零溢出检查。
默认 `material_electron_compact_directory: data/ct_electron/carbon_compact`，无需实验环境变量；
主存预算64 GiB，缺包或版本不符明确报错。多批次保留bank，只读取一次。
当前v7内嵌起点、共享续接终点；在RTX2080Ti相对v5反向对比，主存峰值36.70→31.03GiB，
常驻吞吐3417→3358 primary/s（−1.7%），电子核时间约−1.1%。这是内存优化，不宣称提速。
显存仍约9.5GiB。Proton整批性能基本持平，电子核时间约+1.7%；详见报告。
TOPAS终验仍单独进行，不能把本次无损回归作为新的TOPAS验收。
本机将默认目录链接到NVMe缓存；其他机器可直接制包，或修改YAML路径。
旧v5包需用 `scripts/migrate_carbon_endpoint.py` 无损迁移到独立目录，原包保留；
迁移要求源/目标同文件系统。旧v4先用 `scripts/migrate_carbon_bitmap.py` 转成固定审计身份的v5。
具体命令和复现说明见报告。可选zstd只减少磁盘占用，不减少解压后的运行主存；默认使用raw v7。

[参数化电子响应原型](docs/electron_parametric_20261002.md)仍是独立的近似研究；
该原型仅通过与训练相同步长设置的均匀肺电子峰谷检查；后续发现 CT 的步长与体素边界会改变响应，
因此尚不能据此验收 CT。[短程响应试验与限制](docs/electron_macro_trials_20261002.md)持续记录独立验证，尚未切换正式 CT 路径。

## 目录和数据

- src/、include/：当前完整核心；共享代码规模很小，未做额外物理重构。
- config/：上述八个入口；carbon CT minibeam 尚在验收中。
- data/：配置依赖的运行数据、校验清单和必要侧文件。
- inputs/：CT 二进制、spot 表、束流光学表；没有原始 DICOM 或历史剂量结果。
- scripts/：构建/运行入口。
- docs/package_manifest.json：每个数据文件的来源、SHA256、大小与配置变更。
- AGENTS.md、failed.md：项目约束和已有失败路线，继续开发时必须保留。

原始物理数据独立复制，不依赖旧工作树；共享核反应/EM数据只存一份。
本机C12派生紧凑缓存使用上述NVMe链接，便携部署时可在相对目录重新生成。
CT 电子响应的 replay 原始轨迹属于运行依赖，必须保留；提取日志和中间产物不带入。
运行时清单已改为相对路径并更新受影响的 SHA256。部分不可变侧文件中的绝对路径
仅记录提取来源，不是运行时依赖，保留它们以维持原始 provenance 和校验值。

## Git 工作树

分支：minibeam-minimal-20261001。这是稀疏 Git 工作树，共享原仓库的 Git 历史；
本次尚未提交。稀疏检出只影响本目录的物理文件，不删除历史 benchmark/evidence。
不要执行 git sparse-checkout disable，否则会展开仓库历史跟踪的全部目录。
data/、inputs/、build/、out/ 中新增的大文件默认忽略，源代码、配置和
数据来源清单可以正常提交。新工作树的运行数据独立，但 .git 仍依赖公共 Git 目录。

本次整理执行独立编译和依赖检查，不重新运行蒙卡或 TOPAS 对照。

检查结果：独立 GPU 编译成功；7 个配置预检通过；577 项运行数据 SHA 引用通过；
124 个核心源文件与原工作树一致。运行数据/CT 输入共 13.32 GiB，详情见
`docs/packaging_check.json`。未运行新的蒙卡剂量或吞吐测试。

后续开发已补齐电子队列的CMake宏，并将水/CT的铜与空气段合并为共同实现，
下游水/CT物理仍分别特化。性能候选、回归和最终运行验证记录在上述共享路径报告中；
`packaging_check.json` 保留初始打包检查记录。回归计时可使用
`scripts/compare_minibeam.py`（需NumPy与PyYAML），所有GPU性能构建仍固定使用FP32剂量。

四个 broadbeam 默认统一开启 `broadbeam_electron_tracking: true`，范围为密度≤0.1 g/cm³及周围5 mm。CT 使用 proton/carbon 专属出生包＋shared FP32 传播包；均匀水选区为空，不加载电子包。设为 `false` 即局部沉积，不启用历史 joint/tail 响应。只接管主粒子电离电子，带电核次级的电子损失仍局部沉积。详见[默认开关与验证](docs/broadbeam_tracking_defaults_20261004.md)；2080Ti 用上面的连续批次命令保持原总历史数。

两个 CT broadbeam 预设现默认启用低密度出生电子的无偏俄罗斯轮盘（`broadbeam_electron_low_density_survival: 0.5`）、电子 worker 任务结束账本结构调整和 524288 批次调度；物理、步长、选区和出生包不变。RTX2080Ti RT07575 同日 ABBA：proton 90.4k→138.7k primary/s（+53.4%），carbon 53.1k→87.2k（+64.3%）；全网格 local 1%/1mm 仍>99%，BODY 噪声不变，体外空气单位时间精度 ×1.04/×0.91。设为 1.0 即关闭轮盘。CT 入射面空气仍有约6%的电子建成不足（上游世界空气电子未模拟）。详见[空气/低密度电子优化](docs/broadbeam_air_roulette_20261004.md)。

在此基础上，broadbeam CT 主粒子改用主机证明配置后的专用 kernel（stage 12/13/14），次级粒子使用生产专用路径，每步单地址原子账本改为寄存器累计，每批主机准备从约0.55 s降到约0.15 s；物理与整数审计不变。RT07575 同日 ABBA：proton 136.7k→164.2k primary/s（+20.2%，主核4.36→3.31 s），carbon 86.9k→104.4k（+20.2%，主核2.60→1.70 s、次级3.25→2.80 s）；gamma 与原二进制一致。详见[主粒子/次级粒子优化](docs/broadbeam_primary_secondary_20261004.md)。

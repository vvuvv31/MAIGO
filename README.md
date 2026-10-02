# MAIGO minimal runtime worktree

面向 proton / carbon 的水模体、Schneider CT、铜多缝准直器输运。
此目录从 minibeam 的 bce9806 创建，并复制原工作树当时未提交的最新
src/、include/、failed.md。初始整理保留原输运源码；后续水/CT公共路径
合并与运行验证见 [minibeam共享路径报告](docs/minibeam_shared_20261001.md)。

## 构建与运行

本机需 SYCL CUDA 编译器、CUDA 驱动/工具链、CMake、OpenSSL 开发库和 Python 3。
默认编译器为 `/home/wuwei/sycl_workspace/llvm/build/install/bin/icpx`；
换机器时设置 `CXX=/path/to/icpx`。运行依赖同一 SYCL 安装的 lib/。

```bash
cd /mnt/sdb/wuwei/MAIGO_minibeam
./scripts/build.sh
./scripts/run.sh                             # 列出配置
./scripts/run.sh proton_water_minibeam
./scripts/run.sh proton_ct_minibeam
./scripts/run.sh proton_water_broadbeam
./scripts/run.sh proton_ct_plan_fe
./scripts/run.sh carbon_water_broadbeam
./scripts/run.sh carbon_water_minibeam
./scripts/run.sh carbon_ct_plan_fe_joint
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
输出根目录须不存在或为空；暂不支持断点续跑。可追加 `--plan-preflight` 先检查而不运行输运。
不要同时使用 `--histories`、`--plan-manifest` 或 `--sequential-spots`。
该功能降低进程启动和重复读取开销，不会自动提高单批队列容量；电子 group size 和 queue chunk 仍由 YAML 控制。
详见 [原生多批次说明与验证](docs/native_batches_20261002.md)。

所有预设使用相同入口；把上例的 `proton_ct_minibeam` 换成 `proton_ct_plan_fe`、
`carbon_ct_plan_fe_joint`、`proton_water_broadbeam`、`carbon_water_broadbeam`、
`proton_water_minibeam` 或 `carbon_water_minibeam` 即可，输出目录也应换成新目录。
普通 `--plan-manifest` 和逐 spot GPU 运行也持有进程级物理缓存。
CT 几何复用不再依赖 minibeam 电子库开关；联合电子响应、水响应和中子包共用只读输入缓存。
Carbon CT 联合响应预设仍会在原有质量失败后停止，不会为了执行完批次数而忽略门禁。
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
在上述 262,144 基线上，当前预设使用无损电子记录预解码 `tiered_decoded`：
两组反向顺序 2M 配对的完整吞吐进一步提高 12.95%，输运吞吐提高 26.07%。
FP32 计分、原始响应文件和全部回放物理保持不变；
详见 [CT 电子响应优化](docs/ct_response_20261002.md)。
在每事务及时补充任务版本上，当前 CT 预设将次级队列预留设为 8M、响应热预算设为 4 GiB：
RTX 2080 Ti 两组反向 2M 配对，完整吞吐进一步提高 9.95%，电子吞吐提高 73.63%，
采样显存峰值降低 554 MiB。该容量配置验证至每次 2M 历史，电子任务队列容量不变；
详见 [CT 电子包显存分配优化](docs/electron_memory_20261002.md)。

## 配置范围

| 配置 | 场景 | 说明 |
|---|---|---|
| proton_water_broadbeam | 170 MeV 水开放野 | FE 专用核，核反应及带电次级，电子局部沉积 |
| proton_water_minibeam | 170 MeV，3×3 cm 入射，铜多缝＋水 | Urban、电子响应、研究用中子输运 |
| proton_ct_minibeam | 170 MeV，铜多缝＋肺 CT | Urban 密度缩放、材料电子响应 group32；中子关闭 |
| proton_ct_plan_fe | RT07575 proton 优化计划 | FE core+tail；原计划第一分片，12,531,500 histories，不是完整计划 |
| carbon_water_broadbeam | 200 MeV/u 水开放野 | 从 unified water 配置派生为 FE 专用核；本次未做新剂量对照 |
| carbon_water_minibeam | 250 MeV/u 单 spot，铜多缝＋水 | 碳原生弹性、Urban 主粒子、电子响应；保留较细的旧步长 |
| carbon_ct_plan_fe_joint | RT07575 carbon 6,481,909 histories | FE 专用核＋研究用联合电子响应 |

非 minibeam 配置使用 FE。Minibeam 保留各原配置的散射、电子响应和核反应选择。
水参考为 Water_75eV，停止本领比例保持 1.0。

这些配置继承原来的研究限制：CT group32 的严格 local gamma 验收尚未全部完成；
carbon CT 联合电子响应仍可能报告 unvalidated_electron_joint_response；
中子包的 runtime_validated: false 状态原样保留。
本目录没有新建或验证 carbon CT minibeam 专用参考配置，也不把 proton
材料电子响应包用于 carbon。现有源代码与碳 CT/铜/水所需模块均保留。

## 目录和数据

- src/、include/：当前完整核心；共享代码规模很小，未做额外物理重构。
- config/：上述七个入口。
- data/：配置依赖的运行数据、校验清单和必要侧文件。
- inputs/：CT 二进制、spot 表、束流光学表；没有原始 DICOM 或历史剂量结果。
- scripts/：构建/运行入口。
- docs/package_manifest.json：每个数据文件的来源、SHA256、大小与配置变更。
- AGENTS.md、failed.md：项目约束和已有失败路线，继续开发时必须保留。

数据文件独立复制，不依赖旧目录的软链接或硬链接；共享核反应/EM 数据只存一份。
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

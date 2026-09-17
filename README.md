# MAIGO

基于 TOPAS / Geant4 提取数据的碳离子凝聚历史 GPU Monte Carlo，目标是复现
*A Data-Driven Fragmentation Model for Carbon Therapy: GPU-Accelerated Monte Carlo
Dose Recalculation* 所述方法。**仅用于研究，不用于临床治疗计划。**

当前患者计算使用 SYCL、Schneider 25 分区材料和 CINEL03 相关末态；
GPU 不在运行时调用 Geant4。当前水生产入口使用共享 CINEL03 框架和统一 EM 包。
CT 包含 H/O 靶通道，不等于已经替代水中所有 stopping、XS、MCS 和 fluctuation 数据。

当前开发与提交目标为 `master` / `origin/master`（原 `fred` 开发线）；
迁移前的 `master` 保留为 `legacy`。历史归档中的分支名不代表现行工作流。
commit / push 仍需用户明确授权，详见 [AGENTS](AGENTS.md)。

## 文档入口

- [完整文档导航](docs/README.md)
- [源码结构](docs/structure.md)、[物理模型与局限](docs/TOPAS_GPU_Physics_Model.md)
- [Planning / geometry](docs/planning.md)、[Scoring / Gamma](docs/scoring_validation.md)
- [当前结果索引](docs/results.md)、[Materials and Methods](mm.md) / [中文](mm_zh.md)
- [活动数据与恢复](data/ACTIVE_DATA.md)、[执行规则与数据最低版本](AGENTS.md)
- [Schneider 计划进度](plan/README.md)、[电子响应计划进度](plan2/README.md)
- [历史记录](docs/archive/README.md)、[FRED 论文解读](docs/FRED_Carbon_Fragmentation_Model.md)

## pristine 分支

`pristine` 是最小可运行快照：只保留 `src/`、`include/`、`config/`、`docs/`、构建文件、
Git 跟踪的小型数据表，以及数据校验/下载工具；`tests/`、`startup/`、`benchmark/`、
`evidence/`、`plan*/`、`scripts/` 等研究、测试和历史脚本不在此分支。

大型物理数据包（统一 EM、δ矩、Schneider v2.1 二进制）与 RT07575 CT 输入不在 Git 中，
以 `pristine-v1` release 资产发布。下载并解包：

```sh
./tools/fetch_release_data.sh
```

解包后运行生产前仍需校验 Schneider v2.1 数据：

```sh
python3 tools/verify_schneider_v2_1_data.py
```

CT 输入解包到 `data/ct/`（`rt07575_packed.bin`、`RT07575/shard_01/spots.csv`、
`beam_model.csv`）；运行 CT 配置前将其路径填入 `ct_grid_file`、`tps_spots_file`、
`tps_beam_model_file`。水 benchmark 只需 `data/` 中的表。

## 构建

需要 CMake ≥3.22、C++20、Ninja；GPU 需要支持 NVIDIA 的 oneAPI/DPC++ SYCL 工具链。
本仓库当前验证硬件为本地 RTX 2080 Ti（sm_75）。先按本机安装设置编译器及共享库路径。

```sh
cmake --preset oneapi-nvidia-release -DCARBON_CUDA_ARCH=sm_75
cmake --build --preset oneapi-nvidia-release --parallel 8
```

CPU 子集可用 `cpu-debug` preset 构建，但不作为患者 CT GPU 验证的替代后端。
SYCL/GPU 在沙盒外执行；禁止远程/集群 GPU。未指定环境时 TOPAS 经本地 sbatch；明确指定远端时可在该处运行。
并发总预算为 192 CPU 线程 / 160 GiB，具体约束见 AGENTS。

## 数据与运行

统一 EM 已按 [AGENTS.md](AGENTS.md) 中的授权例外接入正式运行：
[水生产配置](config/unified_water_production.yaml)、
[RT07575 生产配置](config/rt07575_unified_em_production.yaml)。
原发 C12 和全部 18 种已支持带电离子共用水 / Schneider EM 核心包与派生矩表。
当前默认采用**两矩 Gamma δ 聚合＋解析 Poisson 分步修正**：保留材料特异受限
stopping/range、原生受限涨落（scale=1）及 MCS，取消逐 δ 碰撞抽样与 δ 时钟限步；
δ 能量局部沉积，不运行电子空间 tracking。仅 δ stopping>0 时追加 1% 组合平均能损限步。
非弹性事件重放保持开启；独立核弹性仍是研究选项。公式和近似边界见 [mm_zh.md](mm_zh.md)。
两个生产预设默认开启 `secondary_step_chunking: true`（每 16 次完整次级循环
保存状态并压紧存活索引）；设为 `false` 可关闭，详见
[接入验证](benchmark/runtime_breakdown_20260914/SEGMENT_PRODUCTION.md)。
默认 `CARBON_EM_EXACT_INDEX=ON`：按浮点指数桶缩小 EM 二分查找，不改节点或插值；
关闭用预设 `oneapi-nvidia-index-off`。
使用前需重新构建二进制，旧构建不能执行新配置开关。
低密度阈值区和患者 Gamma 精度验收尚未完成，质量报告保留提示。
核弹性仍受原有研究模式限制；联合核弹性计算继续使用研究配置。
新 EM 二进制约 1.29 GiB，需单独复制，尚未上传到 Release；
详见[数据说明](data/ACTIVE_DATA.md)和[实现说明](docs/physics/unified_em_v1.md)。

安装核心数据后，还需生成 206,977,416 字节的派生矩表；旧 Release 不含该表，
Git 只提供生成器与固定 SHA256 清单。需要 Python、NumPy：

```sh
python3 tools/build_delta_moments.py
```

每次 Schneider CT 运行前：

```sh
python3 tools/verify_schneider_v2_1_data.py
python3 tools/verify_unified_em_data.py
```

任一缺失、SHA/size 不符、schema 降级或 registry 不完整都必须停止。
最新最低栈是 v2.1 / 14-projectile；文件名中的 stopping v1 和 delta-tail v1
仍是当前数据，不能仅按名字判断过时。大文件及外部 water 包不是普通 clone 就能获得，
依照 manifest 和 [活动数据说明](data/ACTIVE_DATA.md) 安装，不从 trash 自动 fallback。

新计算从上述当前生产配置出发，按病例核对 CT、spot、beam model 和几何。
[2026-09-05 冻结配置和结果](benchmark/topas10x/gpu_current_20260905.md) 仅供历史比较；
不要覆盖原有配置、日志和剂量。
运行接口为：

```text
./build/oneapi-nvidia-release/carbon_mc --config <准备好的新运行配置> --device cuda
```

配置是项目的轻量 `key: value` 格式，不是任意 YAML。
先核对实际逐 spot histories、几何、数据和输出路径，再运行。
大任务必须分片，任一 secondary overflow 必须缩小分片重跑。
不要把旧示例配置当作当前 CT 验证配置；旧 fixtures 已部分归档。

## 当前吞吐与 CT benchmark

正式接入回归：RT07575 6,481,909 粒子，wall **68.76 s**、程序 elapsed **65.45 s**，
约 **99.0k histories/s**，零 overflow。b1–b4 全部 10 个 200k 用例通过运行质量检查；
100/200/300 MeV/u 零能散峰高误差约 −0.0715% / −0.1098% / +0.0424%。
详见[接入验证](benchmark/benchmark20260915/delta_partition_production/README.md)。

最近 RT07575 单片测试：4,861,226 粒子用时 **48.88 s**，程序吞吐 **103.0k/s**，
采样显存峰值 **9,517 MiB**，零 overflow。把原发批量增至 131,072 未见明确收益，
保留 `history_chunk_size=34816`；本次剩余任务目标上限为每片 490 万粒子。
次级队列固定 3,200 万条，原发全部完成后才处理次级；当前代续跑状态为 264 字节/条
另加索引，不能以原发阶段约 4.4 GB 占用判断整片显存余量。该分片上限不是对任意病例的安全保证。

本次三病例全粒子数比较**运行中**：

| Case | GPU / TOPAS 各自总粒子数 | GPU 分片安排 |
|---|---:|---|
| RT07575 | 129,638,170 | 20 个已完成小片＋14 个剩余大片 |
| RT06423 | 151,087,660 | 31 片 |
| 20022516 | 177,173,040 | 37 片 |

原始 TOPAS 及 GPU 结果按病例保存在 `benchmark/ctbenchmark20260915/<case>/`，
后处理保存在 `benchmark/ctbenchmark20260915/result/`。这些本地大数据不是普通 clone 的内容。
每个 spot 的整数总粒子数保持一致，测试剂量不计入；显存不足或 overflow 须拆小重跑。

比较 BODY 内 TOPAS ≥10% 全局最大剂量的体素，输出 3%/3mm、2%/2mm、1%/1mm、
3%/0mm 的 global/local Gamma。搜索步长依次为 0.3、0.2、0.1 mm；0mm 不做空间搜索。
等中心处三解剖面输出 1D 曲线、2D 剂量及 GPU−TOPAS 差值，差值色标固定为
±5% TOPAS 全局最大剂量。累计 Gy 不拟合归一、不做剂量配准。
本批 TOPAS 启用独立核弹性，GPU 最新生产配置未启用；报告保留这一物理范围差异。

## 验证边界

2026-09-05 三病例共 60/60 shards accepted、零 overflow；457,898,870 histories。
Global 1%/1mm 为 96.74–98.79%，Global 3%/0mm 为 94.27–99.97%；
完整 Global/Local 指标及严格定义见 [结果索引](docs/results.md)。

这些结果绑定冻结 executable 和输入，包含当时的 entrance-mask candidate；
不表示当前工作树任意新候选已验证，也不表示 plan2 电子纵向响应完成。
不以 Gamma 通过率声称全物理等价、任意材料/能区泛化或临床准入。

测试入口及已归档 fixtures 的限制见 [structure](docs/structure.md)。
本轮全粒子数计算仍在运行；未完成结果不作为已通过验收的证据。

## License

[GPL-3.0-or-later](LICENSE)。

旧 FRED 功能及解析碎裂代码的移除范围、通用模块替代和验证结果见[清理报告](docs/fred_cleanup.md)。

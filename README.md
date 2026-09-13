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

## 构建

需要 CMake ≥3.22、C++20、Ninja；GPU 需要支持 NVIDIA 的 oneAPI/DPC++ SYCL 工具链。
本仓库当前验证硬件为本地 RTX 2080 Ti（sm_75）。先按本机安装设置编译器及共享库路径。

```sh
cmake --preset oneapi-nvidia-release -DCARBON_CUDA_ARCH=sm_75
cmake --build --preset oneapi-nvidia-release --parallel 8
```

CPU 子集可用 `cpu-debug` preset 构建，但不作为患者 CT GPU 验证的替代后端。
SYCL/GPU 在沙盒外执行；禁止远程/集群 GPU。TOPAS 经本地 sbatch，
并发总预算为 192 CPU 线程 / 160 GiB，具体约束见 AGENTS。

## 数据与运行

统一 EM 已按 [AGENTS.md](AGENTS.md) 中的授权例外接入正式运行：
[水生产配置](config/unified_water_production.yaml)、
[RT07575 生产配置](config/rt07575_unified_em_production.yaml)。
原发 C12 和全部 18 种带电离子共用水 / Schneider EM 包。
低密度阈值区和患者 Gamma 精度验收尚未完成，质量报告保留提示。
核弹性仍受原有研究模式限制；联合核弹性计算继续使用研究配置。
新 EM 二进制约 1.29 GiB，需单独复制，尚未上传到 Release；
详见[数据说明](data/ACTIVE_DATA.md)和[实现说明](docs/physics/unified_em_v1.md)。

每次 Schneider CT 运行前：

```sh
python3 tools/verify_schneider_v2_1_data.py
```

任一缺失、SHA/size 不符、schema 降级或 registry 不完整都必须停止。
最新最低栈是 v2.1 / 14-projectile；文件名中的 stopping v1 和 delta-tail v1
仍是当前数据，不能仅按名字判断过时。大文件及外部 water 包不是普通 clone 就能获得，
依照 manifest 和 [活动数据说明](data/ACTIVE_DATA.md) 安装，不从 trash 自动 fallback。

三病例复现以 [2026-09-05 冻结配置和结果](benchmark/topas10x/gpu_current_20260905.md)
为起点，使用新输出目录；不要直接覆盖其中的 shard 配置、日志和剂量。
运行接口为：

```text
./build/oneapi-nvidia-release/carbon_mc --config <准备好的新运行配置> --device cuda
```

配置是项目的轻量 `key: value` 格式，不是任意 YAML。
先核对实际逐 spot histories、几何、数据和输出路径，再运行。
大任务必须分片，任一 secondary overflow 必须缩小分片重跑。
不要把旧示例配置当作当前 CT 验证配置；旧 fixtures 已部分归档。

## 验证边界

2026-09-05 三病例共 60/60 shards accepted、零 overflow；457,898,870 histories。
Global 1%/1mm 为 96.74–98.79%，Global 3%/0mm 为 94.27–99.97%；
完整 Global/Local 指标及严格定义见 [结果索引](docs/results.md)。

这些结果绑定冻结 executable 和输入，包含当时的 entrance-mask candidate；
不表示当前工作树任意新候选已验证，也不表示 plan2 电子纵向响应完成。
不以 Gamma 通过率声称全物理等价、任意材料/能区泛化或临床准入。

测试入口及已归档 fixtures 的限制见 [structure](docs/structure.md)。
本轮文档整理未重跑 Monte Carlo，不修改冻结数值。

## License

[GPL-3.0-or-later](LICENSE)。

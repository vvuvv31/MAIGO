# 两矩 δ 聚合与解析分步修正：正式接入验证

2026-09-15 用户验收后接入水/Schneider 生产选项，原发 C12 与全部 18 种已支持离子共用。默认原生受限涨落 scale=1、精确 CT 边界和次级 64 步续跑保持开启。

## 实现与依赖

见 [mm_zh.md](../../../mm_zh.md) 第 3 节。取消离散 δ 抽样和 δ 限步，用接受谱第一、第二矩匹配 Gamma；对受限线性均值应用 Poisson 分段期望修正，保留完整 δ 跳变漂移。仅 δ stopping>0 时追加 1% 组合平均能损限步。
原生 range 分支不重复修正。Gamma 不保留零碰撞原子和高阶矩，δ 仍局部沉积。

派生表现在位于 `data/em/unified_em_delta_moments_v2.bin`，不再依赖 scratch。`tools/build_delta_moments.py` 从固定核心包重建并得到与候选完全一致的 SHA256；运行时拒绝缺失、损坏及错源数据。大二进制不加入 Git，旧 Release 用户需运行生成脚本。

## 验证

- 8 项针对性 CTest 全部通过：Gamma 统计、Poisson 分段期望、矩表拒绝检查、生产配置、原生步长与均值/涨落/离散谱组件。
- b1–b4 全部 10 例，每例 200k primaries，固定原 TOPAS；三维记分横向求和，与能量账本闭合，全部零 overflow、EM 无错误。
- 生产与已验收候选体素剂量最大差异小于峰值的 0.000065%；RT07575 小于 0.000030%。见 `integration_comparison.json`。这是实现一致性检查，不是与 TOPAS 的 Gamma。
- 基准沿用全离子弹性研究配置；患者生产配置仍不启用独立核弹性。患者 BODY Gamma、低密度阈值闭合尚未完成。

## RT07575 1/20 shard

6,481,909 primaries，按同一 spot 权重拆两段，第二段 seed+1000003。Wall **68.763 s**，程序 elapsed **65.447 s**，**99041.2 histories/s**；primary 19.744 s，secondary 37.267 s。零 overflow。Wall 是两个进程耗时之和，不含队列等待，合并另计。单次测量不代表重复运行置信区间。

## 峰值结果与图

[全部图表](benchmarks/plots/README.md)：b1–b3 含 IDD、core/halo sigma 及误差；全部案例含九深度线性/log profile。IDD relative error 显示范围 ±5%，数据不裁剪。

|Case|峰高误差 %|
|---|---:|
|b1_100|-0.0715|
|b1_200|-0.1098|
|b1_300|+0.0424|
|b1_400|-0.8366|
|b2_150|-0.1174|
|b2_250|-0.5089|
|b2_350|-0.7617|
|b3_layers|-0.8153|
|b4_soft_lung|-0.4737|
|b4_soft_bone|-1.3788|

## 复现

本地准备参考数据后依次执行 `prepare_benchmarks.py`、`benchmarks/run_gpu.py`、`benchmarks/analyze.py`、`benchmarks/plot_benchmarks.py`、`run_split_shard.py`。参考数据及体模来自 benchmark20260914/production_continuation，需单独保留；Git 不含这些大数据或 dose raw。`compare_integration.py` 还依赖本地已验收候选剂量。GPU 仅在本地 RTX 2080 Ti 运行。配置和二进制 SHA 见 input_manifest 与各 gpu_status。

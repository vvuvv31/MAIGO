**最新代码：proton CT minibeam 与 TOPAS gamma 对比（2026-10-01）**

当前工作区重新编译并新跑 GPU 1000 万入射质子，复用匹配的 TOPAS 1000 万参考。主照射野局部 3%/0.2 mm 二维 gamma 为 **93.2625%**；全局 3%/0.2 mm 为 **99.9911%**。肺区谷底平均剂量比 TOPAS 低 **2.0236% ± 0.2650 个百分点（1 SEM）**。本次结果不能解读为全物理或逐体素 1% 一致。

本次只进行验证，没有修改输运实现或生产预设。`scripts/build.sh` 从当前未提交工作区完成构建，源文件哈希在运行前后均核验一致。二进制 SHA256 为 `4cc6a1caf5fc3e94274779f5515c2559d1eb51c5270fbd9dde92f391d98bac7e`，与上一轮交付一致。剂量评分 `CARBON_DOSE_FP32=ON`、`CARBON_DOSE_FP64=OFF`；TOPAS 原有双精度 EnergyDeposit 文件只作为输入数据读取。

采用 `config/proton_ct_minibeam.yaml`：170 MeV 质子，30×30 mm 平源，60 mm 厚铜准直器，15 条缝，宽 0.5 mm、间距 3.6 mm，Urban v2，当前 CT 主输运特化和精确延迟电子 CDF，group32，history/queue chunk 均为 65536，保留响应库缓存。两边均为 20 个独立种子的 50 万粒子批次。GPU 保持 research 模式。

TOPAS 4.2.p3 / Geant4 11.3.2，原生完整物理参考来自 `/mnt/sda/wuwei/real_ct_full_10m_20260929/`，具体批次在 `manifest.json` 的 `topas_reference` 中。此次没有重跑 TOPAS。CT 的 30,592,800 个体素逐一核验：材料 section、FP32 密度和几何一致；网格 1200×42×607，间距 0.1×2×0.5 mm，边界原点 (−60,−42,0) mm。TOPAS 输出与 GPU 均为 x 最快的 z,y,x 顺序。

比较口径：

- 以 TOPAS 为参考。TOPAS EnergyDeposit 按每个体素质量转换为 Gy，两边均除以实际入射历史数；不拟合剂量、不做峰值配准或独立归一化。
- BODY 包括肺内低密度体素，排除外部空气和分离床板；沿 |y|<15 mm 内的 BODY 体素求平均，获得 x–z 二维剂量图。**这是二维投影 gamma，不是三维逐体素 gamma。**
- 主照射野 |x|<15 mm；TOPAS 投影最大剂量的 3% 为低剂量阈值，共 123,651 点。最大值取完整投影图，不只取局部 ROI。
- 局部 gamma 的剂量分母为对应参考点的 TOPAS 剂量；全局 gamma 分母为 TOPAS 投影最大剂量。gamma≤1 为通过。
- PyMedPhys 0.41.0，线性插值（scipy），max_gamma=2，不随机抽点、不提前停止。0.2 mm 指标搜索步长为 0.02 mm；1/1、2/2、3/3 指标步长分别为 0.05、0.1、0.1 mm。

| Gamma 标准 | 主照射野通过率 |
|---|---:|
| 局部 3% / 0.2 mm | 93.2625% |
| 全局 3% / 0.2 mm | 99.9911% |
| 全局 1% / 1 mm | 99.9450% |
| 全局 2% / 2 mm | 99.9846% |
| 全局 3% / 3 mm | 99.9960% |

同一局部 3%/0.2 mm 标准：扩展区域 |x|<25 mm 为 93.7977%（136,805 点）；峰条带为 95.8303%（15,205 点）；谷条带为 87.5100%（11,305 点）。峰条带是各缝中心 ±0.2 mm、|x|<15 mm；谷条带是各相邻缝中点 ±0.2 mm、|x|<12.6 mm。将阈值提高到 10% 后，主照射野通过率为 94.9679%，但更多低剂量谷区被排除。全局高通过率不能代替谷区精度检查。

同位置局部 3% 剂量差通过率（无距离搜索，非有限 DTA gamma）为 57.5466%。独立批次给出的逐点剂量差标准误中位数为参考剂量的 3.1769%，56.73% 的评价点标准误超过 3%。因此逐点超差混有显著历史统计噪声；该标准误不等于 gamma 通过率的置信区间。

下表为完整 ROI 剂量比较，不施加 gamma 的低剂量阈值。峰指中央峰 |x|<0.2 mm，谷按上述谷条带合并；入口区是各横向位置从 BODY 入体开始的 20 mm，肺区为 CT 深度 100–200 mm。± 为两引擎独立批次传播得到的 1 SEM，不包含有限电子响应库的系统/采样不确定度。

| 区域 | ROI | (GPU−TOPAS)/TOPAS |
|---|---|---:|
| 入口 20 mm | 中央峰 | -0.0391% ± 0.4703 pp |
| 肺区 100–200 mm | 中央峰 | -0.3505% ± 0.6256 pp |
| 入口 20 mm | 谷区 | -0.7142% ± 0.8273 pp |
| 肺区 100–200 mm | 谷区 | -2.0236% ± 0.2650 pp |
| 入口 20 mm | 全照射野 | -0.1219% ± 0.1261 pp |
| 肺区 100–200 mm | 全照射野 | -0.1502% ± 0.1170 pp |

肺区谷底 −2.02% 的平均偏差明显大于其历史统计标准误，不能把所有差异归为噪声。当前 GPU 的 CT 中子剂量尚未输运，光子不继续输运，电子空间响应仅覆盖主输运循环，带电次级粒子电子损失仍局部沉积；group32 和有限响应库也保留研究近似。这些是解释差异时需要考虑的因素，本次比较未分解各因素的贡献。

与之前 `proton_ct_split_20261001/full10m` 的 group32 结果使用同一组 GPU 种子和同一 TOPAS 参考：局部 3%/0.2 mm 和同位置 3% 剂量差通过率完全相同，6 个 ROI 的偏差变化最大仅 1.845×10⁻⁸ 个百分点。本轮实现和调度优化在这些检查上没有可见精度退化；已有谷区偏差也没有被消除。

全部 20 批运行质量检查通过，EM 失败计数 0，队列溢出 0，最大相对能量残差 3.9841×10⁻⁷。TOPAS 原有无效索引未评分能量的最大入射能量比例为 5.51×10⁻¹⁴。GPU 总进程时间 251.027 秒（含首次加载及输出），39,836 入射质子/秒；这是本次验证任务的计时，不是重新进行的硬件性能基准。

结果和复现材料：

- [gamma / 剂量差图](../out/ct_topas_gamma_20261001/gamma_maps.png)
- [峰谷深度曲线及统计带](../out/ct_topas_gamma_20261001/peak_valley.png)
- [横向剂量曲线](../out/ct_topas_gamma_20261001/lateral_profiles.png)
- [主结果 JSON](../out/ct_topas_gamma_20261001/summary.json)、[各阈值及 ROI gamma](../out/ct_topas_gamma_20261001/gamma_extended.json)、[运行审计](../out/ct_topas_gamma_20261001/validation.json)
- [完整输入与参考清单](../out/ct_topas_gamma_20261001/manifest.json)、[源文件哈希](../out/ct_topas_gamma_20261001/source_sha256.json)、[几何审计](../out/ct_topas_gamma_20261001/geometry_audit.json)
- [执行脚本](../out/ct_topas_gamma_20261001/run.py)、[基础分析](../out/ct_topas_gamma_20261001/analyze.py)、[gamma 图及扩展指标](../out/ct_topas_gamma_20261001/diagnostics.py)、[逐点统计](../out/ct_topas_gamma_20261001/statistics.py)

Python gamma 依赖路径为 `/mnt/sda/wuwei/maigo_gamma_python_deps`。分析脚本可对保存的结果重算；`run.py` 需要新的隔离输出目录，防止覆盖已有批次。原始 GPU FP32 剂量保存在 `out/ct_topas_gamma_20261001/gpu/out/batch000` 至 `batch019`。

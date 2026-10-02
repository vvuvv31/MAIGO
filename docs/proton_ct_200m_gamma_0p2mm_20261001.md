**Proton CT minibeam：两端各 2 亿粒子，local gamma / 0.2 mm**

直接复用已通过运行审计的 GPU / TOPAS 各 200M 剂量，无需重新进行蒙卡。GPU 为修复 CT 外边界舍入后的 FP32 版本，全部 200M 来自同一程序。

沿用 BODY 内 |y|<15 mm 平均的二维 x–z 剂量、主照射野 |x|<15 mm，以及 TOPAS 完整投影最大剂量的 3% 低剂量阈值。TOPAS 为参考；local 分母为各参考点剂量。按实际入射质子数归一化，无剂量拟合、配准或平滑。**本结果是二维投影 gamma，不是三维逐体素 gamma。**

| Local gamma | 主照射野 | 峰条带 | 谷条带 |
|---|---:|---:|---:|
| 1% / 0.2 mm | 92.1627% | 94.0085% | 93.2373% |
| 2% / 0.2 mm | 96.4155% | 99.9342% | 96.9197% |
| 3% / 0.2 mm | 97.9345% | 100.0000% | 98.4729% |

主照射野均评价 124,814 点，gamma≤1 为通过。峰条带为缝中心 ±0.2 mm、|x|<15 mm；谷条带为相邻缝中点 ±0.2 mm、|x|<12.6 mm。各区域均施加同一 3% 低剂量阈值。

PyMedPhys 0.41.0，scipy 线性插值，interp_fraction=10，即搜索步长 0.02 mm，max_gamma=2；评价所有点，不随机抽点，不在通过后提前停止。剂量原始网格间距为 x=0.1 mm、z=0.5 mm；0.02 mm 是插值搜索步长，不是新的蒙卡评分分辨率。max_gamma=2 对通过率判据不构成截断偏差；图中大于 2 的值统一显示为 2。

同一评价点集合的 local 3% / 0 mm 通过率为 89.2961%。增加距离容差允许在邻近位置寻找匹配，因此 0.2 mm gamma 与同位置剂量差是不同指标。

[Gamma 图](../out/ct_topas_200m_20261001/gamma_0p2mm/gamma_maps.png) · [完整指标及输入哈希](../out/ct_topas_200m_20261001/gamma_0p2mm/summary.json) · [分析脚本](../out/ct_topas_200m_20261001/gamma_0p2mm/analyze.py) · [两亿粒子运行报告](proton_ct_200m_20261001.md)

复算：`PYTHONPATH=/mnt/sda/wuwei/maigo_gamma_python_deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 out/ct_topas_200m_20261001/gamma_0p2mm/analyze.py`。

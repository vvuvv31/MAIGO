# 铜段步长减半检查

Cu/slit ceilings halved to 0.025 mm in both engines, water remains 0.05 mm. Three 1M batches each. GPU paired source/seeds against previous three Cu0.05 runs; TOPAS independent seeds. Step study only; not a high-stat replacement for main benchmark.

| 深度 mm | GPU减半变化 pp ± 1SE | TOPAS减半变化 pp ± 1SE | 0.025 mm两端谷区差 | 近似95%区间 |
|---:|---:|---:|---:|---:|
| 20 | +0.52 ± 1.73 | -0.86 ± 1.27 | -0.53% | [-9.46%, +8.41%] |
| 40 | -5.56 ± 2.63 | -0.29 ± 2.20 | -1.18% | [-12.07%, +9.70%] |
| 60 | -4.29 ± 2.07 | -1.21 ± 1.99 | -0.38% | [-7.10%, +6.34%] |
| 80 | -2.15 ± 0.74 | -5.56 ± 2.05 | +3.63% | [-3.97%, +11.23%] |
| 100 | -4.24 ± 1.49 | -3.12 ± 0.97 | +1.19% | [-9.17%, +11.55%] |
| 120 | -0.74 ± 0.68 | -1.89 ± 0.70 | +1.99% | [-1.13%, +5.11%] |

不同TOPAS种子下的变化仍含统计噪声。GPU配对能降低一部分源抽样方差，但改变步数后散射随机数对应关系也会变化。只用此表的点估计不能把0.025 mm宣称为1%解决方案。`GPU_15M_baseline_plus_paired_shift_vs_new_TOPAS`仅为方差降低的估计值，未冒充实际15M的0.025 mm运行。

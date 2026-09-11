# Electron full20 20260910：等中心图（电子包，float march）

与 `gpu_current_20260909_profiles` 同口径：2D图以原始DICOM HU灰度图为底，jet色图50%透明度叠加。CT窗为−1000～1000 HU；逐层方向、位置、间距和尺寸与剂量网格验证一致。

二维图为3×3：行=轴位/冠状位/矢状位，列=TOPAS剂量(Gy)/GPU电子剂量(Gy)/百分比差值。双方剂量共用色标。1D误差与2D差值均为100×(GPU−TOPAS)/TOPAS全体积峰值，显示范围−5%～+5%，保留正负。CSV/NPZ保留未截断值。无配准/拟合归一化。

GPU为电子 joint r3 ON（float march 生产构建），同源同步长；剂量绝对Gy。

## Gamma（≥10% 掩膜，绝对剂量；细 = 0.25 mm 自适应加密）

| 病例 | Global 1%/1mm（粗→细） | Local 1%/1mm（粗→细） | Local 3%/0mm |
|---|---:|---:|---:|
| RT06423 | 99.82 → 99.94 | 93.35 → 99.16 | 97.08 |
| RT07575 | 99.74 → 99.88 | 90.08 → 98.87 | 96.88 |
| 20022516 | 99.56 → 99.97 | 88.90 → 99.50 | 85.66 |

冻结无电子版对照（粗→细 / Local30）：RT06423 87.46→96.87 / 92.25；
RT07575 83.34→95.38 / 89.15；20022516 84.98→98.25 / 81.35。
剂量在 `/mnt/sda/wuwei/electron_full20_float_20260910/<病例>/gpu_sum.raw`，
逐片 `dose.raw` + `execution.json` + `gamma_coarse.json`；参考为冻结
`/mnt/sda/wuwei/ct_previous_full20_20260909/<病例>/topas_sum.raw`。
状态 EXPERIMENT（joint 未验证），非生产验收。

- RT06423：[三轴 profile及百分比误差](RT06423/isocenter_profiles.png)、[三解剖面百分比差值](RT06423/isocenter_planes.png)。
- RT07575：[三轴 profile及百分比误差](RT07575/isocenter_profiles.png)、[三解剖面百分比差值](RT07575/isocenter_planes.png)。
- 20022516：[三轴 profile及百分比误差](20022516/isocenter_profiles.png)、[三解剖面百分比差值](20022516/isocenter_planes.png)。

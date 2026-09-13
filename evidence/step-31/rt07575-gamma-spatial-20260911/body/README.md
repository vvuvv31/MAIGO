# RT07575：仅 RTSTRUCT BODY 内 Gamma（2026-09-11）

按用户要求，将评估范围限制到 RTSTRUCT 的 BODY（ROI 10），并保留原先 TOPAS参考剂量>=全剂量场最大值10%的阈值。
评估体素数350401，原459684，排除体外109283体素。不是HU阈值筛选，BODY内的低密度体素仍保留。

RTSTRUCT 与CT FrameOfReferenceUID一致；原轮廓24层、间隔3mm，CT35层、间隔2mm。
先在DICOM坐标栅格化闭合轮廓，再按切片位置线性插值有符号面内距离场，以体素中心正负决定BODY归属。
所有CT层位于轮廓z范围内，无外推。原始DICOM HU推导出的材料分区与恢复到TOPAS坐标后的CCTG材料数组100%一致。
插值方式已明确记录，不宣称与TPS专用轮廓栅格化算法逐体素完全相同。

仅参考查询点限于BODY；Gamma仍搜索原始完整GPU剂量场，不将体外剂量清零或裁切。
全局归一化保持原TOPAS全剂量场最大值。空间搜索加密到0.0625mm；0mm为同体素剂量差。

| 判据 | Global | Local |
|---|---:|---:|
| 3%/3mm | 100.0000% | 100.0000% |
| 2%/2mm | 100.0000% | 99.9997% |
| 1%/1mm | 99.9304% | 99.8644% |
| 3%/0mm | 99.9951% | 97.1769% |

细化仍是三线性插值场的有限网格搜索，不代表精确连续最小化。
完整逐级搜索结果在analysis.json；不得与旧0.5mm搜索数值直接归因为模拟改善。

此前细化local3%/3mm的1408个失败体素全部在BODY外。限制BODY后，体内local3%/3mm全部通过；
local1%/1mm剩475个失败体素，约0.136%。本次未更改物理模型、未重跑输运。

文件：body_mask_zyx.npy、failure_masks.npz、analysis.json、body_analysis.png。
复现：在仓库根目录运行 `python3 evidence/step-31/rt07575-gamma-spatial-20260911/body_analysis.py`。

# 当前 GPU × TOPAS 10x：三病例完整 benchmark（2026-09-05）

状态：完成。60/60 分片 accepted，overflow=0；总 histories=457,898,870。
仅研究范围验证，不是物理包升级或 plan2 电子响应方案完成声明。

## 统一 Gamma 口径

TOPAS DoseToMedium 原始 float64 Sum replicas 求和；GPU 完整分片剂量求和。
按每个 replica 的逐 spot histories 四舍五入后精确分配，无 LS 缩放，无剂量拟合配准。
以下为 reference ≥10% 峰值的**全阈值区域**，不是抽样结果。
距离搜索为 0.5 mm 三维球内格点 + 三线性插值；不是解析连续最小化。
3%/0mm 为同一体素的剂量差判据。JSON 另含 default_rng(42) 的 50,000 点结果。

## Global Gamma：33 / 22 / 11 / 30（通过率 %）

33 = 3%/3mm；22 = 2%/2mm；11 = 1%/1mm；30 = 3%/0mm。
Global 的剂量差容差以 TOPAS 全体积峰值为基准；Local 以当前参考体素剂量为基准。

| 病例 | 3%/3mm | 2%/2mm | 1%/1mm | 3%/0mm |
|---|---:|---:|---:|---:|
| RT06423 | 100.00 | 99.96 | 98.58 | 99.97 |
| RT07575 | 99.97 | 99.57 | 96.74 | 99.28 |
| 20022516 | 100.00 | 99.94 | 98.79 | 94.27 |

## Local Gamma：33 / 22 / 11 / 30（通过率 %）

| 病例 | 3%/3mm | 2%/2mm | 1%/1mm | 3%/0mm |
|---|---:|---:|---:|---:|
| RT06423 | 99.88 | 99.26 | 87.50 | 92.28 |
| RT07575 | 99.61 | 98.44 | 83.32 | 89.14 |
| 20022516 | 99.94 | 99.61 | 84.97 | 81.36 |

| 病例 | GPU / TOPAS Histories（两者相同） |
|---|---:|
| RT06423 | 151,087,660 |
| RT07575 | 129,638,170 |
| 20022516 | 177,173,040 |

## 过等中心的三轴 1D profile

从上述完整统计量 **3D DoseToMedium** 提取患者 X/Y/Z 三条直线剖面，
不是 1D scorer，也不是横向积分 IDD。等中心不取最近体素：固定另两轴的精确
等中心坐标，用三线性插值；沿取线轴按原始体素中心采样（X/Y 0.5 mm、Z 2 mm）。
横坐标为相对等中心距离，纵坐标为累计绝对 Gy；GPU/TOPAS 使用同一坐标，
无 LS 缩放、峰值归一化、平滑或拟合配准。三条线不能代替全 3D Gamma。

等中心来自冻结 TOPAS `run_full_plan.txt` 的 Patient placement：
TOPAS 被动旋转的逆变换为 `R(+RotZ) × (world − Trans)`，世界等中心为零。
患者体积中心使用 `MHD Offset + (DimSize−1)×Spacing/2`，避免半体素偏移。
20022516 使用已恢复到原患者坐标的剂量，不额外叠加运行时 +1 mm 坐标重标记。

| 病例 | 患者坐标等中心 X / Y / Z（mm） |
|---|---|
| RT06423 | −10.8690 / 93.4088 / 609.2755 |
| RT07575 | −12.2636 / 124.5515 / −781.5517 |
| 20022516 | 61.1405 / −162.1587 / −418.5819 |

![三个病例过等中心三轴剂量剖面](gpu_current_20260905_profiles/threecase_isocenter_profiles.png)

单病例图下排另外显示 `(GPU−TOPAS)/TOPAS全3D峰值 × 100%`，不是 Local 相对误差：

- [RT06423：三轴及差值](gpu_current_20260905_profiles/RT06423_isocenter_profiles.png)
- [RT07575：三轴及差值](gpu_current_20260905_profiles/RT07575_isocenter_profiles.png)
- [20022516：三轴及差值](gpu_current_20260905_profiles/20022516_isocenter_profiles.png)

原始曲线为同目录 `{case}_{X,Y,Z}.csv`（共 9 份）；坐标、输入及 CSV SHA256 见
[profiles_manifest.json](gpu_current_20260905_profiles/profiles_manifest.json)。
脚本：`python3 tools/plot_topas10x_isocenter_profiles.py`；已有输出拒绝覆盖，
复现时使用 `--output <新目录>`。此次仅后处理，未重跑蒙卡或改动物理。

## 运行与质量

| 病例 | 程序 Elapsed 之和（分钟） | Accepted 分片 | Overflow | 最大物理能量相对残差 |
|---|---:|---:|---:|---:|
| RT06423 | 18.54 | 20/20 | 0 | 8.105e-6 |
| RT07575 | 16.78 | 20/20 | 0 | 1.185e-5 |
| 20022516 | 21.36 | 20/20 | 0 | 8.259e-6 |

耗时为各已接受分片日志 Elapsed 之和，不含准备、Gamma 和失败启动；不是端到端墙钟耗时。
仅本地 RTX 2080 Ti / sm_75。无 TOPAS 新运行、远程 GPU、物理参数调整或数据包替换。
使用同一 executable SHA256：`6ae10bb7e11bed70a9ce602a6ba25d141cfa7ad641158162f747a00b12570c0a`。
包含之前已有、尚未提交的 entrance-mask candidate；本轮未改 transport 源码。

## 几何与失败记录

- RT06423：沿用现有 packed CT。用当前 Schneider parser 重建交叉核对，密度与 section 全体素一致。
- RT07575：从病例 DICOM 生成相同形式的无损轴重排；417×0.5 mm 的半范围为 104.25 mm，源变换使用该几何值，没有按剂量调偏移。五个 TOPAS replica 的 Schneider SHA 与仓库一致。
- 20022516：native CT 原 Z 起点 −1 mm 被现有 delta-tail scorer 的 origin=0 门禁拒绝。新运行把 CT 和源同时 +1 mm 平移，密度/section/LUT payload 字节不变，全部相对几何不变；没有关闭 delta 表或放宽门禁。审计在 r3/coordinate_rebase_audit.json。
- 初始 float64 参考读取、SYCL 库环境及输出目录收集问题均保留旧记录；首个成功 RT06423 分片经质量/计数验证后恢复使用，未重复计入。
- 当前 delta-tail 提取能点 150/200/225 MeV/u。跨病例结果不等于新的能区、密度或几何响应验证通过。
- 常规 Gamma 较高，但严格 Local 仍未完全 match；20022516 的 3%/0mm 最弱。没有据这些结果认定剩余误差的唯一物理来源。

## 产物

- RT06423：[患者坐标 GPU 剂量](/mnt/sda/wuwei/topas10x_threecase_20260905_r2/RT06423/gpu_sum_patient.mhd)、[参考剂量](/mnt/sda/wuwei/topas10x_threecase_20260905_r2/RT06423/topas_sum.mhd)、[完整 Gamma JSON](/mnt/sda/wuwei/topas10x_threecase_20260905_r2/RT06423/gamma.json)。
- RT07575：[患者坐标 GPU 剂量](/mnt/sda/wuwei/topas10x_threecase_20260905_r2/RT07575/gpu_sum_patient.mhd)、[参考剂量](/mnt/sda/wuwei/topas10x_threecase_20260905_r2/RT07575/topas_sum.mhd)、[完整 Gamma JSON](/mnt/sda/wuwei/topas10x_threecase_20260905_r2/RT07575/gamma.json)。
- 20022516：[患者坐标 GPU 剂量](/mnt/sda/wuwei/topas10x_threecase_20260905_r3/20022516/gpu_sum_patient.mhd)、[参考剂量](/mnt/sda/wuwei/topas10x_threecase_20260905_r3/20022516/topas_sum.mhd)、[完整 Gamma JSON](/mnt/sda/wuwei/topas10x_threecase_20260905_r3/20022516/gamma.json)。

患者坐标输出仅做已声明的体素排列/坐标映射，不插值、不调剂量。
原 GPU 坐标剂量保留为各目录 gpu_sum.mhd/raw；逐片 dose、quality_report、energy_ledger、配置和日志全部保留。

[总证据与 SHA 索引](/mnt/sdb/wuwei/MAIGO/evidence/step-31/topas10x-current-20260905/summary.json)

脚本：tools/run_topas10x_gpu_benchmark.py、tools/evaluate_topas10x_gpu_gamma.py、tools/prepare_topas10x_ct.cpp。
运行需本地 SYCL 库路径 /home/wuwei/sycl_workspace/llvm/build/install/lib，并使用 --device cuda。
新运行使用新输出目录；禁止覆盖此次冻结结果。4 项 Gamma/分片测试通过，数据 verifier 16/16。
无提交或推送。

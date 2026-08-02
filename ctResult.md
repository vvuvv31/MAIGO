# CT 蒙卡结果总表

本文汇总截至 **2026-08-03** 的患者 CT、CT 中 LET、CT minibeam 和
RT07575 minibeam plan 验证。详细几何推导见 `ctplan.md`，minibeam 的逐步物理
修复见 `minibeam.md`，LET 后续路线见 `futureStep.md`。本文只保留最终结果、
关键中间结论、运行参数和可复现路径。

## 1. 结果口径

不同阶段使用过两种参考，数值不能直接混表：

1. **10M plan-shape 验证**：GPU 对 `physical_dose.mhd`（TOPAS 单 spot Dij
   乘优化权重），在高剂量区拟合一个全局 scale。它验证几何、权重顺序和剂量
   形状，不是独立 full-plan TOPAS 盲测。
2. **严格同粒子 full-plan**：GPU 与 TOPAS 使用完全相同的正权重 spot integer
   L4 allocation，GPU scale 固定为 1。两边先用 RTSTRUCT BODY 去除体外空气，
   gamma mask 为 `BODY ∩ TOPAS dose >= 10% BODY Dmax`。这是当前 dose/LET 的
   权威结果。

除非单独注明，gamma 为 3D、评价剂量三线性插值；有 DTA 的常规搜索步长为
0.5 mm，亚毫米 minibeam 为 0.1 mm。LET gamma 使用 **TOPAS dose 的 10% mask**，
不使用 LET 自身阈值。

## 2. Case、几何与粒子预算

| Case | 解剖 | TPS角 | GPU几何 | spots（active） | Dij histories/spot | 当前严格 full-plan histories |
|---|---|---:|---|---:|---:|---:|
| RT06423 | head | 90° | `tps_90`, patient −X depth | 1102（1015） | 50k | 15,108,664 |
| RT07575 | head | 90° | `tps_90`, patient −X depth | 917（853） | 50k | 12,963,817 |
| RT06541 | head | 270° | 保留 L7/L8 偏转 | 983（924） | 50k | **不采用：TOPAS数据有问题** |
| 20022516 | lung | 0° | `tps_gantry_y`, patient +Y depth | 1549（1234） | 100k | 17,717,177（TOPAS已到，GPU待跑） |

20022516 完整物理计划名义粒子数为 1,771,717,720；集群 TOPAS 使用整数
`K=100`，实际输运 17,717,177 histories，dose 乘 100，LET_d 是比值而不缩放。
任意机架角另可使用 opt-in `tpsSource: true`；默认旧 CT example 路径不变。

## 3. 普通 CT GPU 生产参数

当前最终 dose+LET 使用 `physics_profile: best`。RT06423/RT07575 的配置分别为
`config/beam_ct_fullplan_rt06423_let_soft_tissue.yaml` 和
`config/beam_ct_fullplan_rt07575_let_soft_tissue.yaml`。

| 参数 | best 值 |
|---|---|
| device | CUDA，NVIDIA TITAN RTX，driver CUDA 12.6 |
| `maximum_step_mm` | 0.1 mm |
| `maximum_relative_energy_loss` | 0.001 |
| primary / secondary cutoff | 0.1 / 0.1 MeV |
| voxel spacing | 0.5 × 0.5 × 2.0 mm³（beam frame 为 0.5 × 2 × 0.5 mm³） |
| dose response scale | 0.982，三病例共享，不按患者拟合 |
| straggling | enabled，scale 1.2 |
| MCS | enabled；普通 CT 当前 `enable_ct_material_mcs=false` |
| nuclear chain | primary attenuation + direct secondary + charged transport + cascade |
| cascade generations | 2 |
| neutral transport | off（正式 neutron/gamma package 尚未建立） |
| stopping power | Geant4 11.3.2；50 isotope particle-specific tables |
| material tables | lung / Schneider soft tissue / bone ion stopping power与截面 |
| primary/cascade final state | 100k soft-tissue INCL++ correlated packages |
| secondary queue | 65M；batch 65,536；energy sorting enabled |
| memory budget | 84%；估算约 9.4 GiB |
| LET scorer | enabled，primary C-12 与 all-hadron LET_d |

`medium` 保留完整 dose chain、最大相对能损 0.005；`fast` 保留 dose chain、最大
相对能损 0.01 并禁止 LET。两者不允许与 minibeam 同时启用，在完成更多跨病例
TOPAS 门禁前属于 preview；最终 dose/LET 应使用 best。

## 4. 严格同粒子普通 CT full-plan：dose

以下是 BODY 内、absolute scale=1 的权威 GPU/TOPAS 结果：

| Case | histories | NRMSE | integral Δ | G/L 3%/3mm | G/L 2%/2mm | G/L 1%/1mm | G/L 3%/0mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| RT06423 | 15.109M | 2.379% | +0.704% | 99.9997 / 99.9866 | 99.9815 / 99.4565 | 91.3153 / 71.2856 | 80.7940 / 49.4963 |
| RT07575 | 12.964M | 2.484% | +0.581% | 99.9994 / 99.9442 | 99.9040 / 98.8121 | 90.8177 / 67.9095 | 78.2618 / 41.9709 |

两例 IDD peak bin 完全一致，IDD correlation 分别为 0.9999977 和 0.9999984。
严格 3%/0 mm/local 对逐体素噪声和低剂量容差非常敏感；它明显低于有 DTA 的
gamma，但不表示几何或射程失败。

### 4.1 时间与速度

| Case | GPU elapsed | GPU throughput | GPU kernels（primary/secondary） | TOPAS 56T execution / wall | TOPAS throughput | wall speedup |
|---|---:|---:|---:|---:|---:|---:|
| RT06423 | 237.24 s | 63.69k/s | 138.12 / 93.40 s | 76,027.65 / 76,039.1 s | 198.73/s | 320.5× |
| RT07575 | 212.17 s | 61.10k/s | 116.00 / 90.57 s | 42,506.77 / 42,517.4 s | 304.98/s | 200.4× |

两例 secondary/cascade queue overflow 均为 0；GPU energy-balance error 分别为
2.28e-6 和 4.03e-6。

## 5. 严格同粒子普通 CT full-plan：LET_d

定义为 dose-averaged electronic LET，单位 `MeV/mm/(g/cm3)`。当前 all-hadron
LET 的相关性很高，但绝对均值仍系统偏高；global gamma 接近饱和，local gamma
明显落后于 dose。

| Case | TOPAS/GPU mean | mean relative bias | Pearson r | G/L 3%/3mm | G/L 2%/2mm | G/L 1%/1mm | G/L 3%/0mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| RT06423 | 41.898 / 43.644 | +8.624% | 0.99494 | 99.9946 / 90.9841 | 99.9205 / 73.0092 | 82.1993 / 41.8365 | 87.8820 / 19.7911 |
| RT07575 | 39.081 / 41.147 | +10.234% | 0.99451 | 99.9965 / 86.1031 | 99.9366 / 68.8561 | 81.3376 / 38.5691 | 88.7004 / 17.2605 |

primary C-12 LET 已较好：RT06423/RT07575 的 Pearson r 约 0.9965，平均偏差约
-0.94% 和同量级。all-hadron 剩余误差主要来自 fragment species、材料条件化
末态及低能离子 LET，而不是主碳射程。

已验证的 LET 改进：

- secondary local-deposit cutoff 从 1.0 降至 0.1 MeV，吞吐约下降 12.8%；
- 50 isotope 表相对 32 isotope 只有小幅增益；
- material ion stopping power + 100k INCL++ package 是当前 best 主线；
- case-specific LET 乘子、按患者拟合 SP/MCS 不作为生产方案。

## 6. 历史四病例 10M plan-shape 验证

本节使用 `physical_dose.mhd = TOPAS Dij·x`，高剂量区拟合全局 scale，因此只用于
几何和形状回归。

| Case | G/L 3%/3mm | G/L 2%/2mm | G/L 1%/1mm | G/L 3%/0mm | NRMSE |
|---|---:|---:|---:|---:|---:|
| RT07575 | 99.322 / 98.158 | 97.662 / 95.247 | 85.852 / 59.577 | 79.084 / 41.780 | 2.407% |
| RT06423 | 99.858 / 98.490 | 97.943 / 92.788 | 79.694 / 53.385 | 82.451 / 45.516 | 2.275% |
| RT06541 | 99.912 / 98.759 | 98.282 / 92.070 | 71.464 / 44.927 | 77.879 / 37.793 | 2.576% |
| 20022516 | 99.778 / 98.969 | 98.316 / 94.853 | 79.117 / 53.565 | 79.150 / 34.485 | 2.482% |

RT06541 仅保留这份历史 Dij-shape 结果；后来复制的独立 full-plan TOPAS 数据已
确认有问题并删除，不能用来声称严格同粒子 match。

20022516 10M GPU 用时 52.20 s、191.58k histories/s，primary/secondary kernel
42.22/7.34 s，overflow=0。它证明 lung 的 `tps_gantry_y`、逐 spot L7/L8 偏转
和低密度 CT 路径闭环成立。

## 7. 20022516 lung 的直接 TOPAS/GPU 子集验证

选取 5 个代表性 spot，GPU/TOPAS 严格同为 100k histories、scale=1：

| 指标 | 结果 |
|---|---:|
| integral Δ GPU/TOPAS | +0.704% |
| G/L 3%/3mm | 99.834 / 99.670% |
| G/L 2%/2mm | 98.486 / 94.958% |
| NRMSE / IDD correlation | 5.073% / 0.999385 |
| GPU | 0.875 s，114.34k/s |
| TOPAS 40T | 1207.0 s wall，83.4/s |
| speedup | 约 1380× |

新复制的完整 lung TOPAS 使用 17,717,177 histories、56 tasks，execution
156,934.43 s、wall 156,950 s（43.60 h），平均约 112.90 histories/s、有效
48.37 CPU cores。dose、primary-C12 LET 和 all-hadron LET 文件均已到本地；
其中未计分导航能量至多 0.02198 MeV，可忽略。**尚未运行严格同粒子 best GPU
及统一 BODY dose/LET gamma，因此不能把旧 10M 拟合结果当作该 full-plan 结果。**

## 8. CT 网格、材料与性能结论

- CCTG origin 必须是首体素 low edge，不是中心；旧 half-voxel 约定会造成
  0.25 mm minibeam phase error。
- v1 CCTG 不得把诊断用全 1 `mass_sp_za_rel` 当 Schneider mass-SP；修复后
  lung/bone 使用配置中的 absolute material tables。
- 2×2×2 mm CT 下采样对普通 full-plan 的端到端加速可忽略，次级输运主导；
  不作为生产优化。
- source 必须保留 L7/L8 微角和 SAD，不能把所有 spot 固定为理想 90°/0°。
- GPU/TOPAS 若不使用相同 Geant4 版本的 SP、XS 和 final-state package，残差
  同时包含版本差异，不能只归因于 GPU kernel。

## 9. 异质 CT/minibeam 材料验证

200 MeV/u、100k histories 的 lateral / longitudinal / combined 多材料 phantom：

| 场景 | GPU throughput | TOPAS 40T | depth L1 | integral ratio | G/L 3%/1mm | ΔR80 |
|---|---:|---:|---:|---:|---:|---:|
| 横向多材料 | 54.80k/s | 234.70 s | 1.326% | 0.9996 | 100.00 / 94.44% | +0.120 mm |
| 纵向多层 | 53.18k/s | 247.16 s | 1.288% | 1.0027 | 98.77 / 95.71% | -0.015 mm |
| 横纵组合 | 51.89k/s | 233.53 s | 1.095% | 1.0063 | 100.00 / 96.15% | -0.024 mm |

最复杂场景提高到 1M 后，production 为 93.83k/s、depth local 3%/1mm
96.15%；high-accuracy 为 23.23k/s、99.36%，约慢 4.04×。严格步长改善材料
界面和部分 lateral gamma，但不能修复深部 fragment-dominated 区域。

## 10. 20022516 lung CT 单平面 minibeam

共同几何：200 MeV/u，50×50 mm 平面源，60 mm 铜准直器，15 条 slit，宽
0.5 mm、pitch 3.6 mm，准直器出口到 CT 60 mm，dose grid 0.5×0.5×2 mm³，
LET off。必须把 TOPAS source/snout 平移到 patient-local 中心，否则会穿过
完全不同的肺路径。

| GPU 参数 | production | high-accuracy |
|---|---:|---:|
| `maximum_step_mm` | 0.2 mm | 0.1 mm |
| `maximum_relative_energy_loss` | 0.005 | 0.001 |
| copper max step | 0.25 mm | 0.25 mm |
| copper MCS scale | 0.785 | 0.785 |
| copper nuclear/reaction products | on | on |
| neutral mode | first interaction | first interaction |
| dose scale | absolute equal-history；不拟合 | absolute equal-history；不拟合 |

### 10.1 统计与物理结果

| 比较 | integral Δ | NRMSE | ΔR80 | G/L 3%/0.3mm | G/L 3%/0.5mm |
|---|---:|---:|---:|---:|---:|
| 1M GPU production vs 1M TOPAS | +5.626% | 5.657% | +0.243 mm | 89.249 / 47.656% | — |
| 1M GPU high-accuracy vs 1M TOPAS | +2.742% | 3.674% | +0.105 mm | 90.703 / 48.272% | — |
| 10M GPU vs 5M TOPAS | +2.372% | 2.894% | — | 95.039 / 53.332% | — |
| 20M GPU vs 10M TOPAS | +2.530% | 2.740% | — | **96.462 / 55.464%** | **96.955 / 62.428%** |

20M/10M 使用严格 history scale=0.5，不拟合 GPU。两个 5M TOPAS seed 自比的
3%/0.3 mm 为 99.095/59.294%，两个 10M GPU seed 自比为 98.653/67.215%。
因此 raw local 的主要限制是有限统计、0.5 mm scorer 对 0.3 mm DTA 的欠采样，
以及约 2.5% 的 PVDR/积分系统差。uncertainty-aware local 3%/0.3 mm 为
1σ 67.874%、2σ 81.231%。

### 10.2 时间

| histories | GPU | TOPAS 40T |
|---:|---:|---:|
| 100k | 3.165 s（31.60k/s） | 291.50 s |
| 1M production | 约21.1 s（47.4k/s） | 2259.91 s execution |
| 1M high-accuracy | 57.42 s（17.42k/s） | 同一1M参考 |
| 5M | — | 11034.71 / 11001.65 s（两个seed） |

## 11. RT07575 优化 minibeam full-plan

计划有 1943 spots、1617 个正权重 spot、`sum(w)=43165.516`，四个 subfield
组成两个 opposed angles。minibeam 参数：100 mm copper、17 slits、宽0.7 mm、
pitch 3 mm、half-length 50 mm；angle02 slit array offset 1.5 mm。输运启用
copper EM、nuclear attenuation、charged reaction products、neutral first
interaction、4 cascade generations和粒子特异 SP。LET 本轮关闭。

| GPU 参数 | 值 |
|---|---:|
| numerical step / relative loss | 0.1 mm / 0.001 |
| energy cutoff / secondary local deposit | 0.1 / 1.0 MeV |
| dose output scale | 0.9858617637 |
| copper density / radiation length | 8.96 g/cm³ / 12.8628 g/cm² |
| copper max step / MCS scale | 0.25 mm / 0.785 |
| copper straggling | off |
| copper survivor energy-loss base scale | 0.956（另有100–400 MeV/u通用能量表） |
| water primary stopping-power scale | 0.9958 |
| low-energy MCS transition | 180 MeV/u |
| low-energy primary / fragment MCS scale | 0.20 / 1.00 |
| electronic build-up | fraction 0.06，MFP 0.5 mm，lateral sigma 0.5 mm |
| maximum cascade / neutral generations | 4 / 1 |

高统计参数：每个 seed 的 angle01/angle02 为 79,737,018 + 49,759,530；
secondary queue 58M、neutral 70M、memory fraction 0.82，估算显存
20,063/18,133 MiB，实测约18.6 GiB，所有 overflow=0。

### 11.1 sparse Dij 阈值审计

旧 `dose_limit=2e-6 Gy/spot/voxel` 删除了 87.281% nnz；其单 voxel 最坏累计
遗漏 0.08633 Gy，等于 BODY peak 的 2.950%，几乎占满 3% 容差。GPU 逐 spot
施加相同阈值后，BODY integral 从 +2.357% 变为 -0.165%，但 gamma 没提高：
3%/0.3 mm 从 89.591/66.494% 降到 88.979/66.306%。阈值解释背景积分，不是
严格 gamma 的主要误差源。builder 已把 cutoff 降为 2e-7 Gy并记录实际 dropped
dose sum；最终物理调参应使用无阈值 reference。

### 11.2 GPU–GPU 粒子数收敛

两组完全独立 seed，BODY、10% threshold、scale=1。0 mm 使用全部体素；
0.3/0.5 mm 收敛表使用固定50k样本。

| histories/seed | G/L 1%/0mm | G/L 2%/0mm | G/L 3%/0mm | G/L 3%/0.3mm | G/L 3%/0.5mm | NRMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 129.50M | 83.43 / 38.26 | 97.72 / 66.35 | 99.736 / 82.522 | 99.945 / 96.219 | 99.973 / 97.998 | 0.774% |
| 258.99M | 92.62 / 51.30 | 99.58 / 80.44 | 99.985 / 92.488 | 99.995 / 98.811 | 99.999 / 99.416 | 0.551% |
| 388.49M | 96.11 / 59.70 | 99.89 / 87.15 | 99.999 / 96.085 | 100 / 99.540 | 100 / 99.783 | 0.451% |
| 517.99M | 97.64 / 66.17 | 99.98 / 91.00 | 100 / 97.720 | 100 / 99.766 | 100 / 99.898 | 0.390% |
| **647.48M** | **98.49 / 71.06** | **99.991 / 93.557** | **100 / 98.664** | **100 / 99.899** | **100 / 99.960** | **0.349%** |

最终双向 exact 3%/0 mm local 为 98.6552% 和 98.6721%。新增16个角度任务
共用时 27,771.09 s（7.714 GPU h）；连同已有 batch，每个 seed 为647,482,740，
两组总计1,294,965,480 histories。NRMSE 基本按 `1/sqrt(N)` 下降，证明同模型
残差主要是统计涨落。下一步可让 TOPAS 使用同一粒子数；若 GPU/TOPAS 明显低于
该 GPU/GPU 上限，差异来自 TOPAS统计或物理模型，而不是 GPU history 数。

## 12. 已排除或不能作为生产结论的结果

- RT06541 独立 full-plan TOPAS 数据有问题，已删除；只保留旧 Dij-shape 回归。
- 把实际 Dij 50k/spot 误认为100k曾造成每粒子剂量约1.928倍错位；不是物理模型。
- 按能层、按病例拟合 scale 可抬高 gamma，但属于过拟合，不用于生产。
- 给 GPU 和 TOPAS 同时应用旧 sparse cutoff 不能提高 gamma。
- 出现 secondary/cascade/neutral queue overflow 的高统计 run 一律作废。
- 0.5 mm scorer 上的0.3 mm gamma是固定网格回归，不等价于0.3 mm独立测量精度。

## 13. 权威输出与待办

主要机器可读结果：

- `out/fullplan_result/summary.json`：RT06423/RT07575严格同粒子 dose+LET；
- `out/ct/20022516/full_plan_10M/match_physical/match_metrics.json`：lung 10M Dij-shape；
- `out/ct/20022516/topas_local_compare_100k/`：lung直接5-spot验证；
- `out/ct/20022516/minibeam_plane_e200_20M_vs_topas_10M/`：lung CT minibeam；
- `out/ct/RT07575/minibeam_plan/gpu_gpu_650m_seed_convergence/summary.json`；
- `out/ct/RT07575/minibeam_plan/gpu_gpu_650m_seed_convergence/convergence.json`。

下一项应先处理 `ct/fullplan_result/20022516`：转换TOPAS dose/LET、生成BODY
mask、运行严格相同17,717,177 histories的best GPU，然后用与两例head完全一致
的33/22/11/30 dose和all-hadron LET gamma口径更新本文。其后再决定是否启动
RT07575 minibeam的约650M TOPAS正式验证。

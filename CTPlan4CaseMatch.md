# 四个 CT Plan 的 GPU / TOPAS-Dij 剂量匹配

## 1. 结论

在统一的比较口径下，RT07575、RT06423、RT06541 和 lung case 20022516
的 GPU 计划剂量均与参考剂量良好匹配：

- 四例 **global 3%/3 mm 均 ≥99.32%**，**local 3%/3 mm 均 ≥98.16%**；
- 四例 **global 2%/2 mm 均 ≥97.66%**；
- 新的 lung case 20022516 为 **global 2%/2 mm 98.316%**、
  **global 3%/3 mm 99.778%**；
- 20022516 的三维剂量峰体素完全一致，沿 patient Y 的 IDD 峰只差一个
  2 mm 输出体素；
- lung case 没有因为不同入射轴或低密度肺组织出现整体匹配崩溃。

这里的“TOPAS 参考”准确地说是：**TOPAS 单 spot Dij 经优化权重 `x`
线性叠加得到的 `physical_dose.mhd`**，不是另一次独立运行的 full-plan
TOPAS 蒙卡。因此本验证覆盖逐 spot TOPAS 物理响应、计划权重、GPU 输运和
剂量空间映射，但不应表述成两个独立 full-plan 随机模拟之间的盲测。

## 2. 统一比较口径

| 项目 | 设置 |
|---|---|
| GPU histories | 每例 10,000,000 |
| 参考 | 各 case 的 `code/physical_dose.mhd`（TOPAS Dij·x） |
| 剂量网格 | 2×2×2 mm³ |
| 剂量阈值 | reference maximum 的 10% |
| 剂量归一 | 阈值以上体素最小二乘拟合一个全局 scale |
| Gamma | 3D、评价剂量三线性插值、0.5 mm 搜索步长 |
| 2%/2 mm 与 3%/3 mm | global 和 local 都报告 |
| 1%/1 mm 与 3%/0 mm | 严格诊断指标，不作为临床验收结论 |

所有报告的 gamma 点数覆盖阈值以上的全部体素；没有为这四例做随机抽样。
全局 scale 是比较流程的一部分，所以 gamma 主要评价剂量形状与空间一致性；
“按计划粒子预算的绝对响应”另由拟合 scale / 理论 scale 比值报告。

## 3. Case 与几何

| Case | 解剖 | TPS RotZ | GPU 深度映射 | spots（active） | Dij histories/spot |
|---|---|---:|---|---:|---:|
| RT07575 | head | 90° | patient X (`tps_90`) | 917 (853) | 50,000 |
| RT06423 | head | 90° | patient X (`tps_90`) | 1102 (1015) | 50,000 |
| RT06541 | head | 270° | patient X (`tps_90`，保留偏转) | 983 (924) | 50,000 |
| 20022516 | lung | 0° | patient +Y (`tps_gantry_y`) | 1549 (1234) | 100,000 |

20022516 与前三个 head case 的关键区别不是简单的病例名，而是：

1. 中心束在 TOPAS world 中沿 +Y，不能照搬 patient-X-depth 的体素 packing；
2. GPU 网格采用 `GPU(x,y,z) = patient(x,z,+y)`；
3. source pose 仍要经过 TOPAS passive `Patient/RotZ` 的逆变换；
4. 每个 spot 的 L7/L8 小角度偏转和 emittance 必须保留；
5. 束路经过低密度肺组织及更多密度界面，对亚体素位置和局部剂量更敏感。

本次为代码增加 `spots_geometry_mode: tps_gantry_y`，没有改变既有
`tps_90` case 的实现路径。

## 4. 主要 Gamma 结果

10% reference-maximum 阈值，单位为通过率 `%`：

| Case | Global 2%/2 mm | Local 2%/2 mm | Global 3%/3 mm | Local 3%/3 mm |
|---|---:|---:|---:|---:|
| RT07575 | 97.662 | 95.247 | 99.322 | 98.158 |
| RT06423 | 97.943 | 92.788 | 99.858 | 98.490 |
| RT06541 | 98.282 | 92.070 | 99.912 | 98.759 |
| **20022516 lung** | **98.316** | **94.853** | **99.778** | **98.969** |

按常用的 global 3%/3 mm 或更严格的 global 2%/2 mm 观察，lung case 与三个
head case 处于同一精度水平。若人为要求 local 2%/2 mm ≥95%，20022516
为 94.853%，略低 0.147 个百分点；RT06423 和 RT06541 也低于该阈值。
因此不应只用 local 2%/2 mm 的单一阈值判断几何是否正确。

## 5. 严格诊断指标

| Case | Global 1%/1 mm | Local 1%/1 mm | Global 3%/0 mm | Local 3%/0 mm |
|---|---:|---:|---:|---:|
| RT07575 | 85.852 | 59.577 | 79.084 | 41.780 |
| RT06423 | 79.694 | 53.385 | 82.451 | 45.516 |
| RT06541 | 71.464 | 44.927 | 77.879 | 37.793 |
| **20022516 lung** | **79.117** | **53.565** | **79.150** | **34.485** |

20022516 的 global 1%/1 mm 和 global 3%/0 mm 都落在已有 head cases 的
范围内。它的 local 3%/0 mm 最低；0 mm 不允许任何空间补偿，而 local
准则在低剂量体素给出的剂量容差又更小，因此会放大肺内密度界面、统计噪声、
0.5 mm CT 到 2 mm dose block-average，以及两个输运模型细节差异的影响。
这个结果指出了进一步改进的方向，但不支持“lung 几何轴映射错误”的判断。

## 6. 剂量形状与绝对响应

| Case | 高剂量 NRMSE | 3D cosine | IDD correlation | IDD 峰差 | 积分剂量差 | 拟合/理论 scale |
|---|---:|---:|---:|---:|---:|---:|
| RT07575 | 2.407% | 0.998713 | 0.999820 | 0 bin | −0.037% | 98.067% |
| RT06423 | 2.275% | 0.998828 | 0.999764 | 1 bin | −0.056% | 97.622% |
| RT06541 | 2.576% | 0.998264 | 0.999591 | 0 bin | −0.223% | 97.214% |
| **20022516 lung** | **2.482%** | **0.998470** | **0.999887** | **1 bin** | **+0.099%** | **98.595%** |

一个 bin 为 2 mm。20022516 的 reference 与 scaled GPU 最大剂量分别为
5.8865 Gy 和 5.8110 Gy，最大剂量体素坐标均为 `(147, 63, 20)`。
其最小二乘 scale 为 174.6818；由计划粒子预算
1,771,717,719.95 / 10,000,000 得到的理论 scale 为 177.1718，
两者相差约 1.405%。这一绝对响应偏差比三个 head case 更小，没有看到
lung case 需要异常的病例专用重标定。

## 7. 20022516 正式运行

正式 10M 配置：

```text
config/beam_ct_20022516_10M.yaml
```

运行结果：

| 项目 | 值 |
|---|---:|
| 总 histories | 10,000,000 |
| active spots | 1234 |
| wall time | 52.197 s |
| throughput | 191,583 histories/s |
| primary kernel | 42.217 s |
| secondary kernel | 7.338 s |
| secondary queue overflow | 0 |

GPU 输出和匹配结果：

```text
out/ct/20022516/full_plan_10M/dose_10M.mhd
out/ct/20022516/full_plan_10M/match_physical/match_metrics.json
out/ct/20022516/full_plan_10M/multiplanar_gamma/
```

三组 axial、coronal、sagittal 剂量/剂量差/2D gamma 图均使用相同剂量范围。
单个二维切面的通过率可能低于完整 3D gamma，因为 2D gamma 不允许从相邻
切面寻找匹配点；它们适合定位局部差异，不应替代上面的 3D 汇总值。

## 8. 复现命令

```bash
/tmp/maigo_review_off_make/carbon_mc \
  --config config/beam_ct_20022516_10M.yaml \
  --device cuda

python3 validation/scripts/match_gpu_to_physical_dose.py \
  out/ct/20022516/full_plan_10M/dose_10M.mhd \
  ct/20022516/code/physical_dose.mhd \
  --output-dir out/ct/20022516/full_plan_10M/match_physical \
  --patient-shape 960 607 42 \
  --histories 10000000 \
  --mapping beam_y \
  --no-flip-x --no-flip-y \
  --gamma-points 100000 \
  --gamma-resolution-mm 0.5 \
  --strict-gamma
```

注意：`20022516` 的 `beam_y`、无 X/Y flip 是由 TOPAS source、patient pose
与 CT reorientation 共同确定的离散几何约定，不是为了提高 gamma 做的自由
平移或逐病例搜索。本比较没有施加任何自由三维 translation。

## 9. 最终判断

20022516 的 lung 结果通过了与前三例相同口径的匹配验证。其 global
2%/2 mm、global/local 3%/3 mm、NRMSE、cosine、IDD 和绝对响应均与已有
head cases 同量级，说明新增 Y-depth 几何闭环有效，也没有发现肺组织导致的
系统性剂量失配。

更严格的 local 3%/0 mm 仍是四例中最低项。若下一阶段继续提高精度，应优先
按失败体素与 HU/密度梯度、材料类别、束流深度和能层做相关性分析，再决定是
改进 CT 材料映射、界面步进/边界处理还是核反应模型；不建议直接增加
case-specific 剂量校准，因为当前四例的绝对 scale 已表现出较好的一致性。

# RT07575 head proton minibeam：电子包开启/关闭全计划对照

94.4M/臂结果：患者内local1%/1mm、search0.02mm开启99.562942%、关闭99.491267%，都通过>99%；search0.1mm分别97.595913%/97.440706%，均未通过。local3%/0.2mm从23.6M的92.224037%/92.182540%提高到97.670822%/97.579746%，仍低于绝对99%。关闭后的吞吐提高57.619%，患者内dose sum变化+0.003972%，局部RMS改变0.156034%。本病例的开关剂量差小于大部分GPU与TOPAS的差，不能据此断言其他患者/网格无需电子包。

原487个有效spot、CT、铜准直器、能量/发射度及4Gy粒子归一化保持一致。先完成23,608,643粒子/臂，再按用户要求运行94,434,572粒子/臂的新独立种子campaign。参考为原TOPAS加权物理DIJ（AllChargedUrbanPhysics），没有重新生成TOPAS或拟合剂量/坐标。患者内BODY∪GTV且TOPAS≥10%峰值，共185559体素，网格0.5×0.5×2mm。

## Gamma及搜索步长

| 统计量/判据 | 开启电子包 | 关闭电子包 | 关闭−开启 |
|---|---:|---:|---:|
| 23.6M，local1%/1mm，search0.1mm | 97.696151% | 97.544177% | -0.151973 pp |
| 23.6M，local1%/1mm，search0.02mm | 99.742400% | 99.698209% | -0.044191 pp |
| 23.6M，local3%/0.2mm，search0.02mm | 92.224037% | 92.182540% | -0.041496 pp |
| 94.4M，local1%/1mm，search0.1mm | 97.595913% | 97.440706% | -0.155207 pp |
| 94.4M，local1%/1mm，search0.02mm | 99.562942% | 99.491267% | -0.071675 pp |
| 94.4M，local3%/0.2mm，search0.02mm | 97.670822% | 97.579746% | -0.091076 pp |

历史236M旧实现的local1%/1mm、search0.1mm为98.861818%，也低于99%；它不是相同物理实现的纯统计量控制。相同23.6M剂量仅细化搜索到0.02mm，两模式就超过99%。微束横向剂量梯度很大，必须同时区分蒙卡统计量和有限gamma搜索网格的影响。细化搜索没有改变蒙卡剂量或DTA=1mm。

gamma采用有限球形格点、trilinear插值、完整评价dose体积内搜索，禁止外推。0.1与0.02mm结果分别列出，不以改搜索隐去原结果。

## 吞吐

ABBA：每臂原12个批次，前6批为A1/B1，后6批为B2/A2；同模式两个段使用不同种子，无重复样本。每个6批进程复用context/packages，剔除各自首批计时。RTX2080Ti sm75，CUDA_DISABLE_PTX_JIT=1。未锁频，报告本机观测值，不能视作跨硬件常数。

| 统计量 | 开启 primary/s | 关闭 primary/s | 吞吐提升 | wall减少 |
|---|---:|---:|---:|
| 23.6M/臂 | 49087 | 75763 | 54.344% | 35.209% |
| 94.4M/臂 | 49436 | 77920 | 57.619% | 36.556% |

关闭移除了出生/入队/电子/电磁光子输运及包初始化，并非只跳过最后一个kernel。铜/空气及患者离子输运、涨落、散射、核反应和带电碎片仍保留。

## 两种GPU剂量变化

关闭−开启，采用共同患者内TOPAS≥10%peak区域；dose sum是区域体素剂量求和，不是总能量。

| 统计量 | dose sum变化 | RMS局部剂量变化 | P95绝对局部变化 | paired voxel SE RMS/参考 |
|---|---:|---:|---:|---:|
| 23.6M/臂 | +0.003776% | 0.248815% | 0.500249% | 0.225372% |
| 94.4M/臂 | +0.003972% | 0.156034% | 0.269396% | 0.112769% |

GPU单体素相对参考SE RMS：23.6M开启3.481832%、关闭3.478215%；94.4M开启1.741362%、关闭1.739527%，接近四倍粒子数后标准误差减半。每臂12个独立、近等数量批次提供描述性SE，原TOPAS DIJ逐体素方差不可用。GPU对GPUgamma与绝对TOPASgamma分别在JSON保留，不能用相互匹配代替TOPAS验收。

## 输入、归一化与实验范围

94.4M由原浮点完整计划权重以Decimal最大余数法在整个campaign分配一次，再平衡到12批。不是将旧四组结果重复相加。新seed为202610990000000+b*1e9+active_rank*1000003；两个模式逐spot数量、pose、seed相同，和23.6M种子独立。

每批真实history剂量在输出时乘原处方入射粒子数/整个campaign总histories。合并只求和一次；94.4M的dose_output_scale为23.6M的1/4。没有乘批次数、再平均或用TOPAS拟合倍率。使用既有packed_xneg轴映射恢复患者ZYX。

开启是当前minibeam患者内共享strict26 electron/photon输运（group32），与broadbeam的低密度选择性输运不同。离子使用continuous26，患者ion cap0.1mm、Cu0.05mm，Cu主质子/碎片涨落保持启用。关闭采用既有局部计分分支，对δ/电子能量不再独立输运；核中性产物仍是未追踪能量汇，没有改为局部沉积。

发布程序本来拒绝minibeam缺少共享电子包。为显式消融，只在out中的experimental_config.cpp解除这一条validate限制，并与同一现有CUDA transport静态库链接；两臂用同一实验binary，未改生产配置/代码、未引入新的运行时兼容或诊断模式。原发布binary仍拒绝关闭配置，日志见release_rejection.log。

## 审计与产物

- 23.6M/臂：两模式各12批全部quality accepted、零overflow，配对source/seed相同，离子/核计数与ledger匹配=True。
- 94.4M/臂：两模式各12批全部quality accepted、零overflow，配对source/seed相同，离子/核计数与ledger匹配=True。

原始来源：out/rt07575_proton_current_23m_20261008/；本轮完整日志/配置/输出：out/rt07575_proton_minibeam_no_electrons_20261009/，高统计量在x4/。comparison/report.json及x4/comparison/report.json包含数值、GTV指标、SE和完整审计；两处dose_comparison.png供查看。write_report.py生成本报告。

首个实验config对象编译补齐了与静态库一致的pipeline宏；x4首次启动因错误cwd导致默认当前布局离子CSV解析失败，发生在输运前。失败日志保留在x4/initial_launch_wrong_cwd/，修正cwd后正式运行，不计入剂量与性能。

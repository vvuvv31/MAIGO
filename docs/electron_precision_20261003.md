# 独立电子 FP32 基线与后续精度消融

按用户最新指示，先完成全部独立电子运行路径的 FP32 基线、测量吞吐，再相对这条基线逐项恢复 FP64。之前从 FP64 逐项改成 FP32 的混合精度实验已冻结于 `out/electron_fp32_20261003/mixed_frozen/`，不再作为当前实施顺序。

剂量计分始终是 `CARBON_DOSE_FP32=ON`、`CARBON_DOSE_FP64=OFF`。当前范围是 **proton CT minibeam 的独立电子出生、joint、带电短步、光子和 CT 几何**，不是把整个应用、其他粒子输运或 CPU 报表的所有 double 全部删除。原始包的磁盘格式、SHA 和内容校验保持不变；GPU 运行视图转换为 float。队列接口保留原 ABI，其中 double 字段只作窄化/拓宽；错误快照和 CPU 最终汇总也不属于电子输运算术。

## 配置与实现

- `material_electron_precision: 0`：原电子计算精度，用作性能和剂量对照，计分仍为 FP32。
- `material_electron_precision: 4`：完整 FP32 电子运行路径；要求独立 charged 包和 queue 模式。旧实验的 1/2/3 不属于当前配置接口。
- group32、每批 1,983,920 primary、电子队列 16,384 lanes、primary chunk 262,144；charged 默认最大步长仍为 **0.05 mm**，没有放大到超过 0.25 mm。
- 原始独立包逐个加载、校验、转换并释放原视图；转换后的四个视图在同一进程跨批次复用。没有加载或回退到旧电子轨迹库。
- CDF 使用 uint64 阈值，随机数保留整数位作分类选择；避免直接用 float 累积概率吞掉稀有正能量分支。连续飞行/角度的随机数为严格开区间 float。
- 直接保留弹性散射的 `1-cos(theta)`，避免小角度在 FP32 中消失；方向在 FP32 中归一化。
- CT 定向体素归属比较实际表示的物理面，修正索引而不移动粒子。三角函数先用分裂的 pi/2 常数在 FP32 中做象限约化，再调用小角度 native 运算，避免通用数学库的 FP64 大角度分支。
- 电子网格能量审计使用线程私有 FP32 分项累加，结束后在 CPU 汇总；电子 worker 内没有 FP64 全局原子加。primary/其他粒子原有能量审计不在本次改动范围内。
- `scripts/generate_electron_fp32.py` 从现有参考方程生成运行头文件，precision 特有修正在生成器中显式列出。SYCL 构建运行 `--check` 防止参考方程改变后仍使用过期生成文件；需要时先运行生成器。

## 数值与完整性检查

边界探针覆盖 RT07575 的 504 个内部 x 面、面上及左右相邻 FP32 坐标、两个方向，共 3,024 例。简单改类型产生 588 个错误归属，修复后为零；没有通过 epsilon 位移修复。

三角函数检查覆盖 [0,2pi] 的 1,048,577 个 GPU 输入，象限约化版本最大绝对误差 3.485e-7，sin²+cos² 最大误差 5.001e-7。直接 native 版本未达到预设误差门槛，已保留在 failed.md。

概率阈值的指数/尾数整数实现与 `floor(ldexp(float_probability,64))` 在 8,323,078 个覆盖全部相关指数的输入上完全一致。CDF 分类没有改为低分辨率 float CDF。

最终 GPU 密度检查为 57 组中间密度、116,736 包、11,430,056 步，invalid/coverage_missing/step cap 均为零。组合出生/joint/charged/photon 检查含 335,872 次源抽样：187,671 次原生零产额没有生成电子，其余 148,201 个实际包全部沉积或逃逸，零非法状态。整数区间重映射另经 100,000 例 uint128 参考核对，无差异。

`full_v0_*` 是已发现边界归属问题的中间版本，不能用它约 7.5 秒的电子耗时代替最终基线。

## 验证边界

这是研究运行，不能称为新的 TOPAS 剂量验收。全 FP32 会改变随机轨迹、拒绝采样和步数，因此不要求电子工作计数逐位等于原 FP64；仍要求主粒子/统一 EM 审计、能量、所有任务完成、零非法状态与零溢出，并报告峰谷和局部剂量变化。

Urban 极小能损阈值是已知敏感点：此前单独窄化阈值前的 mean 会改变离散分支，孤立参数格的均值曾相差约 40%。按照最新指示，全 FP32 基线不暗中保留原 FP64 gate；后续恢复该项时必须独立测量速度和局部剂量，不能用全局平均掩盖局部分支差异。

历史冻结对照：原接受电子实现 SHA256 `1581fd7dd3434380b7bb785678926f1ef1b8cc29e5f2e353ca68d5f08bd33872`；旧混合精度 SHA256 `a577d75308318ed3e395d7cf066d76dc22a441d0c4ce33411180f6fa355ef0f3`。旧混合精度屏幕结果保留在 `out/electron_fp32_20261003/screen/report.json`，与当前全 FP32 测量分开。

## RT07575 最终计时与剂量

RTX2080Ti，交替顺序 reference A → full A → full B → reference B，再运行新二进制的 mode0 对照。每个进程有两个各 1,983,920-primary 批次，第二批已复用包；计时包含该批的正常输出。两次进程复测复用相同输入/种子，因此剂量统计仍是每臂 **3,967,840 个不同 primary**，不能把重复计时当作更高独立粒子统计。

| 指标 | 原电子精度 | 全 FP32 |
|---|---:|---:|
| 第二批耗时，两次复测 | 26.5045 / 27.1858 s | 22.6439 / 23.0360 s |
| 第二批平均耗时 | 26.8451 s | 22.8399 s |
| 连续批次吞吐 | 73,902 primary/s | **86,862 primary/s** |
| 电子队列平均耗时，每批 | 16.6475 s | **12.7896 s** |
| 首批初始化耗时 | 5.84 / 5.56 s | 6.42 / 6.35 s |

连续批次吞吐提升 **17.54%**，电子耗时减少 **23.17%**；两个进程的完整墙钟吞吐提升约 **14.07%**。这不是180k/s目标已达成，也不是所有能量/CT场景的统一吞吐。

中间版本把 host-mapped 光子记录放到 GPU 转换，Nsight 显示转换核独占约22.9s；已改在CPU转换同一批经原始校验的主机记录，初始化从约29s降至约6.4s。与GPU转换控制相比，最终所有电子/joint/photon工作计数完全相同。阈值位运算在孤立核中更快，但整例计时未分辨出额外收益，不单独宣称加速。

所有10个正式计时批次通过研究质量、主粒子/统一EM审计、能量及零溢出检查。全FP32的每批完成数严格等于生成数（4,917,548 / 4,905,394），零invalid和coverage_missing。RT07575最大相对能量残差4.942e-8。新二进制mode0与冻结旧程序BODY相对L1差5.937e-8，属于FP32计分累加差异。

**剂量以下均对比原精度独立电子版，不是TOPAS：**

- BODY相对L1：**0.03640%**。
- 20–160mm分段汇总峰/谷平均剂量变化，最大绝对值 **0.005064%**。
- 在参考剂量≥1% BODY最大剂量的体素中，局部绝对相对差的95% / 99% / 99.9%分位为 **0.199% / 0.747% / 2.609%**；最大 **29.54%**。

区域均值满足当前筛选门槛；不能据此宣称所有体素≤1%，也尚未证明这些局部差异均为蒙卡噪声。质量报告显式保留 `unvalidated_electron_fp32_transport`。

默认 `config/proton_ct_minibeam.yaml` 已设为4。该预设（肺CT、170MeV）另跑连续2×500k primary，缓存复用、全部完成、能量及零溢出检查通过，最大相对能量残差3.646e-7。

## 编译指令与复现

早前小网格 Nsight probe 启动的是 electron RoundedRangeKernel；其 SASS 中 **FP64算术、FP64比较和FP64原子指令均为零**，仍有队列ABI/错误快照所需的float↔double转换。该 wrapper 为255寄存器、1600B静态stack。后续完整批次 NCU 确认 16,384 lanes 使用直接 range 内核（242寄存器、1280B），两者不能混用；详见下文校正。原文件加载/校验与一次性转换核不属于此指令断言。

最终二进制SHA256：`c7465f3f11586a38d3ce987a7ad68efdc28613f62bf19007271869f3a0ace3b4`。

- 原始逐批结果：`out/electron_fp32_20261003/full_screen/report.json`
- 汇总：`out/electron_fp32_20261003/full_summary.json`
- 最终源码/指纹：`out/electron_fp32_20261003/full_final_source/fingerprint.json`
- 实际内核SASS审计：`out/electron_fp32_20261003/full_images/final_actual_worker.json`
- 默认预设检查：`out/electron_fp32_20261003/default_preset_report.json`
- 边界/CDF、GPU三角函数、密度及组合包检查：`scripts/check_electron_fp32_math.cpp`、`scripts/check_electron_fp32_trig_gpu.cpp`、`scripts/check_electron_full_fp32_gpu.cpp`、`scripts/check_electron_full_fp32_packets.cpp`。

后续逐项恢复FP64应固定这份全FP32输入、包与步长，以它为每一项的共同对照；不重新采用“FP64为基线、逐项改FP32”的顺序。优先分开考察小能损阈值/涨落、位置边界、响应插值及旋转，剂量计分始终FP32。本轮没有将这些后续开关或测量冒充为已完成。

## 完整批次硬件剖析校正

`out/electron_diagnostics_20261003` 的 NCU 直接采集了冻结最终二进制的
16,384-lane 完整批次。此时运行的是直接 range 内核（242 registers、1280 B
static stack），不是早前小网格 probe 的 rounded wrapper（255/1600 B）。
两个 SASS 均无 FP64 算术；完整批次保留 7/26 处 float↔double 转换。
此前 wrapper 的资源数只适用于该 wrapper，不能外推完整批次。

匹配第 2、3 次电子 launch 的平均 NCU 指标（计数器回放时间不作吞吐验收）：

| 指标 | 原参考 | 全 FP32 |
|---|---:|---:|
| FP64 pipe 活跃度 |76.422%|3.796%|
| FP64 pipe 指令数 |11.9822B|0.4562B|
| long scoreboard 等待 |70.203%|39.111%|
| short scoreboard 等待 |0.211%|0.466%|
| math pipe throttle |0.043%|0.126%|
| local load sectors |5.1999B|4.8424B|
| local store sectors |4.8135B|4.4220B|
| 活跃 warp 占用率 |24.720%|24.743%|
| 每 warp 指令活跃线程数 |13.375|9.025|

FP64 pipe 的剩余计数不是 DFMA/DADD 等纯算术计数，不能将其直接解释为
仍有 FP64 输运算术；SASS 保留类型转换，其对动态 pipe 计数的贡献尚未单独归因。
局部内存流量只下降约 7–8%，占用率未提升，说明状态/访存及分歧仍须优化。
不同精度轨迹并非逐位相同，这两次 launch 的计数差异不完全等同于每步指令差异。

针对剩余 FP64 pipe 计数又补采了动态 SASS 算术计数：同一完整批次电子
launch 的 DADD、DFMA、DMUL predicated-on 计数均为 **0**。证据
`out/electron_diagnostics_20261003/dynamic_fp64_arithmetic.json`。这排除了
将 3.8% pipe 活跃度直接当成残留 FP64 加乘算术的解释，未声称已逐项归因
所有共享管线指令。

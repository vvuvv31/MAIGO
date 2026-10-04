# 独立电子运行路径：proton CT minibeam 已接入

主程序与 `config/proton_ct_minibeam.yaml` 已切换为独立 birth/joint/charged/photon 路径。RT07575 实际运行已由文件打开跟踪确认**旧电子库零读取**；不是仅完成 standalone，也不是旧 continuation 的重命名。该路径仍为 research/non_production，尚未完成 TOPAS gamma 或全部局部剂量的正式验收。

所有 GPU 剂量计分/原子累加均为 FP32（ON），FP64（OFF）；double 仅用于物理量、查表和几何。旧库移除验证二进制 SHA `5ac3bf7f06b8371a30e7522df79aee286fc109609d160a466addbb7bd4cff2d7`。

## 运行包

基目录：`out/electron_standalone_20261003/`。

- **最终包**：`charged_complete512_v2/index.txt`，SHA `ad86b286de05aff663667793c241826e3f46472c992d4687be02a50fc9fe2e14`。
- 82 个实际密度节点，**561,134,640 B（约 535 MiB）**。这仅是 charged 组件，不包括出生、joint、光子包。
- 汇总 8,664,576 次原生 eIoni/eBrem 碰撞，保留 **17,434,630 个正能量 parent/local/electron/photon 分支**，不做样本裁减。
- 附带 native stopping/range、生产阈值、精确导出的最低电子能量、Mott 弹性小表、Urban 常数，以及硬碰撞率的分段三次表达。
- 编译工具 `scripts/compile_electron_response.py`；设备加载器 `include/carbon/electron_charged_bank_device.hpp`。加载器校验索引/逐节点 SHA、文件长度、节点身份、曲线范围、分支/CDF、非法物理值。

粗 257 能量点的线性硬碰撞率会在阈值处制造假阳性。因此新包直接重建 native spline 各区间的三次函数，再按原生规则取 `max(raw_lambda,0)`。82 节点原生函数重建最大绝对误差约 **2.1122e−14 /mm**。对每个能量单元内三次函数的端点及导数根检查正支持，补提实际有 native 支持但缺样本的单元后，没有正碰撞率通道缺口。该结论是支持覆盖，不代表有限能量终态插值精度。

初版 `charged_complete512` 保存了从四舍五入日志推断的 `0.0001 MeV` 截止值。原生实际值为 **`9.9999999999999991e-05 MeV`**，精确 100 eV 输入的零损失抽样暴露了差异。最终 v2 包保存原生导出的值，初版仅保留为诊断证据，不再作为接入候选。

## 独立步进的物理处理

`include/carbon/electron_charged_step.hpp` 在抽能损前先决定真实自由飞行、下一体素面、射程和步长限制。没有裁切已经抽好的完整角度/损失，没有拒绝重抽跨界样本。默认最大步长 **0.05 mm**，API 拒绝超过 0.25 mm 的设置。

- 弹性散射使用已经验证的 native GS single-scattering/Mott 公式。弹性碰撞率在当前短步内取初始能量值；默认再限制平均损失不超过初始 KE 的 0.5%，这是显式的新步进近似。
- 硬碰撞使用全能量域上界抽 proposal，再用步末实际能量的 native rate 做 null-collision 接受检验；能量下降后不会强行调用一个不支持的过程。
- 连续能损使用 native restricted stopping/range 的分段线性插值、Urban 涨落和完整的 AlongStep 最终能量/停止结算。原始涨落大于入射 KE 时执行原生结算，不重抽。
- W 与物理 KE 分离：按能量概率选择连续沉积、继续电子、离散 parent/child/local 分支，W 不变。光子交给已验证的独立光子链，产生电子后返回本 charged 链。没有用局部沉积代替缺失分支。
- 终态仍按能量单元中的原生联合能量份额/角度样本抽取，方位角均匀；有限能量插值属于待验证近似。
- 实际 CT 密度在同一 Schneider 组成的相邻密度节点间插值质量 stopping/range/截面，再恢复实际密度；硬碰撞终态按过程率加权混合，cut 线性插值。不外推密度、不裁剪物理 KE。Urban 参数在同组成节点间相同；Mott 角分布使用下节点，其同组成密度变化小于 9.44e−7 相对值。
- 精确体素面由 `origin + integer * spacing` 身份判定后决定方向归属，修复小数网格除法舍入带来的零步；没有 epsilon 位移。

## 验证结果及边界

### 全节点组件与计分

`charged_gpu_final.json`：167,936 个包、17,088,705 步、5,543,675 次弹性散射、833 次硬碰撞、71,600 次体素面到达。无 invalid、coverage missing 或步数耗尽。最大步长 **0.0409094 mm**；连续部分能量结算最大残差 **1.11e−16 MeV**。FP32 单位能量权重计分闭合。19 个光子是明确的能量交接，不是沉积或被丢弃。

`charged_reader_checks/report.json`：82 节点各 10,001 个能量点的参数/正率过程抽样检查；错误 SHA、截断、尾随字节、非有限三次区间、损坏 CDF 均被拒绝。

`tests/electron_charged_step.cpp`：小数原点/间距的双向面归属、外表面逃逸、切向、棱角同距跨面、物理最低能量终止、缺表/缺分支明确失败，以及最大步长限制。

### 原生完整 AlongStep

`alongstep82_final/report.json`：82 节点 × 11 能量 × 6 长度条件，共 **5,412 条件**，每条件 native/新实现各 50,000 次，总计**每方 270,600,000 次**。调用实际 native `AlongStepDoIt`，检查生产阈值、模型、停止规则、PIXE 无沿步子粒子及逐样本能量闭合。

全部通过逐条件均值检查，门槛为 `7 × 差异标准误 + 0.0002 × 入射KE`。额外 0.02% 入射能量容差明确用于 stopping/range 插值，不是剂量校准，也不是 CT 验收门槛；保留每个条件的实际差异、标准误和停止计数。最大逐样本能量残差 **1.11e−16 MeV**。该测试不代替分布形状/局部剂量验收；原始 Urban 分布的独立检查见前一报告。

### 12 个完整电子—光子界面场景

原生参考 `interface_12` 覆盖肺↔软组织、软组织↔骨，以及法向/80°/90°入射。软组织→骨两个掠入射场景提高至 **20,000 native histories**，存于 `interface_native20k`；其余仍 2,000 histories。

最终候选 `interface_gpu_final_*` 每场景 65,536 个能量包，共 **786,432 个包、940,508,747 步**；1,400 次独立光子链调用、159,710 次后续跨界，无 invalid，FP32 两侧沉积+真实逃逸权重闭合。这里两侧 deposited count 对应能量加权包，不是 analog 电子个数。

`interface_comparison_final.json` 按原生完整 history 计算标准误：

- 软组织→骨 80°：上游能量占比差 **−0.321 个百分点**，差异标准误约 **0.335 个百分点**。
- 软组织→骨 90°：差 **+0.070 个百分点**，标准误约 **0.317 个百分点**。
- 12 场景当前最大标准化差约 **1.39**；最大绝对差在骨→软组织 90°，约 **−1.18 个百分点**，其原生参考仅 2,000 histories。

将平均损失步限从 0.5% 收紧至 0.1% 的对照，每场景同为 65,536 包，共 1,359,353,814 步，无 invalid。`interface_step_refinement.json` 使用同包随机种子的配对差异标准误，最大标准化差约 2.26，未见显著的两侧能量占比漂移。收紧步限的对照使用 v2 截止值修正之前的包；最终 v2 在默认步限下重跑，所有端点/计数与修正前完全一致。本次抽样没有恰好落在那个临界点，故阈值正确性以最终原生 AlongStep 专项检查为准。

**上述结果尚不能证明 1% 局部剂量或峰/谷精度，也未计算 RT07575 gamma。没有测得可用于宣传的主程序吞吐提升。**

## 实际 CT 接入与独立性

新增配置 `material_electron_charged_index_file` 和 SHA pin，要求 fresh birth/joint/photon、research proton CT minibeam 队列。主程序在初始化及预读阶段均跳过 legacy bank、packed/prepared cache 和 continuation 映射。joint 离开均匀支持时交接真实位置/方向/KE/权重，独立 charged 步进可再进入 joint；光子产生电子后也进入独立链。正能量分支缺支持、非法状态或事务上限耗尽均明确失败，不做局部沉积或假逃逸。

当前默认包位于 `data/ct_electron/proton_independent_v1/`，从已校验提取物复制并逐文件比对 SHA，不依赖 `out/` 软链接。完整 [运行包清单](proton_independent_runtime_manifest.json) 记录 168 文件、源路径、SHA 和字节数：总计 **4,137,523,122 B（3.8534 GiB）**，包括 birth 0.306 GiB、joint 1.326 GiB、charged 0.523 GiB、photon 1.699 GiB 及索引。这不是纯解析模型；仍保留新提取的联合响应和条件终态样本。

旧库保留作历史对照，其他未切换模式的兼容实现仍保留。当前独立接入范围是 **proton CT minibeam**，不意味着 carbon/water 所有预设也已移除旧包。运行示例：

```bash
./scripts/run.sh proton_ct_minibeam --histories-per-batch 500000 --batch-count 2 \
  --batch-output out/proton_independent_example
```

加载缺失、SHA 错误或支持范围不足会报错，不会自动回退。新 birth/joint/charged/photon 缓存跨批复用。

## 连续密度验证

`density_midpoints57/` 对全部 57 个相邻密度对的真实中间密度执行原生参数探测。E>100eV 的最大相对误差：elastic rate **0.266%**、restricted stopping **1.900%**、range **1.746%**；最大 stopping 差在低密度节点47/48、rho=0.05099555、E=0.09675956MeV。这是模型插值误差，不等价于患者剂量变化，也没有据此宣称 1% 剂量准确。

`density_gpu.json`：57 对、116,736 包、11,431,295 步，invalid/gap/cap 均为0，FP32 能量计分闭合。独立 joint→charged 状态交接测试不提供旧 source 数组，仍通过；几何/终止规则测试通过。

## RT07575 主程序结果

`rt07575_independent_smoke/` 完成 **1,983,920 primary**，group32。`strace -f -e trace=openat,open` 确认旧 index、packed 和 prepared 文件零读取；legacy material views=0，日志明确 `legacy_loaded=0 legacy_fallback=0`。全过程只读新四组件和非电子物理包。

`rt07575_independent_20m/` 一进程完成同二进制旧 charged 对照1批和独立路径10批；独立臂共 **19,839,200 primary**。对照与历史 matched shard 的全网格相对 L1 为 **8.01e−8**，primary/EM 审计一致。独立10批全部 research accepted，无 failure、队列溢出、charged/photon coverage gap 或 invalid；物理能量相对残差最大 **6.89e−8**。全部批次无 source/boundary restart。

比较对象是 **fresh birth + joint + photon + 旧 charged continuation**，不是 TOPAS。入射暴露相同，没有拟合归一化。BODY 相对 L1 **0.14786%**，峰谷为限定中心场区域内按深度汇总：

| 深度 | 峰变化 | 谷变化 |
|---|---:|---:|
| 20–40 mm | −0.04502% | −0.00799% |
| 40–80 mm | −0.01582% | +0.00549% |
| 80–120 mm | +0.00017% | +0.00316% |
| 120–160 mm | +0.00027% | −0.00083% |

各区配对批次标准误保存在 `peak_valley_sampling_SE_percent`；最大0.03447个百分点。部分小偏差已可统计分辨，不能把所有差异都解释成噪声。该标准误不包含固定训练包误差。

| BODY 内参考剂量门槛 | 体素数 | 局部差≤1% | 局部差≤3% | 最大局部差 |
|---|---:|---:|---:|---:|
| BODY 最大值的1% | 435,483 | 96.6998% | 99.9199% | 21.7179% |
| BODY 最大值的10% | 206,068 | 99.7938% | 99.9898% | 5.0566% |

这是零距离逐体素相对剂量差，不是含距离搜索的 gamma。更高历史数显著降低差异，但不能证明剩余异常全由噪声造成。**未通过“所有局部体素均在1%内”的声明，也未开展本独立路径的正式 TOPAS gamma 验收。**

输出包括 `isocenter_plane.png`（穿过 isocenter 的束轴纵向平面）、`lateral_profiles.png/json`（40/80/100/117.0136/130/150mm），重算命令：

```bash
python3 scripts/compare_independent_charged.py \
  --out out/electron_standalone_20261003/rt07575_independent_20m --shards 10
```

## 运行成本与下一步

RT07575 十批独立臂耗时合计 **307.37s**（逐批wall之和），约 **64,544 primary/s**；平均每批30.74s，其中电子20.34s、输运循环29.40s。包含额外旧 charged 对照的整个进程332.19s；不能把332.19s全部归给候选。初次单独带strace运行 setup5.16s、电子19.26s、进程34.05s，不能作为严格冷盘或无跟踪性能测量。

单散射 charged 代价较高，当前实现**没有获得吞吐提升**。原旧charged路径电子阶段历史测量约2.7s/同量级批次，但对照冷热驻留预算不同，不以此报告受控加速比。当前收益是消除大体积旧轨迹库依赖和初始化，下一步应研究短步输运成本、查表与调度，同时保留横向散射和峰谷剂量。

独立运行目标与正式剂量验收分开记录。后续优先检查局部异常、对 TOPAS 验证，以及连续密度/终态插值误差；不得用切回旧库或局部沉积掩盖新路径问题。

此前 standalone 源码与原生二进制指纹见 `charged_final_source/fingerprint.json`；主程序指纹见 `rt07575_independent_smoke/build.json`。失败/修正记录于 `failed.md`。

默认肺 CT/170MeV 另以原生连续批次完成 **2×500,000 primary**，两批均 research accepted、零溢出/非法状态/覆盖缺口；文件跟踪再次确认旧电子库零读取，第二批 charged/photon/joint cache hit。证据 `default_preset_check/report.json` 与 `files.trace`。此项验证默认入口及缓存，不是该场景的剂量对照。

后续保留物理模型的[电子内核优化](electron_throughput_20261003.md)已接入当前构建：RT07575直接ABBA驻留吞吐64.95k→72.07k primary/s，电子吞吐+17.98%，整数审计一致。本文前面的20M剂量验收边界与初始运行时间仍对应旧库移除时的二进制；最新构建和源指纹另列于性能报告。

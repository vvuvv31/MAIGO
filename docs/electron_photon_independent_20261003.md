# 独立光子处理进展

本轮继续“完全摆脱旧库”的目标。完成独立原生光子数据提取、跨 CT 光学厚度算法和条件相互作用抽样原型。**目标仍未完成：主程序已加入可选新光子模块，已通过 RT07575 光子替换的配对检查；电子界面回退仍在加载旧库。** 依赖状态明确记录在 `out/electron_standalone_20261003/INDEPENDENCE_STATUS.json`。

## 新模块

- `photon_optical_transport.hpp`：一个 `−ln(U)` 光学厚度穿过实际 CT 体素，逐段扣除 `μ×距离`，直到相互作用或真实几何出口；材料变化不重抽 RNG、不在界面沉积，也不依赖有限长度源轨迹。
- `photon_response.hpp`：native 光电/康普顿/瑞利截面决定过程；条件 CDF 按能量份额选择局部沉积、继续光子或子电子，保留能量—方向关联；统计权重 W 不再乘分支概率。空通道明确返回失败，不能当成沉积或逃逸。
- `photon_response_data.hpp`：独立 schema/SHA/长度/CDF/能量及方向范围校验和只读载入，数据结构不引用旧 electron response/轨迹图。
- 新原生 scorer `PhotonResponseNtuple` 仅记录 primary photon 相互作用及全部子粒子出生；后代正常运行，不杀轨迹、不消费 RNG。可选截面查询有独立的逐字节不变性验证。

没有使用整步轨迹裁切、局部电子替代、丢弃光子分支或患者逃逸假定。光子飞行不受带电粒子 0.25 mm 步长上限约束；提取时所有带电后代的 native MaxStep 为 0.05 mm。

## 提取证据

`out/electron_standalone_20261003/photon_bank100k/report.json`：

- 82 个实际 Schneider 材料/密度节点，各 100,000 原生光子历史，共 **8,200,000 histories**。
- 257 个对数能量源节点，10 eV–0.65 MeV；原始 opt4，无 GS override，production cut 0.05 mm。
- **24,000,500** 次 primary photon 相互作用、**22,912,518** 个子粒子出生，逐步/逐历史能量账闭合。有限训练盒 source escape=0。
- 原始记录总计 6,613,562,160 B，属于离线训练证据，不是拟交付的 runtime payload。
- 每节点约 4,200 条衰减率：对数网格加元素壳层阈值附近节点。三种保留过程均有独立 native 截面，pair conversion 在本能量范围为零。
- 82 节点/25 组成的 `rate/rho` 比较最大相对差 **1.112e−15**。只证明本范围 photon attenuation 的密度线性，不外推到碰撞终态或电子散射。

独立 0.1 MeV 肺参考 5,000 histories：总衰减率 0.00507490226 /mm，首飞行平均光学厚度 **0.989742 ± 0.013824**（1 SE）；理论过程概率 phot/compt/Rayl=0.017600/0.951065/0.031335，观测=0.0204/0.9492/0.0304。相互作用状态和所有 child 出生逐条匹配完整 312-byte 原生记录。开启/关闭截面提取后的完整记录和精简记录各自 SHA 完全相同，见 `photon_scorer_invariance.json`。

原生二进制 SHA：`aaaae1de81550224dbb36010156b9b60aa6db9638b8b2e4e1c342d7efee5c613`。构建源/命令/原生输入在 `photon_native/build_manifest.json`。

## CPU/GPU 代码验证

- 宿主 1,000,000 条分层解析射线：包含透明层、反向入射、恰在材料面、棱角同时穿越、无覆盖和非法状态；通过。
- GPU 1,000,000 条肺/软组织分层射线，使用上述真实材料衰减率：位置相对解析值最大误差 **3.98e−13 mm**；FP32 原子计分合计严格等于 1,000,000。见 `photon_optical_gpu.json`。这是光子飞行算法校验，不是完整 photon/electron dose 验收或主程序吞吐。
- 三个代表节点导出未额外压缩的条件相互作用原型，运行包分别约 **27.40/28.96/17.74 MiB**（节点57/12/32）。每个正能量后继都保留，原始轨迹不进入这些包。
- 肺 0.1 MeV 的 1,000,000 次 sampler 抽样，其分支概率、能量矩和角矩均在算子直接积分的 6 SE 内；方向单位长度检查通过。
- 相同能量与独立 native 5,000-history 首碰撞参考比较：电子分支能量权重差约 −0.001827，参考 SE 0.002137；光子分支差约 +0.001817，参考 SE 0.002149。完整矩比较在 `photon_operators_v2/node57/sampling_report.json`。这些有限统计比较不等于 CT 局部 1% 验收。

所有 GPU 计分保持 FP32；表格/几何 double 运算不是 FP64 剂量评分器。

## 覆盖缺口及下一步

普通 analog 提取虽有 10 万历史/节点，但低概率过程仍可能一个样本也没有。在已检查的代表材料中，最差能量箱未覆盖的过程概率约 **0.606%–0.666%**。例如低能区某些瑞利/康普顿通道、较高能量处少见的光电/瑞利通道。没有把缺口降为零，也没有用邻近过程代替。

第一次 compiler 把训练源能量放在 bin 边缘，FP32 源舍入产生人工空箱；改为以源节点为中心、相邻节点几何均值为边界，不改变物理样本。保留了失败原型和散列，见 failed.md。上述约 0.67% 是修正后的真实有限统计缺口。

因此暂不生成“全覆盖已验收”标签，也未启用主程序替换。下一步是调用当前 Geant4 的公开 process 接口，按材料/指定能量/指定过程提取条件终态，补齐稀有通道。已检查 v11.3.2 的 `G4VEmProcess::PostStepGetPhysicalInteractionLength`、`PostStepDoIt` 和生产 cut 处理；必须保留 native null-collision 接受/拒绝及次级过滤，不能只调用 model 后遗漏 process-level 行为。官方源码备份在 `geant4_source/`，来源：

- https://github.com/Geant4/geant4/blob/v11.3.2/source/processes/electromagnetic/utils/src/G4VEmProcess.cc
- https://github.com/Geant4/geant4/blob/v11.3.2/source/processes/electromagnetic/utils/src/G4EmCalculator.cc

随后把独立光子与 charged joint/界面模块连接，移除 legacy 初始化、加载与映射，最后运行无旧库 RT07575 的能量/队列/峰谷/局部剂量验收。现有混合结果不作为最终目标已完成的证据。

## 复现入口

```bash
python3 scripts/build_photon_response_native.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/extract_photon_response.py \
  --out <新输出目录> --nodes $(seq 0 81) --histories 100000 --workers 6
OPENBLAS_NUM_THREADS=1 python3 scripts/compile_photon_response.py \
  --source <提取目录> --out <新算子目录> --nodes 57 12 32
```

`check_photon_optical_gpu.cpp` 是独立 SYCL/FP32 验证程序；`check_photon_response.cpp` 及 `check_photon_response_samples.py` 检查条件 sampler。产物均保留于 `out/electron_standalone_20261003/`，正式该初始阶段主程序二进制是此前 birth-only 集成版本；后续接入结果见末节。


## 后续进展：独立条件包与 GPU 全链测试

已按原生 process 接口完成82节点的条件终态提取，每个有截面的过程/能量箱512次接受采样（257个能量箱）；阈值落在箱内部时使用该箱内有正截面的能量点。通过 GammaGeneralProc 的公开过程查找，不替换 native 物理模型。离线 scorer 会直接调用过程并释放已记录的子轨迹，因此它是独立条件提取工具，不称为只读 analog scorer。

- 条件包：`out/electron_standalone_20261003/photon_complete512/index.txt`，SHA `a91b1e7b6301e3d07b4fd70f1bb2cc9c7dfc4cdea6c7cad789ab831b6427ba41`。
- 原生条件提取共30,144,512次碰撞、19,424,287个子粒子，逐次能量闭合残差最大1.11e−16MeV。756,021个严格零KE子粒子全部保留原始记录（其中108个方向未定义），编译时仅跳过它们的零概率分支。运行包总计1,824,087,920 B（1.699 GiB），其中1,811,887,776 B是条件分支数组。未额外稀疏化。全部82节点的衰减率结点/箱边界/间隔中点检查中，正截面通道缺覆盖为0。每节点53个空通道均为无截面区域。
- 光电、康普顿、瑞利在肺/软/骨100keV的条件分布与独立 analog 参考进行9组比较，201项能量/角度/联合分箱特征及 exact Fisher occupancy 检查通过；最大标准化差异2.804。不能外推为全能量/全密度剂量验收。
- `check_photon_bank_gpu.cpp` 实际加载全部82节点，在三层不同组成/密度几何内抽262,400条10eV–0.65MeV光子链：碰撞283,986次，跨体素181,776次，沉积68,590包，真实逃逸48,633包，交接电子145,177包，无非法/缺覆盖。FP32权重计分闭合。载入观察值2.709s，非严格冷启动测试。
- 条件方向保留能量—极角关联，方位角均匀；当前能量箱及相邻密度条件分布的线性混合均作为显式近似。实际飞行使用实际密度。电子分支交回完整物理位置、方向和KE，统计权重W不变。
- 运行开关 `material_photon_response_index_file` / `material_photon_response_index_sha256`，限定已有独立 birth + joint 的 research proton CT minibeam queue。主程序里新 joint 所有光子出生以及旧 charged fallback 产生的光子都交给新模块。旧 charged fallback 仍需旧库，不宣称完全移除。
- 过渡期条件分支使用 mapped host 存储，避免与仍驻留的旧库竞争2080Ti显存；率表、索引在设备。每进程缓存，不逐批重复加载。后续只有实际无旧库运行才可证明部署独立。

`photon_conditional_native/build_manifest.json` 原生二进制SHA：`1b00687b365b1b0878cdbe2ab5a0d855825814abe6ec7e480740e992ef2f74a5`。本节替代前一节的稀有通道缺口状态，前面的失败记录保留作溯源。


### 首轮主程序结果（外表面几何修复前冻结版本）

`rt07575_photon_screen/report.json`：同进程2对配对批次，每组3,967,840 primary，FP32/group32；主输运/EM审计一致、能量及零溢出检查通过，关闭开关相对前次birth实现剂量L1仅8.09e−8（累加顺序差）。新光子4,384次调用，5,007次相互作用，263,025次跨体素；无缺覆盖或非法状态。BODY相对L1差0.001368%，分深度峰谷汇总最大变化0.002773%。在BODY最大剂量1%阈值上，局部差<=1%占99.9594%，<=3%占99.9933%，最大6.363%；10%阈值以上全部<=1%，最大0.844%。这些是固定电子模型、单独替换光子的配对差，不是相对TOPAS gamma或全路径验收。

单对/两对计时未显示可信吞吐收益：电子总计5.31769s控制、5.32377s新光子。新增模块目的为解除光子轨迹库依赖，不把这一变化称作加速。仍有完整旧电子库被加载用于charged边界。

补充测试发现恰在CT外表面朝外的射线被判invalid，已修正为真实零距离escape，加入向内/向外/切向/非法方向测试。首轮冻结二进制和日志保留，新版本正在进行每组19,839,200 primary确认，结果不混并。


### 修复后的主程序确认结果

`out/electron_standalone_20261003/rt07575_photon_confirm/report.json`，GPU二进制SHA `6b84d126a27ccce114c0493e6bf7f5632f7898a0666143c5ebe10c4a74a5d272`。

- 同进程10对AB/BA批次，每组19,839,200 primary，FP32/group32；20批全部质量/能量/主输运及EM审计/零溢出检查通过。研究开关保留未完成独立剂量验收的近似标记。
- 新光子22,318次调用、25,179次碰撞、1,376,459次跨体素：14,580次子电子交接、1,919次沉积、5,819次真实逃逸，非法和缺覆盖均为0。
- BODY相对L1差 **0.001391%**；20–160mm分段峰/谷汇总变化最大 **0.003177%**（40–80mm谷区），配对runtime采样SE为0.000238个百分点，不包含训练误差。
- BODY最大剂量1%阈值以上435,482体素：局部差<=1%占 **99.997704%**，10个体素超过1%，最大 **1.8372%**；全部<=3%。10%阈值以上206,064体素全部<=1%，最大0.3776%。不宣称全体素1%验收。
- 10批电子核合计：控制27.1159s，新光子27.1488s；输运116.3097/116.2263s。没有分辨出加速收益。两个arm均固定fresh birth和同一joint模型，区别仅为光子处理；不是相对TOPAS gamma。
- 包单独加载的GPU全链确认也通过；观察载入2.005s（页缓存未受控，不作冷启动保证）。20批成对运行进程总墙时265.21s包含双方输运、加载及输出，不能作为单组计划计算时间。

光子替换仍为可选research路径，没有修改正式preset。**旧charged接口库仍加载；完全独立的电子运行目标未完成。**

### charged边界下一阶段的已完成准备

新增 `ElectronBoundaryProbe`、构建及参数提取脚本。`electron_boundary_parameters82_v3/report.json` 覆盖82节点×257能量点（10eV–0.65MeV），参数CSV共4,449,733B。提取实际material/cut的GS Mott修正、scattering-power correction、弹性/第一输运平均自由程、限制停止本领/射程及原生eIoni/eBrem模型截面。由公开参数重构的第一输运平均自由程与原生公开截面乘实际cut修正后的结果最大相对差4.11e−15。不是使用默认水值，也不是把第一输运平均自由程误当弹性碰撞平均自由程。

0.4MeV弹性平均自由程：肺0.003029mm、软组织0.000891mm、骨0.000592mm。原生配置为GS/Mott，global step type=fUseSafetyPlus，skin=3，range_factor=0.08。GS源码的这种边界算法在skin内采用单次弹性散射、在外部限制凝聚步；物理参数支持研究小表边界输运，但尚无GPU算子或相应剂量验收。

读取native GetLambda的原始插值在节点55/59/61共7点出现负值。全部保留为CSV末两列诊断，交付的正截面来自G4EmCalculator对实际production cut的模型求值，没有把负值偷偷截为零。最初的未初始化GS内部currentCouple与插值率提取失败见failed.md，未用失败输出作物理结论。

还需要原生弹性角分布、限制能损涨落、硬碰撞全部子分支，以及继续态/回穿/棱角与真实界面剂量验证。当前4.45MB参数表不是整个电子包大小，不能据此宣称已取代旧库。


后续charged边界的弹性公式、Urban涨落CPU/GPU验证及82节点硬碰撞提取见[独立电子边界组件](electron_boundary_independent_20261003.md)。主程序charged旧库依赖尚未移除。

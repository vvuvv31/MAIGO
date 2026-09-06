# plan2 结论记录（RT06423 section-0 电子纵向响应）

## 2026-09-06 优先修正：数值安全实现，物理验收 BLOCKED

521域外查询已逐条定位（无截断，45histories）：初始168.27–170.52MeV/u，
全部减速后降至148.20–149.997MeV/u，位置200–219.5mm。记录前后raw
逐字节一致。根因是低能覆盖不足；入口2–120mm通过不等于全路径覆盖。
下一步补低于150MeV/u的独立响应参考与留出点，再考虑新版本诊断表，
不clamp、不缩短phantom、不豁免零域外门。证据：
`evidence/step-31/longitudinal-domain-trace/`。

均匀三密度GPU诊断实现已测，但整组门禁FAILED：新显式flag只允许175MeV/u
单束、核off、smoke/scale=1，全网格同一已知密度且section0。HU=-975/-951
实际入口误差+0.324%/+0.134%，与离线预测分箱差<0.022%；默认off逐字节
复现旧结果，混合密度/材料真实launch均拒绝。HU=-951仍有521次域外查询
（298.674904MeV留源），零域外门不豁免。下一步只读定位这些查询能量/深度，
再决定补能量覆盖，不开放患者/界面。见
`evidence/step-31/longitudinal-uniform-gpu/`。

固定质量厚度离线预测已通过pilot：scale=1，冻结175MeV/u参数，
参考密度离线/GPU最大5mm分箱差0.0212%；HU=-975/-951的2–20mm
预测误差+0.318%/+0.126%，最大5mm误差0.452%/0.418%。这些是离线
预测，两个密度的GPU候选仍不生效；不能写成新的GPU/Gamma提升。
下一步限于显式未验收的均匀section-0三密度GPU诊断，异质CT路由不得
开放。固定出生群/joint/界面仍未完成。证据见
`evidence/step-31/longitudinal-mass-thickness/`。

密度参考 pilot 已完成：175 MeV/u、HU=-975/-951，各两份 TOPAS20k
（2355–2358，全0:0）及GPU开关配对20k。2–20 mm基线偏差分别+1.284%/
+0.563%，20mm以后窗口偏差绝对值<0.06%。候选因密度限制不生效，
开关raw逐字节一致、纵向搬运为0；这里只关闭参考采集/域外保护检查，
不关闭Step07物理密度模型门禁。下一步先离线检验固定参数质量厚度预测，
仍需固定出生群、joint及界面验证。见 `evidence/step-31/longitudinal-density-175/`。

后续独立能量 pilot：175 MeV/u、同源/同密度/scale=1，两份 TOPAS 20k
（2353/2354，0:0）与 GPU base/long20k 完成。入口2–20 mm偏差由
+3.944%降至+0.547%，各5mm分箱候选偏差最大0.590%。仅记
PASS_PILOT_ONLY，不是通用核或患者Gamma验收。无参数修改，密度与界面
仍 BLOCKED。详见 `evidence/step-31/longitudinal-holdout-175/`。

后续更新：共享 200 MeV/u writer 失败已定位并修复为横向 δ-tail 的深度诊断
同步缺失（不是 3-D 剂量物理错误）。六个 1k CUDA 回归全部输出闭合，包括
200/120 MeV/u 开关、斜向与贴边；域外 raw 与修复前逐字节一致。
下方关于该 writer 阻塞的描述保留为历史。真实响应的独立物理验证仍 BLOCKED，
原先有能散小样本的 physical-residual 门禁未声称解决。无新患者 Gamma。
见 `evidence/step-31/longitudinal-review-20260906/depth-mirror-fix.md`。

逐病例标定的历史三例不构成通用核独立验证，旧 Gamma 不适用于本次修复代码。
当前候选锁定 CSV/metadata/contract、smoke + scale=1；域外不 clamp，
仅参考密度均匀 section-0 内按真实体素交界分配，跨材料余量保留为诊断近似。
候选质量报告必含 `unvalidated_longitudinal_candidate`，不得发布为 accepted 物理。
真实 CUDA host/device 单测通过；120 MeV/u 域外 on/off raw 完全一致。
200 MeV/u on/off 两者均在 writer 出现相同 bin 闭合失败，即使 baseline quality
accepted 也不等于输出成功。全套旧回归未运行，不宣称全部通过。
未运行新患者 Gamma，未升级数据栈。密度/能量/界面/joint 验证与共享 writer
问题仍未关闭；证据见 `evidence/step-31/longitudinal-review-20260906/`。

> 约定：每完成一个 Step 即向本文件追加一节。旧结论不改写；统计不足标
> INCONCLUSIVE，门禁失败标 BLOCKED 并写明原因。正式新表与 full20 结论以
> plan2/README.md 冻结口径为准。

## Step 01 — DONE：基线冻结（2026-09-05）

- HEAD `c6fe6b44`；工作树非干净：tracked 改动为 plan/README.md、plan2/README.md、
  plan2/steps/13 个旧文件删除、src/transport_sycl.cpp 入口 mask 未提交；
  untracked 为 v2.1 bin 包、电子审计工具/测试、entrance-mask 证据、scratch。
- `verify_schneider_v2_1_data.py` 16/16 通过，无 v1 回退。
- GPU exe：`build/carbon_mc bfd03552`（mask 前陈旧）、
  `build/oneapi-nvidia-release/carbon_mc 6ae10bb7`（与当前 mask 源码同日构建）；
  maskfix dose `0ee4457a` 与 validation.json 一致，但 validation.json 未记录
  exe SHA，记为“证据–executable 对应关系未知”限制。
- TOPAS exe `/home/wuwei/topas/topas-build/topas e1f5ccc0`；
  配置 `split20_rt06423_strict_01_v2_1.yaml d756293f`；bundle pins 按 manifest 通过。
- 冻结 full20 基线不变（Global 1%/1mm 98.426%，Local 1%/1mm 87.786%，
  Global 3%/0mm 99.804%，Local 3%/0mm 93.422%）。后续 A/B/C 必须复用上述
  冻结版本并先补对应记录。

## Step 02 — DONE：电子审计加固（2026-09-05）

- `tools/analyze_electron_deposit_steps.py` 新增：逐 track 身份/出生一致性、
  单事件单 primary、重复 step 拒绝、逐事件残差（防正负抵消）、逐家族残差与
  最差家族身份、KE 有限非负、JSON `allow_nan=False`、未计算 slab 审计写 null。
- 测试 9→17 全过；binary12（job 2338）重算物理量与旧 JSON 完全一致：
  出生 146.739 / 沉积 143.945 / 逃逸 2.795 MeV，>0.5 mm 前向 28.878%，
  逐事件最大残差 4.2e-15，最差家族 (0,10,99) 1.55e-15。

## Step 03 — DONE：分片与聚合（2026-09-05）

- 新增 `tools/run_electron_response_diagnostic.py`（≤12 histories/片、Binary、
  独立目录、config/TOPAS/scorer SHA、CPU≤192/Mem≤160G/磁盘预算预检、
  超预算拒交、不删数）与 `tools/merge_electron_response_diagnostics.py`
  （先合加权直方图/能量/分母再算概率、不平均分位数、steps 不计 histories、
  顺序无关/重复/缺片/SHA/混配置/能量和/预算/非零退出 8 门禁）。
- `tests/test_electron_response_campaign.py` 4 项通过。
- smoke12+seed2 双片聚合：24 histories，能量求和一致，joint 守恒，顺序无关，
  pooled 分位数保持 null（只验证聚合代码，不证明统计收敛）。
- `/mnt/sda/wuwei/delta_longitudinal_audit` 现 16G（含 2335 取消残留约 14G）；
  binary 约 123 MiB/12 histories。演示 campaign 预算 5 GiB / 估计 0.31 GiB / 2 片。
- 本批未改 GPU 物理、未生正式新表、未启动患者运行。

## Step 04 — DONE：父 C12 出生条件与版本化记录（2026-09-05）

- scorer 新增 v2 记录格式（`startup/extensions/CarbonElectronDepositNtuple.*`，
  已同步 `/home/wuwei/topas/extensions/` 并重建
  `/home/wuwei/topas/topas-build/topas 2351cc6e`）：
  21 个 v1 列原样保留，新增 8 列（29 列/196 B）——`schema_version=2`、
  `parent_valid`、`parent_ke_MeV`、`parent_dir_x/y/z`、`creator_process_id`、
  `birth_density_g_cm3`。旧 21 列文件不得静默按新格式解析。
- 父方向定义：同事件内电子首步之前最近一次 C12 pre-step 动量方向
  （last-known-state 近似，非精确产生时刻状态，已在头文件与结论中显式声明；
  不假设父粒子沿世界 +z）。creator 图例：0 primary、1 ionIoni、2 eIoni、
  3 compt、4 conv、5 phot、6 eBrem、7 CoulombScat、8 msc、9 other。
- 途中修复构建注册：扩展 .cc 首行必须保持 `// Scorer for <ClassName>` 格式，
  CMake 生成器取首行派生注册名，注释改动曾导致注册名污染而重建失败一次，
  已恢复并验证。
- 分析器支持 v1/v2 双模式，未知列数明确拒绝；v2 校验 schema 全 2、
  valid 哨兵、单位方向、creator 0–9；输出 missing-parent 计数与按父方向投影
  的纵向/径向量化（+x/+y/+z/−z/斜向旋转测试通过）。测试 25 项全过。
- TOPAS 配对验证（同物理同 seed 917001，各 12 histories，2CPU/4G）：
  job 2339（v1）与 2341（v2）3D 剂量 4.4M 体素值 bitwise 一致（唯一字节差异
  为 dose.csv 注释行中的 config 路径）；v2 逃逸闭合 3.3e-15；
  missing-parent 0/25604；creator 全 ionIoni。
- **语义修正（Step06 实证后追加）**：scorer 缓存的 `parent_*` 经查是同事件
  C12 末步退出态（同事件电子父能恒定，如 G0 event0 全 2328.4），不是产生
  时刻态——Geant4 先走完主径迹才弹堆栈次级。真产生时刻绑定改由分析器离线
  完成（电子顶点到同事件 C12 轨迹段的最近匹配，距离分位数 ~0.0 mm，
  missing 0），匹配父 KE 覆盖真实慢化区间（G0：2355–2436 MeV）。
  原“父 KE 2328–2405 证实慢化”表述作废，以本条为准。
- run 工具元数据补齐：record_schema_version、TOPAS 版本、config 关键字段
  实录，cuts/step-limit/Geant4 版本取不到时写 null+原因。
  数据：`/mnt/sda/wuwei/electron_v2_validation/{v1_ref,v2_new}/`。
- 只改观察与记录，未改物理（配对剂量一致为证）。

## Step 05 — DONE：能量分配契约（2026-09-05）

- 契约文件：`plan2/energy_partition_contract.md`；八问全部答复，未改 GPU。
- 核心事实（代码证据）：GPU stopping 系 `ComputeTotalDEDX` 非受限总量
  （Dump.cc:187）；`deposited_MeV` 为已 straggle 一次的总电子能损
 （transport_sycl.cpp:2950–3011，tags 0/1/2）；v1 搬运
  `moved = deposited × moved_fraction`（:3030，tags 17/18），纯横向终点。
- v1 分母是 scorer 沉积剂量（2.01 mm 外/总量，200 MeV/u 处 10.24%），
  不是电子出生能量；生产阈 TOPAS 默认 0.05 mm，阈下留局部。
- v2 实测（seed 917001）：parent loss 424.1054 = primary_local 276.5991 +
  birth 147.5063；birth/parent-loss = 34.78%，v1 只搬 ~10%，差值系有意
  留局部的近端 δ。禁止叠加 34%、禁止 moved_fraction 改 0.34、
  禁止分母混淆；joint 必须与 v1、与显式输运互斥，不重施 straggling；
  scorer 外三类账与零 overflow 门禁沿用。

## Step 06 — DONE：有限 slab 偏差与几何收敛（2026-09-05）

- 实验（200 MeV/u、HU=-1000、同 seed 917001、各 12 histories、2CPU/4G）：
  G0 基线（job 2341）、G1 横向×2（job 2342，体素 8 mm³）、G2 纵向×2
  z 0–440（job 2343，体素 4 mm³），另 G3 内部样本为 50 mm 内缩切分（离线）。
  工具：`tools/compare_electron_response_geometries.py`（真产生时刻父 KE
  分带、SHA/重复/闭合门禁，5 项测试全过）。
- 结果：横向扩大几乎无变化（逃逸 Δ+0.0002，joint L1=0.038，横向已收敛）；
  纵向扩大变化显著——逃逸率减半（0.0229→0.0119），且在每个匹配父 KE 带内
  一致下降（[2350,2400] 带 0.0221→0.0063），joint L1=0.357。
  G0 的 z=220 出射面截断了纵向响应尾部。
- 判定：按停止条件，**纵向未收敛，不生成 runtime kernel**，保留为有限几何
  诊断。逃逸能量只记账不重分；远尾 bin 未删除。
  数据：`/mnt/sda/wuwei/electron_v2_validation/{v2_new,g1_wide_xy,g2_long_z}/`
  与 `geometry_comparison.json`。
- 尾部定位补遗（Step06 允许的“继续定位未包含尾部”，离线零新运行）：
  新工具 `tools/attribute_escape_faces.py`（6 项测试全过）对三几何做逃逸面
  归因（与审计逃逸总量一致到 1e-9）：G0 逃逸 3.373 MeV 中 90.8% 经 +z 前向
  面；G1 加宽后前向占比 99.6%（横侧逃逸已可忽略，横向收敛的另一证据）；
  G2 延长后前向逃逸 3.06→2.70 MeV（尽管出生翻倍），前向占比 74.0%——
  **未被包含的尾部即前向出射电子，440 mm 仍未包住**。
  各 `face_attribution.json` 与审计交叉一致。

## Step 08 — BLOCKED：候选联合响应数据编译（2026-09-05）

- 阻断原因（前置门禁 05–07 未全部通过）：06 纵向几何未收敛（joint L1=0.357，
  前向尾部 440 mm 未包住）；07 单密度表覆盖 section 0 被拒（mov 差 73–80%）、
  1/rho 缩放不成立（16 倍偏离）。按方案“仅在 05–07 通过后开始”，不得编译
  候选核、不得进 GPU（09–15 连带阻断，保持 TODO）。
- 本步零产出是合规结果，不是欠账。诊断链（01–07 工具/数据/结论）完整可提交。

## Step 07 — DONE：独立能量与密度验证（2026-09-05）

- 方案冻结在先：`plan2/step07_validation_plan.md`（口径/误差法/阈值/INCONCLUSIVE
  规则，事后未动）。HU 取值来自 parser 真实边界（section 0 = [-1000,-98)，
  `src/ct_grid.cpp:293`）：-1000（0.00121，已有复用）、-550（0.46458）、
  -100（0.92794，仍属 section 0）；材料名规则经 TOPAS 源码确认并随预加载存在。
- 运行：150/175(held-out,只验不拟)/225@HU-1000 + 200@HU-550/100，
  各 12 histories 同 seed 917001，jobs 2344–2348 全 exit 0，
  逐片闭合 ~1e-15、missing 0、零 overflow、bundle 验证通过。
- T1 密度（200 MeV/u）：mov 0.3426→0.0922→0.0697，相对差 73–80% > 20% 阈值
  → **检出密度依赖，单密度表覆盖 section 0 被拒**（高密度下束流在 slab 内
  射程停止是主因之一，本身即证非标度性）。
- T2 held-out 175：esc 0.0179 vs 插值 0.0186（±3σ=0.0215 内），
  mov 0.3484 vs 0.3444（±3σ=0.012 内）→ **插值相容（弱证据）**。
- T3 1/rho 缩放：**不成立**。纵向 90% 分位 10.47→0.43→0.29 mm，
  按 1/rho 应为 10.47→0.027 mm（16 倍偏离）；joint 概率 L1 达 0.48–0.53。
- 统计不足项按冻结规则标 INCONCLUSIVE；Step06 纵向未收敛阻断仍然有效，
  不制表。数据：`/mnt/sda/wuwei/electron_v2_validation/e{150,175,225}_hu-1000/，
  e200_hu-{550,100}/`。

## Review 修正 — 重新打开门禁（2026-09-05）

本节覆盖上面有关“Step03/04/06/07 已完整通过”的解释；旧记录保留，不删除原始数据。

### High：Step07 混淆材料区间与密度公式区间

- material section 0 = [-1000,-950)，来自 SchneiderHUToMaterialSections。
- [-1000,-98) 是密度公式第 0 段，不是材料 section 0。
- HU=-550 属于 section 1，HU=-100 属于 section 2；原实验同时改变组成与密度，
  不能作为 section-0 单材料缩放实验。
- 原密度忽略 DensityCorrection。真实文件给出 HU=-1000 的 rho=0.0113160652，
  不是 0.00121；-550/-100 分别约 0.45520017/0.87682963 g/cm3。
- “section-0 的 1/rho 缩放被证伪、16 倍偏差”撤回。跨材料原始数据仍可留作诊断。
- 新增 tools/audit_schneider_response_scope.py；-1000/-975/-951 全部通过 section-0
  核验，-1000/-550/-100 明确非零退出。几何比较 --case-hu 会检查 section 和原始密度。
- Step07 BLOCKED：先另建正确实验设计，不调阈值，不覆盖旧冻结文件。

### High：聚合门禁没有实际核验来源

- 旧 --verify-raw 只检查 SHA 字符串存在，未重算原始文件。
- 旧 --expected-config-sha 遇到缺 metadata 时可直接通过。
- 已修：分析器记录明确 raw 路径；聚合器重算 steps/header/dose SHA，缺路径或
  文件错误均拒绝；严格配置检查要求每片有 metadata/config SHA。
- 新增每片直方图非负、shape、能量闭合、事件覆盖与重新计算剂量闭合检查，
  防止不同分片的误差互相抵消。输出显式 raw_verified/config_verified。
- 磁盘统计按去重的明确 raw 路径求和，不再通过重叠 glob 双算或漏算。
- 旧无 raw 路径的报告必须重新分析后才能使用 --verify-raw，不静默兼容为成功。

### High：Step04 尚未得到精确父出生态

- v2 scorer 缓存的是末态；离线最近轨迹段、线性 KE 插值与位移弦方向是诊断近似，
  不是精确过程产生时刻的父状态。距离接近 0 不能证明 KE/动量时刻精确。
- 分析器已显式输出 parent_conditioning_runtime_eligible=false。
- Step04 IN_PROGRESS；新精确记录格式仍待实现。本轮不改核物理、不重编或替换 TOPAS。

### Medium：joint L1 混入沉积总量

- 旧 abs(H0-H1).sum()/(H0.sum()+H1.sum()) 不是概率分布 L1。
- 已改为 sum(abs(H0/H0.sum()-H1/H1.sum()))，范围 [0,2]；
  总沉积比单独输出。相同形状乘 20 倍必须得到形状距离 0。
- 真实 G0/G2 二进制数据复核：新 L1=0.08640870，总沉积比=2.10567318；
  原 0.357 不能继续当作形状差异。新旧指标定义不同，不沿用旧阈值直接判定。
- 这些 joint 仍是无条件直方图，父 KE 分带只有单独的标量统计。工具已显式标注。
- 延长 slab 也新增了靠近新出射面的电子出生，仍有出射电子本身不能证明固定出生群
  的响应未收敛。Step06 INCONCLUSIVE；需要固定出生条件/位置和独立统计验证。

### Medium：运行器只是计划生成，未实现动态资源和完成闭环

- 旧工具只检查本批申请数量，不检查已有任务，不监控磁盘增长，不落盘实际 job ID
  或更新完成/分析状态。不能支持 Step03 DONE 的原结论。
- 已补正数/有限资源、seed 范围、case ID、绝对输出路径、shell 引号、现有输出体积
  预检；修复 sv:Ph/Default/Modules 元数据漏读。
- 为避免越过共享资源限制，--submit 现在明确拒绝，保留配置/脚本生成；
  不能绕过此限制称运行器已实现自动 campaign。
- Step03 IN_PROGRESS：动态调度、磁盘监控、include-chain provenance 和完成链仍待补齐。

### 验证和边界

- 本轮未运行新 TOPAS/GPU，没有新 Gamma，没有更新 package。
- 真实 G0/G2 原始步骤 SHA 和 corrected density 已复核，新比较结果在
  /tmp/electron_review_geometry.json（诊断，非生产 evidence 冻结点）。
- 新增 tests/test_electron_response_review.py 覆盖真 SHA 篡改、缺 metadata、
  分片误差抵消、跨材料 section、正确 density correction、L1 标度不变、
  vector 参数和不安全提交拒绝。
- plan2/README.md 已同步状态；Step08 继续 BLOCKED，但原因是未满足的真实门禁，
  不是把错误的跨材料实验当成缩放模型被证伪。

## 修复实施：v3 产生步骤绑定（2026-09-05）

- 新增独立 CarbonElectronDepositNtupleV3，旧 v2 类和旧数据不覆盖。
  34 列/232 bytes，schema_version=3；记录准确 parent_step_id、该步
  pre KE/方向及 post KE/方向。GetSecondaryInCurrentStep 捕获当步电子，
  用 G4Track 指针跨堆栈关联，电子首步消费后转 track ID 缓存；每事件清理缓存。
- 不修改任何 track user information、随机数或物理过程。pre/post 是准确
  的产生步状态，但不宣称观察到模型内部未公开的瞬时采样 KE。
- 分析器 v3 不再执行最近轨迹段匹配；逐根校验父 ID/step ID、pre/post KE，
  缺记录、错误能量、错误 schema 均失败。旧 v1/v2 按原格式读取。
- 新测试 4 项（产生步与末步区别、错 step、错 KE、版本伪装）；全套相关
  Python 测试 26+6+7+4=43 全通过，数据 verifier 16/16。
- TOPAS 已实际构建，job2349（2CPU/4GB、12 histories、seed917001）成功。
  25,604 个电子根全部绑定，missing=0；与旧 v2 同种子 3D dose 数值逐值一致。
  birth=147.506317796851、dep=144.133192169176、escape=3.373125627674 MeV；
  电子相对残差 3.32e-15，全局残差 -6.29e-16。
- 数据 /mnt/sda/wuwei/electron_v3_validation/；源码、TOPAS executable、
  配置链与 raw SHA 记录于
  evidence/step-31/entrance-mask-candidate/v3-parent-binding.json。
- 新 step07_validation_plan_v2.md 修正 HU/密度、要求匹配出生 KE/位置并
  分离 production-cut 导致的出生谱变化。尚未运行新密度 campaign。
- runner 的计划生成功能默认指向 v3 且生成明确 Quantity；submit 仍拒绝，
  动态资源与完成监控没有伪装成实现。
- Step04 的产生步骤关联缺陷已实质修复；整体 Step04 仍 IN_PROGRESS，
  后续记录/统计流程与生产条件完整验收尚未结束。Step06/07/08 门禁不解除。
- 本轮无 GPU 物理修改、无新 Gamma、无数据包替换、无提交或推送。

## 续修：受监控 pilot 与闭合审计补洞（2026-09-05）

- 新独立 Slurm 执行器已完成三次真实顺序运行：2350/2351/2352，分别为
  HU -1000/-975/-951，各 1 history、seed918001、2CPU/4GiB，COMPLETED/0:0。
  生成器 --submit 仍拒绝，必须经 execute_electron_response_campaign.py；
  监控已有用户 CPU/内存、目录体积、完成/分析状态，失败只取消自己的活动 job。
- 初次 Slurm 拒绝 --mem=4.0G，未提交成功；改成整数 MiB 后用新 r2 目录，
  旧失败记录未覆盖。新数据位于 /mnt/sda/wuwei/electron_density_pilot_v3_r2/。
- 实测 density 0.0113161/0.0393235/0.0662102 g/cm3 与正确公式差 <3.1e-6；
  分析器输出实测密度，执行器拒绝缺失/错误密度、错 schema/history 和剂量闭合失败。
  3D dose/steps 相对误差 <2.2e-9，v3 产生步绑定全通过。
- 对此前“逐事件/家族已硬门禁”的进一步修正：原审计漏加 event electron birth，
  其残差实际为 null；family 残差只报告，未拒绝超限。已补每 root 一次出生记账与
  逐家族拒绝分支，门限保持 1e-3。回归覆盖后代不重记出生、跨事件电子误差抵消、
  同事件不同家族误差抵消。三份原 raw 重算通过，逐事件电子残差 <3.5e-15。
- 原 analysis.json/metadata/raw 不改；最终加强门禁报告另存
  analysis_closure_verified.json。中间 density-only 报告也保留，但不能替代最终报告。
  冻结索引：evidence/step-31/entrance-mask-candidate/section0-density-pilot-v3.json。
- 输出预算按 pilot 修正：单 history 最大约 50.5MiB ntuple，计划估计改为
  (128+64*histories)MiB/片，12 histories 为 896MiB。它是估计不是硬配额。
- Python 相关测试 50/50、v2.1 verifier 16/16。无新 GPU/Gamma、无 physics/package
  变更、无提交或 push。既有未提交 GPU mask 改动保持原样。
- 限制：当前源码包含 pilot 后新增检查，不能说当时已执行这些检查；它们通过的是
  单独 raw 复核。外部 DICOM 预加载目录尚未逐文件 pin；轮询非硬配额，非协作提交器
  仍可能与资源检查竞争；未实现自动恢复。03/04 继续 IN_PROGRESS。
- 三点各 1 history 仅是输出规模和诊断链 pilot，Step07 正式物理验证仍 BLOCKED；
  Step06 固定出生群几何收敛、实际 production cuts 与条件化统计停止规则尚未验收。
  不得宣称密度缩放成立/失败，也不得据此生成响应包或推进患者 Gamma/full20。

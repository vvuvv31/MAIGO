# Water / Schneider CT 统一输运

当前状态：按用户新指令，默认和 production 均使用共享 CINEL03 框架；water 保留原生材料，不走旧水核路径。CLI 默认 config/unified_water_production.yaml；water TOPAS match 延后，以非阻断注记报告。下方 smoke-only 描述为历史记录，不再控制当前路由。

本轮验证：独立 build/oneapi-nvidia-unified-default 编译成功；v2.1 verifier 16/16；统一配置/生产许可/禁止旧回退/缺靶拒收测试及 CT 电子 YAML 开关测试通过。默认10000粒子生产运行通过，能量残差约1.29e-5；1000粒子试跑残差2.07e-4被安全门正确拒收，未放宽门限。未做完整回归或新Gamma，未commit/push。水默认保留既有75eV stopping，精确G4_WATER匹配仍待办。

## 第二阶段：共用核输运已接入（2026-09-07）

- 新入口 `unified_water_nuclear_transport: true`，示例 `config/unified_water_cinel03_smoke.yaml`。保持 `MaterialPhysicsMode::Water` 和非CT几何；共用CINEL03包、核反应/产物/队列/账本/3D记分实现。
- 共用上下文分离 `uses_cinel03()` 与 `is_schneider_ct()`。水有独立primary/secondary材料率view，不使用25-section行；日志255仅为水诊断sentinel，不参与Schneider索引。
- hazard、post-EM靶选择及miss诊断均走同一材料率接口；局部密度仅在hazard乘一次。原生water使用水stopping/逐同位素表，MCS使用真实G4_WATER X0。bundle中的Schneider stopping仅保持数据pin完整，不用于water能损。
- 默认关闭；强制smoke/C12/3D/步长能损约束和明确v2.1路径。禁CINEL02混用、CT/分层/insert几何混用、CT电子响应套水；CPU明确拒绝，不静默走旧water模型。
- 与CT共用missing/domain/gap/overflow/born门禁；额外 `unvalidated_unified_water_transport` 保证生产未验收时accepted=false。
- 二进制 `build/oneapi-nvidia-unified-water/carbon_mc` SHA `7319fb54d566a2c8d90d855668acbbe295550360905585bb8c09fca7a201127e`。
- 初次10k因示例混用dose_to_medium与旧空dose_output_file键而在启动前拒绝；已删冲突键并保留失败目录。`unified_water_10k_r2_20260907` 实测primary3898/3898、secondary3528=3522+4 stopped+2 null，0 overflow。
- `/mnt/sda/wuwei/unified_water_50k_20260907`：primary19442/19442；secondary17744=17700 replay+29 stopped+15 null；tracks189509、queued63872；lookup missing/domain/gap均0、overflow0。相对能量残差2.98966e-5，voxel/in=0.999999999998。唯一质量失败为预期未验收标记，四个核数据SHA与v2.1完全一致。
- `/mnt/sda/wuwei/unified_water_gen0_20260907`：10k，secondary hazards/replay/queued均0，primary3898/3898，代数开关有效。
- `/mnt/sda/wuwei/unified_water_ct_pair50k_20260907/comparison.json`：同CT/源/seed/电子r3，新旧binary各50k，三维dose逐位相同，核计数与电子replay计数相同；浮点能量诊断和mismatch累加存在微小差异，完整记录，未声称全部ledger逐位相同。分析脚本首次把整数形式MeV当计数而误拒绝，已按单位纠正并只重分析，未重跑/覆盖dose。
- 测试：新water配置/质量门禁、原电子YAML开关、CT禁止旧water键、13815 GPU材料率查询通过；原manifest16/16通过。没有跑完整旧套件或新Gamma；无提交/推送。

## 独立TOPAS水模200MeV/u首轮（2026-09-07）

- Slurm2567，24CPU/24GiB，50000 histories，COMPLETED 0:0，305秒。证据 `/mnt/sda/wuwei/unified_water_topas_200_20260907/{manifest.json,comparison.json,depth_profiles.csv}`。复用已有GPU50k，无新患者Gamma，无物理修改，无commit/push。
- 用户指定7模块、G4_WATER、默认production cuts；Beam方向+z，能散1.0%=GPU0.01。水模z=0–200mm、横向4m宽近似GPU横向无限水。ROI128×128×200mm、64×64×400体素；入口前0.001mm真空源。仅3D DoseToMedium Sum，横向积分，水体素质量2e-6kg；无拟合scale或后验平移。
- TOPAS/GPU能量2253.43865/2237.20279 MeV/primary，GPU低0.72049%；峰位85.75/85.25mm，R80=87.43614/86.92884mm；深度corr=0.996267，未归一深度RMSE=1.78105%参考峰值。
- 碎片尾（参考峰后5mm至200mm）能量173.43491/172.86736，GPU低约0.33%；径向RMS=18.2412/17.4772mm。入口/平台/峰区RMS为2.1344/1.9142、3.6425/3.3851、3.2880/3.1508mm，GPU偏窄。2mm横向体素不能用于亚毫米束宽验收。
- 单能单seed50k不是生产验收，无独立噪声置信区间；TOPAS全物理与GPU有限代数、无显式水电子仍有差别，不将横向差异直接归因某个模型。保留unvalidated门禁。下一步100/300MeV/u（300需加长水模）、重复seed/步长收敛及密度验证。
- 新prepare_unified_water_reference.py仅准备、不自动提交；compare_unified_water_reference.py检查成功状态/输入SHA/三维布局及单位。3项深度交点/截断拒绝/量纲单测通过。INCL++重采样警告保留于原日志。

## 续修：修复统一水实际stopping加载（2026-09-07）

- 明确撤销此前“water primary/secondary已经使用YAML指定stopping”的说法：核开启时transport_sycl.cpp硬编码读取旧ion CSV，C12也走其中species16行。ledger此前只是打印配置路径，不能证明实际生效。改为仅unified_water显式路径加载；CT/旧water保留原分支。上传前校验完整CSV能量网格，缺文件拒绝；日志记录实际路径+SHA，probe增加对应检查。
- 原两表注释均Water_75eV，而新材料和参考为G4_WATER78eV。复用TOPAS现有Carbon/IonStoppingPowerNtuple，在与参考相同7模块/G4_WATER下提取。Slurm2568，1CPU/4GiB，0:0完成，原始目录 `/mnt/sda/wuwei/unified_water78_stopping_20260907`。4001个能量×32种离子，0.01–400.01MeV/u。保持旧数据和CT v2.1文件不变；新表仅候选，尚不能用于430MeV/u默认运行。
- compiled初版7列不兼容现有8列解析器，在GPU运行前纠正并另存compiled_v2，未覆盖初版。两份C12独立提取相符，所有值正且有限，各种类能量网格一致；raw delta原样填入两列，无旧经验修正。转换器/原始/候选有SHA。尚不是支持全能域的水物理bundle。
- 旧binary+新表 `/mnt/sda/wuwei/unified_water78_50k_20260907` 剂量逐位不变，由此抓到硬编码，不能当成修复验收。新binary SHA77166cf4d1247700b17c1b858b8c2479d3e398aba0046eda02f965bc22429135；build路径已重建，不宣称此前7319fb二进制仍在原路径，旧dose/manifest未改写。
- 真修复50k `/mnt/sda/wuwei/unified_water78_fixed_50k_20260907`：仅预期unvalidated门禁，零overflow。R80=87.39507mm（TOPAS87.43614），偏差从−0.50730到−0.04107mm；峰位均85.75mm；深度corr0.99994978、RMSE1.78105%→0.39992%参考峰值。总能量2235.27268 vs2253.43865，仍−0.80614%，未声称所有指标改善；横向仍偏窄，需独立验证。
- 新binary配旧表 `/mnt/sda/wuwei/unified_water75_fixed_control_20260907` 保持旧dose哈希38695131df4866c0876bc32c765512786bd0c25f2cdfab2882102625e37b99af。CT成对50k `/mnt/sda/wuwei/unified_water_stoppingfix_ct50k_20260907` 剂量逐位一致、核整数计数与电子replay计数一致，浮点诊断差异另记，非完整患者验收。
- 复跑候选：`python3 tools/probe_unified_water.py --config /mnt/sda/wuwei/unified_water78_stopping_20260907/compiled_v2/candidate.yaml --out <新目录> --histories 50000`。比较使用compare_unified_water_reference.py的--gpu；不覆盖baseline。下一步100/300MeV/u及重复seed/步长收敛，再决定迁移默认示例/补400–430覆盖。未提交/推送。

## 多能量独立验证（2026-09-07续）

- 新GPU100/300MeV/u各50k，使用同77166cf4二进制、同78eV候选表与核v2.1、同seed；仅能量及300MeV水模长度变化。两者真实stopping路径/SHA验证通过，均只有预期unvalidated门禁、零overflow。目录 `/mnt/sda/wuwei/unified_water78_e100_50k_20260907` 和 `unified_water78_e300_50k_20260907`。
- 独立TOPAS本地Slurm2569/2570均COMPLETED0:0：100MeV24CPU/24GiB、146秒；300MeV72CPU/24GiB、167秒；并发合计96CPU/48GiB。100水模200mm、300水模400mm；横向ROI128mm、2×2×0.5mm³，只有3D DoseToMedium。各50k，源和几何按GPU实际配置生成，无后验坐标/scale拟合。

| MeV/u | 峰位差mm | R80差mm | 总能量偏差GPU/TOPAS | 深度RMSE/参考峰值 |
|---|---:|---:|---:|---:|
|100|0|−0.01380|−0.60335%|0.20714%|
|200|0|−0.04107|−0.80614%|0.39992%|
|300|0|−0.12975|−2.09455%|0.79967%|

- 汇总 `/mnt/sda/wuwei/unified_water_energy_controls_20260907/summary.json`，包含3份comparison和dose哈希校验。100碎片尾积分差−2.11%，但径向RMS=10.4607mm vs13.1790mm（约−20.6%）；300尾积分−1.85%、RMS23.9722 vs24.2340mm。横向/能量问题尚未关闭。
- 结论仅为修复后的射程一致性在3能量成立；无独立seed置信区间，不能把单seed差值全部视为系统误差，也不能宣称患者Gamma改善。当前表上限400.01MeV/u，未补到430；默认配置/生产门禁不变。
- 新prepare_water_energy_controls.py和多能量汇总工具，prepare/compare支持匹配长水模。分析5项单测通过（新增800bin位置和历史数缩放），diff检查通过。无新的kernel/physics改动，无commit/push，无活跃作业。
- 下一步优先：300MeV核关闭对照（同3D网格、同stopping/源）分离EM与核贡献，并用独立seed量化误差；100MeV碎片尾单独处理，不能借全局相关性掩盖。仍不调MCS/beam/scale/package，不移除unvalidated门禁。

## 300MeV EM/核隔离与独立seed（2026-09-07续）

- GPU仍77166cf4，真实78eV表和v2.1未改；两个成功50k：`/mnt/sda/wuwei/unified_water300_em_20260907`（关闭核/次级输运，核计数全零）和`unified_water300_seed2_20260907`（全物理，seed202619071），均零overflow，仅预期unvalidated失败。未修改kernel或生产默认。
- 本地TOPAS2571（24CPU/24GiB,opt4+decay,523秒）和2572（72CPU/24GiB,原7模块,169秒）均COMPLETED0:0；并发96CPU/48GiB。400mm水模、2×2×0.5mm³ 3D DoseToMedium，各50k；核关闭与全物理独立seed对应目录为unified_water_ref300_em_20260907、unified_water_ref300_seed2_20260907。
- EM：TOPAS/GPU3599.774796/3600.086916MeV/primary，GPU+0.00867%；R80差−0.06710mm，深度RMSE0.11349%参考峰值。EM峰后5mm窗仍包含能散的初级尾，不是核碎片；分析保留fragment_tail通用字段名，禁止误读为EM产生碎片。
- 全物理seed2：TOPAS/GPU3174.725691/3119.204667MeV/primary，−1.74884%；原seed1−2.09455%。GPU两seed总量差0.23063%，TOPAS差−0.12185%；均值GPU/TOPAS0.98078197（−1.92180%）。两seed只证明重复性，不构成精确置信区间。
- gen3隔离尝试在启动时被`cinel02_max_secondary_inelastic_generations must be between 0 and 2`拒绝；目录unified_water300_seed2_gen3_20260907保留日志，没有剂量，未放宽限制。代数截断贡献仍未知，不能声称已排除。
- 汇总 `/mnt/sda/wuwei/unified_water300_em_20260907/isolation_summary.json`，由analyze_water300_isolation.py校验各dose SHA后产生。probe新增显式--em-only/--seed且仍保留原闭合/未验收门禁；分析支持零剂量窗口null，明确参考EM模式。5项分析测试通过，diff干净，无提交/推送。
- 判断：总能量缺口主要在开启核反应后出现，单独EM总量不足以解释约1.9%缺口；不等于所有EM空间误差或EM/核耦合均已排除。下一步TOPAS3D按粒子/祖先谱系量化中子、光子及其后代沉积，和直接带电碎片贡献区分。已核查现有OnlyIncludeIfParticleOrAncestorNamed过滤器可用；不能只筛带电粒子就宣称排除了中性贡献，也不能把未追踪中性出生能量直接当成本地剂量。禁止调package/scale/放宽gen上限以强行匹配。

## 300MeV中性谱系拆分（2026-09-07续）

- Slurm2573，72CPU/48GiB，50k，同seed202619072、原7模块/400mm水模/2×2×0.5mm³，COMPLETED0:0，186秒。仅增加3D DoseToMedium谱系过滤：neutron（自身或祖先含中子）、gamma_no_neutron（含光子且不含中子）、neither（两者均无），三者互斥且完备。total与原seed2 TOPAS参考逐位相同。
- 结果 `/mnt/sda/wuwei/unified_water300_lineage_20260907/analysis.json`：total3174.725691、neutron17.814066、gamma_no_neutron0.692557、neither3156.219067 MeV/primary；GPU3119.204667，缺口55.521023。中子/光子谱系合计18.506622，为缺口33.33%；数值扣除后仍37.014400MeV/primary。
- 不把这33.33%宣称为因果解释比例：光子谱系包括EM光子，GPU部分能量可能已经用局部沉积近似；neither也不是primary-only。仍需区分核生带电碎片、初级存活/沉积与中性后代重叠影响，不能直接给GPU加18.51MeV补偿。
- 代码证据transport_sycl.cpp约4037/5907：中性产物只累计NeutralProductKinetic然后continue，未进入队列；约4155未追踪sink含neutral/unsupported/Q且分项不重复加入闭合。中性出生动能不能当作记分区沉积。
- 初版分析要求1e-10峰值闭合失败（实测5.80483e-8）；没有重跑/改dose。源码TsVBinnedScorer.hh:190表明fSum为G4float，再写入double binary。独立analyze_water_lineage_roundoff.py按每体素eps32×(total+各分项绝对值)验证，全部通过；global闭合误差2.41e-10。冻结原分析脚本/manifest不改，新分析记录源码SHA。不是GPU精度优化或放宽物理门禁。
- 新谱系过滤四祖先组合测试、舍入界接受/缺失份额拒绝测试通过。无GPU运行、kernel/physics/package/scale修改、无commit/push。生产默认未变。
- 下一步优先同水模primary survival与primary/带电次级3D沉积拆分，量化剩余37MeV从何而来；若要补中性输运，应独立验证响应而非按当前缺口缩放。100MeV碎片尾偏窄仍开放。

## 300MeV primary/电子/次级分项（2026-09-07续）

- 本地TOPAS2574，72CPU/48GiB，50k，同seed202619072，COMPLETED0:0，186秒。只加3D记分，total与原参考逐位相同。互补组为ParentID0初级、neutron/gamma谱系、无该谱系e-/e+、无该谱系且非电子的次级。目录 `/mnt/sda/wuwei/unified_water300_primary_partition_20260907`，analysis.json/depth_partitions.npz。
- GPU同77166cf4/同表/同seed202619071/50k新增origin开关，目录`unified_water300_origin_20260907`，total与原seed2逐位相同；8个类别在逐体素输出舍入界内闭合。--origin禁用冲突sparse CSV，仅输出3D MHD。无kernel改动。
- TOPAS MeV/primary：primary2007.93139、非中性谱系电子265.25381、其他非中性谱系次级883.03387、中性谱系18.50662；GPU primary2206.35364、次级合计912.85103（He405.76458、p289.11598、C97.56935、B79.23454、Li22.34470、Be18.82187）。
- GPU primary−TOPAS primary=+198.42225，但减去全部非中性电子后为−66.83156；GPU次级比TOPAS其他非电子次级多29.81716。二者不能直接作为物理误差：TOPAS电子分项覆盖初级和碎片所生电子，而GPU能损按父粒子局部归属。剩余37MeV无法由本次粗分类唯一定位到primary或secondary。
- 又一标签差别：CINEL03主分支反应后终止原track，所有带电产物（含C12）进入secondary queue；TOPAS ParentID0过滤可能包含核反应后继续的原track。需在比较前核对状态，不能拿所有C12祖先当primary-only（它会包括整条级联）。
- 本轮没有测到primary survival曲线，也没有实现直接父粒子电子标签，禁止声称两者已排除。下一步最小诊断应在TOPAS按直接父粒子/首次核反应前后记3D能量，或将电子转移能量按出生父track记回诊断通道（仅诊断、不改变输运和正式dose）；同步GPU定义后才判断修复点。不得按当前37MeV做全局补偿。
- 工具prepare_water_primary_partition.py/analyze_water_primary_partition.py、probe --origin已加入；谱系2项测试、py_compile、diff检查通过。全部任务结束，无commit/push，无package/scale/生产门禁变更。

## 电子归回父粒子：完整闭合结果（2026-09-07续）

- 复用已有ChargedOriginDoseToMedium，无TOPAS源码修改/重编译。它将电子沉积归到已知带电父类别，保持实际沉积位置；独立neutron/gamma祖先veto排除中性谱系。ParentID0仍不等于首次核反应前标签，本轮尚未增加该标签或survival曲线。
- 初版Slurm2575（64CPU/96GiB）漏列扩展中的neutral_origin类别，分项少0.005231MeV/primary；保留失败输出，未放宽闭合门禁。完整类别2576（64CPU/96GiB，50k，274秒）COMPLETED0:0，工具额度阻断期间已结束，恢复后仅分析原输出，没有重复提交。
- 完整证据 `/mnt/sda/wuwei/unified_water300_parent_origin_complete_20260907/analysis.json`：total与原TOPAS参考逐位相同，所有分项在逐体素输出舍入界内闭合；其他中性0.005232、未分类0.004319MeV/primary，未分类只占总量1.36e-6。

| 父类别 | TOPAS MeV/primary | GPU MeV/primary | GPU−TOPAS |
|---|---:|---:|---:|
|初级C|2214.97739|2206.35364|−8.62375|
|次级C|98.28386|97.56935|−0.71451|
|B|75.74007|79.23454|+3.49447|
|Be|20.10979|18.82187|−1.28791|
|Li|23.72300|22.34470|−1.37829|
|He|432.79467|405.76458|−27.03010|
|H|287.01979|289.11598|+2.09619|
|其他带电|3.56095|0|−3.56095|

- 氦是当前最大的带电分项差异（GPU低6.2455%），初级低0.3893%。这是分项量级定位，不是已证明He stopping或package错误；不同类别正负抵消，不能按比例补偿。中性谱系仍18.50662MeV/primary。
- 只读当前包母粒子普查：C12包30168事件（H/O4965）、次级包353991事件（H/O53361），所有parent_energy_MeV=0，正动能存活母粒子事件0。排除“当前数据中存活母粒子动能被GPU丢掉”猜测，不代表TOPAS所有可能事件均如此。证据首轮目录package_parent_audit.json，SHA与v2.1一致；packed布局136/476bytes、parent_energy offset152与正能量筛选单测通过。
- 下一步收敛到He出生/输运二选一：用已有CarbonCascadeNtuple核查反应产品出生计数、总KE和分代/靶核；与GPU species ledger比对前核对queued birth/reaction import字段语义，不能把多代重复出生能量当独立beam输入。若出生一致再做He同源输运对照。不得用全局scale、任意角度收窄或调rate来修27MeV缺口。
- 本阶段仅诊断工具与计划更新；无GPU物理、包、默认或门禁修改，无commit/push。工具额度恢复后已确认无残留作业。

## 已完成

### 2026-09-07 同ROI核反应、直接率表与实际出生谱续验

- 仅追加species诊断槽11/12：cinel03_replay_input_fov（成功CINEL03 replay的post-EM入射KE）、cinel03_replay_step_dE_fov（这些步的连续损失）。原0–10含义不变，shape改为18×13；两新项是已有能量的子集，**不能加入闭合总和**。ROI按碰撞后实际坐标、半开边界判断，不使用旧pre-x/post-z混合voxel索引。仅适用于CINEL03成功replay，未声称CINEL02全模型覆盖。
- io形状改由schema推导；OOS摘要与单测的`species*11`改为schema stride，避免追加槽后错列。新binary `build/oneapi-nvidia-water-roi/carbon_mc` SHA fe45e2e060e498455df2f1b730b75f01b860337b2152e0c3e9925ec8b23df052；既有baf744e7和77166cf4保留。无粒子能量/位置/随机数更新修改。
- 水50k同seed：4He全空间171.26220、ROI171.17480、ROI步电离0.132752MeV/primary；TOPAS ROI pre-step150.07491，同轨迹延续仅0.004885。3He ROI22.14476 vs19.00941。框外和步内能量定义差不能解释约21MeV/primary的4He差距。TOPAS包含所有过程记录、GPU是成功重放口径，仍需路径/采样审计，不能称已定位物理bug。
- 独立真实G4_WATER IonCrossSectionNtuple探针：Slurm2581 COMPLETED 0:0，1线程/4GiB/1history、2秒；保持7-module physics list、3D dose仅伴随输出，**不是剂量验收**。12832条32种离子×401能量，0.01–400.01MeV/u。当前runtime MaterialNuclearRates host适配器独立导出14proj查询，不修改任何现用表。
- alpha H/O均在domain内398点，runtime/TOPAS比例0.9991707–1.0004584，最大相对差0.08293%；He3 396点最大0.12868%。域屏蔽/参考为零点单列，不声称低端未覆盖区一致。ASCII输出精度和插值解释小数残差，不调rate。真实GPU host/device查询921 primary+12894 secondary off-node通过，最大元素恢复差仍为7.49e−12/9.32e−12。
- 证据 `/mnt/sda/wuwei/unified_water300_roi_replay_20260907/analysis.json` 和 `/mnt/sda/wuwei/unified_water_ion_rate_probe_20260907/`。新旧水dose、CT50k（electron-r3开）dose均逐位相同，整数核计数/质量状态不变、零overflow；CT输出`unified_water_roi_ct50k_20260907`。共享生产物理未放宽。
- 复用既有fragment_birth_spectrum_output_file开关，从实际GPU队列读取出生粒子，未改采样：`/mnt/sda/wuwei/unified_water300_actual_birth_20260907/analysis.json`。开关前后dose逐位相同。仅比较generation0与TOPAS direct-primary C12产物，双方KE>0.1MeV；TOPAS仍ROI-only/first-tracked，不能称完整顶点普查。
- 初代4He：TOPAS58429/GPU58190个；平均KE403.0844/403.6203MeV；2MeV/u分箱CDF最大差0.01308、cos分箱差0.00490、深度分箱差0.01346。初代3He：12855/12833个；均值262.8033/279.5115MeV，能谱CDF差0.03666、cos差0.03916；有偏硬迹象，不凭单seed认定统计显著性或修package。cos分箱宽0.1，不能排除细角度差异。
- 明确未用字段：GPU该路径的parent_mevu/parent_z/parent-product joint列填的是源配置，不是真实反应母粒子状态；本分析只用真实子粒子KE、dir_z、pos_z和generation。不用误标母粒子列作物理结论。
- 测试：6+5+6项Python；C++ test_continuous_species_track_tally（含ROI边界）、test_cinel02_ledger_schema_and_accumulator、test_out_of_scope_isotope_summary_ledger_fields、test_unified_water_config_and_quality；真实率表host/device测试通过。diff检查通过；不是全ctest/患者Gamma/全20shard验收。
- 下一步优先用**实际出生谱+当前stopping/核率**独立积分CSDA光学深度和首次核反应能量期望，先从无限水假设的无逃逸上界做核查，再纳入实际出生位置/方向。需注明分箱、有限边界和级联相关性；不得用近似上界拟合参数。若kernel反应统计超出可解释范围，再做逐轨迹暴露对照。当前不调整rate/package/MCS/scale、不迁移默认water、不commit/push。

### 2026-09-07 4He输运续验与species诊断修复

- 冻结77166cf4 binary，同50k/seed202619071/gen2，maximum_step从0.25减至0.125/0.0625mm；v2.1输入16/16校验通过、零overflow、仅保留unvalidated_unified_water_transport质量失败。证据 `/mnt/sda/wuwei/unified_water300_step_convergence_20260907/analysis.json`。步长改变也改变随机流，单seed不提供精确置信区间。
- 总MeV/primary：3119.2047 / 3120.2615 / 3117.8115；He-origin：405.7646 / 414.1787 / 407.6509；4He核反应导出171.2622 / 170.6252 / 170.6912。没有看到总剂量随步长单调收敛，不能据此缩短生产步长或调整截面。
- TOPAS既有非侵入记录：ROI内4He核反应15126次，pre-step KE150.07491MeV/primary；43次同轨迹同同位素延续仅0.004885，扣除后150.07002。3He为19.00941、同轨迹延续0.002265。GPU export在post-EM且覆盖全输运空间，不能直接把约21MeV差当成率表错误；需同ROI/同碰撞能量对齐，birth能量和角谱/路径暴露仍需检查。
- 真正发现的问题是**诊断账本**：旧最细步长4He continuous_all仅188.724而queued513.236、export170.691、escape19.004，闭合−26.27%；原步长质子诊断也漏约18.02%。3D dose没有相应丢能，禁止用旧species分项断言物理泄漏。
- 最小修复：`ContinuousSpeciesTrackTally`把两项连续沉积按track本地累计，在while之后统一flush（含replay/miss break），原FOV guard不变；不改变粒子dE、步长、RNG、dose scorer、能量类型或任何physics/package。旧FOV语义限制仍在，不声称此项已对齐TOPAS电子归属。核反应、出生和逃逸等一次性项未改。
- 新独立binary `build/oneapi-nvidia-water-ledger/carbon_mc` SHA baf744e7bd78ab8c3a0cebc3ccf82d50feca7e49562656cc90eecd047a2f761f；旧77166cf4原路径/内容均保留。构建编译选项与旧版相同；初始新目录默认flag不同已在运行前纠正，不存在不同剂量模式的对照混用。
- 新旧3组实际GPU对照：0.25mm水模、0.0625mm水模、电子r3开启CT50k；**3D dose均逐位相同、整数核计数相同、CT电子replay数相同**。最细4He continuous_all恢复323.54988、闭合+1.92945e−5；0.25mm闭合+1.81934e−5；CT4He闭合−6.80388e−6。两水模全18species最大相对闭合误差分别2.80618e−5、3.37667e−5。非生产物理批准。
- 最终验收 `/mnt/sda/wuwei/unified_water300_ledger_fix_20260907/validation_v2.json`（v1报告保留，v2增加质量失败列表不变断言）。CT新结果 `/mnt/sda/wuwei/unified_water_ledgerfix_ct50k_20260907`。新Python测试5+6项通过；C++聚焦test_continuous_species_track_tally、test_unified_water_config_and_quality通过；并非全套ctest验收。
- 下一步：优先补4He **相同ROI** 的反应前后能量/逃逸与路径暴露，判别171对150的差来自核率、路径/出生谱还是级联处理。0.25mm生产参数保持不动；不根据这个不等口径差调package或rate。水统一仍smoke-only、无新Gamma、无commit/push。

### 2026-09-07 氦出生诊断续修完成（不等于物理验收）

- 新增独立`CarbonBirthOnlyNtuple`，原extension未改，只去掉额外XS查询、保留列17为未测零。仓库副本和说明在tools/topas_helium_birth；旧TOPAS binary保存为/home/wuwei/topas/topas-frozen-d50f90504eafe2b2，新binary SHA 1d7ba0a56d3841f2413c7e2adfd7f07445b42f842d777f3abb8c7ad91e3b3338。旧manifest未改，审计旧exe应使用保存副本。
- Slurm2578（无XS查询、Water传播）/2579（无scorer）各50k完成；并发总144CPU/96GiB。无scorer完整复现原参考；去XS查询仍与2577逐位相同，故该查询不是本次dose变化的解释。
- 源码TsVGeometryComponent.cc约354行和日志证实：Water PropagateToChildren与子ROI scorer并用会新增Water_1x1x1平行世界。改为ROI + 同64x64x800分箱，复用原平行copy；Slurm2580（72CPU/48GiB）完成，dose与2579及原参考逐位相同，全部SHA和配对门通过。
- 证据：/mnt/sda/wuwei/unified_water300_birth_roi_20260907/{isolation.json,analysis.json,cascade.header,cascade.phsp}。解析76061次interaction，其中29175次primary；重复/孤立He记录检查通过。修复分析器局部质量数a覆盖命令行对象导致的异常后真实全文件分析成功。
- TOPAS first-tracked primary He KE / GPU primary-queued KE（MeV/primary）：3He 67.56673 / 71.73848（+6.174%）；4He 471.03642 / 469.72252（−0.279%）；6He 2.59138 / 1.70297。TOPAS对应出生计数12855/58497/272；GPU这里没有同口径出生计数，不编造。
- 初级He出生KE总和约541.195 / 543.164，未看到可直接解释氦dose −27.03MeV/primary的总体出生能量缺口。但ROI-only、first-step反应/上下文和cutoff差异尚在，单seed、总量接近不证明谱形/角度/出生位置完全一致，不能排除所有生成误差。
- 下一步优先4He出生后同口径能量预算：核反应前后KE、实际电子沉积、ROI外逃逸、代数截断；需要时做同源4He输运。现GPU4He账本供检查：queued511.765、reaction_import42.043、reaction_export171.262、continuous_all322.214、boundary_escape18.637MeV/primary；这些量不是可直接与TOPAS所有代出生相减的完整剂量闭合。不得因当前数据改package或局部加能。
- 6项诊断测试通过，工具语法及diff检查通过；没有跑新GPU/Gamma、没有改GPU物理/包/默认/生产门禁，没有commit/push。以下旧失败段保留为追溯，截面查询候选现已排除。

### 2026-09-07 氦出生诊断：非侵入门拒绝

- 本地Slurm2577 COMPLETED 0:0，耗时182s，50k histories、72CPU/48GiB；输出672551条cascade记录，未新跑GPU。
- 冻结输入SHA全部通过，但TOPAS总dose与seed2参考有1645013体素不一致，总积分约−0.005716%，最大体素绝对差1.76644e−5 Gy。不是已通过的同轨迹出生对照；不因积分差小就放宽门。
- `tools/analyze_water_helium_birth.py` 在出生分析前硬拒绝，保存`/mnt/sda/wuwei/unified_water300_helium_birth_20260907/noninvasiveness_failure.json`，不输出analysis.json。保留全部原始产物。
- 已读scorer实现：记录时会额外调用`GetInelasticCrossSectionPerVolume`；尚未证明该调用造成轨迹改变，不能称根因。出生记录本身还受first-step分支和父轨迹上下文覆盖限制，不是完整顶点普查。
- 下一步：保留原extension和本轮数据，采用独立诊断scorer去掉额外截面查询，保持相同seed/source/geometry/physics，先验证3D total dose逐位不变。若仍变，查组件敏感探测器挂接/运行差异；不得直接调GPU或package。通过后才比较primary-produced He KE与GPU queued_birth−reaction_import，明确cutoff和first-tracked限制。
- 新增`tests/test_water_helium_birth.py`4/4通过（相等、等积分但不同分布、最小浮点差、非法输入）；git diff --check通过。无物理代码改动，无生产门禁放宽，无commit/push。

- 新 `MaterialNuclearRates` 从v3 primary与14-projectile secondary表生成独立纯水H/O质量率；不是把某个section改成水，也不写回25-section表。
- 每个H/O、每个抛射体、每个能量节点遍历全部25分区验证 `partial / mass_fraction`；零组分必须零rate，非有限/负数/覆盖不足/相对差>1e-8直接拒绝。最高组分section是确定性参考，不按Gamma或平均值拟合。
- 当前真实数据最大相对差：primary 7.48796e-12，secondary 9.31929e-12。所有非H/O partial为零；原通道域保留，查询先mask再插值，total为partials之和。
- `MaterialNuclearRateView`不含CT section索引或密度；密度应在后续hazard接线时只乘一次。没有改动现有primary/secondary kernel。
- 与TOPAS绑定的 `/software/geant4-11.3.2` 提取真实G4_WATER。Slurm2565因共享库环境失败，无输出；2566修正LD_LIBRARY_PATH后0:0完成，1CPU/1GiB。探针是Geant4 NIST属性读取，不是TOPAS剂量模拟。
- 属性文件 `data/water_unified/g4_water_material.json` SHA `60be17929880fe18f1758edc02350b3fa7140b817ab0d21cb75bd87dbc891f31`；来源代码、binary、G4materials库、TOPAS CMakeCache与命令固定于 `material_provenance.json`。
- H质量分数0.11189847784106703、O为0.8881015221589329；rho=1、质量X0=36.082977464022328 g/cm²、I=78 eV。加载器验证SHA/版本/材料名/正值/H:O原子数2:1，不能用近水组织或parser四舍五入原子量替代。
- 本机RTX2080Ti真实数据测试：921 primary + 12894 secondary非节点GPU查询与host对照通过；合成域边界/NaN/错误组分/超差拒绝测试通过。`ctest -R '^material_nuclear_rates$'` 1/1通过；不是完整旧套件、全输运或Gamma验收。
- 原Schneider manifest 16/16通过，原数据文件未改；新水属性只是补充材料信息，不是生产批准。

## 剩余顺序（继续实施，不得跳过）

1. 已完成smoke接入：几何无关核数据上下文、冻结v2.1包、14p registry和共同门禁；旧CINEL02暂不删除。
2. 已完成smoke接入：water primary/secondary hazard/target/replay、真实水材料参数，不伪造section。
3. 已完成50k初步闭合：共享账本/产物/3D记分，零overflow；更广的能量和密度检查仍需后续验证。
4. 已完成CT同输入50k成对回归；尚非全患者/多病例通用验收，不据此切换全部默认配置。
5. 纯水175/100/200/300 MeV/u独立TOPAS对照：3D剂量横向求和的深度曲线、primary survival、横向剖面和碎片尾；固定源/历史数与参考实际模型，不调scale。
6. 纯水电子逐段响应缺口单独补齐。当前r3仅Schneider，不能用近水section代替G4_WATER；未覆盖时开启必须拒绝。核统一不以伪造水电子响应为前提。
7. 上述通过后才迁移默认water YAML并退役旧water CINEL02活动依赖。water stopping等仍是必要输入，不删除。没有授权commit/push。

## 重跑当前阶段

```bash
python3 tools/verify_schneider_v2_1_data.py
cmake --build build/oneapi-nvidia-electron-switch --target test_material_nuclear_rates -j 8
LD_LIBRARY_PATH=/home/wuwei/sycl_workspace/llvm/build/install/lib ONEAPI_DEVICE_SELECTOR=cuda:* build/oneapi-nvidia-electron-switch/test_material_nuclear_rates --real-data
```

以上GPU命令仅在本机沙盒外执行，禁止远端GPU。后续TOPAS任务本地Slurm，总资源上限192CPU/160GiB。

# RT06423 严格 Gamma 改进计划：section-0 电子纵向响应

## 用户最新优先级：异质界面 → 患者11/30（2026-09-06）

主线改由 [界面与患者验证方案](interface_patient_focus.md) 控制；低能补表暂缓。
最新执行顺序：隔离GPU界面验证 → 通过后补齐材料/能量/粒子覆盖 → 患者配对 → 生产。
继续验证：0.02mm双seed及双buffer深度门通过（最大窗口0.474%、单bin0.700%）；
横向RMS残差仍存在，buffer敏感性<0.002mm。P0不关闭，下一步检查独立源组
横向统计敏感性；能量份额bootstrap不能替代3D验收。
用户已批准隔离GPU诊断；schema2逐段重放现已接入该隔离路径，患者/生产仍拒绝。
完整56份独立bulk源数据候选在0.02mm步长下，双向界面深度窗口最大误差
0.469%、单bin最大误差0.657%；这是深度通过，不是完整3D界面验收。
组织→空气首2mm横向RMS为3.393mm（TOPAS3.442mm），步长与响应统计
仍须联合收敛检查；不得通过拟合径向scale消除残差。详见主线最新记录。
界面参考已通过质量世界3D记分、两个材料缓冲厚度及两个seed稳定性检查。
旧纵向候选的跨界留源错误已复现；显式质量厚度/均匀出生修复仍不能通过
组织→空气物理门禁（约+2.31%），默认纵向模式已拒绝异质网格。
独立均匀材料电子联合响应已接入隔离GPU诊断，双缓冲厚度、独立seed及
0.1mm步长下，界面深度窗口误差均<1%、单bin<2%。但3D横向检查仍失败：
组织→空气首2mm RMS约2.38mm，TOPAS为3.44mm，不能以深度积分替代界面正确性。
上述2.38mm是旧净位移候选的历史结果，已由逐段候选实验推进，不能当作当前结果。
未更换正式包。P0仍IN_PROGRESS，患者配对与full20未准入，Gamma改善未证明。
以下低能追踪日志是历史记录，不覆盖这个新优先级。证据见
`evidence/step-31/air-tissue-interface/`。

## 2026-09-06 审查修正（优先于下方历史进度）

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

175 MeV/u 独立留出 pilot 已完成（Slurm 2353/2354；两参考 seed，各20k）：
固定 scale=1、参考空气密度，2–20 mm 误差 +3.944%→+0.547%，
候选 5 mm 分箱最大绝对误差0.590%，预设 pilot 门通过。候选仍为未验收，
仅增加一个域内插值点证据，不解除密度/界面/joint-response 门禁。
下一步：先补 HU=-975/-951 参考密度响应，不重新启用源密度外推。
证据：`evidence/step-31/longitudinal-holdout-175/`。

后续更新：共享 200 MeV/u writer 失败已定位并修复为横向 δ-tail 的深度诊断
同步缺失（不是 3-D 剂量物理错误）。六个 1k CUDA 回归全部输出闭合，包括
200/120 MeV/u 开关、斜向与贴边；域外 raw 与修复前逐字节一致。
下方关于该 writer 阻塞的描述保留为历史。真实响应的独立物理验证仍 BLOCKED，
原先有能散小样本的 physical-residual 门禁未声称解决。无新患者 Gamma。
见 `evidence/step-31/longitudinal-review-20260906/depth-mirror-fix.md`。

数值安全修复已实现；通用纵向物理响应仍 BLOCKED，本方案未完成。
`693655e` 的三病例结果使用不同 scale（0.63/0.75/0.90），属于已标定候选结果，
不是独立验证；RT06423 G30 下降约 0.028 个百分点，不应称为持平。
当前候选仅允许 smoke、scale=1，quality 明确拒绝物理验收；150–225 MeV/u
域外不搬运并记账，只允许参考密度的均匀 section-0 内精确体素路径分段。
异质界面余量留在源体素仅是诊断限制，不是界面输运修复。v2.1 最低数据栈不变。

真实 host/device 数值测试和 120 MeV/u 域外开关剂量逐字节等价已通过。
200 MeV/u 候选开/关均仍触发相同 depth-bin writer 闭合失败，不能称端到端通过。
本轮无新患者 Gamma，无 TOPAS 战役。下一步先复现并定位共享 writer 失败，
再补独立能量、密度、界面及纵横联合响应验证；禁止据肺重分箱结果直接改核心 scorer。
详细证据见 `evidence/step-31/longitudinal-review-20260906/`。

## 0. 任务目标、执行范围与进度控制

本文件按用户要求完整替换原 plan2/README.md。原 README 的旧任务描述和旧进度表不再作为本方案的完成证据；plan2 中其他文件不在本次清理范围。

目标：进一步改善 RT06423 的 Global / Local 1%/1mm 和 Global / Local 3%/0mm 一致性。

核心原则：先把电子响应提取验证完整，再做局限于 primary C12、Schneider section 0 的最小修正。不要直接给现有横向偏移加一个经验纵向偏移。

当前证据支持“纯横向 δ-electron 重分配遗漏了纵向迁移”，但不支持“已经获得可以直接用于 GPU 的通用纵向响应核”。

工作分两段：

1. 完成独立电子响应提取、逃逸闭合和跨条件验证。
2. 前置门禁通过后，实现 GPU 候选修正和配对 Gamma 验证。

第一段失败，不得进入第二段。不允许通过调全局剂量比例、坐标、beam、MCS 或 nuclear package 参数提高 Gamma。

### 0.1 本方案的状态定义

- TODO：本方案中的实施与验收尚未完成。已有旧工具或旧结果不等于本步骤完成。
- IN_PROGRESS：正在实施，必须列出剩余工作。
- BLOCKED：前置门禁失败或缺少外部依赖，必须列明具体原因。
- INCONCLUSIVE：统计量不足，不能判定通过或失败。
- DONE：实现、测试、真实运行和证据审查全部满足本步骤要求。
- FROZEN：受保护的基线，不是允许随意替换的默认候选。

不得依据本文件的创建动作将任何实施步骤标成 DONE。

### 0.2 进度表

| Step | 状态 | 交付内容 | 前置条件 |
|---|---|---|---|
| 01 | DONE | 当前工作树、executable、配置和输入基线冻结 | 无 |
| 02 | DONE | 电子审计拒绝路径与逐事件/逐家族检查 | 01 |
| 03 | IN_PROGRESS | 独立 Slurm 执行器及三密度 pilot 已闭环；外部 DICOM 输入冻结、完整条件元数据仍待补齐 | 02 |
| 04 | IN_PROGRESS | v3 准确产生-step 绑定已实现并通过 12-history 配对；完整记录/统计流程仍待验收 | 03 |
| 05 | DONE | 能量分配契约，排除双重计算 | 04 |
| 06 | INCONCLUSIVE | 原 L1 混入总量；修正后须对固定出生群重新验证几何收敛 | 05 |
| 07 | BLOCKED | 三个 section-0 HU 各 1-history pilot 完成；正式密度实验仍缺固定出生群和几何收敛 | 06 |
| 08 | BLOCKED | 候选联合响应数据编译 | 出生态、同材料密度实验及几何收敛门禁未满足 |
| 09 | TODO | 独立验证入口及互斥路由 | 08 |
| 10 | TODO | GPU 最小重分配实现与单元测试 | 09 |
| 11 | TODO | 独立 phantom A/B/C | 10 |
| 12 | TODO | 患者 50k smoke 与配对单 shard | 11 |
| 13 | TODO | 冻结口径严格 Gamma 与失败点归因 | 12 |
| 14 | TODO | full20 准入判断及完整验证 | 13 通过准入门禁 |
| 15 | TODO | 正式数据升级、证据与分离提交 | 全部必要验证完成 |

Step 编号为本方案局部编号，不覆盖 plan/ 下原有 Schneider workstream 编号。

### 0.3 第一批执行指令

先只执行 Steps 01–03。

第一批必须交付：

1. 当前基线与工作树清单。
2. 加强后的审计及失败测试。
3. 可重现的小分片运行、验证和聚合工具。
4. 明确的磁盘预算与资源预算。
5. 两个已有小分片的聚合验证结果。
6. 下一批父出生条件提取的字段设计。

这一批不得修改 GPU 物理，不得生成正式新表，不得启动完整患者运行。先报告本批验收结果，再按后续门禁推进。

## 1. 必须继承的当前状态

### 1.1 冻结完整 20-shard 基线

剂量：

    /mnt/sda/wuwei/rt06423_delta_tail_escape_full20/aggregate/dose_delta_tail_escape_full20.raw

证据：

    evidence/step-31/delta-tail/strict-gamma-summary.json

冻结名义归一指标：

| 指标 | 通过率 |
|---|---:|
| Global 1%/1mm | 98.426% |
| Local 1%/1mm | 87.786% |
| Global 3%/0mm | 99.804% |
| Local 3%/0mm | 93.422% |

不得拿新的单 shard 与这些完整统计量直接作修复优劣判断。

### 1.2 当前未提交入口 mask 候选

src/transport_sycl.cpp 已有入口 mask 修复：

- 原先漏掉 scorer 最外层 voxel。
- 现在检查存在的邻居，不再无条件排除整个网格外壳。
- 未修改输入物理包；其他已有材料接口处理仍保留。

配对同种子单 shard：

    Global 3%/0mm：98.000% → 98.140%
    Global 1%/1mm：98.136% → 98.138%
    Local 1%/1mm：83.840% → 83.822%

只允许得出：

    入口局部误差改善；
    严格 Gamma 并非所有指标改善；
    尚未通过新的 full20 验证。

### 1.3 当前诊断文件

    startup/extensions/CarbonElectronDepositNtuple.hh
    startup/extensions/CarbonElectronDepositNtuple.cc
    tools/analyze_electron_deposit_steps.py
    tests/test_analyze_electron_deposit_steps.py
    evidence/step-31/entrance-mask-candidate/validation.json
    evidence/step-31/entrance-mask-candidate/longitudinal-diagnostic.md
    evidence/step-31/entrance-mask-candidate/binary12-joint-escape.json

### 1.4 已验证电子闭合

TOPAS job 2338：

    12 histories
    C12：200 MeV/u
    能散：1%
    HU：-1000
    均匀 slab
    EM-only
    3D DoseToMedium

结果：

    电子根出生能量：146.739176590 MeV
    家族沉积：      143.944623071 MeV
    家族逃逸：        2.794553519 MeV
    出生 = 沉积 + 逃逸

家族相对残差约 1e-16。它证明记录与能量审计正确，不证明统计量充分，也不证明响应可泛化。

Binary 与同 seed ASCII 的 3D dose 数值完全一致。ASCII 曾把同一逃逸末步两个不同 z 坐标都舍入成 220 mm，无法证明方向向外；检查没有放宽，改用 Binary 后通过。

原始数据：

    /mnt/sda/wuwei/delta_longitudinal_audit/

当前 12 histories 的 ASCII 约 233 MiB，Binary 约 123 MiB。二进制只缩小记录，不等于已经解决大规模输出问题。

## Step 01：冻结当前工作树与验证入口

### 修改与检查范围

只读检查：

    git status --short
    git diff --check
    git diff -- src/transport_sycl.cpp
    python3 tools/verify_schneider_v2_1_data.py
    python3 tests/test_analyze_electron_deposit_steps.py

读取：

    AGENTS.md
    plan/README.md
    plan2/README.md
    evidence/step-31/entrance-mask-candidate/

记录：

- HEAD。
- 工作树是否干净。
- tracked diff。
- 本轮相关 untracked 文件清单。
- GPU executable SHA256。
- 配置 SHA256。
- TOPAS executable SHA256。
- physics bundle 和各输入 SHA256。
- 当前证据与实际 executable 是否对应；无法对应的证据标明限制。

在本 README 更新进度与执行日志，不篡改旧证据。

### 禁止事项

- 不清空其他 plan 文件或证据。
- 不删除 untracked 大包或 scratch。
- 不为了 clean 状态隐藏差异。
- 不自动 push。
- 不把既有用户改动归入自己的修复提交。
- 不把当前 HEAD 当成未提交工作树实际使用的代码版本。

### 验收

必须能回答：后续每个 A/B/C 组使用哪个 executable、配置、数据版本及源代码状态？

任何一项不能追溯，先补记录，不启动新计算。

## Step 02：加固电子审计及拒绝路径

### 修改文件

    tools/analyze_electron_deposit_steps.py
    tests/test_analyze_electron_deposit_steps.py

### 必须保持的检查

- header 列顺序、histories、entries。
- 二进制记录大小与实际字节数。
- 缺失祖先。
- step ID 连续性。
- 出生覆盖。
- terminal 是否位于外边界。
- terminal 是否向外。
- 3D 剂量与逐步沉积闭合。
- 电子家族出生、沉积、逃逸闭合。
- 非支持粒子、非单位权重拒绝。

不能把失败降级为 warning 后输出成功报告。

### 新增检查

逐 track 验证所有记录中以下字段一致：

    run / event / track / parent / PDG
    birth position
    birth KE

另外：

1. 每事件 primary 数符合本实验定义。
2. 同一 track 没有重复 step。
3. 全局闭合不能掩盖事件间正负抵消。
4. 输出逐事件残差。
5. 输出逐电子家族残差及最差家族身份。
6. 非有限 KE、负 KE 明确拒绝。
7. JSON 不允许 NaN 或 Infinity。
8. 未计算字段写 null 并说明原因，不写 0。

### 限定适用范围

当前逃逸审计仅适用于：

    均匀 box
    真空外部
    无再入
    EM C12/electron/photon
    单位权重

不得宣称支持真实 CT 边界。

### 验收

新增每项失败测试，运行整个分析器测试集。用已有 binary12 重算，既有物理结果不变；新增门禁没有被绕过。

## Step 03：可扩展记录、分片及聚合

### 实施顺序

先采用：

    每片最多 12 histories
    Binary 输出
    每片独立输出目录
    完成 → 验证 → 聚合

先实现自动化分片及聚合，不直接启动上千 histories 的单文件全步记录。

建议新增：

    tools/run_electron_response_diagnostic.py
    tools/merge_electron_response_diagnostics.py

上述是待实现名称，不得把未实现命令写成已运行命令。

### 每片元数据

    唯一 case ID
    唯一 seed
    请求和实际 histories
    Slurm job ID
    配置 SHA
    TOPAS executable SHA
    scorer 源码 SHA
    原始文件路径、大小、SHA
    完成状态
    分析状态

必须预先定义磁盘预算并由工具执行。达到预算后停止提交新任务，保留已完成数据，报告当前统计量。不得自动删原始数据。

### 聚合算法

不能平均分片分位数。先合并：

- 原始加权直方图。
- 出生、沉积、逃逸能量。
- 对应分母。
- 按独立 history 或独立分片组织的统计量。

再计算概率、分位数与误差。不得把 electron steps 当成独立 histories。

### 验收测试

1. 分片顺序打乱，结果不变。
2. 同片重复输入，拒绝。
3. 缺片或失败片，不输出完整 campaign PASS。
4. SHA 不匹配，拒绝。
5. 不同配置误混合，拒绝。
6. 聚合能量等于各片之和。
7. 工具确实执行资源与磁盘上限。
8. 任一失败时退出码非零，报告不伪装成成功。

两份已有小分片足以测试聚合代码，不足以证明物理统计收敛。

## Step 04：补齐父 C12 出生条件和记录版本

### 当前缺口

已有记录能得到电子 birth KE、birth position、沉积 pre/post position、parent/track ancestry。

编译可输运响应还需可靠绑定：

    父 C12 产生电子时的能量
    父 C12 产生电子时的方向
    出生材料与密度
    电子 creator process

不能把所有电子标成名义 200 MeV/u 而忽略实际慢化。

### 修改方案

在现有 TOPAS extension 上新增带版本记录格式，或新增独立 scorer。不得让旧 21 列文件静默按新格式解析。

元数据必须包含：

    schema version
    TOPAS / Geant4 version
    physics modules
    production cuts
    step limits
    材料组成与密度
    几何尺寸
    3D scorer 尺寸
    源位置、方向、能量、能散
    seed

### 父方向定义

使用出生时父 C12 方向定义：

    longitudinal = 位移沿父方向的投影
    radial = 位移垂直父方向的模长

明确方向来自哪个 Geant4 step 状态。不能假定父粒子永远沿世界 +z。增加合成方向和旋转测试。

### 验收

- 同物理、同 seed，新旧 scorer 的 3D dose 一致。
- 沉积与逃逸闭合通过。
- missing-parent 为零。
- 未知 schema 明确拒绝。
- 旋转测试通过。

只修改观察与记录，不修改物理。

## Step 05：能量分配契约，防止电子效应双算

### 当前实现事实

transport_sycl.cpp 目前采用：

    moved energy = deposited_MeV × moved_fraction

再按横向 radius 与随机方位角移动。

旧 v1 moved_fraction 描述提取到的横向尾部，不等于全部电子出生能量比例。

### 明确禁止

    旧横向修正保留，再额外搬走 34%。
    把“电子沉积的 28.9% 前移超过 0.5 mm”当成“全部 stopping 的 28.9%”。
    直接把旧 moved_fraction 改为 0.34。

### 必须输出的能量项

    父 C12 电子过程能量损失
    父 C12 局部沉积
    显式电子根出生能量
    电子家族沉积
    电子家族逃逸

明确 production cut 以下能量归属，建立并验证对应闭合关系。

检查 GPU stopping 对应总电子能损还是 restricted stopping，给出代码、表定义或提取实现证据，不凭变量名判断。

### 交付文档

新增 energy_partition_contract.md，回答：

1. 新模型从哪个现有能量项取能量？
2. 可迁移比例分母是什么？
3. 原局部沉积减去多少？
4. 新位置增加多少？
5. scorer 外能量如何记账？
6. 与旧横向模型是否互斥？
7. 与显式 electron transport 是否互斥？
8. 是否重复施加 straggling？

任一问题未解决，不得改 GPU。

## Step 06：有限 slab 偏差和几何收敛

### 必须理解

有限 slab 内实际沉积的响应不等于无限均匀介质响应核。离开 slab 的电子没有提供后续完整沉积位置，不能把逃逸 KE 随便分配到某个终点。

### 实验

先固定 200 MeV/u、HU=-1000：

1. 原始 slab。
2. 横向尺寸增大。
3. 纵向尺寸增大。
4. 出生位置远离外边界的内部样本。

其余设置不变。在相同出生条件下比较：

    电子出生能量谱
    逃逸比例
    joint radial/longitudinal 分布
    总可迁移能量比例

不得比较不同出生能谱的无条件分布后，把差异全部解释为边界效应。

### 停止条件

几何扩大后分布仍明显变化，不生成 runtime kernel。继续定位未包含尾部，或保留为有限几何诊断。

不能删除远尾 bin 使分布看起来收敛。

## Step 07：独立能量与密度验证

### 训练/验证分离

第一版范围：

    C12
    Schneider section 0
    150–225 MeV/u

建议训练节点：

    150 / 200 / 225 MeV/u

至少保留一个内部能量作 held-out 验证，例如 175 MeV/u。验证数据不得参与参数拟合。

### 密度

section 0 中选三个不同 HU：

- 接近低端。
- 居中。
- 接近上边界但仍属于 section 0。

具体值从 parser 的真实边界取得，不靠记忆硬编码。各 HU 密度使用相同 Schneider 公式。

### 缩放假设

不能直接假定位移与 1/rho 成正比。可以测试该假设，但必须验证：

- 纵向分布。
- 横向分布。
- 横纵相关性。
- 逃逸比例。
- 总能量。

不成立时不能用单密度表强行覆盖整个 section 0。

### 验收标准

验证前冻结以下内容：

    观测量
    ROI
    统计误差方法
    允许差异
    统计不足的处理

不准看结果后放宽阈值。保留现有能量闭合和零 overflow 硬门禁。统计不足标 INCONCLUSIVE，不标 PASS。

## Step 08：编译候选联合响应数据

仅在 Steps 05–07 通过后开始。

### 命名与位置

不覆盖：

    data/schneider/schneider_section0_c12_delta_tail_v1.csv

候选建议：

    schneider_section0_c12_electron_response_v2_candidate.*

放独立实验目录，不修改正式 bundle pins。

### 必须包含

    schema version
    supported projectile / section
    energy / density domain
    energy partition definition
    joint radial/longitudinal distribution
    bin edges / sampling definition
    normalization
    tail / overflow / escape treatment
    完整 provenance

### 采样

不能分别独立抽 radius 和 longitudinal displacement，必须保留联合相关性。

节点间插值策略明确定义并测试，不能机械复用旧一维 quantile 插值。

### 编译器硬失败

以下任一项出现则拒绝：

- 非有限值或负概率。
- 错误归一或缺能量节点。
- 不一致 composition。
- 超出声明范围。
- joint histogram overflow 未解释。
- 缺 raw provenance。
- 重复 campaign 数据。
- 未通过能量分配契约。
- 用有限 slab 逃逸值冒充通用终点分布。

## Step 09：先建独立验证入口

### 预计涉及

    include/carbon/schneider_delta_tail.hpp
    src/schneider_delta_tail.cpp
    或新增独立 electron_response 类型
    src/config.cpp
    src/transport_sycl.cpp
    diagnostics / IO
    tests/

优先独立类型，避免改变旧 v1 文件语义。

### 配置

显式区分：

    已验证 transverse-v1
    候选 joint-v2

同一份能量只能走一个模型。

候选只能从显式验证入口加载：

- 不静默替换正式默认值。
- 不降低 v2.1 nuclear/stopping 最低要求。
- 不删正式 verifier SHA 检查以兼容候选。
- 候选损坏时失败，不退回 water/旧包。
- 正式升级仍遵守 AGENTS.md 的完整门禁。

### 第一版最小范围

    primary C12
    Schneider section 0
    验证过的能量和密度范围

不扩 secondary p/He、其他 section、water 或核反应终态。

## Step 10：GPU 重分配和测试

### 能量约束

本步能量损失确定后：

    本步能量
    =
    保留的局部沉积
    +
    分配到其他体素的能量
    +
    离开 scorer 的能量

每份能量只记一次。不要无故改变粒子慢化、hazard、survival、secondary queue、straggling、MCS 或 beam RNG。

新 RNG tag 使用前审计冲突，给出明确映射。

### 坐标

    destination
    =
    source
    +
    longitudinal × parent_direction
    +
    radial × transverse_direction

测试 +x、+y、+z、-z、斜方向，不只测世界 +z。

### 材料和边界

两端都是空气不代表中间无组织。不能未经审计照搬旧模型的终点 section 检查。

明确处理经过其他材料的路径。若候选不支持：

- 显式计数。
- 单列能量。
- 不声称 heterogeneous 完成。
- 不静默跨组织使用均匀空气响应。

离开 scorer 与跨入其他材料不得混为一类。

### 必须测试

1. 候选关闭时原逻辑不变。
2. water 路径不变。
3. 非 section-0 不变。
4. 不支持能量/密度策略明确。
5. host/device lookup 等价。
6. 联合采样统计吻合。
7. 坐标旋转正确。
8. local/moved/escaped 分项正确。
9. voxel/in-grid 闭合。
10. charged-origin voxel 闭合。
11. 材料交叉计数正确。
12. RNG tag 无冲突。
13. overflow 为零。
14. v1 路径回归通过。

### depth / LET 语义

3D dose 为评价依据，IDD 从 3D 横向求和。审查旧 depth tally 是否仍表示同一物理量。LET 未同步处理时明确标记限制，不得输出语义不一致却不说明的指标。

## Step 11：独立 phantom A/B/C

### 组别

    A：冻结 transverse-v1
    B：transverse-v1 + 入口 mask 修复
    C：B 的代码基线 + joint 候选，替代旧横向重分配

不能把 A→C 的全部提升归因于纵向修复。

### 固定条件

    源
    histories
    seed
    几何 / 网格
    stopping
    核反应设置
    MCS
    能散
    归一

仅声明的候选差异允许变化。

### 顺序

1. 均匀 section-0、EM-only。
2. 入口和出口附近。
3. 内部平衡区。
4. 更大几何。
5. 不同能量。
6. 不同密度。
7. 倾斜入射。
8. 异质界面诊断。

前项失败不跳患者。

### 报告

    3D 总能量
    入口各层横向积分
    中心和外侧 ROI
    纵向 / 横向响应
    逃逸能量
    能量闭合
    primary survival（若核开启）

不能只提交看起来重合的曲线。

## Step 12：患者 50k smoke 与完整单 shard

### 50k

本地 RTX 2080 Ti，运行前：

    python3 tools/verify_schneider_v2_1_data.py

要求：

    accepted=true
    overflow=0
    lookup failure 不增加
    born 守恒
    能量闭合通过
    voxel/in-grid 闭合
    新近似项有计数和能量

overflow 必须拆分重跑，禁止拿溢出剂量做 Gamma。

### 完整单 shard

使用冻结 shard01 的 spots、实际 histories、seed、配置、映射和 Gamma 参数。

复制冻结配置，仅修改声明的候选开关与输出路径。不要手工重建“差不多”的配置。

A/B/C 必须相同样本量。

## Step 13：严格 Gamma，固定评价口径

继续使用冻结的：

    TOPAS reference
    10% reference threshold
    50,000 sampled points
    sample seed 42
    坐标映射
    网格间距
    名义 histories 归一

同时输出 Global/Local 1%/1mm 与 Global/Local 3%/0mm。

### 3%/0mm 的约束

零距离就是同一空间位置剂量比较，禁止邻域搜索、隐式平移、非零距离替代或重新最佳配准。

### Paired turnover

相同点统计：

    原失败 → 新通过
    原通过 → 新失败

按入口/内部/出口、section、剂量带、primary/secondary origin 定位。不能只报净通过率。

### 不确定性

采样点有空间相关性。不能把 50,000 点当成完全独立 Bernoulli 样本来宣称显著性。

结合配对变化、深度或空间 block 统计，必要时第二 GPU seed。新增统计方法也应预先冻结，不随结果选取。

## Step 14：full20 准入和完整验证

必须同时满足：

1. 独立 phantom 通过。
2. 能量分配契约通过。
3. 50k 通过。
4. 单 shard 无 overflow。
5. 严格 Gamma 有可重复改善，或满足预先冻结的非劣标准。
6. 无新未解释能量项。
7. 未改变评价口径制造提升。
8. 候选输入与 executable 已冻结。

若入口改善但 Local 1%/1mm 变差、改善小于运行波动或异质边界出现问题：

    保留候选；
    报告未通过；
    不自动 full20；
    不调一个 scale 补救。

通过准入后，full20 继续按本地 GPU 分 shard 运行；全部 shard 单独通过，零 overflow，再合并并验证完整统计量。不要只因准入通过就将本步骤标 DONE。

## Step 15：正式升级与提交

全部必要验证通过后才允许候选成为新的已验证 electron-response 版本。

按 AGENTS.md 更新：

    manifest
    bundle pins
    metadata
    安装验证器
    最低已验证版本说明
    evidence

不能先修改 AGENTS.md 宣布候选为最低版本，再倒过来跑验证。

建议按实际完成范围拆分提交：

    1. test(diag): harden electron ancestry and escape audits
    2. feat(diag): add reproducible binary response campaigns
    3. feat(topas): record parent-conditioned electron response
    4. feat(data): compile validated joint electron response candidate
    5. feat(ct): add gated section-0 joint redistribution
    6. test(ct): validate independent response phantoms
    7. test(ct): record paired strict-gamma validation
    8. feat(data): promote validated response stack

不为凑列表制造空提交。未完成步骤不提交“完成”证据。未经用户明确要求不 push。

## 全程禁止事项

- 不修改 nuclear package 来修电子纵向响应。
- 不修改 stopping 总量来补空间分布。
- 不调 MCS scale、beam sigma、能散、全局剂量 scale。
- 不改冻结坐标。
- 不把 12 histories 当充分统计量。
- 不把 steps 数当独立 histories。
- 不把有限 slab 终点分布直接当无限介质响应。
- 不独立抽 radial / longitudinal 而丢掉相关性。
- 不让 v1 横向与 v2 联合模型重复作用于同一能量。
- 不把硬失败降为 warning。
- 不把未运行的命令或测试写成 PASS。
- 没有新 GPU run 就没有新 Gamma 结论。
- 不推断授权去删除数据或扩大到无关物理。

## 资源和数据规则

AGENTS.md 是必须遵守的仓库约束，最低 v2.1 stack 不得降级、alias 或静默回退。

    TOPAS：本地 sbatch，数据 /mnt/sda/wuwei
    TOPAS extension、源码与 build：/home/wuwei/topas
    全部 TOPAS 任务合计 CPU ≤192
    全部 TOPAS 任务合计内存 ≤160 GB
    根据计算量按比例分配资源
    InvalidAccount 短暂出现时等待 1–3 分钟复查
    GPU：仅本地 RTX 2080 Ti / sm_75，沙盒外
    禁止远程主机或集群 GPU
    3D scorer；IDD 通过横向求和
    overflow 时拆分并重跑

不讨论通过 FP64 替换 FP32 提高本轮精度，不偏离空间响应修复任务。

## 每批交付模板

每批结束必须提供：

1. 实际修改文件及作用。
2. 未修改的受保护物理路径。
3. 实际执行命令和退出状态。
4. 测试结果及新增失败测试。
5. 数据路径、hash、job ID、资源。
6. 各硬门禁结果。
7. 未完成项、统计不足项和阻塞原因。
8. 对应本表状态更新。
9. 下一步是否满足启动条件。

代码、测试、证据分别报告。能量闭合不等于空间分布正确，单材料正确不等于真实 CT 正确，单 shard 改善不等于 full20 改善。

## 执行日志

- 2026-09-05 续修（以下更新优先于历史 DONE/未实现描述）：
  - 独立 `tools/execute_electron_response_campaign.py` 已实现本地 Slurm 顺序提交、已有用户任务 CPU/内存检查、协作锁、磁盘轮询、实际 job ID、sacct 退出状态及分析状态落盘；失败只取消本执行器尚在运行的任务，保留 raw。生成器 `--submit` 仍拒绝，不绕过监控。磁盘轮询不是硬配额，其他提交器与资源快照之间仍可能竞争；不支持自动恢复或不明确提交的自动重试。
  - Job 2350/2351/2352：HU -1000/-975/-951，各 1 history、seed 918001、2CPU/4GiB，顺序完成，COMPLETED/0:0。初次 `--mem=4.0G` 被 Slurm 拒绝（无实际 job），已修为 `4096M`，失败目录保留，新数据在 `/mnt/sda/wuwei/electron_density_pilot_v3_r2/`。
  - 三点实测密度与含 correction 的公式相对误差 <3.1e-6，均为 material section 0；v3 产生步绑定全部通过。3D dose/step 能量残差 <2.2e-9。
  - 发现并修复审计漏洞：逐事件电子出生能量漏加导致残差 null；逐家族只报告而不拒绝超限。现在每 root 只记一次 birth，逐事件/家族均以 1e-3 门禁拒绝。新增跨事件/同事件家族误差抵消及 descendant 不重复 birth 回归；三份 raw 在加强门禁下另存 `analysis_closure_verified.json`，未覆盖原报告和输出。
  - 最终相关 Python 测试 50/50，v2.1 verifier 16/16。原始文件 SHA、job 信息、原/新报告及限制记录在 `evidence/step-31/entrance-mask-candidate/section0-density-pilot-v3.json`。当前执行器源码含运行后补强，不能冒充 pilot 时的冻结源码。
  - HU -975 单 history raw 约 50.5MiB，替换原固定 320MiB/片估计为 `(128 + 64 × histories) MiB`（12 histories 为 896MiB）；这是预算估计，不是随机输出大小上界。
  - 03/04 仍 IN_PROGRESS：外部 DICOM 预加载输入未逐文件冻结，实际 production-cut/step-limit 元数据与完整统计流程未验收。06/07/08 不解锁；1 history 不构成缩放验证，不制表。本轮未改 GPU 物理、未换包、未运行 Gamma/full20、未提交或 push。
- 方案写入：按用户要求完整替换 plan2/README.md，仅重写本文件。已有实现和诊断作为输入基线，不据此将新实施步骤标 DONE。
- 2026-09-05 第一批 Steps 01–03 完成：
  - 01 DONE：HEAD c6fe6b44；工作树非干净（tracked：plan/README.md、plan2/README.md、plan2/steps/13删、src/transport_sycl.cpp入口mask未提交；untracked：v2.1 bin包、electron审计工具/测试、entrance-mask证据、scratch）。verify_schneider_v2_1_data.py通过16/16。GPU exe：build/carbon_mc bfd03552（mask前陈旧）、build/oneapi-nvidia-release/carbon_mc 6ae10bb7（与当前mask源码同日构建）；maskfix dose sha 0ee4457a与validation.json一致但validation.json未记录exe SHA，标为对应关系未知限制。TOPAS exe /home/wuwei/topas/topas-build/topas e1f5ccc0。配置 split20_rt06423_strict_01_v2_1.yaml d756293f。bundle pins按manifest验证通过。冻结full20基线与§1.1名义指标不变；后续A/B/C必须复用上述冻结exe/配置/数据版本并先补对应记录。
  - 02 DONE：tools/analyze_electron_deposit_steps.py新增逐track一致性、单事件单primary、重复step、逐事件残差（防抵消）、逐家族残差+最差家族、非负有限KE、JSON allow_nan=False、未计算slab审计写null。tests 9→17全过；binary12重算物理量与旧JSON完全一致（birth146.739/dep143.945/esc2.795，forward>0.5mm 28.878%），逐事件最大残差4.2e-15，最差家族(0,10,99)1.55e-15。
  - 03 DONE：新增tools/run_electron_response_diagnostic.py（≤12 histories/片、binary、独立目录、config/TOPAS/scorer SHA、CPU≤192/Mem≤160G/磁盘预算预检、预算超限拒交、不删数）与tools/merge_electron_response_diagnostics.py（先合加权直方图/能量/分母再算概率、不平均分位数、steps不计histories、顺序无关/重复/缺片/SHA/混配置/能量和/预算/非零退出8门禁）。tests/test_electron_response_campaign.py 4项通过。smoke12+seed2聚合24 histories能量求和一致、joint守恒、顺序无关、pooled分位数保持null。/mnt/sda/wuwei/delta_longitudinal_audit现16G（含2335取消14G残留），binary约123MiB/12histories，ASCII约233MiB；新campaign演示预算5GiB/估计0.31GiB/2片。本批未改GPU物理、未生正式新表、未启动患者运行；下一步Step04父出生条件需先过门禁。

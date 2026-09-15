# 当前 package 代码消融

基线提交：8d9ab4d。状态：前四批完成并通过配对；不是全部旧模型代码清理完成。

## 第五批：恒空诊断生命周期删除（候选，完整记账门禁未过）

- 删除56个恒空诊断调用、旧host镜像/回读/零赋值循环，以及27个只剩声明/free的诊断设备指针及尺寸常量。逐一检查不存在分配路径；不删除真实species能量/终止账本、grid或Schneider诊断。
- 旧JSON字段暂保留，由TransportResult默认零初始化提供。输出schema删除是独立后续工作。
- 独立build/oneapi-nvidia-legacy-ablation-diagnostics/carbon_mc，SHA 2a7bbd194eafbc36ed9af2d03a94de91df2e1c4cdefefed6774773796225c6bf。
- 增强compare_package_ablation.py：cinel02诊断/schema/终止字段精确保持；species/grid能量rtol=2e-6、atol=1e-3。3项Python反例/源码防回退测试通过；Step05/20、统一配置、CT开关、bracketing/gap及真实material rates查询测试通过。数据verifier 16/16。
- water10k：/tmp/maigo_ablation_water_diagnostics_20260907，全配对门通过，剂量SHA仍92f1f91b...。
- CT50k：/tmp/maigo_ablation_ct_diagnostics_20260907，运行质量仍仅有原电子实验拒收，零overflow。dose逐位一致（87abdbe6...）、核计数一致、55个非species旧字段精确一致、grid两项能量精确一致。但新增species比较失败：数组索引14（氘核continuous_deposit_all）1752950.125→1752953.75，+3.625MeV、2.06794ppm；索引15（continuous_deposit_fov）1292510.75→1292508.125，−2.625MeV、2.03093ppm。不得宣称第五批全部通过。
- 独立baseline重复CT50k：/tmp/maigo_ablation_ct_diagnostics_control_20260907，同binary/config/seed，dose逐位一致；最大species相对差1.10256ppm。汇总所有6个冻结baseline重复（v3/kernel/loaders/branches/diagnostics/baseline-control）：最大成对波动1.52735ppm；索引14/15的范围分别1.25/0.875MeV。候选两槽差异仍超此次观察范围，暂未放宽2ppm门限。
- 候选代码留在工作树，未替换默认二进制，未commit/push。下一步先解释/处置这两个诊断槽的极小差异，或明确独立标定诊断容差；不能通过挑选重跑或静默放宽门限将此批记DONE。没有剂量/Gamma变化证据，没有新full20。

## 第四批：删除固定关闭的 kernel 分支（完成）

- 实质删除约485行CINEL02旧primary靶核选择/replay、次级H/O率查询、旧次级hazard选择、旧连续损失诊断查询和strict-match分支。保留各处原false分支、共享species账本和队列，不改变求和/RNG/能量逻辑。
- 移除use_cinel02模式变量及所有引用，相关三元表达式固定为原false值。不是仅靠constexpr隐藏旧主分支。
- 源码防回退测试禁止use_cinel02、旧package加载和旧靶核选择调用返回runtime，同时确认共享secondary_rates/species账本仍存在。
- 独立build/oneapi-nvidia-legacy-ablation-branches/carbon_mc，SHA 6b0a771254fdd39d1c0a0c123043da492f1a0b269ccec3099a47bc16eb733a6b。
- Step05/Step20、独立bracketing/gap拒收、统一配置与CT电子开关测试通过；真实material rates的10项版本拒收和921+12894个host/device查询通过；2项Python比较器/源码防回退测试通过。未声称完整ctest通过。
- /tmp/maigo_ablation_water_branches_20260907：water10k生产配置，与8d9ab4d基线dose逐位一致、核计数一致、overflow=0、accepted=true。
- /tmp/maigo_ablation_ct_branches_20260907：真实CT50k电子r3，与基线dose逐位一致、核计数一致、overflow=0；唯一拒收仍是既有电子实验门禁，未放宽。
- 两组dose SHA与前三批相同。无新Gamma/full20，无物理/package修改；未commit/push。

## 第三批：旧次级接口与 CINEL02 独占加载（完成）

- 删除 secondary_rate_table.hpp 的旧13同位素硬编码、旧总率插值、旧采样与legacy grid识别接口（约130行）。
- 删除 tools/run_step20_gpu.cpp（562行）及CMake目标；旧运行器可由Git历史恢复。
- Step05采样测试迁移到实际v2.1的14p注册表、masked partials与独立double插值概率；新增C12 registry与未知抛射体检查。
- Step20包测试迁移到v2.1：14p、masked totals、921节点；alpha+C@200由旧包缺口断言改为当前包必须命中。独立synthetic大gap拒收测试仍保留并实跑通过。
- 删除CINEL02专属package/rate加载、GPU上传与专属诊断buffer分配；保留块外共享species账本。旧selector现在在validate最前明确拒收，use_cinel02为constexpr false。源码中的剩余kernel死分支尚未全部物理删除。
- 删除加载块时编译抓到计时起点被连带删除，已原位恢复并重建；不影响物理。
- 构建 build/oneapi-nvidia-legacy-ablation-loaders/carbon_mc，SHA 0a9f54c322fd48ea2f7153347f9f8e8a8aaabd3e5d5a1c4e04a6e41ce2749423。
- Step05、Step20、真实material rates（含10项旧schema拒收及921+12894个host/device查询）、统一配置、CT电子开关、独立bracketing/gap测试全部通过。非完整ctest。
- /tmp/maigo_ablation_water_loaders_20260907：水10000粒子production，剂量逐位相同、核计数相同、零overflow、accepted=true。
- /tmp/maigo_ablation_ct_loaders_20260907：真实CT50000粒子含电子r3，剂量逐位相同、核计数相同、零overflow；保留既有且唯一的电子实验拒收，未放宽门禁。
- 剂量SHA与前两批完全一致。无新Gamma/full20，未改package与物理参数，未commit/push。

## 已删除

1. SCHNRATE / SCHN2RAT v1/v2 加载兼容、旧网格/端点钳位、弱 metadata 校验与旧 metadata 文件名候选。现在只接受当前核率 schema v3。
2. 配置、上传和 provenance 输出中的旧核率/事件包文件名回退；非 attenuation 的核输运要求显式当前 bundle 和 v3 率表。
3. GPU 次级 v1 注册表分流、硬编码13同位素回退、旧碰撞点 hazard/target sampler 分支、sec_rate_version 字段。

保留当前 CINPKG04 v4 事件包、SCHNRATE/SCHN2RAT v3 核率、SCHNSTOP v1 stopping；它们共同组成当前 v2.1 栈，不能按文件名中的 v1/v2 字样误删。保留纯水 H/O 材料适配，禁止伪造 Schneider 水分区。无 package 数值/物理参数修改。

## 验证

独立构建，均为 RTX2080Ti/sm_75，相同编译选项；未覆盖基线二进制。

- baseline: build/oneapi-nvidia-unified-default/carbon_mc，SHA c87ae63f7e22bc0f7e2c1dc39b016c2dd3ad681c2e9a3e703d68fe9450c5bf5e。
- 第一批: build/oneapi-nvidia-legacy-ablation/carbon_mc，SHA 4ce4631b8c8e92da72a922192d362cb21868a08b9d312eb0ac3ab469eb4bd587。
- 第二批: build/oneapi-nvidia-legacy-ablation-kernel/carbon_mc，SHA ae7973f51e110439418e18f42e3cfdd67c11587e928756fd14957db7672ed60f。
- 10个旧/非法版本header拒收检查通过；确认在版本检查处失败，而非依赖后续缺payload/metadata偶然失败。
- 当前真实数据：921 primary、12894 secondary非节点host/device查询通过，14 projectile registry保留。
- 统一water配置/production许可/缺靶拒收测试、CT电子YAML开关测试通过。
- A/B比较器反例通过：剂量变化、核计数变化、overflow均拒收。
- 两批均分别运行 water 10000 histories生产配置和 CT 50000 histories（现有电子r3开启）：dose.raw逐位一致，核整数计数一致，overflow=0。水 accepted=true；CT保留且仅有既有 unvalidated_electron_joint_response 拒收，未放宽电子验收门禁。
- 水 dose SHA: 92f1f91be19bf4247440ab840f7f456887176405eee8bc1ef711cf1ae47bc228。
- CT dose SHA: 87abdbe68a9527bf79892d5b7a1d24c2afb06352802344886b6adcc7d9dc4d21。

运行产物（本地临时目录，非Git产物；comparison.json记录二进制/配置SHA，子目录含manifest、日志、质量报告和剂量）：

- /tmp/maigo_ablation_water_v3_20260907
- /tmp/maigo_ablation_ct_v3_20260907
- /tmp/maigo_ablation_water_kernel_20260907
- /tmp/maigo_ablation_ct_kernel_20260907

可复用工具 tools/compare_package_ablation.py：--before/--after 指定冻结二进制；--config指定相同输入；--water使用水生产探针；CT --joint指定已有响应。输出目录必须全新。不可覆盖冻结结果；overflow必须拆分双方重跑。没有新Gamma/full20；逐位相同意味着这些配对输入在相同参考和Gamma定义下不会产生指标变化。

## 剩余工作（未完成）

1. transport_sycl.cpp中的use_cinel02主分流、独占加载/上传和旧主replay已删除。仍有旧诊断buffer/空指针记账、未支持几何的旧secondary replay备用块、FRED event-library加载/终态和相关配置/CMake依赖，需继续逐块核查删除。当前使用的FRED-2GR MCS表不是旧event-library，不能按FRED名称整批删除；也不能误删CINEL03共用的species账本、队列与终止记账。
2. 旧13同位素辅助函数和run_step20工具已删除，Step05/Step20已迁移。剩余旧rate常量/默认对象值及tests/test_secondary_rate_table_hardening.cpp等旧fixture仍需迁移，不能声称所有旧测试已经通过。
3. primary CSV attenuation诊断、旧CDF/四分类兼容路径及其单测：明确哪些是独立诊断，哪些是已退休package路径，再分别删除。不可擅自改变最新CT mask/采样顺序。
4. 清理 CMake 旧模型依赖和废弃配置键，保留清晰迁移报错；完整测试套件迁移尚未做，不声称全ctest通过。
5. 每批使用相同基线/当前package做水与真实CT逐位A/B；最终再统一提交。当前批次尚未commit/push。


2026-09-15 更新：旧事件库加载/回放、paper 独立截面选项及 2GR 已退役；上述旧清理边界已过时。随后按用户要求移除了解析碎裂代码、经验参数和专用账本，原发 post-EM null 现在保留轨迹并完成 EM 记分。18 离子索引与指数碰撞距离使用独立通用模块；当前 CINEL03 队列及能量账本保留。范围与验证见[清理报告](../docs/fred_cleanup.md)。

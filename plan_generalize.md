  # 第一阶段：Primary Ion 通用化

  ## Summary

  将 primary C-12 从传输算法中抽离为显式离子定义，同时保持现有工程名、carbon namespace 和 CRPKG/CCAS v1 格式不变。

  验收包含：

  - 现有 C-12 计算在字段映射后数值逐项一致。
  - 使用合成 proton 数据完成 EM、非弹性反应、final state、secondary queue 和 cascade 全链 smoke test。
  - Package 编译不再限定 INCL++ 或固定 TOPAS/Geant4 版本。

  ## Public Interfaces

  采用破坏式 YAML 升级，不保留旧字段：

  primary_atomic_number: 6
  primary_mass_number: 12
  # 可选；省略时严格使用 A * 931.49410242 MeV
  primary_rest_mass_MeV: 11177.929228

  primary_stopping_power_file: data/...
  primary_inelastic_cross_section_file: data/...
  primary_reaction_package_file: data/...
  cascade_package_file: data/...

  - 删除 mass_number 和 tps_particle_type。
  - 重命名 stopping_power_file、nuclear_cross_section_file、reaction_package_file 为上述 primary 专用名称。
  - 要求 Z > 0、A >= Z，可选静质量必须有限且为正。
  - initial_energy_MeVu 仍表示每核子动能，总初始动能为 E/u * A。
  - 所有仓库内 YAML、CLI 帮助、配置测试和文档一次性迁移。

  输出接口同步破坏式泛化：

  - primary_c12_* 改为 primary_*。
  - CSV 的 primary_c12_MeV、primary_c12_letd_* 改为 primary_MeV、primary_letd_*。
  - fragment 字段明确使用 secondary_carbon、secondary_proton 等名称，避免 proton primary 与 proton secondary 混淆。
  - minibeam、validation scorer、MHD suffix 和绘图脚本同步使用通用 primary 命名。

  ## Implementation Changes

  - 将 CPU、legacy SYCL 和 current SYCL primary 路径中的 Z=6、碳有效电荷常量和 C-12 MCS 参数替换为 PrimaryIonDefinition。
  - 提取 host/device 共用的 Barkas 有效电荷公式：
    z_eff = Z * (1 - exp(-125 * beta * Z^(-2/3)))。

  - MCS 和 straggling 接收显式 projectile rest mass；未配置静质量时保持现有 A * 931.49410242 运算和浮点顺序，确保 C-12 结果不漂移。
  - 将 fragment stopping-power 回退从“相对碳有效电荷”改为“相对当前 primary reference ion”；显式 isotope stopping table仍具有最高优先级。
  - Primary stopping、XS、reaction package 始终来自用户配置；启用 attenuation、secondary generation 或 cascade 时分别校验所需文件存在。
  - 保留现有固定 fragment scorer 分类；O/Ne 等 primary 会进入独立 primary scorer，其 Z>6 secondary 暂时归入 other。
  - 保留 cascade_light_ion_xs_scale；将碳专用校准项明确改名为 cascade_secondary_z6_xs_scale，不把它错误解释为任意 primary 的通用模型。
  - CRPKG/CCAS v1 二进制布局不变，runtime 以 YAML 的 primary Z/A 为准，本阶段不做包身份强校验。
  - Package 编译器移除固定 TOPAS 4.2.p3、Geant4 11.3.2 和 INCL++限制；版本、physics model 和 process 名仅作为 provenance 写入 sidecar。
  - 编译器使用 metadata 中的 projectile Z/A 校验 primary continuation，兼容 proton PDG 2212 和通用 ion PDG 编码。
  - Package 仍必须满足“离散 inelastic interaction、入射 MeV/u、局部沉积、相关产物 Z/A/动能/方向”的数据语义。
  - 更新 mm.md：算法描述改为通用 charged-ion transport，明确当前缺失 hadronic elastic、decay 和动态全物种 scorer。

  ## Test Plan

  - 配置测试：缺少 Z/A、非法 Z/A、非法静质量、缺少条件性物理文件均拒绝启动。
  - 数学测试：Z=1/2/6/8/10 的有效电荷、质量、MCS 和 straggling 均有限且满足基本单调性。
  - C-12 parity：修改前先构建 reference executable；修改后用相同 seed、配置和 package 运行 serial、SYCL CPU 与 CUDA。
  - C-12 acceptance：重命名列后所有 scorer 数值、反应计数、queue 计数和能量 ledger 逐项完全相同；backend/log/header 允许因通用命名变化。
  - Proton full-chain smoke：测试中生成最小 stopping CSV、XS CSV、CRPKG v1 和含 proton projectile 的 CCAS v1，开启 straggling、MCS、primary inelastic、secondary transport 和 cascade。
  - Proton acceptance：至少发生一次 primary interaction 和一次 queued/cascade charged transport；无 queue overflow，dose/LET 非空且有限，能量守恒满足现有容差。
  - Package 工具测试：非 INCL 模型名和不同版本 provenance 可编译；非 inelastic schema、Z/A 不一致、非法方向和能量仍被拒绝。
  - 将依赖已删除 100k LFS 包的测试改为生成式小 fixture，使完整 CTest 不再依赖旧外部包。

  ## Assumptions And Deferred Work

  - 第一阶段只证明 C-12 等价和 proton 工程链路可运行，不宣称 proton 物理已与 TOPAS 对齐。
  - 不重命名 carbon namespace、include 目录、CMake target 或 carbon_mc executable。
  - 不升级 package v2，也不在 runtime 强制读取 metadata sidecar。
  - 暂不实现 hadronic elastic、decay、精确 fragment mass table、有效电荷表或动态任意 Z/A 分物种 scorer。
  - Helium、oxygen、neon 的真实表和 package 验证作为下一阶段；第一阶段形成的 primary 接口不得再包含 C-12 特例。
**通用束源参数（2026-09-26）**

非 minibeam 的 CUDA 统一 EM 路径按 YAML 源 Z/A 选择主粒子，允许配置 proton、C12 及包内覆盖的带电离子。切换粒子需要同时指定匹配的 EM、δ 矩、核反应、stopping 及可选弹性/散射数据包。加载器检查身份、质量、SHA 和能区；旧 C12 包不能作为质子主束包使用。minibeam 仍限定 C12。当前尚无新质子数据的端到端剂量验收结果。

质子束源参数片段见 [source_proton.yaml.template](../config/source_proton.yaml.template)：

```yaml
primary_particle: proton
initial_energy_MeV: 150
beam_energy_spread_percent: 0.5
number_of_histories: 1000000
```

`initial_energy_MeV` 是每个主粒子的总动能。上述输入内部为 150 MeV/u、相对 RMS 能散 0.005。C12 的 3000 MeV 总动能则对应 250 MeV/u。原有 `initial_energy_MeVu` 和相对 RMS 的 `beam_energy_spread` 继续可用，但每一对单位不同的字段只能指定一个。

`primary_particle` 支持 `proton`、`deuteron`、`triton`、`he3`、`alpha`/`he4`、`c12`/`carbon12`、`ion`，不区分大小写。原有 `primary_atomic_number` 与 `primary_mass_number` 必须成对提供；它们与已命名粒子同时出现时必须一致。未知名称、非整数 Z/A、非有限/非正能量、非有限质量、冲突能量单位、冲突能散单位和重复键都会报错。

其他离子示例：

```yaml
primary_particle: ion
primary_atomic_number: 8
primary_mass_number: 16
primary_rest_mass_MeV: 14895
initial_energy_MeVu: 100
```

此处质量只是说明输入格式，实际使用必须与新参考定义一致。未知离子未提供质量时拒绝运行，不再自动当成 `A × 931.494 MeV`。命名质子的默认核质量是 938.27208816 MeV；其他受支持轻离子也有各自默认核质量。

为兼容现有 C12，未显式指定质量的旧配置保留既有 C12 源与各 MCS 参考的质量选择。显式 `primary_rest_mass_MeV` 会传入主粒子 Highland、FE、minibeam Urban v2 的质量计算；Urban 的角度和 MFP 辅助函数也使用这个值。统一 EM、通用 Urban 和核弹性包若使用不同的主粒子质量，启动时拒绝，不能静默混用。旧 `urban` 及旧拟合分支不是新 proton 模型的验证依据。

束斑位置、角度、束宽和相关性继续使用现有 `source_origin_*_mm`、正交的 `beam_ux/uy/uz_*`、`enable_emittance_source`、`emittance_sigma_*` 和 `emittance_correlation_*`。使用 TPS/spot plan 时，已有的每个 spot 能量与束流参数接口保持不变。

**主粒子物理与计分字段**

新配置使用以下名称；旧名称作为 YAML 别名保留，同一配置同时给新旧名称会报错。

| 新名称 | 旧名称 |
|---|---|
| `ct_schneider_primary_cinel03_file` | `ct_schneider_c12_cinel03_file` |
| `enable_minibeam_primary_roi_scoring` | `enable_minibeam_primary_c12_roi_scoring` |

主粒子核弹性按实际 Z/A 解析索引，并由发生率和末态采样共同使用。主粒子 CINEL03 设备/缓存字段、纯水核反应投影的粒子身份改为 primary；包的 projectile metadata 必须匹配 YAML。源 Z/A 也进入派生主粒子核反应缓存的身份判断。

主粒子分量标签由实际 Z/A 决定：旧 C12 继续输出 `primary_c12`，质子为 `primary_proton`，其他粒子为 `primary_z<Z>_a<A>`。ROI 文件名遵循相同标签，能量分组按实际 A 换算。真实 C12 次级粒子的分类不变。

minibeam 的 Cu/水 production cut 可分别用 `minibeam_copper_production_cut_mm` 和 `minibeam_water_production_cut_mm` 指定，默认均为 0.05 mm。这不是 `maximum_step_mm`。Urban loss-range 参考检查由粒子、活动电离过程、材料和 cut 共同决定：proton 默认 `proton`/`hIoni`，C12 保留 `C12_Z6_A12_charge6`/`ionIoni`。必要时用 `primary_urban_reference_particle` 与 `primary_urban_ionisation_process` 对齐新导出器的实际元数据；参考名称必须是 proton 或以与束源一致的 `_Z<Z>_A<A>_charge<Z>` 结尾，电离过程也必须匹配该粒子；冲突时报错。这些字段不会改变束源 Z/A，也不会解除物理支持限制。

新参考状态字段使用 `VALID_ACTIVE_STEP_CONTEXT`；旧 `VALID_ACTIVE_C12_STEP_CONTEXT` 只接受 C12 身份。改变 metadata 文本不是生成或验证新质子参考数据的方法。

**本次范围与接入边界**

非 minibeam 接口现支持用 YAML 参数和包路径切换源；不再全局拒绝非 C12。新包需遵循现有二进制格式及 registry：EMJOINT1 仍包含 18 个带电物种和水/25 个 Schneider 分区，核反应使用 SCHNRATE v3、CINEL03、14-projectile 次级包及 bundle。格式/registry 扩展属于另一个任务，不是任意 Geant4 导出文件都可直接加载。

- 主粒子 rate sidecar 增加 `"projectile": {"z": 1, "a": 1}`。旧 sidecar 缺少该字段仅解释为历史 C12，不从 YAML 猜测身份。
- SCHNSTOP sidecar 的 `projectile.z/a` 必须匹配源，`data_filename` 必须匹配实际文件名，现有网格和材料 schema 保留。
- 核反应 bundle 内 SHA、channels Z/A、rate/package 能区须一致；二进制事件粒子身份和质量也会检查。
- 换 EM 包时提供 `em_delta_moments_sha256` 和 `em_delta_moments_source_sha256`，后者必须等于 `em_package_sha256`。旧 C12 默认哈希继续兼容。δ 矩表格式为 EMDMOMT2，与 EM 节点数及顺序严格一致，须由对应 EM 包重新生成。
- `multiple_scattering_model: highland` 按源 Z/A 和核质量计算；`urban_v2` 使用匹配的 SHA 固定数据包。FE 的非 C12 主束需 `fermi_eyges_parameter_set: species_water`；`fermi_eyges_species` 控制是否选择该物种，未选择者使用 Highland。现有模型名称不代表已实现或验证 TOPAS proton Wentzel 模型。
- 旧碳束上游空气散射、电子响应、CSV attenuation、packaged fluctuation 与核弹性诊断仍限定原适用域。普通质子配置应关闭这些诊断，核弹性使用 `all_ion_elastic_file` 及匹配反冲 stopping 包。

一个配置可以从现有非 minibeam 水/CT YAML 复制，再将源改为 `primary_particle: proton`、总动能 `initial_energy_MeV: 150`，替换所有相关 package 路径、SHA 与 bundle。完整的待填数据模板见 [proton_source_gpu.yaml.template](../config/proton_source_gpu.yaml.template)。本次接口/完整性检查不等于新质子包的数值或物理验收。

`load_primary_source_parameters()` 只描述源，不执行完整配置/物理验证；正式 `load_config()` 仍严格检查所有配置键并执行输运支持验证。新增 `primary_source_parameters` 测试使用 YAML 和合成元数据，检查单位、身份、质量、别名、兼容和拒绝行为，不以旧剂量或旧拟合数据证明质子物理正确。

**已生成的质子数据包（2026-09-26）**

0.1–250 MeV 的非 minibeam 主束包和共享依赖已组装，入口为 [proton_water_fullphysics.yaml](../config/proton_water_fullphysics.yaml)。提取范围、哈希、低能补采样和当前精度限制见 [proton_fullphysics_packages.md](proton_fullphysics_packages.md)。

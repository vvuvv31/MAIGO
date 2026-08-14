# Database scorer extensions

这些 `.cc/.hh` 文件是从当前已经用于数据库验证的 TOPAS extension build 整理出来的源文件。每个 `.cc` 的首行保留了 TOPAS CMake extension discovery 所需的 `Scorer for ...` 标记。

不要手工添加 `TsExtensionManager.cc`。TOPAS 配置 `-DTOPAS_EXTENSIONS_DIR=/path/to/startup/extensions` 后，会从这些文件自动生成 manager 和 scorer 注册表。运行时使用的 TOPAS binary 必须是用同一批源文件重新编译的版本。

| scorer quantity | 用途 |
|---|---|
| `CarbonMaterialPropertiesNtuple` | 材料密度、辐射长度 |
| `CarbonStoppingPowerNtuple` | C-12 stopping-power table |
| `CarbonCrossSectionNtuple` | C-12 非弹性截面/平均自由程 |
| `CarbonReactionNtuple` | primary C-12 反应和一级带电产物 |
| `CarbonCascadeNtuple` | 带电级联 interaction/product 记录 |
| `CarbonNeutralNtuple` | 中性相互作用和产物 |
| `IonStoppingPowerNtuple` | 不同碎片 Z/A 的 stopping power |
| `IonCrossSectionNtuple` | 不同碎片 Z/A 的非弹性截面 |
| `NeutralCrossSectionNtuple` | gamma/neutron 的总截面 |

`build_ct_material_packages.sh` 只需要 `CarbonCascadeNtuple`：同一份相关
interaction/product 记录既编译 charged cascade package，也筛选 primary
track-1 C-12 的首次非弹性反应来编译 reaction package。
interaction 记录还保存 `G4Step::GetTotalEnergyDeposit()`；reaction 和
cascade v1 运行时包直接使用该值，避免把反应 Q 值或核质量差
误当成局部剂量。旧 package layout 不再读取，运行时会直接报错。

其中最后两个 scorer 不在默认模板中启用，但保留在包内，便于单独做诊断。`myHadronLET` 属于独立 LET extension，没有复制到这里。

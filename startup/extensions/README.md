# Database scorer extensions

这些 `.cc/.hh` 文件是从当前已经用于数据库验证的 TOPAS extension build 整理出来的源文件。每个 `.cc` 的首行保留了 TOPAS CMake extension discovery 所需的 `Scorer for ...` 标记。

不要手工添加 `TsExtensionManager.cc`。TOPAS 配置 `-DTOPAS_EXTENSIONS_DIR=/path/to/startup/extensions` 后，会从这些文件自动生成 manager 和 scorer 注册表。运行时使用的 TOPAS binary 必须是用同一批源文件重新编译的版本。

| scorer quantity | 用途 |
|---|---|
| `CarbonMaterialPropertiesNtuple` | 材料密度、辐射长度 |
| `CarbonStoppingPowerNtuple` | source-primary stopping-power table (legacy extension name) |
| `CarbonCrossSectionNtuple` | source-primary 非弹性截面/平均自由程 (legacy extension name) |
| `CarbonReactionNtuple` | primary C-12 反应和一级带电产物 |
| `CarbonCascadeNtuple` | 带电级联 interaction/product 记录 |
| `CarbonNeutralNtuple` | 中性相互作用和产物 |
| `CarbonElasticNtuple` | 参数化 Z/A 的弹性 interaction、primary continuation 和全部可见产物 |
| `ElasticCrossSectionQueryNtuple` | 直接查询指定低能点的 Geant4 弹性宏观截面，仅用于诊断 |
| `IonStoppingPowerNtuple` | 不同碎片 Z/A 的 stopping power |
| `IonCrossSectionNtuple` | 不同碎片 Z/A 的非弹性截面 |
| `NeutralCrossSectionNtuple` | gamma/neutron 的总截面 |
| `PrimaryCrossingCount` | 配置 projectile Z/A 的 primary history 逐 bin crossing count |

`PrimaryCrossingCount` 是一个 binned scorer，不使用 `Fluence x area`。
在 scorer 参数下设置 `i:.../ProjectileZ` 和 `i:.../ProjectileA`；每个
Geant4 event/history 对同一 component bin 最多贡献一次，即使 primary 在
该 bin 内有多个 step 或散射后重新进入。去重集合保存在 TOPAS 的
worker-local scorer 实例中，并在 `UserHookForEndOfEvent` 清空，随后由
`TsVBinnedScorer` 的标准 MT 合并路径合并各 worker 的 bin totals。

对沿 z 方向的水 phantom，使用 `XBins = 1`、`YBins = 1`、`ZBins = N`；
当 phantom 总深度为 700 mm 且 `ZBins = 1400` 时，组合 index 的 z division
就是 0.5 mm depth bin。输出 `Report = "Sum"` 即为每个 bin 的 crossing count。
该 scorer 需要显式的 projectile 参数，例如：

```text
s:Sc/PrimaryCrossing/Quantity = "PrimaryCrossingCount"
s:Sc/PrimaryCrossing/Component = "Phantom"
i:Sc/PrimaryCrossing/ProjectileZ = 1
i:Sc/PrimaryCrossing/ProjectileA = 1
```

`build_ct_material_packages.sh` 只需要 `CarbonCascadeNtuple`：同一份相关
interaction/product 记录既编译 charged cascade package，也筛选 source-primary
track-1 的首次非弹性反应来编译 reaction package。项目名称仍保留 Carbon，
但 exporter 使用实际 source ion 的 Z/A。
interaction 记录还保存 `G4Step::GetTotalEnergyDeposit()`；reaction 和
cascade v1 运行时包直接使用该值，避免把反应 Q 值或核质量差
误当成局部剂量。旧 package layout 不再读取，运行时会直接报错。

其中最后两个 scorer 不在默认模板中启用，但保留在包内，便于单独做诊断。`myHadronLET` 属于独立 LET extension，没有复制到这里。

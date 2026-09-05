# MAIGO 未决问题

记录 2026-08 水箱 200 MeV/u GPU vs TOPAS 验证中仍打开的物理/计分/编包问题，以及代码里对应的缺口。  
对照：TOPAS 4.2.p3 / Geant4 11.3.2，`g4ion-inclxx`。验证几何 200 MeV/u \(^{12}\mathrm{C}\)、100k、seed `20260714`、0.5 mm bin、400 mm 水。

状态：`open` 仍缺；`partial` 有开关或对照实验，生产默认未改；`closed` 已证实不是这条路。

---

## 1. 离子 EnergyDeposit 与 δ 电子归属 — partial

**现象。** GPU 连续能损把 δ 电子能量记在离子上（无限制 \(dE\)）。TOPAS 离子 `EnergyDeposit` 是 `G4Step::GetTotalEnergyDeposit()`，高于 production cut 的 δ 进电子 scorer。同一物理总剂量下：

| 列 | 去 δ 前（200 MeV/u 验证包） | 去 δ 后 |
|---|---:|---:|
| 总 IDD | +0.10% | +0.14% |
| 初级 \(^{12}\mathrm{C}\) | +7.16% | −0.57% |
| B / Be / Li | +6.2 / +3.7 / +5.1% | +0.6 / −1.5 / −0.5% |
| He | +0.90%（此前只对 He 开过） | +0.90% |
| p | +7.63% | +3.84% |

**已做。** `restrict_fragment_species_energy_deposit`：初级和指定 Z 的物种列写 \((1-\delta)\times dE\)；输运和总 IDD 仍用无限制账。δ 来自 `let_delta_electron_fraction_file` / 粒子种 SP 表。验证配置：`z_min=1`、`z_max=6`。结果见 `benchmark/scorer/results/water_200_ion_restricted/`。

**未做。**

- 生产 CT 默认关此开关；正式剂量仍是无限制连续能损，没有显式电子。
- 没有电子物种列；去 δ 后的能量不单独记，只从离子列拿掉。
- 质子去 δ 后仍 +3.8%，不能全用 δ 解释。
- `electronic_buildup_fraction` 是另一套入口近似，默认关，不等价于显式 δ 输运。

**代码。** `include/carbon/transport_config.hpp`（`restrict_fragment_species_*`）；`src/transport_sycl_legacy.cpp`（`restricted_primary_dose_device`、`unrestricted_secondary_dose_device`）；`src/detail/sycl_score_device.inc`（`fragment_species_step_deposit_MeV`）。

---

## 2. 400 MeV/u 包在 200 MeV/u 箱里碎片产额偏低 — open

**现象。** 同一套旧 `CarbonCascadeNtuple`、同一 INCL++：

- 400 MeV/u 长程 1M 包、200–204 MeV/u 箱：B 产额 0.116/反应，C 0.162，N 0.033，O 0.012。
- 200 MeV/u 束现场一级 inelastic（180–200 MeV/u）：B 0.153，C 0.246，N 0.147，O 0.119。

GPU 用 400 MeV/u 1M 包时 B/Be/Li fluence 约 −30/−26/−20%；换成从配对 TOPAS 200 MeV/u ntuple 编的包后变为 +0.9/−2.3/+0.6%。

**结论。** 缺口是编包样本（400 MeV/u 束切能量箱），不是 ntuple 漏记同一条 track。

**未做。** 生产 0–400 包仍是单次 400 MeV/u 长程束。若要兼顾 200 MeV/u 的 B/C/靶残核份额，需要按能量分层或把 200 MeV/u 束样本并进对应箱。

**产物。** `data/packages/topas_200MeVu_water_inclxx_100k_{primary,cascade}_3d.bin`（100k，仅 0–204 MeV/u）；对照 `benchmark/scorer/results/water_200_from_paired_ntuple/`。

---

## 3. CarbonCascadeNtuple continuation — closed（对 200 MeV/u B/Be/Li）

**假设。** G4 在 `ionInelastic` 后可能留下弹核；旧 ntuple 只记新 track，GPU 却杀掉弹核。

**实验。** ntuple 补记 `primary_continuation` / `projectile_continuation`，重跑 1M。一级包只多 **2404** 条，全是 \(^{12}\mathrm{C}\)，入射中位约 348 MeV/u；200 MeV/u 箱 B/Be/Li **与旧包相同**。旧 cascade CSV 同顶点后续几乎为空。反应账 `入射−次级KE−local` 约 81 MeV/反应，没有藏着的 GeV 级残核。

**结论。** 不必再为 200 MeV/u 的 B/Be/Li 改 ntuple。continuation 对高能准弹性留下 C-12 仍有用，可保留。GPU legacy 仍把负 PDG 的 C-12 当次级碳，未当初级续航。

**代码。** `startup/extensions/CarbonCascadeNtuple.cc`；`startup/package_tools/compile_reaction_package.py`。

---

## 4. 质子物种列仍偏 — open

去 δ、中子血统记入 species、200 MeV/u 包之后，p 剂量仍约 **+3.8%**（曾 −16%，中子血统修复后 −3.8%，换 200 包后又偏高）。

可能来源（未拆开）：

- `charged_dose_category`：只有 Z=1 A=1 进 p；d/t 进 `other`。TOPAS p 列是 named `proton`。分类已对齐，不是主因。
- 没有独立核弹性包；弹性反冲质子只存在于 TOPAS。
- 中子次级质子份额随包/代数变化。
- 包内 π⁻（PDG −211）被丢掉（Z=A=0），能量不进物种。

**不要**用弹性包去“修”B/Be/Li；弹性主要动剩余 p。

---

## 5. LET_d 峰处仍差 — open

初级入口 LET_d 已对齐（约 −0.1%）。剂量峰处初级 / 全强子 LET_d 曾差约 **−34%**（缺 `let_delta_electron_fraction_file` 时更糟）。验证已加载 δ 表；峰处分母/掩膜、restricted vs unrestricted、以及 all-hadron 是否含电子定义，仍未闭环。

全强子入口也曾约 −15%。图：`water_200_ion_restricted/compare/letd.png`。

---

## 6. 物种合计与物理总剂量 — partial

Host 曾把 `fragment_dose` 再加进 `deposited_energy`，而中子血统能量已在总剂量里 → species 合计虚高约 0.6%。  
`restrict_fragment_species_energy_deposit` 开启后，总 IDD 改用无限制次级账 + 无限制初级，避免把已去 δ 的离子列再加一遍。

生产路径（开关关）仍是 `dose_host + sum(fragment)`。中子血统同时写 `neutral_origin` 与 `fragment_dose` 时，生产合计仍可能双计。测试里 voxel 平面和 vs IDD 闭合曾失败（`carbon_tests`：voxel sum 6.39 vs IDD 4.84），需确认是否此账。

---

## 7. 没有独立核弹性 — open

TOPAS：`g4h-elastic_HP`。GPU：无独立 elastic process；角度部分靠 MCS 和 inelastic 末态。  
补 elastic 包会改变“不发生 / 弹性 / 非弹性”的抽样，主要影响剩余质子和轻反冲，不是 B/Be/Li 的 30% fluence 洞。

---

## 8. 没有显式电子 / 光子输运 — open

δ 电子、轫致、湮灭等不逐步输运。`electronic_buildup_*` 和 δ 分数表只是局部代理。生产剂量是无限制离子连续能损。要和 TOPAS 电子列或受限离子列比，必须走验证开关，不能当成 production 物理。

---

## 9. 中性粒子与 π — partial

中子：`neutral_transport_mode: full` + 400 MeV/u CNPK 后，总 IDD 尾巴改善，p 补上约 11 MeV/primary。中子血统一度不写 `fragment_dose`（已修）。  
π⁻ 在一级包里约 2.2%/反应，KE 被丢，进 untracked。  
验证 ntuple 29 列没有 `local_deposit`，编包时记 0（约 0.55 MeV/反应量级，对 B 洞可忽略）。

---

## 10. 级联代数与稀有核素 — open

验证用 `maximum_cascade_generations: 4`。生产文档曾写默认 2，超出部分变 residual heat。  
200 MeV/u 100k 级联包只有 30 种弹核（1M 400 MeV/u 有 39 种）。稀有 Z/A 没有 cascade XS 时不再反应，能量当地沉积。

---

## 11. Fluence 跨 bin 曾用 Δz 而非 Δs — closed

`score_track_length_mm` 单 bin 加 `step_mm`，跨 bin 曾加 z 重叠。已改为 `overlap/|u_z|`。对近轴初级影响很小；大角度碎片略有影响，解释不了 B 的 −30%。

---

## 12. 对比脚本图例 — closed

`plot_two_panel` 曾写死 “TOPAS 5-seed mean” / “GPU water_200”，配对重跑图也被标错。已改成 TOPAS / GPU。重画脚本：`benchmark/scorer/plot_ion_restricted.py`。

---

## 13. 几何 / 坐标系约定（勿再混）

- TOPAS 病人：先 RotZ 再 Trans。不要把 IEC 机架 90° 当成 TPS 90°。
- PBS 与 Time Feature 分开，不要混一条验证链。
- 额外 scorer 只在 `CARBON_VALIDATION_SCORERS` + `scorer_mode: validation`。

---

## 建议顺序

1. 生产 0–400 包：按能量分层或混入 200 MeV/u 束样本，再在水箱上确认 B/C 份额。  
2. 质子：拆弹性 / 中子次级 / π，再决定是否做 elastic 包。  
3. LET_d 峰处：核对 restricted 定义、分母掩膜、all-hadron 物种。  
4. 生产路径的 species 合计双计与 voxel–IDD 测试。  
5. 不要再为 200 MeV/u 的 B/Be/Li 改 ntuple。

# 八个预设的数据提取扩展

此目录收集当前八个预设需要的 TOPAS/Geant4 导出器、编译脚本、离线打包工具和材料输入卡。源码从周边仓库筛选、复制并适配当前目录；执行不需要访问原仓库。`source_provenance.json` 记录来源与安装后 SHA256，`coverage.json` 逐文件对应当前 `data/manifest.json` 的全部资源。

这是离线提取工具。编译不会改变 `config/`、GPU 运行时或安装的 `data/`。没有加入中性粒子追踪包；核反应记录仍观察完整 Geant4 反应以审计能量闭合，内部带电碎片 Z/A 注册表仍必需。原始记录、构建产物及校准数据放 `work/`，不随源码发布。

## 1. 八个预设与数据范围

| 预设 | 主粒子专属包 | 额外材料/场景输入 |
| --- | --- | --- |
| proton_water_broadbeam | proton 核反应、弹性/反冲、出生参数、水 stopping | Water_75eV |
| carbon_water_broadbeam | carbon 核反应、弹性/反冲、出生参数、水 stopping | Water_75eV |
| proton_ct_broadbeam | proton 主粒子表 | Schneider、RT07575 head proton CT/spots/beam model |
| carbon_ct_broadbeam | carbon 主粒子表 | Schneider、RT07575 head carbon CT/spots/beam model |
| proton_water_minibeam | proton 主粒子表 | Cu stopping/rate/cascade/elastic、空气数据 |
| carbon_water_minibeam | carbon 主粒子表 | Cu/空气、carbon water spots |
| proton_ct_minibeam | proton 主粒子表 | Cu/空气、独立 lung benchmark CT |
| carbon_ct_minibeam | carbon 主粒子表 | Cu/空气、独立 lung benchmark CT |

共同包族包括 EMCNT001 continuous26、delta moments、次级核反应及 Urban。每个预设实际展开的文件路径以 [preset_dependencies.json](../docs/preset_dependencies.json) 为准。RT07575 是 head case，CT、源、权重、放置、准直器及归一化必须成套使用；lung minibeam 不能称为 RT07575 全计划验证。

覆盖清单区分物理提取、离线派生、静态定义和外部输入。CT、计划 spots、beam model 不是物理扩展输出；这里不加入 DICOM 转换或 TOPAS 文本计划 replay。

**源码齐备不等于可以仅凭源码逐字节重建当前包。** 必须匹配 TOPAS/Geant4 数据集、物理模块、cut、seed、统计量及离线校准语料。两项具体前提：

- 碳主核反应冻结编译器要求原生产记录按 stride=20 重建恰好 30113 个基础事件，再加审计过的低能补点。次级编译器需要 campaign manifest、raw validation 和对应原始记录。

全量 raw/corpus 未复制，未执行全量生产重提取或新剂量验收。

## 2. 目录与依赖

| 路径 | 内容 |
| --- | --- |
| `topas/common/` | 核反应捕获、Schneider rate/stopping、主粒子材料查询、材料 JSON |
| `topas/unified_em/` | 全离子 native EM；与 common 的同名 stopping scorer 分开链接 |
| `topas/all_ion_elastic/` | 弹性联合分布/反冲，含冻结 carbon 实现 |
| `copper_ion_transport/` | [碳微束铜碎片弹性提取与涨落验证](copper_ion_transport/README.md) |
| `native/` | Geant4 原生工具 CMake 项目 |
| `tools/` | 各包族 campaign 驱动、编译器和验证器 |
| `package_tools/` | native recorder 协议与当前 CINPKG04 写入 |
| `reference/` | HU 映射、26 composition 卡、82 joint 材料定义、注册表/bundle 模板 |

`cinel02.py`、`worker_*.cinel02` 是原始 recorder 协议名称，不是旧运行时包读入路径。EMJOINT1 等格式只用于离线中间计算；最终 ion EM 必须是 EMCNT001。

已验证：Linux、OpenTOPAS 4.2.3、Geant4 11.3.2、CMake、C++ 编译器。原生 CMake 固定 Geant4 11.3.2 EXACT。TOPAS 源码需要当前 native 反应 recorder 接口；扩展不会自动为不同版本移植内部接口。Qt、GDCM 等按 TOPAS 要求安装，环境脚本需设置 Geant4 数据集及动态库路径。Python 依赖见 `requirements.txt`。

## 3. 编译与候选目录

从仓库根目录执行，替换下面 `/path/to/...`：

```bash
python3 -m venv extensions/work/venv
source extensions/work/venv/bin/activate
python3 -m pip install -r extensions/requirements.txt
source /path/to/env_topas.sh
export MAIGO_TOPAS_SOURCE=/path/to/OpenTOPAS-4.2.3
export MAIGO_G4_CMAKE=/path/to/geant4-11.3.2/lib/cmake/Geant4
export MAIGO_TOPAS_PREFIX=/path/to/topas-dependencies
bash extensions/build_native.sh --build-dir extensions/work/build/native \
  --geant4-dir "$MAIGO_G4_CMAKE" --jobs 6
for profile in nuclear-copper unified-em elastic-carbon recoil-carbon elastic-proton; do
  bash extensions/build_topas_extensions.sh --topas-source "$MAIGO_TOPAS_SOURCE" \
    --build-dir "extensions/work/build/$profile" --profile "$profile" \
    --geant4-dir "$MAIGO_G4_CMAKE" --cmake-prefix "$MAIGO_TOPAS_PREFIX" --jobs 6
done
```

| TOPAS profile | 用途 |
| --- | --- |
| nuclear-copper | 核反应、材料 CSV、Schneider stopping/rates、Cu cascade |
| unified-em | native ion EM，替换 common 同名 CarbonStoppingPowerNtuple |
| elastic-carbon | 冻结 reference/carbon 弹性 |
| recoil-carbon | 冻结 reference/carbon 反冲 stopping |
| elastic-proton | Water_75eV proton 行/反冲替换 |

每个 profile 必须有独立 build 目录，脚本拒绝混用。源码暂存到 build 下 `maigo-extension-source/`，TOPAS 源目录不变。`--stage-only` 只选择源码；`--cmake-arg ARG` 可重复；默认 Qt=ON，无 Qt 时显式 `--topas-use-qt OFF` 并确认其他依赖。

原生生成六个目标：`material_metadata`、`extract_ct_urban`、`extract_copper_urban`、`extract_air_urban`、`extract_air_ionization`、`validate_urban`。validator 使用当前包查询器做离线 FP32 验证，不构建 CPU 输运替代路径。

```bash
export MAIGO_KIT_WORK="$PWD/extensions/work"
export MAIGO_NUCLEAR_TOPAS="$MAIGO_KIT_WORK/build/nuclear-copper/topas"
export MAIGO_EM_TOPAS="$MAIGO_KIT_WORK/build/unified-em/topas"
export MAIGO_NATIVE="$MAIGO_KIT_WORK/build/native"
export MAIGO_STAGE="$MAIGO_KIT_WORK/packages-candidate"
python3 extensions/prepare_staging.py --out "$MAIGO_STAGE"
```

候选目录必须不存在，只创建目录并复制 HU/bundle 模板。下文输出均放候选区，不指向仓库根目录；生成器通常要求新的输出目录。

## 4. continuous26 ion EM

```bash
python3 extensions/tools/unified_em/prepare.py --output "$MAIGO_KIT_WORK/ion-em-raw"
python3 extensions/tools/unified_em/run_extract.py \
  --campaign "$MAIGO_KIT_WORK/ion-em-raw" --topas "$MAIGO_EM_TOPAS"
python3 extensions/tools/unified_em/compile_unified_em_package.py \
  "$MAIGO_KIT_WORK/ion-em-raw" "$MAIGO_KIT_WORK/ion-em-native.bin"
python3 extensions/tools/unified_em/synthesize_analytic_density.py extract \
  --source "$MAIGO_KIT_WORK/ion-em-native.bin" \
  --output "$MAIGO_STAGE/data/em/ions/composition_bases.npz" --lossless
python3 extensions/tools/unified_em/build_continuous_density.py \
  --bases "$MAIGO_STAGE/data/em/ions/composition_bases.npz" \
  --sections extensions/reference/schneider_sections.metadata.json \
  --helper "$MAIGO_NATIVE/material_metadata" --output "$MAIGO_KIT_WORK/continuous26-packed"
```

最后输出目录内是 `continuous26.bin`、`delta_moments.bin` 及 `.metadata.json`，审核后成套复制到候选 `data/em/ions/`。用 `verify_unified_em_data.py PACKAGE` 检查两包；单独 moments 可用 `build_delta_moments.py --source ... --output ...` 派生。native 多密度只用于离线拟合，最终连续密度规律不储存 transport density states。保留当前 native spline/conditional 规律及明确的 hard-rate 校准近似。

## 5. 核反应及 rates

```bash
python3 extensions/tools/primary_source/campaign.py prepare \
  --root "$MAIGO_KIT_WORK/proton-primary" --histories 20000
python3 extensions/tools/primary_source/campaign.py run \
  --root "$MAIGO_KIT_WORK/proton-primary" --topas "$MAIGO_NUCLEAR_TOPAS" --workers 4
```

默认四个能量是初始采集，不够替换生产包。按覆盖报告使用 `--energies`、`--targets`、`--coverage` 补点；`--final-state-xs-factor` 默认 1，不能擅改校准。

确定性 rates 是 0.1–460.1 MeV/u、步长 0.5、921 节点。需要实例化全部 25 Schneider 组成的外部 DICOM 材料试件；当前 CT `.bin` 不能替代这个 DICOM 目录：

```bash
export TOPAS_BINARY="$MAIGO_NUCLEAR_TOPAS"
export MAIGO_SCHNEIDER_DICOM_DIR=/path/to/all-25-sections-DICOM
export MAIGO_RATE_CAMPAIGN_ROOT="$MAIGO_KIT_WORK/rates"
python3 extensions/tools/schneider/xs_dump_v2_1.py
```

生成 15 份 TOPAS 输入及 Slurm 脚本，默认不提交。可直接运行各 `*_param.txt`；或检查 partition/资源设置后 `--submit`。`TOPAS_G4_DATA_DIR`、`TOPAS_LD_LIBRARY_PATH` 来自安装环境。`proton_dump.json` 是主包输入，`h1_dump.json` 是次级 registry 同 Z/A 项。

```bash
python3 extensions/tools/primary_source/compile.py \
  --campaign "$MAIGO_KIT_WORK/proton-primary" \
  --rate-json "$MAIGO_KIT_WORK/rates/raw/proton_dump.json" \
  --output "$MAIGO_STAGE/data/nuclear/schneider/proton"
```

`--campaign` 可给多个补点目录。保持正 rate 的终态支持、有效域、能量闭合，禁止缺节点静默清零。

碳/次级冻结配方先提供基础 raw 与补点目录；后者含 `campaign_manifest_all.json`、`raw_validation_all.json` 和声明的 raw。`validate_v2_raw.py --campaign-root ... --manifest ... --out ...` 可审计新的 manifest。`templates.py` 提供原始采样模板常量；冻结编译器不是从零采样 campaign 生成器。

```bash
export MAIGO_PRIMARY_CAMPAIGN_ROOT=/path/to/pinned-carbon-production-raw
export MAIGO_V2_CAMPAIGN_ROOT=/path/to/audited-secondary-and-carbon-fills
export MAIGO_RATE_DUMP_ROOT="$MAIGO_KIT_WORK/rates/raw"
export MAIGO_ASSET_OUTPUT_ROOT="$MAIGO_STAGE"
python3 extensions/tools/schneider/compile_primary_package_v2_1.py
python3 extensions/tools/schneider/compile_v2_package.py
python3 extensions/tools/schneider/compile_primary_rates_v2_1.py
python3 extensions/tools/schneider/compile_rates_v2_1.py
```

最终 `nuclear/schneider/{proton,carbon,secondary}/`：终态 CINPKG04、主 rate SCHNRATE v3、次级 rate SCHN2RAT v3。v2_1 是冻结配方名。14 项次级 registry、target 顺序与 support/null evidence 约束保持。模型、覆盖或统计量更改属于新候选。

## 6. 弹性/反冲及 Schneider stopping

```bash
python3 extensions/tools/general_ion_elastic/prepare_campaign.py \
  --hu-file "$PWD/extensions/reference/hu_to_material.txt" \
  --dicom-dir "$MAIGO_SCHNEIDER_DICOM_DIR" \
  --output "$MAIGO_KIT_WORK/elastic-carbon" --samples 512
python3 extensions/tools/general_ion_elastic/run_campaign.py \
  --campaign "$MAIGO_KIT_WORK/elastic-carbon" \
  --topas "$MAIGO_KIT_WORK/build/elastic-carbon/topas"
python3 extensions/tools/general_ion_elastic/compile_all_ion_elastic.py \
  "$MAIGO_KIT_WORK/elastic-carbon" "$MAIGO_STAGE/data/elastic/schneider/carbon/all_ion.bin"
```

反冲另建 campaign，prepare 加 `--recoil-only`，run 使用 `recoil-carbon/topas`。raw 的 `recoil_stopping.bin` 已为 ELRSP001，安装时配套生成 bytes、registry、输入来源的 metadata。`verify_all_ion_elastic.py --data DIR` 检查 18 projectiles、26 materials、13 targets 及所有抽到 isotope 的反冲覆盖。

冻结 carbon 保留原 accepted bank 的 G4_WATER reference 行；当前 proton 导出使用 Water_75eV。不要在重构中悄悄替换冻结行。运行时水 reference 仍是 I=75 eV 的 Water_75eV。proton 组装仅替换 proton 行，保留 reference 其余 17 行：

```bash
python3 extensions/tools/primary_source/assemble.py \
  --reference-bank "$MAIGO_STAGE/data/elastic/schneider/carbon/all_ion.bin" \
  --work /path/to/proton-elastic-recoil-water-campaign --output "$MAIGO_STAGE"
```

`--work` 必须含 `elastic/raw/z1a1.bin`、`elastic/provenance.json`、`recoil/raw/recoil_stopping.bin`、`recoil/provenance.json`、`tables/proton_water.phsp`（4001×5）。proton 弹性/反冲均用 `elastic-proton` profile。目标弹性目录必须为空。

common `CarbonSchneiderStoppingPowerDump` 输出 25 sections、4302 节点（0.01–430.11 MeV/u）的主 stopping CSV/JSON：

```bash
python3 extensions/tools/schneider/compile_schneider_stopping.py \
  --raw-csv /path/to/carbon-stopping.csv --raw-json /path/to/carbon-stopping.json \
  --output-dir "$MAIGO_STAGE/data/stopping/schneider" --projectile-z 6 --projectile-a 12
```

质子改 Z=1/A=1。输出 `carbon.bin`/`proton.bin`、metadata 和离线 audit CSV。全离子 `IonSchneiderStoppingPowerDump` 用 `compile_schneider_ion_stopping.py --raw-csv ... --raw-json ... --campaign-manifest ... --out-bin ...` 编译为离线审计，不自动安装退役运行时包。

## 7. Cu、水、空气表和 cascade

```bash
python3 extensions/tools/tables/prepare.py --out "$MAIGO_KIT_WORK/material-tables" \
  --topas "$MAIGO_NUCLEAR_TOPAS" --samples-per-node 512
python3 extensions/tools/tables/compile.py --campaign "$MAIGO_KIT_WORK/material-tables" \
  --output-root "$MAIGO_STAGE" --delta-reference "$PWD/data/stopping/water/ions.csv"
```

prepare 创建 water/air/copper × proton/carbon 表查询及 G4_WATER 材料 JSON probe。一条 track 触发遍历查询网格，不意味着一个反应事件就能构建 cascade。Cu elastic 必须直接来自 `PrimaryMaterialDump`，不能用 step-difference 替代。

编译保留显式参考 CSV 的逐 Z/A delta fraction 校准，不猜测缺数据。water 输出两个 `ions_*_candidate.csv`，Cu ion rates 也分 context 输出候选；比较后按原配方选择共享 `water/ions.csv` 和 `copper/ion_cross_sections.csv`。G4_WATER material JSON 是 composition/radlen 记录，不用于把输运水 I=75 eV 改为 NIST 默认。

Cu 原始模板：`extract_copper_physics.txt`（C12）、`extract_copper_fragment_cascade.txt`（逐 projectile）。复制到新 campaign，设置 seed、threads、histories、source Z/A、总能量 MeV，运行 nuclear-copper TOPAS。模板统计量较大，执行前查看；不是剂量 benchmark。每个需要的 Z/A 和能量分别采集，保留日志/recorder。

```bash
python3 extensions/tools/minibeam_copper/compile_copper_fragment_cascades.py \
  --input 1,1,/path/to/proton-cu-raw \
  --output "$MAIGO_STAGE/data/nuclear/copper/proton/cascade.cinpkg" \
  --metadata "$MAIGO_STAGE/data/nuclear/copper/proton/cascade.metadata.json"
```

重复 `--input Z,A,ROOT` 汇总碳及碎片；`--preserve-projectile-package Z,A,PACKAGE` 要求对应 payload 保持。必须覆盖运行时带电 projectile 有效域。

`compile_copper_physics.py RUN_ROOT OUTPUT --native-elastic-file FILE --native-elastic-metadata JSON` 保留 C12 冻结审计配方。其离线历史产物名映射为当前 `stopping/copper/carbon.csv`、`nuclear/copper/carbon/primary_rate.csv`、`elastic/copper/carbon/primary.csv`、`nuclear/copper/carbon/cascade.cinpkg`。tables 已生成的 canonical 表不要重复用另一来源覆盖。

```bash
source /path/to/geant4-11.3.2/bin/geant4.sh
"$MAIGO_NATIVE/extract_air_ionization" "$MAIGO_KIT_WORK/air-ionization" 1 1 32769
```

审计 native probes/范围/非负有限值（阈值以下允许零）后，`air_ionization_rate.csv` 安装为 `ionization/air/proton_rate.csv`。

## 8. Urban 全 CT/Cu/空气

```bash
for medium in ct copper air; do
  python3 extensions/tools/urban/run_campaign.py \
    --binary "$MAIGO_NATIVE/extract_${medium}_urban" \
    --out "$MAIGO_KIT_WORK/urban-$medium" --medium "$medium" --workers 2
done
python3 extensions/tools/urban/pack_reference.py "$MAIGO_KIT_WORK/urban-ct" \
  "$MAIGO_KIT_WORK/urban-ct-packed" --edges "$MAIGO_KIT_WORK/urban-ct"
python3 extensions/tools/urban/pack_reference.py "$MAIGO_KIT_WORK/urban-copper" \
  "$MAIGO_KIT_WORK/urban-copper-packed" --edges "$MAIGO_KIT_WORK/urban-copper" --materials G4_Cu
python3 extensions/tools/urban/pack_reference.py "$MAIGO_KIT_WORK/urban-air" \
  "$MAIGO_KIT_WORK/urban-air-packed" --edges "$MAIGO_KIT_WORK/urban-air" \
  --materials G4_AIR --species-count 2
for medium in ct copper air; do
  dir="$MAIGO_KIT_WORK/urban-$medium-packed"
  "$MAIGO_NATIVE/validate_urban" "$dir/urban_mcs_candidate.bin" "$dir/probes.bin" "$dir/mfp_probes.bin"
done
python3 extensions/tools/urban/merge_packages.py \
  --ct "$MAIGO_KIT_WORK/urban-ct-packed/urban_mcs_candidate.bin" \
  --copper "$MAIGO_KIT_WORK/urban-copper-packed/urban_mcs_candidate.bin" \
  --output "$MAIGO_STAGE/data/scattering/urban/ct_copper.bin"
cp "$MAIGO_KIT_WORK/urban-air-packed/urban_mcs_candidate.bin" "$MAIGO_STAGE/data/scattering/urban/air.bin"
```

CT 是 Water_75eV + 全部 25 Schneider 组成；CT/Cu 各 52 species，空气仅 proton/carbon。dense 默认32769、held-out4097 节点、offset0.37。导出 vectors/mfp/mfp_edges，validator 阈值5e-4，须 `FP32_REFERENCE=PASS`。合并检查 registry/mass。

Schneider radiation-length JSON 可从 CT native 元数据 section、density、radlen 派生并对齐当前 schema，质量 radlen=`radlen_mm/10*density`；HU 映射来自 reference。来源 metadata 与 section0 delta-tail 校准审计需按实际新 campaign 重建，不能直接把旧时间戳当成新采样证据。

## 9. 电子／光子包移除

当前运行时仅保留离子 EM 能损和局部沉积，不再提取或安装电子／光子输运包。continuous26 和 delta moments 的提取仍按第 4 节执行。

## 10. 检查、安装和验证范围

```bash
python3 extensions/verify_kit.py
python3 scripts/check_data_manifest.py
```

后两条默认检查当前安装资源。电子检查器支持 `--root` 指向具备 config/docs/data 的独立候选仓库。候选先完成包字段/能量/覆盖/completion/zero-overflow 审计及匹配剂量检查，再成套安装 payload、companions、bundle/registry；仅在主动接受替包后，按 manifest 检查器 `--help` 更新摘要。

保持 fluctuations、Cu primary/fragment fluctuations、broadbeam0.5mm/minibeam0.1mm ion step caps、FP32 atomics，以及 campaign largest-remainder 权重、per-spot seed、正确归一化。本工具不改调度/源行为。

### 碳离子 TOPAS 剂量参考的核弹性过程

`g4h-elastic_HP` 不会为 GenericIon 注册核弹性。与当前启用碳离子核弹性的 GPU 对比时，TOPAS 必须另外加载官方 `G4IonElasticPhysics`。`topas/common/CarbonIonElasticPhysics.{hh,cc}` 将该构造器暴露为 TOPAS 模块，并在构建过程后检查 GenericIon 弹性是否实际注册；它不修改 Geant4 截面或散射模型。上述构建 profile 会自动 staging 此扩展；独立 PBS 剂量验证构建则将这两个文件与 PBS 扩展一起放入 `TOPAS_EXTENSIONS_DIR`。

在已有物理模块列表中增加 `"CarbonIonElasticPhysics"`，并同步增加 `sv:Ph/Default/Modules` 的元素数量。例如 RT07575 head 原七模块列表补为：

```text
sv:Ph/Default/Modules = 8 "g4em-standard_opt4" "g4h-phy_QGSP_BIC_HP" "g4decay" "g4ion-inclxx" "g4h-elastic_HP" "g4stopping" "g4radioactivedecay" "CarbonIonElasticPhysics"
```

启动日志应包含 `CARBON_REFERENCE_ION_ELASTIC GenericIon process=... registered=true`。完成后检查：

```bash
python3 scripts/check_topas_carbon_reference.py /path/to/topas.log
```

该检查仅验证 GenericIon 弹性注册，不代表剂量接受。2026-10-07 RT07575 夜间参考缺少这一过程，不能继续作为与当前 GPU 核弹性配置一致的参考；原始结果必须保留，补齐后重新模拟和计算 gamma。

剂量门槛：broadbeam local1%/1mm >99%、search0.1mm；proton minibeam local3%/0.2mm、search0.02mm，相对匹配旧基线下降≤1个百分点。绝对 TOPAS gamma 分开报告；carbon minibeam 剂量精度验收 deferred，runtime audits 仍必需。clean regression 是匹配 GPU 实现检查，不是新 TOPAS acceptance。性能实验前读 `../failed.md`，smoke 不支持性能结论。

2026-10-07 实测：原生六目标、TOPAS electrons/nuclear-copper 编译通过；其余四 profile 检查 staging。electron composition、G4_WATER material JSON、air proton ionization probes 通过。Urban air 两 species 完整 dense/held-out 流程8192 steps、零 failed queries，最大 query 相对误差约2.57e-6。另检查 Python/CLI/shell、来源摘要、材料卡与安装资源。记录见 [extensions_20261007.json](../docs/extensions_20261007.json)。这些结果不代表全量重提取、剂量接受或性能测试。

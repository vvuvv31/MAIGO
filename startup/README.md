# TOPAS 数据库提取启动包

这个目录把用于构建 GPU 物理数据库的 TOPAS scorer extension 和一键提取脚本集中在一起。默认提取 100、200、300、400 MeV/u 的 C-12 在指定材料中的数据；每个能量独立运行，便于在 TOPAS 或机器异常时只重跑一个能量。

## 目录

- `extensions/`：数据库提取所需的 TOPAS scorer 源码。TOPAS 的 CMake 会根据每个 `.cc` 文件第一行的 `Scorer for ...` 自动生成 `TsExtensionManager`，因此这里特意没有放生成的 `TsExtensionManager.cc`。
- `templates/database_extraction.txt.in`：自包含的水箱/均匀材料参数文件模板。
- `run_topas_database_extraction.sh`：顺序生成参数文件并运行 TOPAS；默认只做 dry-run，实际计算需要显式加 `--run`。
- `build_ct_material_packages.sh`：生成 lung/bone/Schneider section-7 的 400 MeV/u INCL++
  cascade n-tuple，并直接编译 primary reaction 与 charged cascade runtime
  package；默认 100k histories/material、最多 56 threads，同样默认 dry-run。
- `package_tools/`：将 TOPAS cascade ASCII n-tuple 校验、拆分并编译成
  `CRPKG`/`CCAS` binary 的标准库 Python 工具。
- `output/`、`work/`：运行结果和临时参数文件目录，已加入 git 忽略规则。

## 首次使用前手动配置

打开 `run_topas_database_extraction.sh` 顶部的配置块，至少修改：

```bash
TOPAS_BIN="/absolute/path/to/OpenTOPAS-install/bin/topas"
TOPAS_ENV_SCRIPT="/absolute/path/to/topas-setup.sh"  # 没有就留空
TOPAS_G4_DATA_DIR="/absolute/path/to/G4DATA"
TOPAS_LD_LIBRARY_PATH="/absolute/path/to/topas/lib:/absolute/path/to/geant4/lib"
TOPAS_THREADS=40
HISTORIES_PER_ENERGY=20000
PHANTOM_MATERIAL="G4_WATER"
```

`TOPAS_LD_LIBRARY_PATH` 和 `TOPAS_G4_DATA_DIR` 为空也可以；如果 `TOPAS_ENV_SCRIPT` 已经设置了这些变量，则可留空。脚本只负责运行已经编译好 extension 的 TOPAS，不会偷偷替换用户的 TOPAS/Geant4 物理库。

如果需要编译这些 extension，使用与目标 TOPAS、Geant4、GDCM 完全相同的构建环境，并将 `TOPAS_EXTENSIONS_DIR` 指向本目录：

```bash
cmake -S /path/to/OpenTOPAS -B /path/to/topas-build \
  -DTOPAS_EXTENSIONS_DIR=/absolute/path/to/MAIGO/startup/extensions
cmake --build /path/to/topas-build --parallel 8
```

生成的 TOPAS 必须包含下表中的 scorer 名称；否则运行时会报 `Unknown scorer quantity`。仅仅把 `.cc` 文件放在运行目录中不会把 extension 加入已有的 TOPAS binary。

## 运行

先查看将要执行的命令：

```bash
./startup/run_topas_database_extraction.sh --dry-run
```

确认 TOPAS 路径、材料和 history 数后顺序运行四个能量：

```bash
./startup/run_topas_database_extraction.sh --run
```

常用覆盖参数：

```bash
# 只跑两个能量，并把每个能量的 history 改为 100k
./startup/run_topas_database_extraction.sh --run --energies 200,400 --histories 100000

# 开启 32 种碎片/中性粒子的表格；文件和计算量会明显增加
./startup/run_topas_database_extraction.sh --run --species-tables
```

CT lung/bone/Schneider section-7 final-state package：

```bash
./startup/build_ct_material_packages.sh --dry-run
./startup/build_ct_material_packages.sh --run \
  --topas-bin /path/to/extension-enabled/topas \
  --materials schneider_section07,lung,bone \
  --histories 100000 --threads 56
```

该 TOPAS binary 必须由 `startup/extensions/` 编译，并使用与 runtime 数据
一致的 TOPAS 4.2.p3、Geant4 11.3.2 和 `g4ion-inclxx`。raw n-tuple 与中间
CSV 写入 `startup/work/ct_material_packages/`；最终 reaction/cascade v1
package 写入 `data/packages/`。当前 v1 把 Geant4 实际 local deposit 写入每个
interaction，是 CT dose 运行时的必要输入；旧布局不再兼容。

结果位于 `startup/work/e*`，每个 scorer 通常有一对 `.header`/`.phsp` 文件；`startup/output/run_manifest.tsv` 记录能量、材料、history、线程数、退出状态和输出文件。模板使用 ASCII n-tuple，因为级联/材料表包含字符串列；后续可用项目中的解析/编译脚本转换为 GPU lookup package。

## 默认 scorer

核心模式默认开启：

- `CarbonMaterialPropertiesNtuple`：密度和辐射长度；
- `CarbonStoppingPowerNtuple`：C-12 0.01--400 MeV/u 的电子/总 stopping power；
- `CarbonCrossSectionNtuple`：C-12 的 H/O 以及材料宏观非弹性截面；
- `CarbonReactionNtuple`：primary C-12 反应和一级碎片；
- `CarbonCascadeNtuple`：按 parent Z/A、incident energy、相互作用序列记录带电级联末态；
- `CarbonNeutralNtuple`：中性粒子反应与产生物。

`--species-tables` 另外开启 `IonStoppingPowerNtuple`、`IonCrossSectionNtuple` 和 `NeutralCrossSectionNtuple`。这些表用于提高不同碎片和中性粒子在 GPU 中的条件化精度，但会增加 TOPAS scorer 开销和输出体积。

本启动包不包含 `myHadronLET`；LET scorer 是剂量/数据库提取之外的独立扩展，避免把 LET 的输出开关和数据库生成流程耦合在一起。

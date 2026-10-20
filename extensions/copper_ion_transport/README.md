# 碳微束铜中带电碎片输运

这是独立的 Geant4 11.3.2 离线提取/验证工具，不链接到 GPU runtime。物理列表采用 QGSP_BIC_HP + EM option4 + IonINCLXX；直接查询各带电粒子实际挂接的弹性过程、截面和适用模型。未增加中性粒子输运。

## 编译与提取

先加载 TOPAS/Geant4 环境，确保 Geant4 数据集与动态库可用。从仓库根目录执行：

```bash
source /path/to/env_topas.sh
bash extensions/copper_ion_transport/build_extract.sh \
  --geant4-dir /path/to/geant4-11.3.2/lib/cmake/Geant4 \
  --build-dir extensions/work/build/copper_ion_transport \
  --urban-package data/scattering/urban/ct_copper.bin \
  --nuclear-package data/nuclear/copper/carbon/cascade.cinpkg \
  --output-dir extensions/work/copper_ion_candidate \
  --jobs 4
```

输出 `copper_fragments.bin`、`fragments.csv`、各自的 metadata 。工具将原 Urban 物种与当前铜 CINPKG04 中全部正能量带电 direct/continuation 产物合并（当前 324 种），防止 Na-24 等铜反应碎片缺少数据。输出目录必须是新目录，失败时保留候选供检查。轻粒子使用 native proton/deuteron/triton/He3/alpha 定义，其他使用基态离子。每物种 101 个对数能量节点（0.001～4641.59 MeV/u）、每节点 128 个自然铜弹性事件。固定随机种子 10070917。铜 Urban 包独立提取 8193 个 dense 节点与 1025 个 offset=0.37 held-out 节点，连同精确 MFP 边界探针验证；每项 FP32 相对误差须 ≤5e-4，才产生可安装包。元数据记录版本、物理列表、物种、网格、验证结果和文件摘要。原水/CT Urban 包不替换。

原铜 δ cut/spin 常量提取已移除；完整电子响应由 beamline_em 单独提取。提取器拒绝缺失过程、缺失模型、非两体末态和写入失败。

安装位置为 `data/elastic/copper/carbon/fragments.csv`、`data/scattering/urban/copper_fragments.bin` 及各自同名 metadata。先检查候选并验证，再复制到这些位置、重新构建 GPU runtime，更新 `data/manifest.json` 与 `examples/preset_dependencies.json`。两个碳微束预设通过 `minibeam_copper_fragment_elastic_file` 和 `minibeam_copper_fragment_urban_package_file` 引用这些包。

## 数值验证

`build_extract.sh` 自动调用 `validate_urban`，检查候选 Urban 包与 held-out
native Geant4 探针；失败时不生成可安装包。它检查的是 Urban 查表精度，
不验证涨落采样或完整计划剂量。统一铜涨落引擎的验证范围和已有结果见
主仓库 README 中的 Cu/air 涨落说明。

## 运行时范围

铜／空气平均能损、CSDA 和涨落统一使用 [beamline_em](../beamline_em/README.md) 提取的非受限响应。此处的物种 Urban／弹性包只负责散射与核弹性，原受限能损矢量不参与 runtime 的能损或射程查询。

`minibeam_copper_fragment_native_transport: true` 开启原生碎片 Urban 和物种弹性，碳默认开启、质子默认关闭。主粒子完整涨落开启，Cu／空气碎片完整涨落默认关闭；没有单独的硬 δ 抽样。真实路径用于全电子能损和核光学深度，几何路径用于位移。源质量、核反应及中性产物 sink、束线归一化保持独立。GPU 运行必须通过 completion/energy/coverage/zero-overflow 审计；有限弹性库收敛性及碳微束剂量验收仍需独立评价。

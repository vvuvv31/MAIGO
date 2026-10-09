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
  --constants extensions/work/copper_ion_candidate/copper_ion_constants.hpp \
  --jobs 4
```

输出 `copper_fragments.bin`、`fragments.csv`、各自的 metadata 和指定的常量头文件。工具将原 Urban 物种与当前铜 CINPKG04 中全部正能量带电 direct/continuation 产物合并（当前 324 种），防止 Na-24 等铜反应碎片缺少数据。输出目录必须是新目录，失败时保留候选供检查。轻粒子使用 native proton/deuteron/triton/He3/alpha 定义，其他使用基态离子。每物种 101 个对数能量节点（0.001～4641.59 MeV/u）、每节点 128 个自然铜弹性事件。固定随机种子 10070917。铜 Urban 包独立提取 8193 个 dense 节点与 1025 个 offset=0.37 held-out 节点，连同精确 MFP 边界探针验证；每项 FP32 相对误差须 ≤5e-4，才产生可安装包。元数据记录版本、物理列表、物种、网格、验证结果和文件摘要。原水/CT Urban 包不替换。

常量头文件记录 native Cu 的 0.05 mm 电子产生阈值、I、Fermi 能量、电子密度和各物种 spin/磁矩。提取器拒绝缺失过程、缺失模型、非两体末态和写入失败。

安装位置为 `data/elastic/copper/carbon/fragments.csv`、`data/scattering/urban/copper_fragments.bin` 及各自同名 metadata，常量头为 `include/carbon/copper_ion_constants.hpp`。先检查候选并验证，再复制到这些位置、重新构建 GPU runtime，更新 `data/manifest.json` 与 `docs/preset_dependencies.json`。两个碳微束预设通过 `minibeam_copper_fragment_elastic_file` 和 `minibeam_copper_fragment_urban_package_file` 引用这些包。

## 数值验证

```bash
extensions/work/build/copper_ion_transport/validate \
  data/scattering/urban/copper_fragments.bin > extensions/work/copper_native_validation.jsonl
```

验证器用 native G4IonFluctuations/G4UniversalFluctuation 对照 FP32 移植的受限涨落方差及采样矩，并用独立数值积分验证包含 δ 转移的总能损矩（含平均约 26 个候选/步的情况），覆盖质子、d、alpha、Li7、C12、O16、Ti48 和多个能量/步长。超出当前 Urban 因子表的测试点明确标为 OUTSIDE_BANK；运行时缺失覆盖会拒绝剂量，不会替代模型。验证结果只说明相应离线采样比较通过。

## 运行时范围

`g4_ion_delta` 用于两个碳微束：主 C12 与带电碎片使用 native 受限平均能损、G4 ion/universal 涨落和 δ 转移；碎片采用 Urban 相关角度/位移，并与按物种铜核弹性竞争。MSC 的几何步长用于位移，真实步长用于能损和光学深度。级联达到原有代数上限后停止新的非弹性反应，但继续弹性/EM 输运。

质子两个预设保留 `g4_universal_delta` 原有实现。C12 主粒子的 A×amu 校准运动学不变；碎片按 native 物种质量处理。δ 电子与铜核反冲能量仍记入束线移除能量，不向患者局部沉积。水/CT continuous26、strict26、电子/光子输运及核非弹性包不变。

有限弹性事件库的收敛性、铜 δ 电子的显式输运以及碳微束剂量验收仍未完成；GPU completion/energy/coverage/zero-overflow 审计必须通过，不能用 smoke 测试宣称 TOPAS 剂量精度或性能。

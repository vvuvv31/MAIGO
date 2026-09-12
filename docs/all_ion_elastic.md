# 全离子弹性输运（研究配置）

当前实现覆盖原发 C12 与既有 18 种带电离子，对 Schneider 的全部 13 个靶元素进行弹性反应；支持 Schneider CT 和统一水。不是仅 H，也不再把各向同性 CM 分布推广到所有靶。

## 数据与采样

- `data/schneider/all_ion_elastic_v1.bin`：18 projectile × 26 材料（25 Schneider + 水）× 13 元素反应率，137 个非均匀能量节点，0.001–6000.11 MeV/u。
- 每个 projectile/element/energy 节点有 512 个联合样本，保存靶同位素、核质量、`t/tmax`。靶同位素由 Geant4 截面存储器抽样，末态来自实际挂载模型的 `ApplyYourself`。
- 实际模型：proton 使用 hElasticCHIPS；d/t/He3/alpha 使用 hElasticLHEP；其余离子使用 NNDiffuseElastic。提取配置显式启用 `CarbonIonElasticPhysics`，不能假定历史 TOPAS benchmark 含有同一模块。
- 先按当前材料的元素 partial-rate 选靶，再按两邻接节点的 rate 加权抽样节点。重用联合靶同位素与动量转移样本，使用当前动能下的相对论两体运动学构造末态。
- 元素末态在保留天然同位素组成的纯元素材料中提取。材料依赖性、节点密度及有限样本数仍需进一步验证。

原发使用总弹性+非弹性宏观率消耗剩余光学深度，并按率选择反应类型。次级弹性距离与非弹性距离竞争，弹性不受非弹性代数上限限制；弹性反冲继承相应代数。步内 EM 能损后总弹性率归零的候选明确记为空碰撞，保留能量和轨迹；真实域外查询使整次运行失败。

## 反冲与能量守恒

修复了反冲方向不守恒、弹性提前碰撞后误推到 CT 边界，以及 cutoff 以下氢反冲丢失能量。两体运动学接受实际核静质量，守恒总四动量。

已有 18 种物种中的反冲进入通常的带电队列，继续适用的 EM、弹性和非弹性过程。其余反冲通过 `elastic_recoil_stopping_v1.bin` 覆盖的 37 个天然靶同位素获得材料特异总 stopping，显式减速与 MCS，而不是将高能重反冲全部在出生点沉积。**这些额外反冲物种是 EM-only 扩展，不宣称也已覆盖它们的后续核反应。** 总 stopping 包含电子与核 stopping；其中核 stopping 仍是凝聚近似，不生成进一步的显式碰撞级联。低于能量 cutoff 的残余能量局部沉积。

队列同时记录反冲出生与排队；溢出记录计数和能量，失败分片必须拆分重跑。缺失弹性查询或高能反冲 stopping 覆盖使运行失败，不回退旧数据。

## 启用

使用 `config/unified_water_elastic_research.yaml` 或 `config/rt07575_elastic_research.yaml`。后者保留本机病例输入绝对路径，迁移时需更新。

```yaml
run_mode: research
all_ion_elastic_file: data/schneider/all_ion_elastic_v1.bin
all_ion_elastic_sha256: 30a861486a8b21718815d1954dd8440e4dbad44f5afa224e67970c57413159e9
elastic_recoil_stopping_file: data/schneider/elastic_recoil_stopping_v1.bin
elastic_recoil_stopping_sha256: 96d48062b9a976caabb9774534ae486a079157e18ee37e4cbce261e5b2c3a4b2
```

CT 还要求 Schneider stopping 与 secondary exact-faces；LET 关闭。不能同时开启旧 `ct_elastic_diagnostic` 或 `enable_nuclear_elastic`。`run_mode: production` 暂不允许这套候选数据；闭合通过不等于全部临床剂量指标已经验收。

```bash
python3 tools/verify_schneider_v2_1_data.py
python3 tools/verify_all_ion_elastic.py
# 完整来源审计额外需要本机提取原始数据及可执行文件：
python3 tools/verify_all_ion_elastic.py --audit-provenance
```

旧 v2.1 manifest 与文件保持不变。11.3.2 Release 尚不包含这两个新 elastic 数据包；迁移新实现时需另行复制，不能只依赖旧 Release。

## 验证范围

相对论两体 420 组四动量测试通过；真实数据 GPU/host 2808 组查询对照通过；水和 RT07575 各 50k 闭合以及 RT07575 一个原始完整分片（6,481,909 histories）通过且零溢出。详情与具体构建 SHA 见 `evidence/step-31/all-ion-elastic-20260911/`。

仍需相同弹性物理列表的 TOPAS 对照、能量节点/末态样本收敛和全病例全统计验证，才能转为默认生产配置。现有 TOPAS 参考上的单分片 Gamma 比较只作为剂量变化诊断。

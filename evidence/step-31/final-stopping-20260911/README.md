# Final stopping repair — 2026-09-11

最终组合：primary C12 predictor-midpoint ON、secondary Schneider material-specific stopping ON、secondary exact-faces ON。
只运行这个组合，不新增开关消融运行。旧 v2.1 核数据及原发 stopping 固定文件保持不变。

## 修复

- 配置正式键与诊断别名冲突时拒绝运行；输出实际正式键、材料表路径与 SHA。
- 次级步首与中点使用同一 Schneider section × isotope 表，并只乘一次局部密度。
- 材料表开启后 CT 内域外查询使运行失败，记录首个 Z、A、section、E、rho，不回退水表。
- TOPAS 提取真实 unrestricted electronic stopping；原始逐行验证，除参考密度后入表；严格核验 schema、registry、grid、SHA。
- 补充主机/GPU 共用查表函数和能域/端点验证。

## 发现的能域问题

原 430.11 MeV/u 候选表在首个完整分片中出现 174 次域外查询。
记录的首例是质子：Z=1, A=1, section=5, E=438.569916 MeV, rho=0.993171 g/cm³。
失败运行的剂量未输出、未合并。目录保留：
`/mnt/sda/wuwei/final_stopping_20260911/20022516_domain_audit`。

350 MeV/u C12 撞击静止质子的相对论两体弹性最大反冲动能约为 1353.42 MeV：
`Tmax = 2 mp (gamma²-1) / [1 + 2 gamma mp/MC + (mp/MC)²]`（质量用 MeV/c²，c=1）。
这不是非弹性全产物的统一上限。扩展候选表采用保守范围 0.01–6000.11 MeV/u，步长 0.1 MeV/u。

## 提取与运行

TOPAS local sbatch job 4001，COMPLETED 0:0；1 CPU、10 GB 请求，耗时 88 秒。
二进制与全部原始数据：`/mnt/sda/wuwei/ion_section_stopping_20260911_extended/`。
附带 `stopping.metadata.json` 记录原始数据、提取二进制、组成材料及 SHA。
C12 与 pinned SCHNSTOP 在重叠 E>=5 MeV/u 域内最大相对差异 0.0006214279574。
主机/GPU 4500 查询最大相对差异 1.19159e-7；主机格式/域检查通过。

50k 闭合通过，物理相对残差 4.79504427138e-6，零溢出。
完整 20 分片通过，共 177,173,040 个原发粒子；全部质量报告 accepted，零溢出。
最大物理能量闭合相对残差见 `summary.json`，约 5.24e-6。
完整输出：`/mnt/sda/wuwei/final_stopping_20260911/20022516_extended/`。
冻结验证二进制保留为该目录的 `validated_carbon_mc`，SHA 见 summary。
`lung-final-shard01.yaml` 是实际运行的最终配置，含本机绝对路径。

```bash
python3 tools/run_final_stopping_validation.py \
  --binary /mnt/sdb/wuwei/MAIGO/build/oneapi-nvidia-stopping-final/carbon_mc \
  --table /mnt/sda/wuwei/ion_section_stopping_20260911_extended/compiled/schneider_ion_section_stopping_v1.bin \
  --reference /mnt/sda/wuwei/ct_previous_full20_20260909/20022516 \
  --out /mnt/sda/wuwei/final_stopping_20260911/NEW_OUTPUT --full20
```

扩展材料表仍是独立候选包，不覆盖 v2.1 权威 manifest；一个肺病例验证不等于全部 CT 病例已验收。

运行二进制在启动前冻结。运行期间仅修正源码中 stopping 日志的硬编码节点数
（4302 → `kSchneiderIonEnergies`）及注释；输运算法未变。当前运行日志的旧节点数
不代表实际表大小，实际大小由二进制表头、metadata 和严格加载器校验为 60002。

## 剂量比较

同一 TOPAS 参考、>=10% 最大剂量掩膜、绝对剂量尺度不调整。
空间 Gamma 使用 0.5 mm 球形搜索网格及三线性插值，非连续精确最小化。
0 mm 指标是同体素剂量差，通过率不能与 3 mm Gamma 混称。

| 指标 | 旧冻结 | 已有 midpoint | 已有 exact-faces | 本次最终组合 |
|---|---:|---:|---:|---:|
| global 3%/0 mm | 94.2793% | 94.2805% | 99.8042% | 99.8003% |
| local 3%/0 mm | 81.3497% | 80.6202% | 91.4058% | 90.8082% |
| local 3%/3 mm | 99.9386% | 99.9384% | 99.9386% | 99.9382% |
| local 1%/1 mm | 84.9795% | 84.9552% | 85.2884% | 85.4511% |

最终组合相对旧冻结和已有 midpoint 明显改善同体素结果；相对已有 exact-faces，
local 3%/0 mm 降低约 0.598 个百分点，而 local 1%/1 mm 提高约 0.163 个百分点。
不能声称材料表与中点积分对所有指标都带来提高，也不能把组合收益全部归因于 stopping。
这些历史结果仅被读取比较，没有新增开关消融模拟。

执行完整分析：
```bash
python3 tools/evaluate_final_stopping.py \
  --run /mnt/sda/wuwei/final_stopping_20260911/20022516_extended \
  --reference /mnt/sda/wuwei/ct_previous_full20_20260909/20022516
```
分析脚本拒绝覆盖已有报告；如需复算应使用独立结果目录。

最终源码构建通过，并使用相同最终组合再次通过 50k 闭合；final_build 的二进制 SHA
与质量结果见 summary.json。完整 20 分片仍绑定保留的冻结验证二进制 SHA。

此前恶化的 20–50% 参考剂量区间：midpoint 历史结果的局部误差 RMS 为 3.677%，
本次为 2.026%；低估超过 3% 的体素比例从 8.276% 降为 4.207%。
但相对已有 exact-faces 的 RMS 1.955%、低估比例 1.771%，这两个指标仍略差。
详细结果见 band-comparison.json，全部基于既有或本次完整 3D 剂量，无额外模拟。

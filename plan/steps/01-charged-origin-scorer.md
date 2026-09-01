# 01：Charged-origin Scorer 语义统一（COMPLETED）

## 目标

让 TOPAS 与 GPU 的 species dose 使用相同 origin 归因，禁止把 electron attribution 差异拟合进核模型。

## 实施步骤

1. 在 `/home/wuwei/topas` extension 增加 charged-origin 3D DoseToMedium scorer。
2. 每个带电核按当前 track 的 Z 建立 category；electron/positron 等非核带电后代继承已知 charged ancestor，neutral 后代进入 neutral-origin。
3. delta electron/positron 继承最初带电母离子；neutral 产生的电子进入 neutral-origin。
4. TOPAS 同时输出无过滤 total、旧 particle-filtered species、新 charged-origin species、neutral-origin。
5. GPU 保留 total 和 charged-origin species，并新增 neutral/unsupported/unclassified ledger。
6. 分析器默认只比较无过滤 total 与 charged-origin species；旧 scorer 只作 attribution diagnostic。

## 测试

- 1k 人工事件：primary ionization、fragment ionization、neutron/gamma→electron。
- 10k TOPAS smoke：逐 voxel 与积分 closure。
- 200 MeV/u 100k：确认 Primary C attribution 偏差显著缩小且 total 不变。

## 验收

- [x] TOPAS origin + neutral + unclassified 对 total 积分 closure ≤1e-6。
- [x] GPU category closure ≤1e-5。
- [x] electron/positron 不再造成 species sum 缺少 4%--10%。
- [x] scorer header 和 analysis JSON 明确记录 semantics。

## 完成证据（2026-08-31）

- Extension：`startup/topas_charged_origin_extension/ChargedOriginDoseToMedium.{cc,hh}`。
- 10k smoke：TOPAS job 278；origin sum 对 total 积分差 `-2.96e-8%`，最大 voxel 差相对峰值 `7.42e-8`，错误日志为空。
- 旧/新口径对照：job 279；旧七类 particle-filtered sum 比 total 少 `7.0443%`，Primary C charged-origin 比旧 scorer 高 `7.8009%`。
- 100k full-grid：jobs 280/281，200x200x800、0.4x0.4x0.5 mm；两组 total 逐 depth bin 完全一致，错误日志为空。
- 100k TOPAS origin closure：积分差 `6.38e-9%`，最大 depth-bin 差 `3.05e-8 Gy`，unclassified `7.02e-8 Gy`。
- G1 对 charged-origin：Primary C `-0.041%`、Secondary C `-1.95%`、B `-8.11%`、Be `-9.75%`、Li `-2.46%`、He `+0.89%`、Z1 `+0.18%`。
- 报告：`plan/artifacts/charged-origin-e200-100k/analysis.json`，并保存 smoke/group A/group B 配置。
- GPU scorer 语义复核：TOPAS 对每个带电核按该 track 当前 Z 分类；只有 electron/positron 等非核带电后代继承已知 charged ancestor。核 cascade 不跨代继承祖先 species。
- 正确 GPU 修复：`charged_dose_category()` 对所有 Z=1 isotope（p/d/t）统一返回 Z1；核后代继续按当前 Z 分类。跨代核 origin 实验已撤销。
- 200 MeV/u、100k、G1、seed 2026095100：Z1 `24.3753 Gy` 对 TOPAS `24.0174 Gy`，差 `+1.49%`；other charged 仅 `1.9e-6 Gy`；charged total 差 `-0.499%`。
- 当前 species：Primary C `-0.49%`、Secondary C `+0.29%`、B `+2.25%`、Be `-7.23%`、Li `-2.97%`、He `+1.01%`、Z1 `+1.49%`。
- 单元测试：p/d/t 均映射 Z1；`ctest` 2/2 通过。
- 可信分析：`plan/artifacts/cinel02-hybrid-currentz-e200/analysis.json` 与 `species_idd_comparison.png`。

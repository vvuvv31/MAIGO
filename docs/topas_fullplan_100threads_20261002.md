# RT07575 proton minibeam full plan：TOPAS 100线程吞吐

2026-10-02，AMD EPYC 9965，100物理核心，每核心一个逻辑CPU，TOPAS 100 workers。
实测持续吞吐约 **65,000 incident primary/s**。采用相同完整优化计划，完成200k预检、两个独立1M重复和5M长smoke。

| 粒子数 | 进程总耗时 s | 首个history开始 s | 输运及run收尾 s | 该段吞吐 primary/s | 中段10%–90%吞吐 primary/s |
|---|---:|---:|---:|---:|---:|
| 200,000 | 90.683 | 82.861 | 4.654 | 42,972 | 60,372 |
| 1,000,000 A | 101.527 | 82.328 | 17.017 | 58,766 | 64,680 |
| 1,000,000 B | 100.743 | 81.982 | 16.476 | 60,695 | 64,536 |
| 5,000,000 | 161.199 | 81.965 | 76.522 | **65,340** | **66,589** |

1M两次中段速率相差0.22%，平均忙碌核心97.42/96.43；5M中段平均99.94核。
5M持续输运超过一分钟，峰值RSS12.23GiB。输入按顺序覆盖不同能量/位置spot，局部窗口速率会随spot变化；
全计划吞吐应采用覆盖整个计划的平均值，而不是最快窗口。

TOPAS的`Execution Real`包含大量物理表准备：5M日志Execution157.198秒，但首个history在进程开始约82秒后才出现。
不能直接把Execution或短smoke的总wall当成稳态追踪时间。5M的完整进程吞吐仅31,018/s，主要是启动尚未完全摊薄。

## 0.1%重跑时间估算

原0.1% full plan为236,086,432 incident primaries。若TOPAS一次启动连续运行：

- 按5M全输运平均65,340/s外推，纯输运约60.22分钟。
- 加首批准备与退出，约 **61.6分钟**；实际排期可预留 **60–65分钟**，不含Slurm排队和后处理。
- 这是smoke外推，没有重跑整个0.1%；逐小批重启TOPAS会反复付出约82秒的准备开销，不能用上述单次启动时间估计。

## 输入、资源与检查

- 原始参考目录：`/mnt/sdb/wuwei/MAIGO_review_fix_20260923/benchmark/protonCT/RT07575_minibeam/`。
  二进制与原DIJ参考一致，SHA256 `4b4df74e15e08a4ece5393d9e6ec3f91fd660b5e5c4f4f5753a2852fb918d010`。
- 使用当前冻结的优化后`spots_optimized.csv`原始权重，largest-remainder整数化；保留原行、能谱、光学。
  1M和5M都含487个有效spot；200k预检因极小权重取整含485个。
- 把DIJ逐spot模式改为完整计划累积，保留原CT、Schneider材料、固定铜准直器与0.05mm Cu step、
  8个物理模块（含AllChargedUrbanPhysics）、0.05mm cuts、10GeV EM上限和全部次级输运。
  保留Patient DoseToMedium Sum与Copper EnergyDeposit两个scorer。
- 所有运行正常退出0、实际history数精确匹配；剂量有限非负且非零，417×505×35网格、Gy单位一致。
  核对Urban去除伴随Coulomb过程、100线程和10GeV表上限；铜沉积约占标称入射能量85.25%–85.37%。
  5M原生导航未计分告警合计0.00025874MeV，占标称入射能量4.47e-13；保留日志，无门禁豁免。
  本轮测吞吐，没有重新计算TOPAS/GPU gamma。
- Slurm jobs22872、22877均COMPLETED/0，申请100CPU、32GiB、nomultithread。
  Slurm账本显示200逻辑CPU是预留SMT兄弟的结果；执行前按(socket,core)去重并限制到100个逻辑CPU，
  TOPAS实际100workers（输运时含master共101线程），不是200workers。CPU affinity与逐秒资源采样已保存。

## 证据

`out/topas_fullplan_100threads_20261002/`：

- `prepare.py`、`execute.py`、`run.slurm`、`manifest.json`：可复核输入与100核运行方式。
- `pilot_200k/`、`fullplan_1m_a/`、`fullplan_1m_b/`、`extended5m/fullplan_5m/`：原始log、逐行history时间戳、
  逐秒CPU/RSS、剂量、铜能量与退出状态；扩展5M有自己的manifest和Slurm脚本。
- `final_summary.json`、`summary.json`、`extended5m/summary.json`、`sacct.txt`、`artifacts_sha256.json`：汇总和校验。

计时定义：全输运从history0的开始日志至`TOPAS run sequence complete`，包含worker启动/排空及run收尾；
中段使用10%至90%事件开始标记，作为排除首尾固定成本的辅助诊断。开始标记不是逐事件完成计数，
因此最终外推采用5M全输运平均，而不是把中段速率称为精确kernel吞吐。

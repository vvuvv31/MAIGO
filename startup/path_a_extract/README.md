# 历史 TOPAS 水涨落与弹性提取工具

保留的脚本用于生成独立 TOPAS 诊断数据，不是当前统一 EM/Schneider v2.1 生产数据的默认构建入口。

- `run_fluctuation.slurm`：C12 水中 EM-only 涨落数据。
- `run_elastic_events.slurm`、`elastic_target_only.txt.in`：C12 与 H/C/O 的旧弹性诊断。
- `stopping_power_water_campaign.csv`：历史 stopping 提取记录。

旧薄靶 FELB 事件库生成链已退役并移入 `trash/fred_cleanup_20260915/`。生产数据仍按 `data/ACTIVE_DATA.md` 及其固定清单校验。

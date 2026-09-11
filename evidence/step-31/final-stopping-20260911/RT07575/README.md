# RT07575 最新配置重跑（2026-09-11）

Primary midpoint ON、secondary Schneider material dE/dx ON、secondary exact-faces ON。
电子处理保留 section-0 transverse delta-tail，未启用材料电子包、joint 或 segment replay。
使用最新构建与扩展至 6000.11 MeV/u 的真实 TOPAS stopping 表，SHA 见 summary.json。
旧 v2.1 数据栈及扩展材料表均先通过校验。

50k 闭合通过；完整 20 分片共 129,638,170 histories 全部通过质量检查，零溢出。
完整二进制、配置、日志和剂量在 `/mnt/sda/wuwei/final_stopping_20260911/RT07575/`。
实际首片配置见 final-shard01.yaml；参考为 `/mnt/sda/wuwei/ct_previous_full20_20260909/RT07575/`。

评估恢复 packed_xneg 坐标映射，使用 TOPAS >=10% 最大剂量掩膜，绝对剂量尺度 1.0。
空间 Gamma 为 0.5 mm 球形搜索网格与三线性插值，非精确连续最小化。
3%/0 mm 为同体素剂量差。八项结果和旧结果差值见 summary.json。
本次未 commit/push，未做全局物理包升级。

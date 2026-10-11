# CT broadbeam primary涨落消融（2026-10-10）

原RT07575 head full plans，1 mm最大步长，电子输运关闭。仅将primary `enable_energy_straggling` 改为false：质子secondary涨落保持true；碳的对照与候选secondary均为false（前一轮已测快配置）。未改变平均阻止本领、射程终止、CT边界、散射、核反应及碎片输运；关闭开关后使用现有连续能损和delta能损均值。

每species每组1600万histories/1200万计时histories，四个配对ABBA+BABA；每campaign首批只排除计时，所有剂量参加计算。整个campaign源分配、per-spot seeds保持成对一致；跨campaign平均已归一化剂量，原计划归一化仅应用一次。

患者内reference≥10%max，历史TOPAS local1%/1mm、search0.1mm。没有剂量拟合或评价掩膜更改。

|粒子|primary涨落开启 /s|关闭 /s|增量收益|开启gamma|关闭gamma|
|---|---:|---:|---:|---:|---:|
|proton|423,346|581,400|+37.33%|99.947003%|99.947626%|
|carbon|171,242|196,223|+14.59%|99.465548%|99.482642%|

## 检查与统计解释

- proton：四个配对比值 [1.3595928583999846, 1.3691437724774476, 1.383621539877704, 1.3808419189318957]；患者评价区域剂量和变化+0.056304%，相对TOPAS偏移+0.002965%→+0.059270%。两组剂量相对L1差0.00944759，含有限统计误差。
- carbon：四个配对比值 [1.1410271173737172, 1.1448616201817448, 1.1576726444887764, 1.1400016659618368]；患者评价区域剂量和变化+0.002521%，相对TOPAS偏移-0.641788%→-0.639283%。两组剂量相对L1差0.003776，含有限统计误差。

64个保存批次均通过energy/EM/coverage/completion/zero-overflow运行检查，实际步长上限标记均1 mm。同一已验证GPU executable保持不变。两种粒子四配对均提速、累计患者gamma均>99%；微小gamma升高不表示精度改善。

配置消融通过本轮门槛；尚未覆盖其他患者、LET或minibeam，也未做8000万/组确认。历史TOPAS体素方差不可用；本机RTX2080Ti未锁频。碳增益相对于仅关闭secondary涨落的快候选，不能把它直接当作相对默认完整涨落的收益。

实验配置位于 `out/broadbeam_primary_fluctuations_20261010/{proton,carbon}_candidate.yaml`。默认八预设未修改，仍保留primary/secondary涨落。默认CT broadbeam步长仍1 mm。

复现命令、binary/config hashes、逐批结果和配对summary位于同目录；`run.py`串行执行两种粒子。现有quality报告的first-order说明文字假设涨落开启，实验实际开关以归档配置及manifest为准；本次未改动该固定说明或输运源码。

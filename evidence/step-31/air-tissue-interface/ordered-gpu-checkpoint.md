# 隔离GPU逐段重放检查点（2026-09-06）

状态：深度门通过；完整3D界面尚未验收；患者和生产禁止接入。

## 可复查输入与输出

- 原始参考：`/mnt/sda/wuwei/air_tissue_mass20_20260906`，质量世界3D DoseToMedium，双seed。
- 全56源候选：`/mnt/sda/wuwei/electron_ordered_all_runtime_20260906`。
- CSV SHA256：`55a1d0e3396742903bfd0b640513bf43d2401dffb592fe7956d7d27b8bc447ad`。
- metadata SHA256：`22e8ed6ab62e493441be2d7de8cce4a1c4e774c43dacddb54b2401c1f43c412a`。
- 路径binary SHA256：`d43f136d5b633afbed2e0af006897817ac86876804fafd001feadfb508cab6ce`。
- 最新比较：上述候选目录下 `gpu_step_convergence_002.json`；三label
  `gpu_ordered_all`、`gpu_ordered_all_step01`、`gpu_ordered_all_step002`。
- 独立电子阻止本领探针：`/mnt/sda/wuwei/interface_electron_stopping_20260906`，
  Slurm2464/2465完成，1001点，MeV/mm；尚未用于GPU输运修正。

## 结果与限制

0.02mm步长，两方向各20k GPU histories；深度ROI固定界面±5mm。
空气→组织最大窗口/单bin误差0.346%/0.638%，组织→空气0.469%/0.657%。
这是横向积分的深度门，不替代横向RMS、外侧能量份额或完整3D验收。
组织→空气首2mm RMS为3.393mm，TOPAS3.442mm；2–5mm为3.802/3.897mm。
参考和响应数据统计、步长误差仍待联合评估；不能以单seed细步长宣称完全收敛。

完整逐段候选与同CSV折叠路径对照已实际运行；不是未生效开关。
候选限定两材料和C12能量范围，不替换正式Schneider v2.1栈。
质量报告应只含 `unvalidated_electron_joint_response`，绝不能转述为accepted=true。

本检查点对应未提交工作树，不宣称clean源码冻结或生产复现资格。
下一步为细步长独立seed/源级统计验证；通过隔离界面后补覆盖，之后才患者与生产。

## 继续验证：细步长双seed与双buffer

本轮新增四次GPU运行，各20k histories：mass20双向seed1906176，
mass10双向原seed；步长0.02mm，同一binary/全56源响应，数据验证16/16。
无物理代码修改，不重编译运行binary，不改冻结参考或原始剂量。

新增报告均在上述runtime目录（独占创建，未覆盖旧报告）：

- `gpu_fine_step_seed_audit.json`：0.1/0.02mm及细步长第二seed，含TOPAS逐seed横向矩。
- `gpu_fine_step_buffer_gate.json`：`PASS_INTERFACE_DEPTH_WINDOWS_ONLY`，不代表完整3D通过。
- `gpu_fine_step_buffer10_radial.json`：较短buffer的横向结果。
- `source_fraction_uncertainty.json`：按campaign分层、源文件block bootstrap，2000次、seed1906177。

所有细步长比较中窗口最大误差0.474%、单bin最大0.700%；质量失败仅预期
unvalidated，未将accepted=false改成通过。GPU界面±5mm各窗口RMS的buffer变化
小于0.002mm；不能用外边界位置解释以下残留。

| mass20窗口 | GPU细步长两seed RMS (mm) | TOPAS两seed RMS (mm) |
|---|---|---|
| 空气→组织前2–5mm | 4.525–4.530 | 4.393–4.395 |
| 空气→组织前0–2mm | 4.261–4.262 | 4.138–4.204 |
| 组织→空气后0–2mm | 3.359–3.393 | 3.409–3.474 |
| 组织→空气后2–5mm | 3.793–3.802 | 3.880–3.915 |

这是观测跨度，不是双seed置信区间。部分横向残差超过所观察到的GPU seed波动；
不能把全部残差归为MC噪声，也不能据此唯一归因于材料电子阻止本领。

源统计保留实际能量损失权重，不能平均各源文件比率（175MeV/u源的最高bin
暴露量小，简单平均会严重失真）。例如170–175MeV/u的非局域份额：空气
0.3490、bootstrap区间[0.3471,0.3509]；组织0.1007、[0.0985,0.1027]。
这些区间只描述能量份额，不是横向RMS区间；没有计入路径压缩或材料模型误差。

新增横向审计2项、源统计1项测试通过；已有路径4项、响应4项、界面pilot2项通过。
P0保持IN_PROGRESS。下一步按预先固定的campaign/材料分层划分独立源组，
在同一0.02mm GPU配置检验横向响应样本敏感性（不选择有利组替换全样本候选）；
然后才判断是否必须引入逐段电子能量/材料条件响应。不得凭深度通过补生产覆盖。

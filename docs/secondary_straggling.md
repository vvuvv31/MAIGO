# 次级能损涨落默认值与验证状态

2026-10-12 按用户要求，将八个发布预设及 `TransportConfig` 的默认次级能损涨落设为关闭。患者与铜中 primary 涨落保持开启；微束同时关闭铜碎片受限涨落。配置仍可显式开启次级涨落用于对照。

```yaml
enable_energy_straggling: true
enable_secondary_energy_straggling: false
# 四个微束另有：
minibeam_copper_enable_energy_straggling: true
minibeam_copper_fragment_enable_energy_straggling: false
```

## 已验证的范围

下表中的“已验证”指所列 RT07575 **头部 CT 全计划**的配对实验，不表示全部水模体、其他患者或 LET 已验收。默认 CT 微束的肺部 benchmark 与这些头部实验不同。三项实验均关闭显式电子输运，不拟合剂量或做位置配准。

| 模态 | 状态 | 实验与剂量结果 | 测得吞吐变化 |
|---|---|---|---:|
| Proton broadbeam | 已验证：RT07575 CT 配对实验 | 16M/组，primary ON，患者 cap 1mm；历史 TOPAS local1%/1mm，99.954796%→99.954485% | +1.636% |
| Carbon broadbeam | 已验证：RT07575 CT 配对实验 | 16M/组，primary ON，患者 cap 1mm；历史 TOPAS local1%/1mm，99.559277%→99.556428% | +27.22% |
| Proton minibeam | 已验证：RT07575 CT 配对实验 | 16M/组，四配对 ABBA/BABA，患者/Cu/air cap 0.5/0.05/0.2mm，primary ON；独立 TOPAS1B，BODY≥0.4Gy，local3%/0.2mm，86.010032%→86.014023% | +0.816%，配对95%区间 −0.329%～+1.962% |
| Carbon minibeam | **Under verification / 验证中** | 有关闭患者次级及 Cu 碎片受限涨落的吞吐初筛；独立匹配物理剂量验收尚未完成 | 不将初筛视为正式性能/剂量验收 |

质子微束实验同时关闭患者次级及铜碎片涨落，源浮点字符串、spot 配额及种子成对一致；487-spot 优化后计划按整组16M最大余数分配一次，12批剂量各按 W/16M 归一并求和。全部24批 accepted、零溢出、最大能量残差 8.70e-8。

质子微束 control/candidate 暖态吞吐为160,328/161,636粒子/s（RTX2080Ti），四个配对速度比为0.99849、1.01261、1.01448、1.00690，尚无明确提速收益。GTV平均剂量变化 −0.000192%，配对95%区间 [−0.006382%, +0.005999%]；BODY评价区逐体素差异RMS为1.139232%，配对差异SE RMS为1.134367%。该统计量下未分辨出明显额外剂量偏差。

上述质子微束86%的绝对gamma属于16M筛选，不能与保持全部涨落的GPU1B结果99.761789%直接比较并解释为关闭涨落导致下降。1B参考模拟与16M配对筛选是不同统计量。该默认值调整由用户决定，不基于显著速度收益的结论。

实验依据：

- [质子宽束配对实验](broadbeam_preserve_primary_20261010.md)
- [碳宽束配对实验](carbon_ct_secondary_fluctuations_20261010.md)
- [质子微束配对结果](proton_minibeam_secondary_straggling_20261012.json)；原始输入、CSV与逐批审计在 `tmp/rt07575_proton_minibeam_primary_only_fluct_20261012/`。
- [碳微束历史初筛](carbon_minibeam_opt_20261010.md)

历史报告保留其当时“默认开启”的状态说明；当前默认值以上文及发布 YAML 为准。

## 开关语义

模体/患者 continuous26 次级路径关闭受限连续损失涨落及聚合δ矩抽样，保留相应平均能损、全部带电次级输运、散射、核反应和中性产物能量sink。它不关闭核反应或散射的随机性，也不改变束流初始能量展宽。

质子铜碎片路径关闭当前受限/总损失涨落调用，保留平均损失。`minibeam_copper_fragment_native_transport: true` 的原生铜碎片路径关闭的是 **restricted fluctuations**：仍执行硬δ Poisson/能谱/形状因子抽样，保留原有硬电离能损。其开关不等同于完全确定性总能损。开启时必须保持原损失与RNG序列；primary 调用显式保留完整涨落。

生产配置继续要求 primary 涨落开启；次级开关可独立设置。Cu数据、原生Urban、species弹性包、平均能损、核反应与全部运行审计要求保持有效。

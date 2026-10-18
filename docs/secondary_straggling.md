# 次级能损涨落默认值与验证状态

2026-10-12 按用户要求，将八个发布预设及 `TransportConfig` 的默认次级能损涨落设为关闭。患者与铜中 primary 涨落保持开启；微束同时关闭铜／空气碎片的完整能损涨落。配置仍可显式开启次级涨落用于对照。

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

模体／患者 continuous26 次级路径关闭能损涨落，采用非受限电子平均能损，并保留全部带电次级输运、散射、核反应和中性产物能量 sink。当前实现已删除受限／δ 分项抽样与介质电离时钟，具体切换范围见 [非受限输运](unrestricted_dedx.md)。它不关闭核反应或散射的随机性，也不改变束流初始能量展宽。

铜／空气碎片关闭的是完整电子能损涨落，此时使用非受限平均能损，不再保留硬 δ 抽样。主粒子则使用全均值、非受限方差和 Tmax 的一次 Universal/Ion 抽样。八预设默认开关保持一致；旧的受限＋硬 δ 分项实现已移除。历史验证数值对应当时冻结模型，不自动作为当前全区域切换的剂量结论。

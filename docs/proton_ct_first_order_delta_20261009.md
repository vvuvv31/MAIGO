# Proton CT broadbeam：主粒子一阶δ均值（2026-10-09）

用户允许 broadbeam 对 TOPAS 匹配适度下降，患者内 local1%/1mm >99%（search0.1mm）验收。本轮保留质子 CT local-electron primary 的一阶δ均值近似：省略每步两次邻近能量响应评估；不是逐位不变的物理实现优化。

## 实现

`UnifiedEmPoint::prepare`增加默认关闭的FirstOrderDelta模板参数。stage15/16的质子CT宽束local-electron主粒子启用它，将delta_slope置零，省略E±0.001E的continuous_rate/continuous_moments计算。显式能损原来的δ均值为max(0,δ0−0.5×(restricted_mean+δ0×h)×delta_slope)×h，当前主粒子采用δ0×h。

当前能量的continuous26反应率、δ均值与方差、restricted stopping及其斜率修正、受限能损涨落、δ涨落抽样保留。所有带电次级粒子、water、电子开启CT与其他管线默认保留原斜率。主/次级离子涨落开启，0.5mm步长上限、5% combined mean-loss guard、0.1MeV cutoff、边界及核反应竞争、核弹性/碎片输运、中性产物sink、FP32计分保持原配置。未改动hard-rate与conditional-terminal的显式校准近似或数据包。

`TransportResult::primary_first_order_delta_enabled`由实际专用kernel eligibility设置。只有实际执行该primary路径时，quality报告声明proton_ct_primary_first_order_delta；generic/fallback、水和电子开启CT不声明。没有增加运行配置开关或旧路径。

## 组件与GPU验证

327,600个species/material/density/energy条件、2,620,800次抽样，覆盖18物种、26材料、7个密度位置、100能量、4个步长比例和涨落开关。步长按native range、0.5mm与5%guard限制；非delta_slope参数精确相同，其他物种的loss/RNG相同，没有无效能损。对质子，相对总mean loss的δ均值变化p99=0.014550%，最大0.496107%。组件结果不代表完整轨迹的误差上界。

RTX2080Ti、CUDA AOT、禁用PTX JIT，完整RT07575 head计划870行/540正权重spot，原始浮点weight。CT、beam model、spot文件哈希与历史TOPAS输入匹配，placement/source optical参数保持一致；不用拟合缩放或registration。TOPAS参照为历史DIJ物理剂量按原始优化权重合成，没有新TOPAS运行，也没有其体素方差。

ABBA和BABA共四个独立16M campaign/arm。相同seed两臂source分配和每spot seed CSV逐字节相同；每个campaign全局largest remainder分配后分批，连续batch复用context/packages。每臂64M独立入射历史，首批剔除后暖批计时60M。合并64M剂量取四份已经按原计划入射粒子数归一化的剂量平均，不能求和当成单次计划。

| 项目 | 原δ斜率修正 | 主粒子一阶δ |
|---|---:|---:|
| 暖批primary/s | 321,366 | 362,522 |
| 64M患者内TOPAS local1%/1mm | 99.914893% | 99.915828% |
| mask剂量总和相对TOPAS | +0.010373% | +0.012820% |

合并提升 **12.81%**；四对speed ratio：1.114625, 1.145845, 1.150579, 1.104704，均快。GPU未锁频，温度/频率与时间存在变化，不保证跨机器固定收益。原/新有效EM能损抽样数几乎相同（减少约0.00579%）；改动主要减少每次准备的计算。

患者内mask为BODY∪GTV且TOPAS≥全体积peak10%，320,771体素。四个单独16M候选campaign和64M合并结果均通过>99% gate。候选−control的患者mask剂量总和变化+0.002447%。两臂128个性能/剂量批次quality、energy/EM/coverage/completion和零溢出审计通过，最大物理相对能量残差5.4279733e-07。源/种子相同，但物理近似改变轨迹后不要求核事件、步数或随机数消耗与旧模型逐位一致。

native stage16寄存器234→224，stack2376B保持；静态资源不作为动态spill或stall证据。未取得NCU硬件计数器。

## 最终程序与适用范围

正式性能候选SHA256 6cade6920a7dbf91429582c6acaee608cbd0fc288dae5e77b52c1c3e8d251722。随后加入host运行结果/quality近似标记，最终ELF SHA256 4bb5cdeb7126284ce05217b1115d3211a1baf4fe24e9ede6a289e7eea6b32120；三个嵌入GPU设备镜像均与正式候选逐字节一致，所以最终GPU实现就是所测实现。

最终原始full-plan每臂2M比较的是已测一阶候选与最终程序，检查dose/audit/source一致；水和重新开启电子CT各每臂40k与本轮之前基线比较，确认默认关闭模板参数及quality标记，均通过，同位置local1%均100%，最大剂量L1≤5.127e-8。这些是匹配实现回归，不是新TOPAS验收或吞吐结果。

当前可运行程序为build-proton-broadbeam/carbon_mc。完整五管线build/carbon_mc本轮未重新编译。config/proton_ct_broadbeam.yaml仍显式关闭电子并使用已接受的5%guard；预设的既有整数spots与正式验收的原始浮点full-plan不同，不能声称两者逐字节相同。精度qualification限RT07575 head患者内部，不外推肺、其他物种或minibeam。

证据：out/proton_ct_first_order_delta_20261009/，含冻结旧/新binary/source、component checker、ABBA/BABA、原始dose/ledger/quality/source CSV、device identity、focused validation与final build。

最终预设2×1M执行通过，quality只在实际启用路径记录first-order delta。共142个检查批次通过，最大物理相对能量残差3.76651631e-06；manifest（173资产/8预设）、dependency inventory一致性与git diff --check通过。

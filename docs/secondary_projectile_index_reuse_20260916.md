# 次级 projectile registry 索引复用实验（2026-09-16）

## 结论

把每次continuation入口已计算的`(Z,A) → secondary projectile registry index`复用于步内hazard、
He-4审计和步后碰撞查询，没有可推广的吞吐收益。RT07575五次交错配对中，Elapsed中位变化
为−0.065%，次级速度中位变化为+0.232%，两者均落在正负波动内；实际wrapper寄存器从187
升到228。候选由`CARBON_SECONDARY_REUSE_PROJECTILE_INDEX`控制并保持默认关闭，不进入组合。

这个实验只复用粒子寿命内不变的物种registry索引。步前与步后的能量、材料、密度、masked
partials、总反应率和靶核采样仍分别重新查询，没有复用任何随物理步变化的数据。

## 资源与性能

RTX 2080 Ti/sm_75，context-pointer range入口：

|指标|control|索引复用|
|---|---:|---:|
|wrapper registers/thread|187|228|
|wrapper stack/thread|264 B|264 B|
|wrapper constant0|408 B|408 B|
|Elapsed配对中位变化|—|−0.065%|
|次级速度配对中位变化|—|+0.232%|

五对Elapsed变化范围为−0.478%至+0.536%，次级速度变化范围为−0.659%至+0.897%。这说明
显式复用延长了索引live range并明显增加寄存器，但重复查找原本不是可测的吞吐瓶颈；编译器
也可能已在局部路径消除部分重复工作。

## 正确性

每次运行均为3,240,963 histories、1,034,976,717 steps、1,344,312次核相互作用，整数审计
一致、quality通过、overflow为0。候选五次3D剂量均值相对基线均值的最大绝对差为峰值的
0.00001705%，低于本轮五次基线的0.00002558%全局波动包络。

机器可读结果见`benchmark/secondary_projectile_index_reuse_20260916/analysis.json`。患者BODY
Gamma和低密度production-cut阈值精度验收仍未完成。

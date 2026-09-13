# TOPAS 离子能损表与积分阈值控制

**历史诊断，已被取代（2026-09-13）：当前验收固定默认 TOPAS，修复 GPU 的联合电磁处理。下文加密表/降低阈值的实验不作为默认参考或当前推荐配置。**

此扩展保留本轮已运行验证的实现，使参考端数值设置可显式复现；不是GPU stopping修正，也不会自动改变现有benchmark或CT默认配置。

将 DiagnosticIonLinearLoss.cc/.hh 纳入 TOPAS extensions 构建，在已有模块列表末尾加入 "DiagnosticIonLinearLoss" 并增加 Modules 数量。例如纯EM：

```text
sv:Ph/Default/Modules = 3 "g4em-standard_opt4" "g4decay" "DiagnosticIonLinearLoss"
u:Ph/DiagnosticLinearLossLimit = 0.0001
i:Ph/DiagnosticIonDEDXBins = 2560
```

模块必须在EM过程构建之后执行。它仅对GenericIon进程管理器中的ionIoni调用SetLinearLossLimit和SetDEDXBinning，要求恰好匹配一个过程；不会修改电子、质子或alpha各自的独立能损过程。验证时必须检查日志中GenericIon的实际表区间数与linLossLim，不能只确认配置项存在。它不新增弹性、非弹性或电子过程；完整物理运行需保留原有模块。

历史诊断使用的组合为2560区间与0.0001阈值。不要只在原160区间表上降低阈值：小步反演可被正反射程表不一致污染。此组合不是所有几何、材料、能量与粒子的普遍收敛保证。

## 2026-09-12 验证范围

TOPAS/Geant4 11.3.2、Water_75eV；GPU保持中点积分及straggling_scale=1。每组50000 histories。剂量始终用三维scorer后横向求和。下表为100/200/300MeV/u零能散、0.5mm深度网格的峰值误差，定义100*(GPU峰值/TOPAS峰值-1)：

| 物理 | MeV/u | 原参考 % | 加密表及小阈值参考 % |
|---|---:|---:|---:|
| 纯EM | 100 | -2.2312 | +0.9071 |
| 纯EM | 200 | -3.9157 | +0.1903 |
| 纯EM | 300 | -1.4506 | +0.2100 |
| 完整物理 | 100 | -2.7292 | +0.3093 |
| 完整物理 | 200 | -4.7746 | -1.2094 |
| 完整物理 | 300 | -2.5646 | -0.5269 |

200MeV/u纯EM的0.1mm网格峰差从+0.7624%到+0.4544%。完整物理200/300MeV/u峰位±3mm积分GPU仍低约1.44%/1.45%，不能认为剩余核物理差异已经解决。每组只有一个种子，没有独立重复的置信区间，也未验证全部CT、非零能散或所有带电离子的默认设置。

本地可复现资料：scratch/water_range_roundtrip_20260912（确定性表审计）、scratch/water_range_fix_200_20260912、scratch/water_range_fix_validation_20260912、scratch/water_range_fix_full_20260912。各目录保留输入、日志、数值与图。这些大体积诊断数据不随本扩展自动提交。

后续应保留旧参考，给加密参考明确版本；再做原发存活率/通量及反应深度分解。仅凭primary-origin剂量下降不能认定为核反应率偏高，因为该分类还包含关联电子沉积与输运差异。

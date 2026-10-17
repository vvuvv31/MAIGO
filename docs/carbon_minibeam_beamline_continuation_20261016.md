# 碳 CT minibeam：铜／空气原生步分段续跑

本轮保持所有物理模型与步长不变，只改变专用 CarbonMinibeam CT 上游主粒子的执行顺序。RT06423 Head2 原优化90°/270°完整计划，空气0.2mm、铜0.05mm、患者0.5mm；主涨落、Cu native hard delta、Urban、核过程和原生电离时钟保留。

## 实现

每64个原生准直器迭代保存128字节状态，按当前铜／空气材料与16个能量区间重排。保存位置、方向、能量、两套Urban状态／boundary标志、空气电离时钟／步号、铜共享核光学深度／步号、铜接触及全部能量账目。RNG仍由原始spot seed、history和原生step确定，不重新分配。source抽样是stateless，resume重复求值同一源身份／incident值，但从保存坐标和能量继续，不重复准直器前空气。

活跃数低于8192或经过64次续跑时，按原有100000步上限完整收尾，不截断轨迹。不活跃／终止history也只进入compact分区一次。状态槽使用chunk内原始history标识；不保存跨kernel失效的视图指针。铜子粒子、患者主次级均保留其现有物理和调度。其他七个预设与generic不启用这个上游续跑。

状态＋两个uint32索引约为每history136字节：2M默认上限约272MiB，A6000实验4M约544MiB，另有小型计数和分组数组。

## 两设备配对实测

2080Ti全五管线原生sm_75；A6000专用CMB原生sm_86，架构内比较的build selection匹配。所有执行CUDA_DISABLE_PTX_JIT=1，编译结束后开始计时，两卡同步运行。每campaign两批，首批排除；本地每批约3M，远端约6M。初始AB与独立种子反向BA；未锁频。全部72M粒子执行，不跨设备合并剂量或统计样本。

|设备|对照 histories/s|续跑 histories/s|合并暖态提升|AB／BA分别提升|
|---|---:|---:|---:|---:|
|RTX2080Ti|49,211.8|63,711.8|29.465%|26.443% / 32.536%|
|RTXA6000|95,344.6|110,310.3|15.696%|14.412% / 16.982%|

上游主粒子kernel累计时间下降：2080Ti 65.349%，A6000 53.891%。完整wall包含队列、分组、内存与输出开销。两卡采用不同batch/chunk/build selection，不能用这些数字推导纯硬件倍率；也不宣称其他病例固定收益。

当前2080Ti约63.7k/s，尚未达到100k/s。患者主粒子约占当前暖态总wall的36.8%，成为最大的单列阶段；患者次级约21.6%、铜直接子粒子约15.6%、上游约15.4%。其余包含铜后续级联，不能当成纯CPU／IO。进一步达到100k/s需要同时处理患者与碎片，单独继续压缩上游的收益上限已降低。

## 同源实现检查

32个完整计划field-batch运行／能量／EM／覆盖／完成／零overflow审计全部通过。每设备、每seed内的source/非能量整数/离散Urban、EM、核、elastic、live-ion计数相同。最大dose relative L1=7.673e-08，最大相对能量残差=8.370e-07。剂量差为FP32累加顺序级别。

正式命名后的原生sm_75／sm_86完整计划bridge共8个field-batch全部通过；八个发布预设及generic carbon CT／water minibeam也全部通过两批20k同源回归。全部非能量整数一致，零overflow。正式候选与计时原型的差异限于标识符／说明和补全launch数显示；不增加物理近似。

这不是新TOPAS剂量验收；carbon minibeam独立精度状态仍保留。暂停的1B任务未恢复。当前基准、源码快照、构建／输入／binary hashes、逐campaign剂量与审计均在`tmp/carbon_mb_upstream_continuation_20261016/`。

旧profile在空气0.05mm下的步数和周期占比不能作为当前值。当前原生上游剖析见 tmp/carbon_mb_upstream_current_profile_20261016/report.md：17.533%的history实际同时过铜和狭缝，贡献47.584%的狭缝步；这支持按实时状态重排，而不是重试失败的静态spot／直线入口排序。

正式整合全部检查最大dose relative L1=3.110562e-07，最大相对能量残差=5.837314e-06。上述小批回归不用于性能宣称。精确测试的CLI／静态库已安装到`tmp/remove_electron_20261015/build/`。完整证据与hash索引：`tmp/carbon_mb_upstream_continuation_20261016/COMPLETE.json`。

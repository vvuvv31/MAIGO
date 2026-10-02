# 水/CT minibeam 公共路径与性能验证

本轮保留构建修复和公共路径合并。两项微优化均未取得足够稳定的端到端收益，
已从交付源码撤回；不宣称本轮实现了稳定的吞吐提升。

## 构建修复

补回 `CARBON_CT_ELECTRON_QUEUE`、`CARBON_FE_JOINT_QUEUE`、
`CARBON_ELECTRON_REPLAY_REUSE`、`CARBON_ELECTRON_CROSSING_ROWS32` 的 CMake 定义。
此前 cache 显示 ON，但没有对应编译宏；CT 在输运前拒绝执行。
现在实际编译命令包含四项。OFF 配置也验证了前三项宏保持未定义、FE 项为0，
与源码中的 `#if defined` / `#if` 语义一致。所有构建均为 FP32 dose ON、FP64 OFF。

修复后、合并前的源码和有效基线独立保存在
`out/minibeam_unify_20261001/baseline_source/`、`baseline_build/`。
原打包二进制保存在 `shipped_carbon_mc`，不把它作为不可运行CT场景的性能对照。

## 公共架构

- 把原内联源采样、铜/空气输运和介质入口交接抽入
  `src/detail/sycl_primary_beamline.inc`，保留原步骤、RNG和能量操作。
- `PrimaryWaterEntryState` 更名为 `PrimaryMediumEntryState`，布局仍为64字节；
  条件铜响应研究ABI没有改变。
- compact水/CT共用stage3/9的编译实例。上游Urban选择独立于下游介质策略；
  host准入已要求split路径使用全局Urban。
- stage4继续按水/CT分别编译。材料与边界、电子响应、group32抽样、中子与评分配置保持原样。

静态CUDA资源检查确认，旧的CT stage3/9重复实例不再产生；
当前工作不是将患者输运套用均匀水替换，也没有合并水响应与CT材料响应模型。

## 已完成的独立回归

修复构建前后质子水10k、合并前后质子水10k/CT10k/碳水10k：
入口相空间逐字节相同，整数ledger/transport/EM审计一致，quality accepted，队列无溢出。
相对剂量L1最大4.82e-8。所有运行保留研究模式的原物理限制。
碳预设使用绝对spot权重，小样本夹具显式复制单spot并修改其history权重；
不能只修改 `number_of_histories`。夹具也会核对实际完成历史数。

公共路径r1的CT两组500k反向顺序运行，完整墙钟变化+7.899% / -7.164%，合计+0.352%。
铜/空气和患者主核时间基本不变，启动加载变化主导完整墙钟，且当时CPU仍在编译另一候选。
因此不宣称合并本身带来吞吐收益；合并作为共同维护与后续优化基础保留。

## 水响应审计分片：未采用

独立r1二进制支持同核1/64分片对照，不改变逐步固定点整数转换。
RTX2080Ti，两组独立4M、反向顺序：

| 分片 | 第一组完整秒数 | 第二组完整秒数 |
|---|---:|---:|
| 1 | 28.9563 | 29.4008 |
| 64 | 29.1893 | 29.0375 |

两对吞吐变化-0.798%/+1.251%，合计约+0.224%，未显示稳定有用收益。
整数审计、quality/能量/队列通过，剂量L1≤1.853e-7。
已撤回分片源码和运行开关；`candidate_r1_carbon_mc` 与
`shards_rejected_transport_sycl.cpp` 仅作冻结证据。
“有大量原子操作”不能单独证明原子争用是瓶颈。

## 短步几何候选：未采用

候选仅用于proton compact路径：≤0.05mm步按横向可达区间限制slit索引，
两侧各保留额外邻居；沿用原有限slab/圆柱求交和FP32边界规则。
长空气步继续完整扫描。没有修改步长、Urban safety、核反应或粒子保留规则。

冻结候选的 `tests/minibeam_short_step_geometry.cpp` 覆盖狭缝壁及邻接浮点坐标、
掠射、旋转、有限端面、圆柱边缘、长空气路径及随机几何。
GCC和项目Clang分别完成1,441,000次比较，返回值逐位一致。
CPU几何比较不替代完整GPU输运验收。

同一二进制三组4M OFF/ON：完整墙钟合计+0.848%，三对+0.218%/+2.155%/+0.168%；
beamline平均12.0636→11.8449s，水主核10.4778→10.5263s。
随后用另外三组独立4M，直接比较修复后的旧二进制与合并+几何候选：
134,777→135,402 histories/s，合计仅+0.464%，三对-0.348%/+1.706%/+0.038%。
没有足够稳定的端到端收益支持默认采用；不能把局部计时改善直接认定为整机加速。
所有精确整数审计、quality/能量/队列通过，剂量L1≤1.851e-7。

几何候选和开关已从默认源码移除，完整源码/测试保留于
`out/minibeam_unify_20261001/geometry_rejected_source/`，冻结二进制为 `geometry_carbon_mc`。
当前主源码继续使用原始完整slit扫描。

## 复现实验

`scripts/compare_minibeam.py` 对每组使用相同种子，交替执行顺序；
脚本需NumPy与PyYAML；常规构建/运行入口没有新增这些依赖。
每个运行目录保存配置、二进制SHA、完整日志、GPU温度/时钟、quality、剂量和计时。
输出目录必须不存在，以防覆盖旧证据。`--phase` 用于小样本入口相空间比较。
完整进程墙钟包含初始化与输出；同时保留beamline、介质和响应队列计时。
整数审计必须一致，浮点剂量差异单独记录，不以逐位剂量相等作门槛。

```bash
python3 scripts/compare_minibeam.py \
  --baseline out/minibeam_unify_20261001/baseline_build/carbon_mc \
  --candidate build/carbon_mc --preset proton_water_minibeam \
  --histories 4000000 --seeds 100113501 100113502 100113503 \
  --output out/my_water_pairs
```

本轮原始证据：`out/minibeam_unify_20261001/`。
失败或未获稳定收益的候选均另记于 `failed.md`。

## 最终交付二进制验证

最终 `build/carbon_mc` 仅保留公共路径合并与构建修复，已移除两项未采用的微优化。
使用FP32 dose scoring/atomics（FP32 ON、FP64 OFF），水参考为Water_75eV，stopping scale为1.0。
二进制SHA256：`8c52a6b117579bdb2a381d67811e1d0d047a65d275c204115c36f099ef91148a`。

| 最终回归 | 历史数 | 剂量相对L1差异 |
|---|---:|---:|
| 水入口相空间 | 10,000 | 4.7852e-8 |
| 质子水模体 | 4,000,000 | 1.8253e-7 |
| 质子CT | 500,000 | 3.5475e-9 |
| CT内联Cu次级粒子（Cu child queue OFF） | 5,000 | 3.2195e-10 |
| 碳离子 | 10,000 | 4.1844e-8 |

五项回归均通过研究模式quality检查，整数ledger/transport/EM审计精确一致，
队列溢出为零，能量门槛通过；最大相对能量残差为9.1013e-7。
启用入口相空间检查的用例均逐字节一致。
最终源码规范化对照、二进制与源码清单核对、Python语法检查及 `git diff --check` 均通过。

最终单次完整墙钟：水模体4M为28.17895秒（141,949.94 histories/s），
CT500k冷启动为38.99887秒（12,820.88 histories/s）。
这些是交付回归计时，不是成对性能实验，不作为加速证据。
本轮没有建立稳定的端到端吞吐提升。

详细结果见 `out/minibeam_unify_20261001/delivery_checks.json`；
交付状态、源码清单与规范化源码对照分别见同目录的
`delivery_complete.json`、`delivery_manifest.json`、`delivery_source_check.json`。

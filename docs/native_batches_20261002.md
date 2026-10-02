# 原生单进程多批次（2026-10-02）

## 使用

```bash
export LD_LIBRARY_PATH=/home/wuwei/sycl_workspace/llvm/build/install/lib:${LD_LIBRARY_PATH:-}
./build/carbon_mc \
  --config benchmark/protonCT/RT07575_minibeam_GPU_0p1pct/batch0000.yaml \
  --histories-per-batch 2000000 --batch-count 10 \
  --batch-output out/rt07575_native_10x2m
```

总计 20,000,000 incident primaries；上述配置保留 group32。它采用该配置源文件的相对 spot 权重；
该文件来自此前 0.1% campaign 的一个已整数化子批次，不应宣称与未取整的原始优化权重完全相同。
需要严格保留原始优化权重时，应给定原始源文件和相应权重。

三个新参数必须同时使用。`--histories-per-batch` 对 TPS histories/MU 两种模式、TOPAS channels 都表示整批总数，
会覆盖原始绝对历史数/全局 histories scale；保留相对权重。普通非计划源直接运行指定数量。
不与 `--histories`、`--plan-manifest`、`--sequential-spots`、canonical-config 导出或 secondary replay 混用。

`--plan-preflight` 检查配置、源转换、整数分配、种子范围与输出冲突，不创建输出，不做输运。
输出根目录必须不存在或为空，不自动覆盖已有结果，不提供续跑。运行中某批质量失败时停止后续批次；
根目录清单是计划，成功与否以各批 `quality_report.json` 的 `accepted` 和进程退出码为准。

## 源分配、输出与缓存

- 先按全体 N×H 粒子进行 largest-remainder 分配，再将各 spot 分配数均衡拆到 N 批；每批严格 H。
  允许单批低权重 spot 为零；合计各 spot 相对理想分配偏差小于一个粒子。
- 种子是 `base_seed + (batch_index × source_spot_count + spot_index) × 1000003`；
  普通源的 source_spot_count 为 1。零历史 spot 被省略时不改变其余 spot 的种子编号；检查整数溢出。
- 每批独立 `batch000000/` 目录，剂量/LET/诊断输出按原文件名重定位，质量与能量报告也在各批目录。
  `batches.csv` 记录批次、历史数、种子和输出目录；`primary_batches.csv` 记录 history ranges、种子及 20 个源参数。
  `base_config.txt` 保存输入路径和基础归一化配置，实际批次覆盖以清单和源表为准。
- `SyclTransportContext` 复用现有 Unified EM、Schneider/CINEL03、CT 电子响应库与 CT 数组。
  新增类型化只读 host cache 保存 Urban、弹性/反冲、铜 CINEL03、铜/空气 stopping、离子 stopping/XS、loss-range 等解析结果。
  键含绝对路径、文件大小、mtime、适用的预期 SHA pin 与依赖；loader 首次校验不跳过，异常不进入缓存。
- 常驻 CT 电子 bank 只在兼容的 research CT 配置启用；CT 几何和密度/材料数组的跨批复用已独立于该开关，适用于普通 CT broadbeam。水模体也使用同一批次入口。
  每批重新初始化剂量、粒子状态、队列和审计计数；FP32 ON、FP64 OFF，输运与物理近似不变。
- 不是“所有初始化降为零”：部分设备表上传、工作区分配、轻量 air metadata/rate 读取、完整性检查及输出仍按批执行。
  后续共用缓存扩展已覆盖水响应、carbon 联合响应、中子碰撞/反冲停止本领包；仍不承诺所有可选诊断和输出都不再读文件。该实现没有自动扩大单批粒子数或队列容量。

不同批次的原始总能量/Gy 文件可累加；若要换算回某个 full-plan prescription，须按实际总 incident primaries 和原计划权重计算缩放。
不要把单批剂量各自缩放到完整处方后再直接求和。

## 验证

验证证据保存在 `out/native_batches_20261002/`。`tests/native_batches.cpp` 检查整个计划的整数分配、
极小批次与零历史 spot、可重复且不重复的源种子、缓存类型隔离/异常、计数与种子溢出、冲突参数，
以及 proton/carbon water 与 TPS CT 的配置兼容。

实测：RTX 2080 Ti，RT07575，group32，3×1,983,920 primaries，相同完整 spot 权重与逐批种子；
旧版用已有的 `--plan-manifest` 同进程运行，新版直接用三个批次参数。

| 项目 | 旧版 manifest | 原生批次 |
|---|---:|---:|
| 首批准备（秒） | 21.231 | 20.255 |
| 第2批准备（秒） | 1.453 | 0.823 |
| 第3批准备（秒） | 1.330 | 0.571 |
| 三批输运总计（秒） | 29.258 | 29.346 |
| 三批进程 wall（秒，含退出） | 54.770 | 52.483 |
| 第2/3批完整 wall（秒） | 11.392 / 11.428 | 10.749 / 10.634 |

这是一次功能/耗时对照，非多轮稳定性能结论；旧版运行期间主机上还有新版链接作业，启动时间尤其不宜作精确因果比较。
输运耗时基本不变。新增缓存首批11 misses，后两批均11 hits/0 misses；现有大电子库也命中。
两版本 tracked device peak 均4.884692 GiB，连续三批不增长（该计数不包含所有电子库/驱动显存）。
避免每批另起进程的收益来自摊薄约20秒冷准备；旧版 manifest 本来已经有部分跨批复用，不能把全部收益归因于本次改动。

三批均 `accepted=true`、零 queue overflow；Urban、Unified EM、弹性及级联整数审计逐项相同。
逐体素剂量相对 L1 最大8.07e-8；在参考峰值0.1%以上体素的最大相对差2.01e-6；
最大能量残差8.32e-8。符合仅 FP32 累加顺序差异，不引入新的输运/电子近似。

另外，proton/carbon water broadbeam、proton TPS CT 各2×1000通过；
carbon CT joint-response 首批保留既有 `unvalidated_electron_joint_response` 失败，退出1且第二批没有剂量输出，验证了失败即停止。
该 carbon CT 结果不算质量通过，不关闭原有门禁。十二项 CLI/预检检查通过，预检无输出。

主程序 SHA256：`9b9b47f7fa31a5187aa9fb7f297bcfa691dfb481f0b07d412178b4c11a18d2e7`。
原始日志、逐批 quality/energy/dose、清单、源分配与对照统计位于 `out/native_batches_20261002/`；
`comparison.json`、`smoke_checked.json` 和 `preflight_checks.json` 是汇总证据。

水模体两批还分别与旧二进制的独立同种子运行对照：proton 相对 L1 ≤2.79e-9，carbon ≤9.89e-8，
确认每批计分没有沿用上一批结果。证据 `water_parity.json`。

运行 host 分配/缓存测试（先完成正常 FP32 构建）：

```bash
g++ -std=c++20 -O2 -no-pie -Iinclude tests/native_batches.cpp src/batch_run.cpp \
  build/libcarbon_core.a -lcrypto -pthread \
  -L/home/wuwei/sycl_workspace/llvm/build/install/lib \
  -Wl,-rpath,/home/wuwei/sycl_workspace/llvm/build/install/lib -lsycl \
  -o out/test_native_batches
./out/test_native_batches \
  benchmark/protonCT/RT07575_minibeam_GPU_0p1pct/batch0000.yaml \
  out/native_batch_test_must_not_exist
```

2026-10-02 后续扩展：所有 GPU 入口持有进程级 context；普通 manifest、水响应、中子包和 broadbeam CT 缓存见 [共用缓存说明](shared_batches_20261002.md)。以上表格和二进制 SHA 是首次原生批次实现的历史结果。

# RT07575 runtime breakdown — 2026-09-14

当前工作树隔离快照；本地 RTX 2080 Ti / sm_75，100 万原发，完整次级统一 EM、种类分组、原生涨落，生产配置不含独立核弹性。history_chunk_size=34816。

源码和配置见 `scratch/runtime_breakdown_20260914`，原始运行见 `scratch/unified_em_perf_20260913/runtime_current_1m_*`。此处不是之前含弹性的 1M 实验。

未插桩基准：19585.9 histories/s（程序 Elapsed=51.057 s）；进程墙钟 52.775 s。

| 不重叠阶段 | 秒 | 完整进程占比 |
|---|---:|---:|
| Input/source preparation | 0.4410 | 0.84% |
| Transport setup (includes uploads) | 4.5060 | 8.54% |
| Primary kernels | 25.5945 | 48.50% |
| Secondary kernels | 21.6397 | 41.00% |
| Loop host/synchronization/other GPU work | 0.0087 | 0.02% |
| Readback/host finalization | 0.0565 | 0.11% |
| Quality/output | 0.4417 | 0.84% |
| Other process time | 0.0869 | 0.16% |

下列 CUDA 传输时间来自 Nsight 采集，是上述阶段的内部活动，不能再次相加：

| CUDA transfer | 秒 | bytes | calls |
|---|---:|---:|---:|
| Host-to-Device | 0.138604 | 1791399990 | 45 |
| Device-to-Host | 0.004420 | 57741256 | 37 |

Nsight 程序 Elapsed 增加 2.01%；输入/二进制 SHA、步数和 EM audit 已核对。

CT load/preprocess、包加载等嵌套 wall scopes 保存在 results.json，不将其与父 scope 重复求和。此配置读取 packed CT，不包含从原始 DICOM 重新建网格的全流程。没有独立 WET map kernel，CT 材料查询/边界推进在输运内；剂量累加也融合在输运 kernel 中。

## Kernel 内采样诊断

诊断构建 Elapsed=52.415 s，较未插桩基准变化 2.7%。每约 2048 个粒子步抽一个 lane，记录 SM clock 周期；不消耗物理 RNG。

以下是抽样 lane elapsed cycles 比例，包含 warp 分歧、等待和插桩影响，不是各物理过程的独立 GPU wall seconds；嵌套 EM 子项不与外层相加。

### Primary

| Region | sampled cycle share | intervals |
|---|---:|---:|
| CT/material lookup and initial step state | 4.51% | 1075545 |
| EM table/range/rate preparation | 22.66% | 1075545 |
| Geometry clamps/nuclear rates/pre-loss work | 32.52% | 1075545 |
| Unified EM loss | 29.67% | 1075545 |
| Scoring/electron-response branches | 2.23% | 1075545 |
| MCS | 4.38% | 1075545 |
| Advance/face handling | 2.08% | 1075545 |
| Nuclear resolution/loop bookkeeping | 1.95% | 1075545 |

### Secondary

| Region | sampled cycle share | intervals |
|---|---:|---:|
| CT/material/stopping lookup and initial state | 4.78% | 330793 |
| EM table/range/rate preparation | 34.12% | 330667 |
| Geometry clamps/nuclear rates/pre-loss work | 15.70% | 330667 |
| Unified EM loss | 32.78% | 330667 |
| Post-EM geometry/scoring | 5.24% | 661196 |
| MCS | 4.46% | 330529 |
| Inelastic replay/scoring branches | 2.49% | 330667 |
| Elastic/loop bookkeeping | 0.44% | 330529 |

### Primary EM nested

| Region | sampled cycle share | intervals |
|---|---:|---:|
| Mean loss | 24.25% | 1075545 |
| Fluctuations | 49.64% | 1075545 |
| Delta clock/acceptance/spectrum | 26.11% | 1075545 |

### Secondary EM nested

| Region | sampled cycle share | intervals |
|---|---:|---:|
| Mean loss | 20.02% | 330667 |
| Fluctuations | 56.01% | 330667 |
| Delta clock/acceptance/spectrum | 23.97% | 330667 |


后续已进一步拆分几何/核率和重复平均能损，见 [DEEP.md](DEEP.md)、[结论](CONCLUSIONS.md) 与 [深层分解图](deep_breakdown.png)。

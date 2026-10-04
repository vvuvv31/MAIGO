# Carbon CT minibeam 三包路径吞吐初测

2026-10-04，RTX 2080 Ti，同一肺 CT、250 MeV/u C12、3×3 cm 平坦源和铜准直器。新独立 carbon + shared 包与旧 endpoint v7 路径分别在单进程中连续运行 4×100,000 primary。Group32；primary chunk / electron queue chunk 均为 16,384，lanes=16,384；GPU 剂量均为 FP32。未修改物理参数或进行调度优化。

| 指标 | 新 carbon + shared | 旧 endpoint v7 |
|---|---:|---:|
| 驻留吞吐，包含各批输出 | **3,650 primary/s** | 3,252 primary/s |
| 驻留吞吐，仅 transport_total | 3,713 primary/s | 3,301 primary/s |
| 含首次加载的全进程吞吐 | 3,361 primary/s | 2,244 primary/s |
| 40 万粒子完整墙钟 | 119.00 s | 178.27 s |
| 首批 transport setup | 9.06 s | 51.03 s |
| 驻留电子处理 / 10 万 primary | 6.640 s | 4.313 s |
| 电子运行库 | 3.303 GiB | 29.359 GiB |
| 进程主存峰值 RSS | 7.474 GiB | 31.092 GiB |
| 每秒采样的整卡显存峰值 | 7,815 MiB | 9,763 MiB |

驻留统计使用第 2–4 批，按总粒子数除以 `transport_total + quality_and_output` 累计时间计算；不含首批加载。新路径三批分别为 27.123 / 26.963 / 28.105 s，旧路径为 30.299 / 30.437 / 31.521 s。`transport_total` 包括每批准备、输运、读回，并非纯 GPU kernel 时间。

本次驻留吞吐新路径高 **12.25%**，整个短作业耗时降低 **33.25%**；后者很大一部分来自初始加载。**电子阶段本身更慢约 54%**，不能把整轮收益解释为电子 kernel 加速。新路径 CT primary 阶段也较快，但本轮没有硬件剖析，不能据此归因于某一种内存访问或寄存器变化。

这是一次顺序配对初测，不是 ABBA 性能认证。两臂记录 GPU 温度、时钟和显存；不同运行次序及缓存状态会影响首批加载时间。磁盘三包拆分本身不改变模型；此处比较的是拆分后的新独立电子模型与旧轨迹模型。

## 完整性与范围

新路径 smoke 1,000 primary 通过后才启动计时。两臂共八批 quality 均通过（research / non_production），primary / Unified EM 等打印审计一致，队列溢出为零。新路径全部电子包完成，coverage_missing=invalid=photon_gap=0，legacy_loaded=legacy_fallback=0，第二至第四批命中 retained bank 缓存。能量相对残差最大约 1.30e-6。

未在本轮重新计算 TOPAS gamma 或剂量收敛，运行质量检查不能代替剂量验收。

**不是 RT07575 头部 CT 吞吐。** 测试前重新扫描实际 CT：当前 C12 出生包在肺 CT 无材料/密度缺口，但 RT07575 头部仍有 137,985 个体素未覆盖，缺口涉及部分高密度节点。遵守此前记录的失败条件，没有重跑已知缺覆盖的头部场景，也没有修改骨材料、钳位密度或启用局部沉积来取得速度数字。

## 复现证据

目录 `out/carbon_ct_throughput_20261004/` 包含 `coverage.json`、两臂四批 pinned YAML、`plan.manifest`、`execution.json`、完整日志、GPU telemetry、各批 dose / quality / energy ledger，以及 `summary.json`。新路径使用 `data/ct_electron/carbon_independent_v1` + `shared_independent_v1`。

- 新二进制 SHA256：`f2f7f3e14e8435bb2417b49aaf31dfaada35bb1df60ca4492b3fa7ebf9f22557`
- 旧二进制 SHA256：`ab913f824ddd13968f56a47acd897c9283f0a872a4d2af338c6e2cf9ab47d19a`
- 两臂相同批次种子：1002111001、1003111004、1004111007、1005111010。

正式默认配置及二进制未在测速中修改。

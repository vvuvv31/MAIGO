# Proton minibeam 顺序优化进展（2026-10-04）

## 1. 异常诊断记录直接写入：未接纳

原 FP32 worker 在失败分支构造 MaterialPacketFailure 后整体复制到GPU；候选直接写入目标诊断对象，保留全部字段、默认值及前后状态。10,000个随机/非有限输入夹具逐字段位值相同。

RTX2080Ti，RT07575，group32/chunk262144/lanes16384，FP32剂量。ABBA每进程两批1,983,920 primary，全部编译结束后计时：

| 指标 | 原版 | 候选 |
|---|---:|---:|
| 驻留批次 | 17.813465 s | 17.930318 s |
| 电子阶段均值 | 7.727693 s | 7.766928 s |
| 寄存器 / 静态栈 | 248 / 1392 B | 248 / 1392 B |

吞吐变化-0.652%，没有有用收益。SASS静态指令行并未减少，不能把源码少一个临时对象推断为更低GPU状态成本。primary/EM、electron/joint/photon工作量相同，质量/能量/完成/零溢出/零legacy通过。BODY L1最大5.983e-08，>=1%BODYmax体素最大局部差0.000104014%，符合FP32累积差异量级。

worker源码已逐字节恢复，正式二进制未替换。记录在failed.md，完整源码、二进制、SASS、夹具与逐批证据在out/electron_cold_record_20261004。

## 2. 连续批次不变设备表复用：已接纳，默认开启

当前正式源码仅缓存EM/Urban主机输入，每批仍分配上传设备数组，并重复构建EM energy index。早期独立研究里实现过设备复用，但当前正式源码不含该实现。本次从冻结研究源码移植这一单独机制，保留新的C12入口和电子三包布局，不携带材料查询或铜核特化候选。

不变EM、Urban和air数组由SyclTransportContext持有；输入身份变更或context清理会释放，粒子/剂量/RNG/失败计数和审计仍逐批重置。保留字节计入DeviceMemoryTracker的同一预算。环境开关CARBON_RETAIN_PHYSICS_DEVICE=0用于独立对照，默认候选开启。数据文件路径、pin、大小、mtime及伴随metadata均参与身份。

初轮同场景 ABBA（每进程两批）通过：

| 指标 | 原版 | 候选 |
|---|---:|---:|
| 驻留批次 | 17.809828 s | 17.334340 s |
| 驻留吞吐 | 111,395/s | 114,450/s |
| 驻留批次准备 | 1.113526 s | 0.594878 s |
| 驻留 EM 上传/索引 scope | 0.479844 s | 0.000082 s |
| 两批进程 wall | 42.312963 s | 42.057397 s |

驻留吞吐 +2.743%，两组 +1.267%/+4.217%；两批完整进程 +0.608%，短进程首次加载稀释了收益。电子均时7.7242→7.7722s，没有电子 kernel 提速证据。integer ledger、primary/EM、joint/charged/photon工作完全一致；BODY L1最大5.992e-8、>=1%BODYmax最大局部差0.000103055%、峰谷汇总最大差3.774e-7%，最大能量相对残差4.942e-8。全部任务完成、零队列溢出、零旧电子路径。

同进程驻留EM/Urban/air表合计2.201GiB，逐批tracked峰值仍为4.884814GiB；这是保留原有数组，不是另复制一份。此数不含driver及所有非tracker分配，不能当整卡显存峰值。

回归：proton/carbon water broadbeam、proton CT broadbeam、proton/carbon water minibeam各2x2000，以及carbon软组织独立电子2x100000，均通过质量、能量、整数审计、剂量、零溢出。carbon CT broadbeam旧研究预设仍以唯一既存unvalidated_electron_joint_response标记拒绝，候选和对照一致；保留这一拒绝而非将其计为运行验收。此前TOPAS剂量验收是独立结论。更换EM路径强制3次miss；错误SHA及第二批收紧到0.001GiB均在首批成功后拒绝第二批，未产生第二批剂量；预算报错明确来自retained physics tables。

为排除不同二进制和首批加载的影响，再用同一个候选二进制，CARBON_RETAIN_PHYSICS_DEVICE=0/1，每进程4x1,983,920 primary，ABBA确认：

| 指标 | 缓存关闭 | 缓存开启 |
|---|---:|---:|
| 第2–4批平均 wall | 18.179141 s | 17.843210 s |
| 驻留吞吐 | 109,132/s | 111,186/s |
| 驻留准备 | 1.008482 s | 0.603261 s |
| 四批进程 wall | 79.689236 s | 78.832402 s |

驻留提升**1.883%**，正反两组**1.038%/2.716%**；含首批的四批进程提升**1.087%**。准备开销减少0.4052s/批；电子核均时7.8852→7.9195s，没有电子加速结论。所有整数ledger、primary/EM、electron/joint/photon工作一致，所有批次accepted，完成数闭合、零溢出、零legacy。BODY L1<=4.415e-8，>=1%BODYmax最大局部差0.000068643%，峰谷汇总最大差5.379e-7%，能量相对残差<=6.891e-8。

结论：保留设备物理表复用，默认开启，正式 `build/carbon_mc` 已替换为测试过的二进制。电子0.05mm步长、group32、抽样/输运方法和FP32剂量不变。此轮是代表批次性能与实现一致性验证，未重新运行完整119批计划或进行新的TOPAS物理验收，不能把11.12万/s当作新full-plan实测。

身份与回滚：
- 已安装候选SHA256：`68b759a6921675514d45dfae80fa77d61c69218c1e3990e95a079e9a421d41c7`。
- 两批冻结对照为C12入口已接入、proton物理等价的`f2f7f3e14e8435bb2417b49aaf31dfaada35bb1df60ca4492b3fa7ebf9f22557`；四批对照完全同一候选二进制。
- 上一正式二进制`ab913f824ddd13968f56a47acd897c9283f0a872a4d2af338c6e2cf9ab47d19a`备份为实验目录`formal_before_carbon_mc`。
- 编译后仅修正CachedUnifiedEm关于设备驻留的过期注释，运行代码无后续改动。

证据：`out/physics_device_reuse_20261004/{summary.json,confirm_summary.json,regression/summary.json,negative_tests.log,adoption.json,candidate.patch,source_manifest.json}`。初始测试脚本的两个错误假设及原始日志均保留，并记入failed.md；没有修改运行质量门槛。

## 当前边界与下一优先级

本轮完成两项独立候选：诊断冷分支改写撤回；设备物理表复用接纳。收益约2%，尚未解决电子核及铜/空气主粒子核的主要成本。下一优先仍是按实际local-memory/分支热点缩小电子worker活跃状态，并分别测量铜/空气主输运；本报告不把这些尚未实施的候选列为完成。

# Proton CT broadbeam：GPU连续批次与100线程TOPAS

2026-10-02补测，RTX2080Ti vs AMD EPYC9965的100物理核、100 TOPAS workers。
当前实现持续吞吐 **GPU416,098/s，TOPAS24,673/s，约16.86倍**。
GPU计时包含缓存命中后的每批准备、计分、质量检查与输出；TOPAS排除冷启动、包含整个run的输运和收尾。

| 项目 | 时间/吞吐 |
|---|---:|
| GPU首批3M完整耗时 | 8.1902秒 |
| GPU后两批3M完整耗时 | 7.2992 / 7.1204秒 |
| GPU连续批次吞吐 | 416,098 primary/s |
| GPU后两批输运循环耗时 | 6.3849 / 6.3918秒 |
| GPU仅输运吞吐 | 469,606 primary/s |
| TOPAS 3M总进程 | 205.377秒 |
| TOPAS首个history开始 | 81.344秒 |
| TOPAS从history0至run完成 | 121.589秒 |
| TOPAS全输运吞吐 | 24,673 primary/s |
| TOPAS中段10%–90%吞吐 | 24,919 primary/s |
| TOPAS中段平均忙碌核数 | 99.993 |
| TOPAS峰值RSS | 11.52GiB |

GPU连续批次 / TOPAS全输运 =16.86；以TOPAS中段为分母=16.70。
GPU仅输运 / TOPAS全输运=19.03，仅供阶段诊断，实际运行优先采用约16.9倍。
此处CPU确实接近100核满载；没有把以前20万粒子smoke的初始化时间当成追踪成本。

## 范围与物理

采用当前`config/proton_ct_plan_fe.yaml`的RT07575 CT broadbeam预设，以
`inputs/proton_rt07575/part_01/spots.csv`的12,531,500原始分配作为相对权重，
largest-remainder缩放到3M，共539个非零spot。GPU连续3批，每批3M；逐spot数量与CPU的3M输入逐项相同。
这里是该计划分区的相对权重测试，不是重新计算所有计划分区。

TOPAS沿用已有broadbeam CPU参考的二进制、CT/DICOM、Schneider转换、原束线光学、
7个完整物理模块（opt4、QGSP_BIC_HP、ion-INCLXX、elastic_HP、decay、stopping、radioactivedecay），
10GeV表上限、DoseToMedium Sum scorer，不含铜准直器。TOPAS进程与GPU测试顺序执行，未并行争用本次测试资源。

**该倍数包含模型差异，不能仅归因为硬件。** GPU使用当前broadbeam_fast/FE散射、局部电子沉积，
没有材料电子包阶段，且未输运核中性粒子；TOPAS保留显式电子及完整次级输运。
GPU保留FP32 dose ON/FP64 OFF和所有既有质量/能量/队列门禁。
与此前minibeam约2.8倍的差别涉及束流拦截、不同优化计划、散射/电子处理及物理模型，
没有通过仅切换准直器的消融实验将各项收益单独分离。

## 检查与证据

- GPU三批全部accepted，零队列溢出，最大相对能量残差3.24e-7，剂量有限非负非零。
- TOPAS退出0，实际3,000,000 histories，100workers/100物理核心；剂量网格417×505×35、Gy单位正确。
  原生导航未计分能量占标称入射能量2.31e-12。该轮仅测吞吐，没有新增剂量gamma验收。
- Slurm22883 COMPLETED/0；账本200逻辑CPU是SMT sibling预留，进程affinity已限制到100物理核心各一个逻辑CPU。
- GPU SHA256：`9b9b47f7fa31a5187aa9fb7f297bcfa691dfb481f0b07d412178b4c11a18d2e7`。
- TOPAS SHA256：`1c5ecde8bab5c90cd51fb9f5083bba51e7a51f6c90e2a43e16b69161c8fdf6c1`。

全部输入、脚本、日志、逐history时间戳、逐秒CPU/RSS、剂量及质量报告位于
`out/proton_broadbeam_cpu_gpu_20261002/`；`manifest.json`记录源/二进制hash，
`summary.json`保存计时与检查，`sacct.txt`记录作业完成状态。

第一次分析遇到最后一个僵尸进程资源样本缺少VmHWM字段。保留原分析器，改为从实际存在的峰值RSS样本取最大值，
重新分析原有数据；没有重跑输运、筛掉剂量或改变任何计时/质量门禁。

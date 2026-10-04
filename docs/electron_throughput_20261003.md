# 独立电子路径吞吐优化

后续默认已切换至[全FP32电子基线](electron_precision_20261003.md)，RT07575连续批次约86.86k primary/s。下文保留精度改动前的优化记录与指纹。

目标：proton CT minibeam 保持新 birth/joint/charged/photon 和零旧库回退，向 RTX2080Ti 的 RT07575 连续批次 **180k primary/s** 靠近。所有计分 FP32 ON/FP64 OFF，最大物理步长与模型保持原值（charged 默认0.05mm，minibeam≤0.25mm），group32。

工作目录 `out/electron_throughput_20261003/`；开始前检查 `failed.md`。基线二进制 SHA `5ac3bf7f06b8371a30e7522df79aee286fc109609d160a466addbb7bd4cff2d7`，独立运行包四组件仍为3.853GiB。

## 基线与目标预算

此前19,839,200 primary 的十批独立路径：平均每1,983,920 primary 批次30.74s，电子20.34s，输运循环29.40s。180k/s对应每批11.02s；当前非电子部分约10.4s，单靠微小调度调整不够，电子阶段必须显著缩短。冷启动、完整进程和已驻留批次分别报告。

## 已拒绝：线程内长期累计八个诊断计数

源码/二进制冻结为 `counters_transport_sycl.cpp` / `counters_carbon_mc`；仅将每个charged短步的全局原子计数改为worker私有uint64[8]，退出worker时发布，物理模型/数据/RNG不变。ABBA四进程，每进程两个匹配的RT07575 1,983,920-primary shard。

- 基线两次已驻留批次合计60.6149s；候选111.0902s，候选吞吐35.717k primary/s，明显退化。
- 四个匹配shard/臂的电子时间合计79.6170→179.1702s。
- 所有具名整数primary/EM/joint/charged/photon审计一致；剂量相对L1最大8.01e−8，quality accepted research、零队列溢出。没有物理理由拒绝，仅因速度退化撤回。
- 主电子内核静态资源：255寄存器不变，栈5952→6032B；rounded wrapper6640→6896B。静态栈不等于动态spill流量。

采用用户已授权的ps/sudo运行Nsight Compute，各捕获一个相同源的首个电子内核启动。三次kernel replay，不把受profiling影响的耗时作为性能测量；CPU编译可同时运行。报告 `profile_{baseline,counters}/electron.ncu-rep`。

| 指标 | 基线 | 线程长期累计 |
|---|---:|---:|
| 寄存器/线程 | 255 | 255 |
| 活跃warps / 峰值 | 24.88% | 24.89% |
| local load sectors | 10,854,394,113 | 16,885,135,010 |
| local store sectors | 8,725,820,141 | 13,806,452,716 |
| long-scoreboard stall / issue-active | 30.62 | 29.09 |

sector是L1/TEX局部内存请求计数，不能乘32后宣称同等DRAM传输；long-scoreboard比值也不是百分比。证据显示局部访问大量增加，不支持继续同样的长期计数布局。

## 候选：编译期独立电子内核

固定一次运行的independent标志，在主机派发true/false模板内核。true分支去除旧birth/continuation代码，仍采用原来的每步诊断计数；完整物理和每包随机流保持不变。目标是降低活跃状态、局部访问和指令成本。源代码 `specialized.patch`；完成ABBA配对后保留。

- 同4个shard/臂，电子79.9684→71.9745s，电子吞吐+11.11%；已驻留批次60.8711→56.7410s，primary吞吐+7.28%，达到69.929k/s。
- 整数审计完全一致，FP32剂量相对L1最大8.01e−8，quality/energy/zero-overflow检查通过。
- 实际主电子内核静态栈5952→2592B，寄存器仍255；同一个首电子launch的local load sectors10,854,394,113→6,105,612,700，store8,725,820,141→6,485,806,686；活跃warps仍24.88%，long-scoreboard比值30.80。
- 尚未达到180k/s；不能把静态栈减少56%等同于吞吐提高56%。证据 `specialized_screen/report.json` 与 `profile_specialized/`。

后续优先级由实际测量决定：缩小charged/joint/photon热状态；消除重复几何/参数查询；再评估更细的任务划分。不能因减少理论查表次数就宣称加速。改变joint切换或电子输运近似则需额外峰谷/局部剂量验证，不能仅用相同随机种子作为物理验收。

## 下一候选：借用只读密度节点

`ChargedDensityModel` 不再复制两份完整节点描述，而保存稳定驻留节点的指针；明确接受指针以避免引用临时view，参数查询接口用const引用。材料、能量、插值算式、每包随机流和最大步长不变。57个中间密度的参数CSV逐字节与旧checker一致，条件终态支持检查和charged几何/终止测试通过；GPU检查116,736包/11,431,295步的所有输出计数与旧实现逐项一致；模型描述560→160B。相对于已保留的专用内核，第二组ABBA结果：电子71.1458→67.0264s（吞吐+6.15%），已驻留批次56.1531→54.1104s（primary吞吐+3.78%），达到73.329k/s。两组正反顺序均改善，保留此改动。整数审计一致，剂量相对L1最大8.02e−8，quality/energy/zero-overflow通过。完整进程与驻留批次分别见 `borrowed_views/screen/report.json`。

主电子内核静态栈2592→2336B，寄存器仍255。最终组合另与最初基线直接ABBA确认，总增益见下节；不直接相乘前两轮百分比。

专用内核的旧charged兼容分支另以1,983,920 primary完成回归：primary/EM审计一致，剂量相对L1=7.97e−8，quality/energy/zero-overflow通过。该运行与CPU编译同时进行，时间不作性能结论。证据 `specialized_legacy_regression/report.json`。

## 最终选用与直接确认

当前源码及 `build/carbon_mc` 保留编译期独立内核和借用只读节点两项，拒绝长期私有计数。二进制 SHA `1581fd7dd3434380b7bb785678926f1ef1b8cc29e5f2e353ca68d5f08bd33872`；源码快照 `final_source/fingerprint.json`。物理模型、运行包、步长、group32和每包随机序列不变。

直接ABBA四进程，每进程两个不同种子的1,983,920-primary匹配RT07575计划shard，即**每臂7,935,680 primary**；两次候选与两次基线顺序相反。只把每进程第二批用于已驻留吞吐，完整进程包含加载/首批/输出另报。

| 指标 | 优化前 | 最终代码 | 吞吐增益 |
|---|---:|---:|---:|
| 已驻留批次平均时间 | 30.5461s | 27.5290s | +10.96% |
| 已驻留primary/s | 64948 | 72067 | +10.96% |
| 每批电子阶段平均时间 | 20.1226s | 17.0555s | +17.98% |
| 两进程总wall（各两批） | 132.6739s | 120.5666s | +10.04% |

完整整数primary/EM/joint/charged/photon工作审计一致，各批quality accepted research、能量检查通过、零队列溢出、无coverage/invalid。匹配两批合并剂量：BODY相对L1=5.9274823e-08，汇总峰谷最大变化1.5768751e-06%，BODY 1%最大剂量门槛内最大局部变化0.00010255687%。这些量检查**本轮实现优化的等价性**，不替代新电子模型相对TOPAS的正式验收。

**尚未达到180k/s。** 当前驻留批次非电子部分估计10.47s；180k/s要求整个批次仅11.02s，留给电子约0.55s。这说明只做少量缓存/调度调整不足以实现目标。下一阶段优先拆分charged/joint/photon热状态与队列，减少寄存器/局部访存；若要减少单散射次数，需独立验证受几何限制的凝聚输运/更短步joint响应，保留横向扩散、全部正能量分支和≤0.25mm步长。主输运成本也会限制最终上限，不预先承诺180k/s可达。

结果：`final_screen/report.json`、`final_screen/dose_detail.json`、`selected.json`。

最终二进制的相同首电子launch硬件计数：local load sectors **5,012,623,473**、store **4,661,924,221**，相对初始基线分别减少约53.8%/46.6%；活跃warps24.87%，255寄存器/线程，long-scoreboard比值29.79。状态与访存成本确实下降，但寄存器限制的并发度仍未改善。原始与汇总证据 `profile_borrowed/electron.ncu-rep`、`hardware_summary.json`。

## 后续：分阶段调度筛选（已撤回）

将独立电子拆成出生/joint/光子/计分 handoff 和最多32次charged事务两个内核，跨阶段保留完整电子包、故障快照、RNG与原有joint重试时机。每lane额外保存一个状态，不增加电子任务队列容量。FP32、group32、步长和包均不变。

同一RT07575 1,983,920-primary分片：原版、新二进制OFF、新二进制ON的进程wall为31.2898/31.6272/37.0267s；电子16.1873/16.4351/21.5921s。全部具名整数审计一致，quality/energy/zero-overflow通过，剂量L1≤8.035e−8。单轮完整运行已显示明显退化，因此不追加性能确认，不把它宣称为稳定精确的退化百分比。

两个直接阶段内核仍各占255寄存器，栈为1984/1808B（monolithic2336B）；每个primary chunk约200余轮交接。静态栈下降没有换来更高吞吐；尚未单独测量两个阶段的动态local访问，不能把退化全归因于host同步。候选源码/二进制和资源记录冻结在 `out/electron_stages_20261003/`，正式代码已撤回此拆分，失败条件追加至 `failed.md`。

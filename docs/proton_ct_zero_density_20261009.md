# Proton CT broadbeam：跳过严格为零的阻止本领修正（2026-10-09）

本轮保留仅在质子 CT 宽束电子关闭路径启用的零修正快速分支。电子仍按此前授权采用局部沉积；continuous26、离子涨落、核反应、弹性散射、中性产物能量账目、FP32计分和源归一化不变。

## 实现与数值验证

`UnifiedEmPoint::prepare` 与 `corrected_raw_stopping` 增加默认关闭的编译期参数，仅 stage15/16 的 local-electron CT primary 启用。若 0<bg²≤1、材料和参考的 d0 均为零、两套 x0/x1 均为正，则两个密度效应都严格为零，跳过 log10。若同时两个有效 cut 精确相等且为正，整个修正项严格为零，直接返回；导数的 energy±eps 分别执行其自身条件，不跨能量复用结论。

保留原始质量约定、cut 和低能分支、非零修正、原始 spline 及其导数。没有提高 cutoff 或删除物理过程。其他路径默认参数为 false。

离线组件检查覆盖18种带电粒子、26种材料、7个密度位置、100个能量、4个路径长度及涨落开关：2,620,800组 prepare/loss/RNG 字段逐位一致，约94MB转录完全相同。该检查不是CPU输运模式或性能证据。

## RTX2080Ti 性能

CUDA AOT，禁用 PTX JIT。同配置、同种子配对，ABBA 与另一种子的 BABA；每进程4×1M，首批排除，连续批次复用 context/packages。重复进程重用同一轮的种子，因此这些是匹配工作负载的性能重复，不能当成32M独立历史合并剂量。每臂暖批计时12M。

| 实现 | 暖批 primary/s |
|---|---:|
| 已接受 shared-cell stage16 | 303,485 |
| 新零修正 guard | 309,885 |

合并提升 **2.11%**。ABBA比值1.019787，配对1.004696/1.034824；BABA比值1.022379，配对1.033050/1.011963。四对方向一致。GPU未锁频，温度/频率随运行变化；这是本轮相对观测，不与此前不同温度下的314k/s直接比较，不保证跨机器同样收益。

全部32性能批次通过quality、energy、coverage、completion和零溢出审计。源/种子、核反应/弹性、live-ion terminal及整数EM计数匹配，transport ledger在规定容差内。剂量相对L1≤5.183e-8；参考≥10%peak体素同位置local1%通过100%，因此local1%/1mm通过率下界为100%。这是匹配GPU实现验证，没有新的TOPAS验收。

native stage16主kernel寄存器238→234，stack2376B不变。静态资源不用于推断动态spill/stall；硬件计数器权限仍不足。

## 未采用的实验和适用范围

有限尺寸/spin veto冷分支outline逐位组件与GPU审计正确，但ABBA0.996171，没有速度收益，已撤回。仅跳过密度效应的前版ABBA1.009249/BABA1.012853，有一对略慢，未单独推广。本轮增加整个零修正返回后才取得四对同方向结果。

一次提前于device link的错误启动已取消，并单独保存在excluded_before_link_complete，完全排除其计时与剂量。本轮使用最终链接后的独立SHA执行。

电子关闭的TOPAS依据仍为RT07575 **head患者内**full-plan local1%/1mm、search0.1mm为99.948873%，不能外推肺或任意低密度场景。完整五管线build/carbon_mc本轮未重建；当前可运行程序为build-proton-broadbeam/carbon_mc。

证据目录：out/proton_ct_zero_density_20261009/。最终二进制SHA256：01769e6c8b8b25d83385b04649a0682f997ae6d2889bd305d8ee230a15aeb6ff。

最终匹配检查完成：原始full-plan每臂2M（原始浮点权重）、水与电子开启CT各每臂40k，source/离散审计一致，同位置local1%均100%。合计44个验证批次全部通过，最大物理相对能量残差 4.2484786e-07。安装manifest（173资产、8预设）与git diff --check通过。

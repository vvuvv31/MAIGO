# Proton CT local-electron：同一步 CT / scorer cell 复用（2026-10-09）

本轮继续优化显式关闭电子后的 proton CT broadbeam。最终仅保留 CT 材料读取与剂量计分之间的体素定位复用；核反应率延迟查询和核弹性区间缓存均未选用。电子保持显式关闭、局部沉积，离子 continuous26、涨落、核反应、散射、步进、随机数、FP32计分和归一化保持本轮之前的行为。

## 保留的改动

在 stage15 基础上增加仅限 proton CT local-electron 的 stage16。主机必须逐项证明 CT 与 scorer 的维度、spacing、x/y origin完全相同，且 CT z origin为0，才复用计分得到的 directed cell。这里比较的是精确FP32值，不使用几何容差。非重合网格继续stage15；开启电子、water、carbon与minibeam不选择stage16。

复用前保留**尚未clamp的索引是否有效**，因此CT外的轨迹不会被误当成边缘体素；只有有效cell才按计分线性index读取密度和材料。负方向在精确面的`ceil−1`规则不变。它合并两套完整的cell定位，而不是重复此前未选用的“ct_sample归一化坐标传给DDA”三次除法复用。

独立离线几何检查2,097,152组，其中1,821,681组在CT内，覆盖不同spacing、平移x/y origin、随机位置/方向、精确体素面、nextafter相邻数、零与负零方向和外部cell。调用当前`ct_sample`与候选复用cell，valid/density/material逐位一致。该检查不是CPU输运运行模式，也不用于吞吐宣称。

## 交错性能

RTX2080Ti，CUDA AOT，`CUDA_DISABLE_PTX_JIT=1`。两臂同一电子关闭配置、源、物理与种子。每进程4×1M，首批排除；连续批次复用context/packages。ABBA和独立种子BABA共32M primary，每臂16M，暖批计时每臂12M。

| 实现 | 暖批primary/s |
|---|---:|
| 已接受stage15 | 309,907 |
| 新stage16 | 314,282 |

合并提升 **1.41%**。ABBA1.013287（配对1.009066/1.017479），BABA1.014935（配对1.018780/1.011141）。四对方向一致，保留。GPU未锁频，记录保留温度/频率，不能将该小幅观测当成跨GPU固定常数。

所有32性能批次quality accepted，source/seed、核反应/弹性、live-ion terminal/transport ledger与整数EM审计匹配，energy/coverage/completion和零溢出通过。剂量相对L1≤5.214e-8；参考≥10%peak体素同位置local1%通过100%，给出local1%/1mm的100%下界。

stage15 native primary为240寄存器/2368B stack，stage16为238/2376B。静态资源变化不作为dynamic spill或瓶颈证明。Nsight硬件计数器被本机`ERR_NVGPUCTRPERM`拒绝，没有取得stall数据。

## 未保留候选

- **非弹性核反应率延迟查询**：按原有cache刷新条件跳过即将被丢弃的查表结果。ABBA1.003413，配对0.994914/1.011858；方向混合，没有稳定有用收益，撤回。
- **核弹性区间索引复用**：仅复用binary-search所得索引，每步重新按当前能量执行原来的13-target FP32插值/求和。组件1,358,256组逐位rate/cell一致，但对stage16的ABBA为0.986624（0.981165/0.992045），两对更慢，撤回。

两者完整GPU剂量/审计检查通过，未因数值正确就推广。恢复原来的elastic header、primary源码后，重新编译最终stage16；没有保留实验cache接口。失败记录已补入`failed.md`。

## 剂量适用范围与产物

本轮是匹配GPU实现检查，没有重新运行TOPAS或新增物理近似。显式关闭电子的依据仍为 RT07575 **head患者内部**原始full-plan local1%/1mm（search0.1mm）99.948873%，见[既有双域报告](rt07575_broadbeam_electron_gate_20261009.md)。不外推到肺或任意低密度几何。

本轮证据：`out/proton_ct_shared_cell_20261009/`，包括source/binary冻结、geometry checker、ABBA/BABA、dose/ledger/quality、device images/resources、final build与focused validation。两个撤回候选分别位于`out/proton_ct_lazy_hazard_20261009/`和`out/proton_ct_elastic_cell_20261009/`。

当前专用程序为`build-proton-broadbeam/carbon_mc`；完整五管线`build/carbon_mc`本轮未重新编译。编译与最终匹配检查详见该证据目录的`final_validation.json`。

最终匹配检查完成：原始full-plan每臂2M（完整浮点权重）剂量L1≤5.223e-8、同位置local1%100%，source/离散审计一致；水和重新开启电子的CT每臂40k匹配通过，不用于性能宣称。合计44个最终候选/基线验证批次全部通过，最大物理相对能量残差4.2066543e-07。最终重建ELF与正式性能候选逐字节一致，SHA256 `7d8db6cb83bf83b7d3ef6a1510d5ff0e2bd5fe6f24e34d826c76af61d0e34129`。`git diff --check`和安装manifest验证通过。

后续零修正优化已叠加于stage16，当前专用二进制及性能见[零修正报告](proton_ct_zero_density_20261009.md)。以上SHA与测量保留为该阶段历史基线。

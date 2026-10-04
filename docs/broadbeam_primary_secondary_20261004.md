# CT broadbeam 主粒子/次级粒子吞吐优化（2026-10-04）

## 结论

在[空气电子优化](broadbeam_air_roulette_20261004.md)的基础上，再做四项不改变物理的修改。RT07575、RTX 2080 Ti、同日 ABBA，各两个独立种子：

| 预设 | 原 primary/s | 新 primary/s | 提升 | 全网格 local11 |
|---|---:|---:|---:|---|
| proton_ct_plan_fe（10×1,253,150） | 136,679 | 164,245 | **+20.2%** | 99.933/99.921，与原二进制相同 |
| carbon_ct_plan_fe_joint（7×925,987） | 86,887 | 104,432 | **+20.2%** | 99.870/99.780，与原二进制相同 |

每批平均耗时（第2批起）：

| | primary | electron | secondary | 每批准备 | 每批墙钟 |
|---|---:|---:|---:|---:|---:|
| Proton 原 → 新 | 4.361 → **3.314 s** | 3.695 → 3.688 | 0.355 → 0.265 | 0.571 → 0.175 | 9.169 → **7.630 s** |
| Carbon 原 → 新 | 2.597 → **1.697 s** | 4.134 → 4.111 | 3.248 → **2.802 s** | 0.507 → 0.085 | 10.657 → **8.867 s** |

所有对照运行的 charged/joint/photon/EM/步数/核反应整数审计与原二进制逐位一致；剂量差只来自 FP32 原子累加顺序。积分形式的能量账本（`E_secondary_transport_MeV` 等）逐位一致，grid 账本相对差 ~1e-11。

## 修改

1. **Broadbeam CT 主粒子专用 kernel（stage 12/13/14）**。原 broadbeam 主核（stage 6）通过 lambda 捕获的运行时变量判断约 25 个配置开关（minibeam、Urban、均匀水、slab、LET、诊断计分等），编译器无法删去死代码；先前 NCU 显示 proton 主核 46.5% PC 采样是取指等待。新 stage 在主机逐项证明配置后把这些开关作为编译期常量（与 CT minibeam stage 11 同一机制），13/14 再分别固定 proton/C12 的 Z/A。不满足条件时回退 stage 6，并打印 `[broadbeam-ct-specialize]`。所有 BroadBeam kernel 中 `global_urban`、`enable_minibeam`、`minibeam_water_proton_helium_urban` 本来就恒为 false，现在也直接折叠。
2. **Broadbeam 次级粒子使用生产专用路径**：沿用已有 `kProductionSecondaryPath` 折叠集，只把 all-ion elastic 保留为运行时值（carbon 预设没有该文件）。
3. **每步单地址原子改为寄存器累计**（仅 BroadBeam）：次级粒子每个普通步原有三次全局原子（历史账本、Schneider 诊断 `SecondaryDepositedMicroMeV` 的 uint64、grid in/out 的 double，后两者是全卡单地址），改为在段内累计、暂停/终止前一次写入；µMeV 整数仍按每步截断后累加，诊断总数不变，续跑状态不增大。主粒子的 grid in/out 每步 double 原子改为每历史一次。电子 worker 中最热的 calls/elastic 两个计数器改为每 lane uint32 累计（其余 6 个仍每步计数）。
4. **每批主机准备**：每批对 737 万体素 × 82 个节点的覆盖检查（及 500 万选中体素 × 82 出生节点）改为先算每个 section 的密度上下界再 O(n) 比较，比较式和报错信息不变；5 mm 盒形选区按几何/阈值/邻域缓存（几何变化时随 CT 缓存一起清空）。每批准备 0.51–0.57 → 0.09–0.18 s。

## 回归

- 4 个预设的小样本冒烟（proton CT broadbeam、proton/carbon water broadbeam、proton CT minibeam）：退出码与整数审计和原二进制一致。
- tracking OFF：整数审计一致。
- `tests/broadbeam_electron_tracking.cpp` 通过。
- 二分筛选中，每个候选单独测量；没有收益的候选（次级段长 32/64、in-FOV 深度剂量累计、主粒子出生查询计数）没有并入，已记入 `failed.md`。

## 仍然存在的瓶颈

电子队列现在是两者最大的单项（proton 3.69 s，carbon 4.11 s/批）。Carbon 次级 2.80 s 里约 4.6 亿步，主要是 p（41%）、α（24%）、d（22%）的长径迹；短程步只占约 3%，靠调度/段长已无收益，需要降低每步访存依赖才可能继续提速。

## 复现

证据：`out/perf2_20261004/`（`screen*.py`、`final_perf.py`、`runs/*/execution.json`、`analysis.json`、`final_breakdown.json`、`smoke.json`）。A/B 开关：`CARBON_BROADBEAM_CT_PRIMARY_SPECIALIZE=0`、`CARBON_BROADBEAM_CT_SECONDARY_SPECIALIZE=0` 可在同一二进制上回到通用 kernel（默认开启）。

# Proton CT broadbeam 电子出生能量占比查询（2026-10-09）

保留专用proton CT broadbeam（kernel stage13）的精简continuous26查询。电子出生占比只需要restricted stopping与delta mean；原调用state.prepare还准备range、fluctuation和能量导数。新增UnifiedEmPoint::birth_stopping只计算所需两项，原阻止本领、hard-rate校准和conditional terminal公式保持不变。不改变抽样概率、packet权重、group32、Cu/核反应/电子光子输运，也不修改发布YAML或包。

脚本scripts/check_proton_primary_point.cpp已扩展：26材料×7密度×100能量×4步长×2涨落=145,600 loss/RNG条件，出生mean同时逐能量验证，字段与RNG精确一致，invalid0。独立链接使用icpx -O2 -std=c++20 -Iinclude脚本与src/unified_em_package.cpp，显式-lstdc++ -lm。

GPU为RTX2080Ti sm75、CUDA_DISABLE_PTX_JIT=1，采用最新同一步视图复用实现为基线，保持原RT07575 head CT/源/权重/种子/配置/包。每进程4×1M连续批次，性能剔除首批。ABBA与独立种子BABA，共32M primaries，每臂16M、暖批12M。

| 顺序 | 候选/基线整体吞吐 | 两对比值 |
|---|---:|---|
| ABBA | 1.012513 | 1.004984 / 1.019986 |
| BABA | 1.020050 | 1.021684 / 1.018439 |

全部source、离散/ion/EM/charged/photon/joint审计匹配，quality/energy/coverage/completion/zero-overflow通过。dose L1≤5.28e-8，参考≥10%peak体素同位置local1%通过率100%，提供local1%/1mm的100%下界。不是新的TOPAS验收。GPU温度/频率未锁定，所报1.25%/2.01%为本轮观测，不是固定硬件常数。

范围限定在电子出生占比查询；没将full prepare删出真正需要range/variance/slope的离子步进准备。其它pipeline继续原full prepare。原始源码、可执行文件、两轮结果和数值检查保留于out/proton_ct_birth_partition_20261009/。

两种顺序合计暖批吞吐 151,304→153,769 primary/s，观测比值1.016290。native stage13 primary资源238→241寄存器、2544→2512B静态stack；不据此推断动态spill或实际occupancy。

最终专用构建已恢复且SHA与正式性能候选一致：9e12c3ab25cc7b76f497eb6b61784fd673d8c074e449340fb04695c4b4450c7b。额外两项消融（birth queries per-primary归并、joint首轮重复guard省略）均未测出稳定收益，已回退；failed.md和各out证据目录保留拒绝条件。当前完整电子模型仍约154k/s，没有达到200k或无电子包约299k/s的目标。

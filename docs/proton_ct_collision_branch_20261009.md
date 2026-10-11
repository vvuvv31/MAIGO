# Proton CT broadbeam：低能碰撞分层抽样实验（2026-10-09）

结论：两版候选均不推广，发布物理、配置和碰撞实现保持基线。仅实验减少未选中的原子退激工作，没有删去退激、光子或离子涨落。

## 实现及验证边界

原实现完整生成 Penelope 碰撞和原子退激，再用能量加权 reservoir 选择一个后继。候选只对 ≤0.1 MeV 的电离碰撞先选择 primary、secondary、relaxation、local 预算；选中 relaxation 时仍执行原有 breadth-first 空穴级联及内部 reservoir。V1 顶层用 reservoir；V2 顶层用一次累计概率抽样，单个正预算不抽随机数。

选中 local 时仍计分原 packet 权重 W。原实现把 prompt local 与 cascade local 合为一个 terminal 标签；候选分别选择两部分，因此未使用的 local terminal energy 标签不逐项等价。验证将 local termination 合为一类；继续输运的电子/光子验证物种×归一化能量×cosine 联合直方图。核反应中性产物仍为未追踪能量汇。

实际完整事件生成参考（包含原 reservoir）的 CPU 抽样验证：26材料×3密度×6能量=468条件，每条件每臂12000样本。cuts 使用 scalar26 且按当前 rates26 支持域 clamp。两臂非法事件数均0，最大单bin标准化差 2.72517，预先拒绝阈值6.5。样本覆盖不能证明所有离散数据错误都能在跳过级联时被检测，因此候选并未建立新的完整数据完整性验收。

每个参考/候选采样调用的平均 RNG 次数为 1.195982/1.189243（按测试条件等权平均，不是实际 CT 事件加权）。早期 distribution.csv 的 reference_rng 是完整事件无 reservoir 的 oracle 成本，不能用于宣称相对当前运行时的 RNG 节省；以 sampled_distribution_v2.csv 为完整采样参考。

## GPU 实验

RTX2080Ti sm75，selective proton broadbeam 构建，CUDA_DISABLE_PTX_JIT=1。每进程4×1M连续批次，性能剔除首批；ABBA，V2使用与V1不同的种子。源权重/种子、配置和包相同。未锁频，V1冷到暖1890→1830MHz，V2后段1815MHz，绝不把温度漂移归于算法收益。另一组误在link/ptxas未结束时启动的V2任务已中止，不参与性能结果，保留在 performance_v2_cancelled_compile_overlap。

| 候选 | 整体吞吐候选/基线 | 两对比值 | 4M每臂匹配local1%/1mm |
|---|---:|---|---:|
| V1 顶层reservoir | 0.990272 | 0.982221 / 0.998209 | 99.69308% |
| V2 顶层累计选择 | 0.998545 | 0.993047 / 1.004011 | 99.71539% |

Gamma采用完整参考≥10%峰值体素，0.1mm等距球形搜索和trilinear插值，无剂量拟合、重归一化或配准。V1/V2总剂量比 1.000001916/0.999999418，L1 0.002304/0.002220。这只是GPU实现匹配，不是TOPAS验收，也不是无偏性的高统计证明。四批ROI分解只是描述性检查，未证明同时间统计FOM提高。

所有正式GPU批次quality accepted，能量残差门槛<1e-4；coverage/invalid/overflow为0，expected/completed一致。每批source CSV、核反应/核弹性计数、ion_species terminal及transport ledger保持匹配。电子工作计数因抽样改变而不要求精确一致。

## 为什么没有推广

V2原生electron worker资源166→171寄存器、1632→1712B静态stack；primary仍231寄存器/2384B。这只能说明静态资源变化，不能推出动态spill或实际occupancy。当前一个1M日志约183.2M charged calls、325936 hard collisions；优化只覆盖其中低能电离碰撞，不能把它等同于整个电子输运。实际退激耗时未单独测出，因而不能进一步认定退激完全不值得优化。

V1两对均未更快，V2配对方向相反、总体持平。没有可采用的实现吞吐提升；不继续对同一候选做更多无目的调参。signed correction和碎片roulette本轮未实施。下一轮应定位primary continuous26 preparation中的重复rate/moment/几何计算，先证明能减少实际工作再开发候选。

原始证据：out/proton_ct_branch_20261009/。恢复后的selective构建及小规模匹配检查另记录在 restore_build.log 和 restored_check/。

恢复验证完成：源码与冻结头文件逐字节一致，专用可执行文件SHA256与原基线相同（7dd578659eb46781e6d59e9b81eed8e19f75ebfbf6fa09e1244758644f96deea）。proton CT broadbeam两批、每臂20k匹配检查通过，source/discrete/EM/energy/coverage/completion/overflow审计保持一致，dose L1≤1.38e-8；不用于性能宣称。

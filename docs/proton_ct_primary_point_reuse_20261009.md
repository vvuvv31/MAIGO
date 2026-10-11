# Proton CT broadbeam 同一步材料视图复用（2026-10-09）

仅在专用 proton CT broadbeam primary（kernel stage13）保留本次优化。准备步进参数时保存该步的 continuous26 `UnifiedEmPoint`，能损计算复用它的材料、mean 与 cut。视图不跨步、不进入 secondary resume state；每一步仍重新读取当前材料/密度。没有修改 continuous26 的校准、质量约定、随机数顺序、步长、涨落、核反应、电子/光子或计分精度。

原路径从 prepare 到 loss 重新构造材料视图，含 log-density 坐标和八个材料字段的插值。这次尝试消除同一步的重复构造，与此前被拒绝的跨步/whole-state material cache 不同。其它管线继续使用默认 `PreparedPoint=false` 路径。当前生成的专用可执行文件位于 `build-proton-broadbeam/carbon_mc`；完整五族 build 不在本轮重新编译范围内。

## 验证

`check_proton_primary_point.cpp` 用当前 EMCNT001 包，覆盖26材料、7个密度位置、20个对数能量位置、4个步长（含 range 两侧）和涨落开/关，共29,120组。准备字段、loss字段、valid/failure stage、proposal/accept计数和最终 RNG 完全一致，非法参考/候选结果均0。它是数值组件验证，不是CPU输运运行模式，也不是性能实验。首次独立链接缺少 libm，在显式加上 `-lstdc++ -lm` 后通过。

示例（从仓库根目录）：

```bash
/home/wuwei/sycl_workspace/llvm/build/install/bin/icpx -O2 -std=c++20 -Iinclude \
  scripts/check_proton_primary_point.cpp src/unified_em_package.cpp \
  -o /tmp/maigo_check_proton_primary_point -lstdc++ -lm
/tmp/maigo_check_proton_primary_point
```

RTX2080Ti sm75，当前release包/CT/源与配置，CUDA_DISABLE_PTX_JIT=1；每进程4×1M连续批次，首批从吞吐统计剔除。先ABBA，再独立种子BABA，共32M primaries（每臂16M，性能统计每臂12M）。所有完整批次source CSV、核反应/核弹性计数、live ion species、EM/charged/photon/joint工作审计相同。quality accepted，energy/coverage/completion/zero-overflow检查通过。

| 顺序 | 候选/基线整体吞吐 | 配对比值 |
|---|---:|---|
| ABBA | 1.008154 | 1.001244 / 1.014994 |
| BABA，独立种子 | 1.013004 | 1.022673 / 1.003471 |

合并暖批吞吐约150,169→151,760 primary/s（本轮观测+1.06%）。这是一项小幅收益。GPU未锁频，温度28→80°C，记录中graphics clock最高1905MHz、后段1830MHz；不能把该比例当作固定硬件常数或严格置信区间。

每批dose L1≤5.25e-8；参考≥10%峰值体素同位置local1%通过率100%，提供local1%/1mm的100%通过率下界，不需要额外搜索。该改动保留原抽样算法和RNG顺序，组件检查与完整GPU审计保持匹配；没有引入新的估计量，也没有新的TOPAS验收。

## 静态资源与决定

候选primary native cubin231→238寄存器，2384→2544B静态stack；这些数值不证明spill流量或动态occupancy变化。实际四对完整吞吐均为正、两种顺序整体方向一致，保留限定范围的同一步复用。收益来自哪类动态指令/等待尚未分离，不能宣称寄存器下降或大幅加速。

原始证据在 `out/proton_ct_point_reuse_20261009/`：baseline/candidate源码与可执行文件、check_point.json、两轮status/abba/baba/summary、原生images与resources。最终仅整理注释/缩进，并重建专用target核对二进制一致性。

最终构建检查完成：注释/缩进整理后ELF SHA改变，但逐函数原生SASS（保留指令及编码）与正式性能候选完全相同，详见final_sass_check.json。20k每臂的最终匹配检查通过，source/discrete/EM/completion审计一致，dose L1≤1.45e-8；该小规模检查不用于性能宣称。

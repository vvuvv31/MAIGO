# 次级粒子随机流种子修复（2026-10-08）

本轮修复确定的随机流身份缺陷：主粒子进入水/CT 后出生的次级粒子，其 stream 只包含 spot 内 history 编号、碰撞步和子代序号，没有继承 spot 的随机种子。次级输运使用共享 Philox key=2026，因此不同 spot/批次的上述编号相同就会使用相同随机地址。仅提高批次数或改变主粒子种子，不能保证次级随机数完整独立。这是随机采样正确性问题；不能单凭它断言此前约 1% 的谷区偏差就是该缺陷造成。

本报告属于 HU0 110 MeV 质子、121 个等权 spot、0.7 mm/3 mm 铜狭缝、0.5 mm XYZ 网格受控实验，不是 RT07575 head 全计划验收。TOPAS 仍使用原 2420 万粒子 AllChargedUrbanPhysics 参考。

## 证据与修改

主粒子 `rng_history = global_history - spot.history_begin`，不同 spot 重复从零编号。旧主核非弹性入口 `event_product_stream(rng_history, steps, role, ip)` 和弹性反冲入口 `child_stream(rng_history, branch_tag(...))` 都忽略 `spot_seed`。次级的 EM、Urban、核反应和弹性抽样随后都使用 `(2026, frag.rng_stream, step, dimension)`，导致条件编号一致时的确定性地址复用。

`primary_descendant_stream(spot_seed, rng_history)` 现在先混合完整 64 位种子与局部 history，再分配碰撞/子代身份。修复覆盖主核反应子代、弹性反冲和铜出生子代；孙代沿现有 stream 树继承。铜与模体内出生使用不同分支域。源、主粒子 RNG 地址、逐 spot 种子、权重分配、原有 RNG 维度、队列布局及 CUDA ordering 均保留，不使用 GPU 原子队列位置作为身份。

物理数据包、continuous26 校准、电子 spatial joint、Urban、步长、铜涨落、FP32 几何和剂量累加保持原设置；没有拟合剂量、移动网格、平滑三维验收数据或恢复中性粒子追踪。

独立检查复现旧共享-key地址完全相同，并验证五种种子（含仅高 32 位不同的种子）、1000 histories 和五种子代分支，共 25,000 个样例没有重复身份。545 项狭缝几何检查和 83 项退役接口检查通过。64 位 stream 混合沿用既有哈希式身份设计。

## 修复前的种子重复对照

额外模拟了相同输入、不同主粒子种子的 2420 万粒子。原先 z=5–15 mm 和 35–45 mm 谷区约 −1% 的偏差分别变为 −0.47% 和 −0.39%，名义区间均包含零，说明该幅度不能直接认定为稳定物理偏差。由于本对照使用修复前 RNG，次级随机地址仍可能复用，不能称为完全独立 GPU 对照。

同一固定深度分层样本中，修复前 GPU 种子重复之间的 gamma 约 77%；TOPAS 两组独立 968 万粒子之间约 68%。粒子数不同的结果不能直接比较或当作通过率上限，但说明同一实现也会在低统计量下大量失败。

## 原种子 2420 万粒子完整复验

修复后使用原先两个重复 ×100 批、每批 121,000 粒子及原逐 spot 种子；每批按整轮 12.1M 粒子归一化，累加 100 批，再平均两轮。原 GPU、TOPAS 文件保留，绝对剂量均为 Gy/入射质子。

200 批均 accepted、零溢出，最大相对能量残差 4.63e-06。完整 EM/charged/photon 覆盖及电子完成审计通过；主粒子核/弹性计数与修复前匹配：True。八个默认预设各两批 ×1000 粒子的运行审计通过，源 CSV 逐位一致、主粒子计数一致。此低统计量检查不构成八预设剂量验收或性能证据。次级随机数变化会改变全部预设的部分离散计数，因此没有将严格实现不变清仓回归记成通过或放宽其标准。

|指标|仅狭缝修复后|再修复次级 RNG|
|---|---:|---:|
|3D local3%/0.2 mm gamma|77.78%|77.77%|
|y 平均 30 mm 诊断性 2D gamma|99.97%|99.97%|
|5–15 mm 峰 GPU−TOPAS|-0.07%|-0.07%|
|5–15 mm 谷 GPU−TOPAS|-1.01%|-0.98%|
|35–45 mm 谷 GPU−TOPAS|-1.06%|-1.04%|
|65–75 mm 谷 GPU−TOPAS|-0.18%|-0.19%|

|深度|修复后 3D gamma|
|---|---:|
|0-20mm|93.61%|
|20-50mm|88.93%|
|50-80mm|69.05%|
|80-100mm|59.82%|

原掩膜内实际差异 RMS 为参考峰值的 2.328%，TOPAS 加修复后 GPU 的统计 RMS 预测为 2.320%。GPU 使用两个独立种子重复估计方差，尚未估计空间协方差；此诊断不证明每个失败体素都是噪声，也没有修正 gamma。TOPAS 单体素相对 sigma 中位数仍约 4.27%，大于 local 3% 容差。剩余低三维通过率需要更高的两端统计量才能进一步区分模型偏差；本轮不宣称已达到三维剂量验收。

3D 使用同一 598,155 体素参考掩膜，local3%/0.2 mm、0.02 mm 搜索、无剂量缩放或位移。参考深度区块并行计算时，每个工作进程搜索完整 GPU 体积；`skip_once_passed=True`、`max_gamma=1.00001` 保留通过/失败判定，保存的通过 gamma 是最小值的上界，失败值截断在 1 以上。0.5 mm 网格上的插值不增加独立 0.2 mm 空间采样。2D 平均结果仅作诊断，不替代三维验收。

## 复现与文件

```bash
g++ -std=c++20 -O2 -I include scripts/check_secondary_rng.cpp -o /tmp/check_secondary_rng
/tmp/check_secondary_rng
./scripts/build.sh -DCARBON_CUDA_ARCH=sm_75 # 根据自己显卡设置
export CUDA_DISABLE_PTX_JIT=1
```

本轮文件位于 `out/hu0_proton_minibeam_3cm_20261008/residual_fix/`：`rng_fix_provenance.json` 记录二进制/源/配置哈希；`run_rng_fixed.cpp` 与 `rng_fixed_case/` 是原种子完整 GPU campaign；`rng_results.json`、`full_runtime_audits.json` 和 `presets_validation/runtime_summary.json` 保存结果与审计。`collect_rng_fixed.py` 顺序运行剂量、3D gamma 和统计诊断；`matched_repeat_results.json` 记录额外种子对照及其 RNG 独立性限制。

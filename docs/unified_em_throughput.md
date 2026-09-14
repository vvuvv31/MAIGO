# 统一 EM 吞吐优化、消融与 RT07575 时间分解

本轮以吞吐为主。默认保持原生 EM 步长与全部已有物理过程，未通过改变 TOPAS 参数提高吻合。硬件为本地 RTX 2080 Ti / sm_75。所有吞吐均指原发 histories/s。

## 已保留的改动

- 材料目录索引与轨迹材料缓存；步首 EM 数据复用。
- 将统一 EM 与 legacy kernel 编译路径分离，保留公开输运接口及配置拒绝规则。
- OpenSSL 文件 SHA256 加速：每次仍完整读取、核验所有包，未缓存或跳过哈希。缺少 OpenSSL 时保留便携实现；构建时安装 OpenSSL 开发包可获得加速。
- 修复原发离开 CT 横向边界后误报 EM 缺失域的问题，剩余能量归入逃逸，补齐与次级相同的边界策略。此修复也应用到 CT 对照基线；无效材料、能量域、采样失败仍拒绝剂量。
- 可选 `CARBON_RUNTIME_BREAKDOWN=1` 输出主机阶段时间；CUDA 传输使用 Nsight Systems 单独测量。

未默认采用：轨迹审计汇总（C，收益接近测量波动）、能量指数粗索引（与 kernel 分离组合首轮额外收益约 1–2%，未达到保留门槛）、步长倍率。研究倍率 1.25 的四组 50k 筛选最高仅 +0.47%，停止扩展；倍率默认 1.0，非 1.0 仅允许研究模式。因此没有进入近似模型的患者 Gamma 门槛，也不声称消除了统一 EM 原有低密度 cut-onset/患者 Gamma 待验收限制。

## 消融

A=材料索引/缓存，B=步首数据复用，C=轨迹审计汇总。8 组合 ×3 模体 ×3 轮；已汇总 72/72 次运行，每次 200k、10k/chunk，按预先规定顺序交替运行，GPU 互斥。以下为中位数：

|ABC|b3 histories/s|b4 lung|b4 bone|每例轮数|
|---|---:|---:|---:|---|
|000|6471|5325|7217|3/3/3|
|001|6491|5334|7238|3/3/3|
|010|6957|5823|7760|3/3/3|
|011|6986|5831|7774|3/3/3|
|100|6761|5569|7441|3/3/3|
|101|6727|5550|7415|3/3/3|
|110|7245|6099|7991|3/3/3|
|111|7249|6093|8022|3/3/3|

这些组合来自同一消融源码快照。000 是编译期开关对照，不等于历史二进制的机器码；不要将二者混为同一个性能基线。最终前后比较使用保留的历史统一 EM 二进制。

## 最终模体对照

|案例|基线 histories/s|优化 histories/s|提升|墙钟 前→后(s)|轮数|
|---|---:|---:|---:|---|---:|
|b3_layers|6687|11152|66.78%|36.788→19.595|3|
|b4_soft_lung|5441|8936|64.25%|43.654→24.029|3|
|b4_soft_bone|7446|12645|69.81%|33.762→17.484|3|

额外完成统一 EM 水模 100/200/300 MeV/u、150/250 MeV/u 能散及 350/400 参考案例。保留原 beam/scorer，统一替换为同一 EM 包后比较两版；原 benchmark 中 b1/b2 的历史结果没有覆盖。水模结果见 `water_results.json`。

## RT07575：100k/s 目标与实际结果

|配置/运行|粒子数|程序 histories/s|墙钟 histories/s|墙钟(s)|有效剂量|
|---|---:|---:|---:|---:|---|
|ct_pilot_baseline_fixed|200000|6457|5280|37.876|True|
|ct_pilot_candidate_fixed|200000|11354|10379|19.270|True|
|ct_1m_candidate_fixed|1000000|14154|13828|72.315|True|
|ct_1m_baseline_fixed|1000000|9232|8681|115.199|True|
|ct_1m_profile_fixed|1000000|14128|13455|74.323|True|
|ct_1m_elastic_fixed|1000000|13304|13010|76.866|True|

生产配置不含独立 all-ion elastic；elastic 对照显式使用 research，其他 EM 参数相同。完整保留 RT07575 的 spot 顺序与分布，采用最大余数法将同一输入 spot 权重分配到精确 200k/1M 粒子，10k/chunk。未用只选快 spot、移除慢轨迹或把次级计入原发吞吐的方式接近目标。

生产配置 1M 的 kernel 合计 66.757 s，占完整墙钟 92.3%。即便完全消除其他开销，按本次 kernel 成本也仅约 14980 histories/s；100k/s 需要 kernel 再快约 6.68 倍。这是当前实现的测量推算，不是硬件绝对上限。不会为追目标强行放宽步长或删物理过程。

## Runtime breakdown

以下主表来自不带 profiler 的 1M 优化运行；CUDA 传输来自同配置单独的 Nsight 运行，不把两次运行的时间拼成一个精确总和。

|主机阶段/设备 kernel|时间(s)|说明|
|---|---:|---|
|输入与 source 准备|0.358210|包含配置、CT 读取、TPS spot 几何与上游空气处理|
|输运初始化|4.588332|包含物理包验证、分配与上传|
|原发 kernel|36.596218|含 EM、核过程、几何步进与剂量原子累计|
|次级 kernel|30.160855|同上，全部支持的带电次级|
|读回与主机 finalize|0.055894|含剂量数组读回、类型转换、诊断与结果组装|
|质量检查与输出|0.450639|含 MHD/RAW、能量账与质量文件|

进一步可测的嵌套项：

|请求项|测量/归属|
|---|---|
|CT preprocessing|本次读取已打包 CT，累计 0.059363 s；不包含此前离线 DICOM→packed 文件生成。其时间分散在输入、初始化及输出阶段，不能重复计入总和。|
|Ray tracing / WET|没有独立 WET 图计算阶段。TPS source batch 为 0.175205 s（含 source 几何等）；CT 边界遍历融合在 kernel 内，不能把 source batch 等同于完整 ray tracing。|
|Kernel evaluation|上述原发/次级 kernel，已有事件计时与 Nsight 时间线核对。|
|Dose accumulation|GPU 原子累计融合在 kernel 内，没有独立可分离时间；主机读回/组装与输出时间见上表。未通过关闭 scorer 的有偏差运行估算“剂量累计时间”。|
|Host-to-Device|0.138687 s，1.791 GB，45 次；包含于相应主机阶段。|
|Device-to-Host|0.004402 s，0.058 GB，37 次；包含于相应主机阶段。|

RT07575 的 1M 运行记录了 2,878,453,068 次原发/次级 EM 步进，2,250,786,233 次 δ 提议和 1,984,087,180 次接受：平均每个原发约 2,878 次 EM 步进、2,251 次 δ 提议。这个工作量解释了为何仅放宽宏观 EM 步长几乎没有收益：δ 碰撞距离仍会截短步长。事件数本身不等于各操作耗时，不能据此伪造 kernel 内百分比分解。

优先结论：输运 kernel 是主要瓶颈；CT 预处理、传输优化无法将当前结果推到 100k/s。后续若继续，应先细分 kernel 内的 EM 查表/δ 电子步进、MCS、核采样与 scoring 成本，再决定是否值得进行更大规模的实现重构；本轮到初步优化和分解为止。

## 验证与重现

- 消融及最终等价候选：相同 seed/config、EM 审计整数、原发步数；逐体素剂量差异/全局最大剂量 ≤0.001%，质量加权 IDD 差异/峰值 ≤0.0001%。零 overflow、能量记账通过。
- RT07575 200k 和 1M 的同修复基线/候选审计及核事件一致，剂量差异见 `rt07575_comparison.json`。未计算或宣称新的 BODY gamma。
- 材料选择 56,540 个边界查询逐位一致；host/GPU 各 94,500 个 G4 能损点零失败；候选粗索引 82,741,572 个 GPU 查询逐位一致。SHA 两种后端覆盖 13 个分块/补齐边界，并校验真实 EM 包；配置保护与研究步长区域限制测试通过。
- 首次 CT pilot 无效结果完整保留：4 个原发 C12 的 section=-2 域失败；没有拿这些剂量做性能/精度验收。修复保持残余能量为 escaped，正常域错误仍硬失败。
- 默认 CMake：`CARBON_EM_MATERIAL_CACHE=ON`、`CARBON_EM_STEP_CACHE=ON`、`CARBON_EM_LOCAL_AUDIT=OFF`、`CARBON_EM_SPECIALIZE=ON`、`CARBON_EM_EXACT_INDEX=OFF`、`CARBON_USE_OPENSSL_SHA256=ON`（可用时）。研究步长两个倍率均为 1.0。
- 所有脚本/小结果在 `benchmark/unified_em_performance_20260913`；原始剂量、日志、二进制和 Nsight 报告在 `scratch/unified_em_perf_20260913` 与 `scratch/unified_em_ablation_20260913`。两处均未自动加入 Git。原始 benchmark、TOPAS 参考和物理包未覆盖。
- `ablation.py`、`final_pairs.py`、`prepare_rt07575.py`/`rt07575_pairs.py` 可重现运行；`summarize_ablation.py`、`compare_final.py`、`compare_ct_performance.py`、`runtime_breakdown.py`、`plot_results.py` 和本脚本生成报告。运行前验证 Schneider manifest，仅本地 GPU，发现 overflow 须拆分基线/候选重跑。

本轮未 commit/push。

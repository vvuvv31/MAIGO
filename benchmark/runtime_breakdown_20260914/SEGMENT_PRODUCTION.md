# 次级 64 步续跑：正式生产接入

用户于 2026-09-14 接受已测得的精度差，并要求接入默认生产配置。本次不做 commit/push。

## 默认行为

- `config/unified_water_production.yaml`、`config/rt07575_unified_em_production.yaml` 显式开启 `secondary_step_chunking: true`；结构默认 false，旧配置保持原调度。
- 每次最多 64 次完整次级循环，存活索引稳定压紧，队列少于 8192 直接完成。暂停不做 terminal scoring，保留 RNG、δ 时钟、材料缓存、pending dose 和诊断累计；当前代完成后才处理子代。
- false 路径不分配续跑缓冲；每代临时内存由 DeviceMemoryTracker 管理，失败停止，不能静默降级。该检查不预估其他进程的显存；若不足须减少每 shard histories。默认 state 为 336 B，加每轨迹约 20 B 索引/标志，RT07575 1M 第一代额外约 1.03 GB。
- 同时接入已验收 mean_guard：原发只在可选审计开启时执行重复 mean 查询，实际联合 EM loss sampler 不变。
- 质量报告记录 `secondary_step_chunking_accepted`；生产 YAML 的独立核弹性设置不变。

## 正式源码开关验证

Release、SYCL CUDA sm_75，本地 RTX 2080 Ti。水/CT 生产配置各通过 17 项检查；新开关解析、关闭路径、无效依赖拒绝均通过。
`build/oneapi-nvidia-secondary-continuation/carbon_mc` 用正式源码构建，二进制 SHA256：
`1d32f4fee006f982b93eca963da57cceee8340d74b807d816a25035ec8f21d99`。

|场景|OFF histories/s|ON histories/s|提升|次级 OFF→ON s|最大体素差/峰值 %|
|---|---:|---:|---:|---:|---:|
|ct50k|7703.4|8157.7|5.90%|1.388 → 0.930|0.00000956|
|water50k|9321.6|10150.7|8.89%|0.769 → 0.339|0.00003021|
|ct1m|21559.5|29535.4|36.99%|21.852 → 8.859|0.00009662|

ct50k/ct1m 未启用独立弹性，water50k 使用已有含全离子弹性的水模输入。
所有运行质量 accepted、零 overflow；EM audit、步数、标量整数 energy ledger 在 ON/OFF 及冻结 mean_guard 基线之间一致。
物理配置一致，仅新调度开关不同，故不要求 config SHA 一致。
OFF 日志无续跑池，ON 日志确认 state=336 B；1M 为 28+12 次推进，压紧约 0.0056 s。
OFF 与冻结基线最大差依次为峰值的 0.00000796%、0.00001510%、0.00002174%。

## 精度边界

此前原型含弹性 RT07575 1M 稳定最大差约 0.000955% 峰值（该体素局部约 0.00337%），
用户已接受。差异超过自重复噪声，原因仍未证明。50k 终止状态逐位审计已通过，
但不能推广成所有轨迹逐位相等。未重新计算 TOPAS Gamma，原有患者 Gamma、低密度阈值区、其他可选编译诊断覆盖限制继续保留。

脚本：`run_segment_production.py`；结果：`segment_production_results.json`。
所有 CT 执行前均通过 `verify_schneider_v2_1_data.py`；本次未修改物理数据。

## 日常生产二进制

README 指向的 `build/oneapi-nvidia-release/carbon_mc` 已重新构建，与上述验证二进制 SHA256 相同。
另用含全离子弹性的 RT07575 1M 执行检查：28534.3 histories/s，次级 9.288 s；
相对冻结基线提高 36.39%，最大体素差为峰值的 0.00095151%。
质量 accepted、零 overflow，审计/步数/整数能量账本一致。结果见 `segment_production_release_results.json`。

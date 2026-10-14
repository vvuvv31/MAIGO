# Proton CT broadbeam 弹性总截面预计算（2026-10-12）

主粒子每步的核弹性 hazard 原先从13个靶元素读取相邻能量节点并逐项插值求和。现在仅对关闭电子输运的 CT proton broadbeam 专用主粒子路径，在初始化时按原靶元素顺序求出节点总截面，GPU每步插值两项总值。原分靶截面、靶核选择、角分布、两体运动学、反冲输运和共享核光学深度保留。水和其他管线使用原查询；包格式、数据文件和预设参数不变。

RT07575 head 原优化完整计划，当前代码，电子OFF，主涨落ON，次级涨落OFF，患者最大步长1mm。RTX2080Ti，ABBA+BABA，四组配对；每campaign三批、每批2M，首批只预热。每臂24M模拟、16M暖批计时。每campaign在全6M中一次性按最大余数法分配spot权重，源/seed配对一致；剂量按原4Gy计划粒子权重归一化一次。

| 指标 | 对照 | 预计算 |
|---|---:|---:|
| 暖批吞吐 /s | 441,918 | 450,445 |
| ABBA BODY local1%/1mm | 99.964971% | 99.964971% |
| BABA BODY local1%/1mm | 99.952440% | 99.952440% |

合并增益1.9296%，四配对比值1.00633、1.02769、1.02471、1.01843。未锁GPU频率，不能外推为所有病例/设备的固定增益。Gamma使用历史TOPAS逐spot物理剂量按原优化权重合成的参考，BODY且参考>=0.4Gy，search0.1mm；不是新提交TOPAS完整重跑的验收。各臂均通过当前runtime energy/EM/coverage/completion/zero-overflow检查。

这是数学线性的求和/插值重排，FP32结果并非逐位等价。100,000组件查询最大相对截面差4.4313e-7，额外device数组14,248字节。配对source/seed manifests一致；少量核事件/EM计数因舍入改变，不能声称整数审计逐项相同。首轮三批配对dose L1最大9.003e-6。所有失效、能区检查及反应末态抽样仍执行。

先前尝试删除重复Z-face检查：control445820/s、candidate443401/s，-0.5425%，未采用并写入failed.md。更大结构性提速仍需要减少实际步进和计分工作；这轮结果不支持几倍吞吐承诺。

证据：`tmp/proton_broadbeam_elastic_totals_20261012/{component.json,report.json,confirmation_report.json,matched_audits.json,summary.json}`，原始日志、配置及剂量均在同目录。正式实现：`include/carbon/all_ion_elastic.hpp`、`src/detail/sycl_pipeline_setup.inc`、`src/detail/sycl_pipeline_primary.inc`。

最终集成以编译期分派仅启用该专用路径的总值查询，水、其他粒子及通用路径仍编译为原分靶查询。组件实际C++ view另通过100,000查询及无效能区/物种/材料、其他投射粒子fallback检查，最大相对差4.53956e-7。

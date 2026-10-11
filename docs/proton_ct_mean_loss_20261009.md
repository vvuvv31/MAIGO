# Proton CT broadbeam：平均能损限制2%→5%（2026-10-09）

用户允许 broadbeam 对 TOPAS 的剂量匹配适度下降，以患者内 local1%/1mm >99%（search0.1mm）验收。本轮保留质子 CT 宽束的 combined mean-loss guard 由0.02改为0.05，已应用到config/proton_ct_broadbeam.yaml。不是要求和旧GPU剂量逐位一致的优化。

## 改动及物理边界

只修改现有broadbeam_maximum_mean_loss_fraction；生产解析器已经支持research broadbeam_fast中的[0.01,0.05]，无新开关、数据包或代码路径。primary/secondary同时使用该guard，减少在该限制下的细步进。

maximum_step_mm仍0.5mm，几何/计分边界、核反应竞争、continuous26阻止本领/射程/涨落、独立弹性、带电碎片输运和中性产物能量sink保留。primary与secondary涨落都开启，energy cutoff仍0.1MeV。电子继续此前已授权的局部沉积。不会把该验证推广到其他粒子、肺或minibeam。

## 完整计划、参照和统计

RT07575是head。采用原始优化计划870行、540个正权重spot，原始浮点weight；CT、beam model与spots均与既有TOPAS DIJ输入哈希匹配。保持原患者placement和source optical参数。物理参照是历史TOPAS DIJ按原始优化权重合成的physical dose，没有重新运行TOPAS，其体素方差不可用。

ABBA与BABA，每进程16×1M，四个不同的campaign seed；同一seed两臂的source分配/spot seed CSV逐字节一致。每个16M campaign用全局largest remainder分配，再切批，连续批次复用GPU context/packages。每臂总64M独立原初历史，每臂暖批计时60M，首批剔除。64M剂量取四份已按原计划入射数归一化的campaign剂量平均，不能求和当成一次计划剂量。没有dose fit或registration。

## 测量

RTX2080Ti、CUDA AOT，CUDA_DISABLE_PTX_JIT=1；两臂用同一最终已链接ELF，SHA256 01769e6c8b8b25d83385b04649a0682f997ae6d2889bd305d8ee230a15aeb6ff。

| 项目 | 原2% | 新5% |
|---|---:|---:|
| 暖批primary/s | 307,591 | 325,621 |
| 64M患者内TOPAS local1%/1mm | 99.908346% | 99.910216% |
| 患者内剂量总和相对TOPAS | +0.014591% | +0.014516% |
| valid EM loss draws | 17,369,555,533 | 16,552,561,655 |

合并吞吐提升 **5.86%**，EM能损抽样次数减少 **4.70%**。四对speed ratio：1.003658, 1.077621, 1.083865, 1.069766，均快。ABBA合并+4.05%，BABA+7.68%。GPU未锁频，温度/频率变化较大，第一对只有+0.37%；这是本机本轮观测，不承诺所有硬件固定提升。

患者内mask为BODY∪GTV且TOPAS≥全图peak10%，共320,771体素。所有单独16M campaign的TOPAS gate均通过（99.942%～99.961%）；64M合并候选99.910216%也通过。64M的gamma低于单次16M，说明gamma不能当作单调统计收敛指标，不能据此宣称剩余误差全是随机噪声。64M候选−control剂量总和变化为-0.00007577%。

两臂共128个campaign批次quality accepted、energy/EM/coverage/completion和零溢出通过；最大物理相对能量残差5.42339006e-07。物理步进改变后，不要求核事件/EM计数与旧模型相等，但保持每一臂自身审计有效；source分配和种子匹配。

## 应用与证据

预设当前采用0.05。其输入spots.csv是既有整数化版本；正式TOPAS验收使用上述原始浮点full plan，不能把两个文件宣称逐字节相同。应用后的预设另做两批完成性/质量检查，不用于吞吐或新TOPAS验收。

配置修改无需重编译；专用可执行程序build-proton-broadbeam/carbon_mc支持该配置。完整五管线build/carbon_mc本轮未重建。安装manifest及preset dependency inventory不变，独立生成inventory与仓库逐字节一致。

证据：out/proton_ct_mean_loss_20261009/；run.py、reverse/run.py、combine.py、全部分批dose/ledger/quality/source CSV、command和原始时间、combined.json与应用日志保留。

应用后预设的两批（每批1M）均quality accepted、零溢出，最大物理相对能量残差1.74308426e-07，quality显式记录5% mean-loss approximation。git diff --check与173资产manifest验证通过。

后续主粒子一阶δ优化已叠加到5%guard，最新专用程序及验证见[一阶δ报告](proton_ct_first_order_delta_20261009.md)。本报告的SHA/测量保留为该阶段历史结果。

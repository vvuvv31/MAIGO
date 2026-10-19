# 当前研究版与全 Urban clean 的直接对比（2026-10-19）

正式配对测试的研究工作树为 `/mnt/sdb/wuwei/MAIGO_mb_secondary_fe`，下文 `tmp/urban_vs_clean/` 指该工作树中的实验目录。

基线为未修改 clean `6dda9970c6fd16b3cf0c4a2f8599fe5a9635be9b`，从 git archive 冻结并重新编译。候选为当前研究树的精确源码快照及已测 native 可执行文件。两臂均为原始 Urban 主/次级输运，实验次级 FE OFF；合并状态见文末。

## 测量范围

RT07575 头部原优化完整计划：proton 487 spots、carbon 400 spots（90°/270°）。每种粒子每臂 3120 万 histories，按原权重 campaign-wide largest remainder 分配，两臂逐 spot 源、seed、YAML 相同。RTX 2080 Ti 每批 200 万，RTX A6000 每批 320 万，两卡并行；每进程连续三批、首批排除，每设备 ABBA。热吞吐按后两批总 histories / 总墙钟时间计算，包含源读取、输运、审计和输出。

两臂同一编译器、Release/P+C minibeam 专用构建，BB/generic 均关闭；显式 sm_75/sm_86 AOT，CUDA_DISABLE_PTX_JIT=1，并检查两臂 native ELF。全部编译结束后才进行 GPU 计时。保留主粒子涨落、次级无涨落、unrestricted 阻止本领、原 Urban/核过程与中性能量汇、FP32 剂量、患者 0.5 mm、Cu 0.05 mm，碳 CT 狭缝空气 0.2 mm。

## 直接热吞吐

| 模态 | GPU | 全 Urban clean | 当前研究版 | 吞吐提升 | AB / BA 配对提升 |
|---|---|---:|---:|---:|---:|
| Proton minibeam | RTX 2080 Ti | 335.39 k/s | 363.48 k/s | +8.376% | +8.067% / +8.684% |
| Proton minibeam | RTX A6000 | 487.94 k/s | 538.77 k/s | +10.417% | +10.885% / +9.951% |
| Carbon minibeam | RTX 2080 Ti | 102.33 k/s | 112.33 k/s | +9.778% | +9.026% / +10.506% |
| Carbon minibeam | RTX A6000 | 169.50 k/s | 186.19 k/s | +9.842% | +7.348% / +12.399% |

当前实现仍为全 Urban。收益来自 Cu/air 同步长 range/MFP 查询复用，以及保持原始支持域与 MFP 的紧凑运行时视图；读取时仍完整验证原磁盘包，对不能证明数值安全的曲线保留原插值检查。这里直接测量整套保留修改，不叠加各轮历史百分比。

## 分阶段用时（秒/百万原初粒子）

| 模态 | GPU | 阶段 | clean | 当前 | 变化 |
|---|---|---|---:|---:|---:|
| Proton | 2080 Ti | 铜/空气主粒子 | 1.47368 | 1.31300 | -10.903% |
| Proton | 2080 Ti | 患者主粒子 | 0.93865 | 0.90427 | -3.663% |
| Proton | 2080 Ti | 铜中直接子粒子 | 0.01574 | 0.01594 | +1.308% |
| Proton | 2080 Ti | 介质次级 | 0.37946 | 0.34613 | -8.783% |
| Proton | 2080 Ti | 总墙钟 | 2.98161 | 2.75117 | -7.729% |
| Proton | A6000 | 铜/空气主粒子 | 0.88396 | 0.74705 | -15.489% |
| Proton | A6000 | 患者主粒子 | 0.66638 | 0.62671 | -5.952% |
| Proton | A6000 | 铜中直接子粒子 | 0.00931 | 0.00932 | +0.178% |
| Proton | A6000 | 介质次级 | 0.20218 | 0.18673 | -7.642% |
| Proton | A6000 | 总墙钟 | 2.04942 | 1.85608 | -9.434% |
| Carbon | 2080 Ti | 铜/空气主粒子 | 2.05841 | 1.83258 | -10.971% |
| Carbon | 2080 Ti | 患者主粒子 | 1.47926 | 1.41450 | -4.378% |
| Carbon | 2080 Ti | 铜中直接子粒子 | 2.16283 | 1.94444 | -10.097% |
| Carbon | 2080 Ti | 介质次级 | 2.27534 | 2.07665 | -8.733% |
| Carbon | 2080 Ti | 总墙钟 | 9.77260 | 8.90216 | -8.907% |
| Carbon | A6000 | 铜/空气主粒子 | 1.15087 | 0.99565 | -13.487% |
| Carbon | A6000 | 患者主粒子 | 1.04548 | 0.99317 | -5.003% |
| Carbon | A6000 | 铜中直接子粒子 | 1.10402 | 1.01144 | -8.385% |
| Carbon | A6000 | 介质次级 | 1.45357 | 1.31940 | -9.230% |
| Carbon | A6000 | 总墙钟 | 5.89954 | 5.37094 | -8.960% |

## 审计与适用范围

- proton_minibeam_RT07575：2640 项离散计数比较，差异 0；最大单场剂量相对 L1 差异 1.144e-07，最大能量相对残差 7.691e-08。固定 BODY & TOPAS >0.4 Gy 区域，两版 GPU 同体素 local1% 一致率 100.000000%。
- carbon_minibeam_RT07575：5280 项离散计数比较，差异 0；最大单场剂量相对 L1 差异 9.821e-08，最大能量相对残差 7.69e-07。固定 BODY & TOPAS >0.4 Gy 区域，两版 GPU 同体素 local1% 一致率 100.000000%。

共 124,800,000 正式 histories，另有 160 万 integration histories。全部 energy/EM/coverage/completion/live-ion/zero-overflow 审计通过。所有配对提升均为正：True。

这是 GPU 实现一致性和吞吐比较，未新增 TOPAS 验收。GPU 时钟未锁定，结果限于此头部计划、批量与专用构建；不替代历史大批次、其他病例或完整五管线构建的吞吐。正式配对测试期间，clean 源码及生产可执行文件保持不变。

证据位于 `tmp/urban_vs_clean/`：`PROVENANCE.json`、`CLEAN_SOURCE.json`、`SOURCE.json`、`MANIFEST.json`、`NATIVE_IMAGES.json`、`INTEGRATION.json`、`RESULTS.json`、`BREAKDOWN.json`、`STATUS.json`；包含远端取回的原始日志和逐批结果。

## 相同粒子数的绝对 TOPAS gamma

两版分别使用本轮每种粒子各 3120 万 histories，合并所有设备/独立批次（含首批，计时首批排除不影响剂量统计）。保留已有 campaign dose_output_scale，原始 batch dose 只求和一次；无剂量拟合、重新归一化或配准。共同固定 mask 为 BODY 且 TOPAS >0.4 Gy；TOPAS 为 reference，local 剂量容差 3%，空间搜索步距 0.02 mm。

| 模态 | 版本 | local3%/0.2mm | local3%/0.1mm | local3%/0mm |
|---|---|---:|---:|---:|
| Proton minibeam | clean 全 Urban | 91.63039% | 78.90577% | 56.90903% |
| Proton minibeam | 当前研究版 | 91.63039% | 78.90577% | 56.90903% |
| Carbon minibeam | clean 全 Urban | 92.12993% | 83.77997% | 57.02034% |
| Carbon minibeam | 当前研究版 | 92.12993% | 83.77997% | 57.02034% |

三种标准的通过率变化均为 0.00000 个百分点，逐体素通过/失败状态变化均为 0。两版 GPU 直接比较，这三种 gamma 均为 100%。质子 mask 325,762 个体素，碳 mask 399,247 个体素。

这里的 GPU 统计量为 3120 万，不能直接与历史 10 亿 GPU 粒子数的绝对通过率混为同一结果。该配对结果说明本轮约 8%–10% 的吞吐优化没有改变本轮绝对 TOPAS gamma；并非新的高统计量 TOPAS 验收。

新增证据：`tmp/urban_vs_clean/GAMMA_RESULTS.json`、`GAMMA_COMPLETE.json`、`gamma_compare.py`，以及 `analysis/total/<mode>/gamma_comparison.json`、两版合并剂量和逐体素通过数组。

## 合并到 clean 与完整构建检查

2026-10-19 将接受的同步长查询复用、紧凑 Urban 运行时视图与现有离线因子校验工具选择性合并到 clean。CMake FE 实验选项和次级 FE 输运代码未合并；主/次级仍采用原 Urban，所有 YAML、物理开关、源与步长设置不变。

从合并前 git 快照重新构建全五管线基线，与合并后的完整 Release 程序直接匹配检查。显式 sm_75 AOT，所有 GPU 执行 CUDA_DISABLE_PTX_JIT=1；现有 `build/carbon_mc` 已更新。八个发布预设，以及 generic 质子 CT 宽束/微束、碳 CT/水微束，每配置每版连续两批、每批 20,000 histories 全部通过，共 960,000 histories。这是低统计量实现回归，不重新声称吞吐或 TOPAS 验收。

全部源行/随机种子、离散与 live-ion 审计一致，能量/EM/coverage/completion/zero-overflow 通过；剂量最大相对 L1 差异 2.9794924e-07。额外比较八预设共 3344 个离散计数字段一致。107 项接口与 4 项 package 契约检查通过；全部原始资产清单及输入身份检查通过。

Urban 主机检查覆盖 3,195,279 个状态、9,585,837 次比较，原支持域/MFP 与异常曲线数值 guard 行为保持；原始非法零因子拒绝。已合并的离线工具对正确构造参考通过、错误原始因子拒绝；这属于工具接口检查，并非新增真实 G4 验收。

合并检查证据位于 clean 工作树 `tmp/urban_merge_20261019/`：`PROVENANCE.json`、`SCOPE_CHECK.json`、`NATIVE_IMAGES.json`、`VALIDATION.json`、`HOST_CHECK.json`、`OFFLINE_CHECK.json`、构建日志及逐批原始审计。

# Carbon minibeam 独立电子包：头部软组织优先试验

2026-10-04：按用户要求，首轮从肺 CT 改为 RT07575 头部软组织。已完成均匀软组织中 C12 250 MeV/u 的新旧路径各 2×100,000 primary 对照。新包可运行，峰谷汇总差异较小，**尚未完成真实头部 CT / TOPAS 验收，未切换 carbon 默认预设**。

## 范围与数据

- 使用真实 RT07575 CT 中最常见软组织 section 7 的中位密度 1.0415881872 g/cm³，构建均匀材料体模。不是把患者 CT 的骨或空气替换成软组织后声称是患者结果。
- 网格 600×20×320，体素 0.1×2×0.5 mm；3×3 cm 平坦 C12 源，250 MeV/u，原 carbon 铜准直器，group32，两个固定种子 1002111001 / 1003111004。
- GPU FP32 剂量；候选电子输运 precision=4，保留当前 0.05 mm 电子步长上限、0.005 损失限制、光子和横向输运。
- 共享当前 proton 使用的 Schneider joint / charged / photon 传播包；**不共享 proton 的出生谱**。从已有 C12 v7 缓存无抽稀导出原始出生能量—角度联合 CDF、产额和全部样本，保留 0–255 MeV/u 的 51 个原始 5 MeV/u 通道，当前入口限单能 C12 ≤250 MeV/u。
- 出生包 990,004,008 bytes，63 个材料密度节点，最大电子能量 0.633830106 MeV，落在共享传播包的 0.65 MeV 覆盖内。源仍是旧 C12 提取的每材料 16 条 300 MeV/u 历史，有限出生谱统计不因运行更多 primary 自动改善。
- 新格式 MAIBIR02 显式记录 Z/A，禁止给 C12 加载 proton 出生包；旧 MAIBIR01 仍按 proton 解析。保留包缓存键加入 Z/A。
- 此次仍只处理 primary-loop 电离电子响应；核碎片的电子损失保持原有局部处理，不能称为所有次级离子的电子输运均已接入。

## 剂量结果

参考是同场景旧 C12 endpoint v7 GPU，**不是 TOPAS**。按 incident primary 归一化，不做剂量拟合。

| 指标 | 新路径相对旧路径 |
|---|---:|
| 汇总峰区 | −0.1460% |
| 汇总谷区 | +0.0150% |
| 中央峰区 | −0.1682% |
| 全照射野均值 | −0.00420% |
| 峰区深度曲线最大绝对相对差 | 0.7747% |
| 谷区深度曲线最大绝对相对差 | 0.8113% |

汇总区域使用 |y|<15 mm、|x|<12.6 mm，峰区距缝中心 <0.2 mm、谷区距最近缝中心 >1.6 mm。曲线逐深度比较时使用各自参考曲线最大值 3% 阈值；不能从空间汇总结果推断全部体素满足 1%。

二维 y 平均剂量 gamma，|x|<15 mm，参考二维图最大值 3% 阈值，30,572 个评估点：

| Gamma | Local | Global |
|---|---:|---:|
| 1% / 0.2 mm | 88.015% | 99.987% |
| 2% / 0.2 mm | 96.811% | 100% |
| 3% / 0.2 mm | 99.029% | 100% |
| 3% / 0 mm | 93.887% | 100% |

阈值内二维剂量图局部相对差绝对值 p95=3.292%、p99=6.145%、最大 19.152%。剂量图相对 L1=0.905%。两批初测不足以分离电子随机抽样噪声与模型偏差，不能宣称全部局部剂量通过 1% 验收。

## 运行与性能

RTX 2080 Ti，编译结束后顺序运行，未做 ABBA 性能认证。

| 每批 100k primary | 旧路径 | 新路径 |
|---|---:|---:|
| 第 2 批 transport_total（包已驻留） | 6.437 s | 6.448 s |
| 第 2 批对应吞吐 | 15,536 /s | 15,509 /s |
| 两批平均电子处理 | 0.1311 s | 0.1687 s |
| 首批 transport setup | 3.395 s | 8.326 s |

驻留吞吐基本相同，新路径电子部分约慢 28.7%。均匀材料旧路径只加载所需表，约 79 MB；候选当前加载完整共享包，运行 bank 3.303 GiB（其中映射主存 1.266 GiB），源组件合计 4.470 GiB。因此本试验不支持“均匀软组织更快或更省内存”的结论；全异质 CT 的旧大包情况须另测。

外部进程墙钟旧/新为 26.180 / 21.028 s，但程序内 process_main 为 16.118 / 20.913 s，旧进程有明显启动前延迟。**不以该外部墙钟差宣称加速**。

两臂各两批 quality accepted（research / non_production）、零 queue overflow，打印的 primary / Unified EM 审计一致；候选 1,723,569 个电子包全部完成，coverage_missing=invalid=photon_gap=0，legacy_loaded=legacy_fallback=0。1,000-primary smoke 和错误 projectile / 260 MeV/u 拒绝检查通过。

Proton 同配置旧/新二进制 1,000-primary 回归也通过：primary / EM / joint / charged / photon 整数工作一致，剂量相对 L1=3.2931e-11；正式 proton 可执行文件未替换。

## 完整头部尚缺的覆盖

RT07575 实际 CT 的材料密度扫描：section 0–14 在当前 C12 出生库内；section 15–19 的部分密度越界，section 20–24 缺失，共 137,985 个体素未被该出生库覆盖。这是体素覆盖数，不是被照射体素数或剂量比例。骨材料缺口不能由 soft tissue 的密度缩放、钳位或局部沉积掩盖。

下一步先扩展头部软组织材料及密度条件的独立验证，再补齐高密度骨材料出生节点，检查软组织—骨 / 空气界面，随后开展真实头部 CT 和同条件 carbon TOPAS 对照。当前已有 proton full plan TOPAS 剂量不能直接充当 carbon 参考。

所有试验证据位于 `out/carbon_independent_trial_20261004/`：`soft_comparison.json`、`soft_comparison.png`、`soft_*`、`head_material_coverage.json`、`birth/manifest.json`、`birth/source_identity_check.json`、`proton_regression.json`。候选二进制 SHA256 为 `f2f7f3e14e8435bb2417b49aaf31dfaada35bb1df60ca4492b3fa7ebf9f22557`；出生包 SHA256 为 `d7fd1918fc6b2fed88049a0113b823972853de82da0261b47322f0e412b93df8`。

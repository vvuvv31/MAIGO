# C12 肺 CT minibeam 接入与首轮验收

最新状态（2026-10-03）：**无损紧凑电子包已接入默认，完整肺CT连续group32运行通过，运行bank降至45.92GiB，当前预设峰值主存约47.6GiB，未再发生OOM；持续吞吐约3337 primary/s。** 完整63表源检查、GPU同种子回归和正式入口检查已完成，详见[紧凑包交付报告](carbon_electron_compact_20261002.md)及 `out/carbon_electron_compact_20261002/progress.json`。双方200k的初步gamma已计算，但尚未通过最终TOPAS剂量验收。参数化响应仍是独立近似研究；下面保留早期接入与OOM阶段的历史记录，不代表当前后台状态。

用户确认首轮使用与 proton CT minibeam 相同的肺 CT/细网格，C12 入射能量 250 MeV/u。采用 3×3 cm 单能平坦源、相同铜准直器；不是现有 carbon water minibeam 的 TPS/emittance 源。报告 local/global 1%、2%、3%/0.2 mm 及 3%/0 mm，并分别检查峰谷剂量。本轮初始统计为 GPU/TOPAS 各 2×100,000 primary，不把低统计初测自动认定为最终验收。

## 已完成

- README 的八组模式矩阵将 carbon CT broadbeam 标为 **已验收（RT07575）**，依据 `carbon_joint_acceptance_20261002.md`。
- `src/transport_sycl.cpp` 的 CT compact 入口允许 research C12，使用共享 Cu/air 上游与通用材料 CT stage 4。粒子 Z/A、总动能和 RNG 状态保持原有传递方式；质子专用 stage 11、Cu-child queue 和默认能量排序仍只在原来的质子条件下启用。
- 初版 FP32 二进制 SHA256 `833ca2b3fc9bbb21c40b0ec996df3af078d103c3acd2d36c27e77a68fdbf03a4`，归档于 `rejected_sample_limits/candidate_carbon_mc`；下列已有路径回归使用此版。容量修复版 SHA256 `765239db765e67c00964d6770a68c339aa9c2517c9687a9df37b6af517f5fdfb`，已冻结为 `out/carbon_ct_minibeam_20261002/candidate_carbon_mc` 并重启完整响应验证。
- 现有 proton CT minibeam 和 carbon water minibeam 各 1,000 primary 的旧/新二进制回归通过：Unified EM 审计一致、quality 通过、零溢出。剂量相对 L1 分别 `2.0946e-11`、`1.3114e-8`；峰值归一最大差异分别 `1.7555e-8`、`1.0275e-7`。
- C12 1,000 primary 的 combined→compact 传递诊断通过：电磁审计完全一致，核反应 60 次、主粒子弹性事件 36 次均一致；剂量相对 L1=`2.6255e-10`、峰值归一最大差异=`2.9448e-8`。**此诊断使用局部电子，仅验证通道拆分，不用于电子剂量验收。** 自动流程还会以完整 C12 电子响应重复检查。
- 原生 full-physics TOPAS 1,000 primary smoke 和两批 100,000 primary 均已完成。两批实际墙钟时间约 223.32 s、225.34 s，未计分能量分别 `2.01112e-5`、`1.319e-5` MeV。没有新增 GPU 剂量比较结果可报告。

## C12 专用电子包

材料/密度网格沿用 proton 包对应肺 CT 的 63 个采样点，响应数据重新由 **C12** 独立产生。不是修改 proton 索引中的粒子标签，也不复用 broadbeam joint 的简化路径。

- 数据目录：`data/ct_electron/carbon/`。
- 每个材料/密度点提取 16 个初始 **300 MeV/u** C12，记录从入射到停止的电离及电子/光子输运状态；编译为 5 MeV/u 能量区间。
- 目标能量覆盖为低能至 300 MeV/u，250 MeV/u 是本次束流验收点。数据覆盖、边界查找通过、某入射能量的患者剂量验收是不同层次；不能宣称所有能量均验证。
- 原生提取只隔离主粒子电离，移除大质量粒子的 CoulombScat/nuclearStopping；患者 TOPAS 对照保留 full physics，未使用这个提取专用模块。
- 每个包保留原始轨迹、状态引用、响应表和 SHA pins，严格检查材料、密度、粒子身份、轨迹闭合。Photon continuation 的现有限制仍保留。
- 低密度材料原始轨迹很大，例如首张表约 6.36 GiB，编译耗时约 621 s；采用受限并发，避免一次铺满全部表导致内存不足。
- 63/63 张表已经编译完成，完整索引和 `config/carbon_ct_minibeam.yaml` 已生成。
- 首次 GPU 在输运前报告 `Water response binary header`：最大 C12 表有 16,341,241 个样本、21,928,974 个路径节点，超过加载器原有 10M/20M 上限。仅为 schema6 的 C12 材料表将有界容量扩至 20M/25M；保留原 water/proton 上限、SHA、字节数、通道能量闭合、CDF/路径图和运行内存预算检查。原失败日志及二进制归档于 `rejected_sample_limits/`，没有剂量结果被接受。修复后先独立加载审计实际 C12/proton 包，再重启 FP32 GPU 验证。

## 原自动后续流程（OOM 后已停止）

容量修复后的独立 host 审计已完整加载实际 C12/proton 各 63 张表，均退出 0；错误索引 SHA 拒绝检查也通过。证据为 `loader_audit.json`、`loader_carbon.log`、`loader_proton.log`，FP32 构建日志为 `build_capacity_fix.log`。这些是数据完整性与加载检查，不等同于患者剂量验收。

`benchmark/carbonCT/minibeam_validation/finish_campaign.py` 保留原流程及候选二进制记录；当前已停止，以下不是正在执行的后台任务：

1. 等待全部 C12 表完成；工作进程失败或文件缺失则停止并写入失败状态。
2. 生成 pinned `index.json` 与 `config/carbon_ct_minibeam.yaml`，采用 material electron group32、retained bank 和 decoded cache。
3. 完整响应的 combined/compact 1,000-primary 对照，检查 quality、队列和剂量；失败不继续放大。
4. 同一 GPU 进程运行两批 100,000 primary，复用包。
5. 与已有两批 TOPAS 对照，生成 `comparison.json`、`comparison_maps.npz`、`comparison.png`。

首轮 gamma 沿用 proton minibeam 的 **二维 BODY 内 y 平均图**：`|y|<15 mm`，评估 `|x|<15 mm` 且参考图剂量≥最大值3%的点；不是三维全体素 gamma。按各自实际 primary 数归一化，不做拟合比例或剂量配准。最终接纳仍需评估峰谷差异及蒙卡统计，不从代码运行成功自动推断。

运行脚本：

- `prepare_bank.py`：提取/编译 C12 包，独立 pin 提取程序和编译工具。
- `prepare_gpu.py`：生成 C12 专用配置；transport-only 诊断只允许写入 `out/`。
- `run_topas.py`：参考运行和实际历史数/未计分检查。
- `regression.py`：原 proton CT、carbon water 路径回归。
- `finish_campaign.py`：串接完整响应对照及后续批次。
- `compare.py`：初始剂量、gamma、峰谷比较。

原始输出均在 `out/carbon_ct_minibeam_20261002/`，不覆盖 proton 或历史 carbon 输出。原始验收流程已停止；恢复前必须处理 OOM/内存占用，不得跳过失败的质量或覆盖范围检查。参数化原型的独立电子检查不替代本流程的 CT 剂量验收。

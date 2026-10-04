# Carbon CT minibeam 无损紧凑电子包（2026-10-03）

当前实现和默认预设已安装，正式 `scripts/run.sh` 入口的两批回归通过，第二批命中bank缓存。该项优化不等于TOPAS剂量终验。适用的实测场景为RTX2080Ti、肺CT、C12 250 MeV/u、group32、100k primary/批。实验、冻结源码、日志和剂量位于 `out/carbon_electron_compact_20261002/`，以下称R。

## 实现与内存

不引入新的物理近似：保留电子/光子状态、横向扩散、级联关联、出生分布、CDF和随机候选顺序。GPU剂量始终为FP32 ON / FP64 OFF；原轨迹double字段保持位值，这不是FP64剂量计分。

- 原始250,216,366行中，57,267,808行父C12仅用于离线出生索引。完成原始SHA、谱系、出生和能量验证后移除这些运行时不用的行；保留192,908,620条electron和39,938条photon记录，并重映射全部关联。
- 232B状态分解为144B动态记录、共享64B轨迹字典和32B材料/状态字典，全部原字段可逐位恢复。状态数组58,050,196,912→31,412,809,888 B。
- 相邻能量桶没有任何候选进入/退出事件时共享同一有序候选序列；9,199,475,958个逻辑候选只存3,011,488,463项，逻辑offset/count与抽样rank不变。
- 保留47,831,596个出生样本和逐步原始回放验证得到的child-energy证明；动态输运检查继续执行。
- `MAIGO_CARBON_EM_4`绑定原始SHA、材料身份、相关源码指纹和ABI，每块及manifest均有SHA。流式加载直接写最终主存目标或既有GPU上传暂存区，不同时重建完整原始巨图。

| 指标 | 原始库 | 当前紧凑库 |
|---|---:|---:|
| 运行bank累计分配 | 100,453,630,192 B（93.56 GiB） | 49,309,078,020 B（45.92 GiB） |
| 数组磁盘体积，未压缩 | — | 49,309,053,852 B |
| 可选zstd数组体积 | — | 26,802,846,720 B（约24.96 GiB） |
| 完整进程HWM，当前4GiB热预算 | 曾发生系统OOM | 约47.61 GiB |
| 完整运行采样显存峰值，当前预设 | — | 9723 MiB |

运行bank减少50.91%。zstd只缩小磁盘文件，不进一步降低解压驻留量。主存预算64GiB按完整解压字节计；这仍不是适合小主存设备的包，也不保证任意批次/队列配置都不会超限。

## 加载与持续吞吐

最终预设选择 `tiered_small_decoded`、4GiB热预算、8M次级队列；电子任务队列预算仍为576MiB，group32。小表优先只改变分配/晋升顺序，发布表编号不变。本机快速缓存位于NVMe，YAML使用项目内相对路径。

最终raw格式4×100k确认：首次准备69.919s；各批101.313、29.552、29.341、31.182s，整个进程观测198.496s。第一批之后命中同进程bank缓存，不重复读取49GB数组。过程墙钟由1s轮询观测，阶段/批次使用程序自身时钟。

两组相反顺序比较的驻留批次吞吐收益为+5.876%和+6.090%。合计每arm驻留400k：

| 指标 | 原顺序1GiB | 小表优先4GiB |
|---|---:|---:|
| 驻留批次累计墙钟 | 127.109s | 119.873s |
| 持续吞吐 | 3147 primary/s | 3337 primary/s |

持续吞吐+6.037%；包含相同六批核工作量的电子吞吐+20.619%，整体transport吞吐+5.304%。证据 `delivery_placement.json`、`delivery_confirm.json`、`delivery_performance.json`。

存储筛选中，原顺序1GiB的首次setup为raw NVMe84.701s、zstd NVMe129.925s；后续电子核相同。页缓存状态未控制，不能把这些观测值当作严格冷启动速度比。本机选raw作快速默认，zstd保留为磁盘节省选项。仅扩大原顺序热预算到4GiB，驻留批次只改善0.66%，不作为默认。外部NUMA启动偏好也没有明确收益，未启用默认绑定；相关拒绝和重试条件已写入 `failed.md`。

历史原始库combined 1k曾耗时954.589s，compact曾被系统OOM杀掉。其磁盘、缓存、粒子数不同，不与当前100k直接计算加速倍数。

## 正确性与回归

- v3、固定v4分别完整检查全部63表：原始字段位值、next/child、回放能量证明、全部逻辑候选、出生CDF/关联一致。证据 `host_checks.json`、`host_checks_v4_fixed.json`、`inventory.json`。
- 固定解码器检查全部49.309GB原数组SHA；损坏帧、截断、尾随数据、超出声明输出、raw/zstd双重编码均拒绝。完整帧、128MiB最大解压窗口、精确输出大小和SHA检查均保留。最初的合法“输入耗尽”状态误报已修复，拒绝版冻结在 `rejected_v4_drain/`，未接受其GPU结果。
- 完整CT：v3对历史原库combined/compact；v4对v3以及不同内存放置的连续批次，支持域和全部离散/EM审计一致，局部剂量差仅为FP32累加舍入；能量与零溢出检查通过。
- 正式YAML入口的均匀section19 group32、group1、inline各2批通过。错误proton身份、同时指定proton包均拒绝。group1/inline仅小规模回归，不宣称它们的大批次容量已验收。
- 新提取包的prepare→encode→源校验链已通过单表测试：`encoder_smoke.json`。
- proton CT与RT07575各旧/新2×10k通过。原proton63表及RT扩展82表按新指纹重制，旧键保留；全部有效字段位值一致。部分24B birth_sample仅4B尾部ABI padding不同，CDF double位值、row和source均逐位检查；运行时各块自己的SHA照常验证。证据 `proton_artifact_equivalence.json`。
- 六个其他预设旧/新2×2k回归的剂量/离散审计均一致，五个正常退出并通过原质量门。carbon CT broadbeam joint保留旧版的 `unvalidated_electron_joint_response` 非生产质量失败和exit1；没有为了回归改门，也不推翻其已有独立RT07575剂量验收记录。

proton性能回归为old/new/new/old，每进程3×500k，每arm共3M，比较后续驻留批次：整批墙钟26.020→26.133s，吞吐−0.432%，两次对照方向不一致；transport核累计慢1.713%，电子核慢8.499%。因此只说整批基本持平，不声称各核零退化。新增共享视图的版本存在这项局部回退，未作独立硬件归因；proton剂量最大非零体素相对差1.513e-6（约0.0001513%），全部离散审计/能量/零溢出通过。证据 `delivery_proton_perf_metrics.json`。

## TOPAS比较仍未终验

这是优化与原GPU回放的一致性验收。现有TOPAS比较仍仅双方200k、二维BODY y均值、参考峰值3%阈值、无拟合缩放，不是3D最终gamma：

| Gamma | Global | Local |
|---|---:|---:|
| 1% / 0.2mm | 92.748% | 78.620% |
| 2% / 0.2mm | 97.408% | 84.451% |
| 3% / 0.2mm | 98.926% | 86.780% |
| 3% / 0mm | 86.145% | 17.141% |

中心峰积分偏差+1.867%、谷区−0.279%，两批SEM约4.925/5.096个百分点，尚不能认定收敛到1%。证据 `comparison.json`。本次不改变原能量适用域，也不把250MeV/u单场景验证推广到全部能量/CT。

## 使用与复现

```bash
python3 scripts/prepare_material_electron.py \
  --index data/ct_electron/carbon/index.json \
  --cache data/ct_electron/carbon_compact --carbon-compact
./scripts/run.sh carbon_ct_minibeam \
  --histories-per-batch 100000 --batch-count 4 \
  --batch-output out/carbon_ct_four_batches
```

本机 `data/ct_electron/carbon_compact` 链接至 `/home/wuwei/.cache/maigo/carbon_em_v4_285a84dd_raw`。便携配置不硬编码该机器路径；其他机器可直接在相对目录制包。修改相关算法源码后需重制派生包，无需重新提取原物理数据。

若优先节省磁盘，可另行执行：

```bash
python3 scripts/encode_carbon_electron.py \
  --index data/ct_electron/carbon/index.json \
  --source data/ct_electron/carbon_compact \
  --cache data/ct_electron/carbon_compact_zstd
```

然后将 `material_electron_compact_directory` 指向新目录。显式YAML优先于旧实验环境变量；proton仍用独立的 `material_electron_packed_directory`。转换不覆盖旧文件，不省略原始校验。源验证器为 `scripts/check_carbon_compact.cpp`；迁移工具只接受已审计的v3身份，不静默改旧包版本。

交付算法指纹：`285a84dde87e93f75b1302553dc07e0ba76b44229387478041fb66455fb3fc7e`。
交付二进制SHA256：`fa4d2c225e987f6126b62847fdc15b81292d26b2c360b074bf35ab68ce2281f5`。
主目录安装的是冻结隔离构建的同一二进制，安装前逐文件核对源码；主CMake已重新配置为相同指纹及FP32计分。既有未提交修改已保留，未commit/push。

正式入口最终检查：`R/installed.json`，2×1k完整肺CT，对原v3 compact的剂量支持域、离散审计、能量与零溢出均通过；主源码/指纹/已安装二进制与冻结验收版本逐项一致。

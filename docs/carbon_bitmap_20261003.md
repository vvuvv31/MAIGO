# C12位图候选索引与32位图（2026-10-03）

状态：隔离实现、离线转换、FP32 GPU小回归、完整CT反向对比及proton/其他预设回归已完成。已合并相同源码并安装已验证二进制、切换默认缓存；正式脚本两批交付检查通过。目标是进一步降低运行电子库，同时保持所有原始物理字段、电子/光子级联、横向输运和随机选取序列。

候选代码/证据：`out/carbon_bitmap_20261003/`。实验始终FP32剂量ON、FP64剂量OFF。

## 格式与大小

新的独立格式MAIGO_CARBON_EM_5：原144B动态记录及共享轨迹/材料字典不变；next、child_offsets、children改为经界限检查的uint32；next以UINT32_MAX表示终点。稠密能量候选桶改用uint64位图和每512行uint32前缀rank，稀疏桶保留原uint32行号。相同候选桶共享16B描述符指向的数据。

rank/select先用前缀rank定位512行块，扫至多8个64位字，最后用6次固定二分找到所选bit；不使用与rank成正比的清bit循环。返回的行号必须与原候选同一rank逐项相同。桶边界、候选数量、精确能量筛选和fallback次序均保持。

63表实际数组37,610,799,420B（35.0278GiB），原49,309,053,852B（45.9226GiB），减少23.724%。运行bank元数据及完整进程主存/显存峰值要以GPU测试为准。这里没有实施端点共享，因此不声称已达到29GiB。

## 离线证据

- 全63表迁移：原v4身份固定为285a84dd，所有原始块逐个SHA检查，共同物理数组保持不可变硬链接，新图逐行窄化/反解；全3,011,488,463个存储候选逐项重建并比对顺序，另117,643,225次rank/select边界/抽样检查通过。v4本身已有全63表原始源重建验证；这是完整链式无损检查，不是仅按样本推断全库正确。
- 新候选进一步直接对原始轨迹检查table62、0、19（最小、最大及中间表）的字段位值、所有逻辑候选、谱系、出生CDF和child-energy证明，均通过。
- GCC和项目Clang各通过4,967,876次合成精确select，覆盖空桶、满桶、空512行块、尾部不足64/512行、共享桶、稀疏与稠密切换；越界32位转换、重复和逆序候选均拒绝。
- 损坏图SHA、截断位图、错误格式/身份均正确拒绝。新的原始制包→v5 raw→zstd→原始源检查链也通过小表测试。

整个制包/转换在CPU完成，一张表一张表处理；没有在GPU运行时重建巨图。旧v4包和当前默认始终保留。候选算法指纹9a344a25b97b2184fd6a568fd00de2bd6ea7145721d8ac4e84134152cd818305。

## 完整肺CT实测

RTX2080Ti，FP32，group32，小表优先，热预算4096MiB，二次粒子队列8M。旧→新→新→旧，各3×100k；GPU测试期间没有制包或编译并行。

| 指标 | 旧v4 | 新v5 |
|---|---:|---:|
| 实际运行bank |45.9227GiB|35.0278GiB|
| 进程主存HWM |47.6131GiB|36.7023GiB|
| 常驻后4批合计 |123.7807s|117.4094s|
| 持续吞吐 |3231.5primary/s|3406.9primary/s|
| 6批电子核总时间 |28.0669s|26.2082s|
| 两次setup |77.92/68.12s|55.09/41.38s|
| 两次首批 |104.85/96.45s|83.97/70.64s|

常驻吞吐+5.43%，电子处理吞吐+7.09%。初次配对采样显存峰值9723→9729MiB，压缩主要降低主存；热预算和粒子队列没有缩小。加载/首批时间受页面缓存影响，未建立严格冷盘速度比。

每批离散输运、EM和电子任务审计相同，能量/质量及零队列溢出检查通过，非零剂量支持一致；新旧最大体素相对差6.3813e-7（0.0000638%），属于FP32累加顺序差异。group32、group1、inline小回归及不兼容路径拒绝检查也通过。

证据：`full.json`、`carbon_acceptance.json`、`delivery_smoke.json`，以及各次独立配置、日志、资源采样、二进制SHA和剂量结果。

## Proton与其他预设

Proton CT和RT07575分别2×10k，剂量支持/离散事件/EM一致，能量、质量、零溢出通过；最大体素相对差分别2.6773e-7和2.5749e-7。

Proton CT反向顺序旧→新→新→旧、各3×500k：常驻4批总时间26.4218→26.0781s（吞吐+1.32%，视为基本持平），输运核总时间32.62636→32.62651s，电子核6.36691→6.43236s（+1.03%时间）。审计相同，最大体素相对差1.3354e-6（0.0001335%）。这次不声称加速proton。

六个其他预设的剂量和审计均一致：proton/carbon water broadbeam、proton CT broadbeam、carbon CT joint broadbeam、proton/carbon water minibeam。前述carbon joint预设仍因既有unvalidated_electron_joint_response在第一批后退出1；新旧均只有这个质量失败，零溢出、能量残差和实际第一批输出均检查。这是回归行为一致，不是该通用包的新验收。

证据：`delivery_proton.json`、`delivery_proton_perf.json`、`proton_acceptance.json`、`regression/summary.json`。完整日志记录原有物理模型限制；本次没有重算TOPAS gamma，Carbon minibeam TOPAS终验仍待完成。

## 迁移、交付与复现

算法指纹：`9a344a25b97b2184fd6a568fd00de2bd6ea7145721d8ac4e84134152cd818305`。
已编译/验证的FP32二进制SHA：`c36d887047e7c0c11f289632c50d191d3d40b6ced4378dd75d4dbf16fbf0b500`。
合并前逐文件核对主目录仍与隔离基线相同，再仅复制本次16个修改/新增文件；已有未提交工作保留。主目录部署的是相同源码的隔离构建产物，未声称重新编译主目录；`scripts/build.sh`可据当前源码重建。

默认相对路径不变，本机链接已从`/home/wuwei/.cache/maigo/carbon_em_v4_285a84dd_raw`切到`/home/wuwei/.cache/maigo/carbon_bitmap_v5_20261003`。旧包、冻结旧二进制和主源码备份仍保留在证据目录/原缓存位置。磁盘旧包未删除，因此本机总磁盘占用不会自动按运行库减少的量下降。

已经持有审计过的raw v4时，可迁移到新的同文件系统目录：

```bash
python3 scripts/migrate_carbon_bitmap.py \
  --index data/ct_electron/carbon/index.json \
  --source /home/wuwei/.cache/maigo/carbon_em_v4_285a84dd_raw \
  --cache /home/wuwei/.cache/maigo/carbon_bitmap_new
```

首次从原始轨迹制包使用README中的`prepare_material_electron.py --carbon-compact`。新格式身份随共享源码指纹变化，proton派生缓存也需相应身份；本机63表及RT07575的82表都已准备，旧身份保留。不可把新格式目录给旧二进制，或把旧v4目录给新二进制。迁移工具只接受固定已审计v4身份；v3→v4旧入口已明确报错退役。

长期保留的合成测试为`tests/electron_bitmap.cpp`，验证rank/select顺序、空块、尾块、共享桶、32位溢出等：

```bash
g++ -std=c++20 -O2 -Iinclude tests/electron_bitmap.cpp -o /tmp/maigo_electron_bitmap_test
/tmp/maigo_electron_bitmap_test
```

下一步可独立研究端点共享再减少约5.67GiB；当前没有实现，也未把预计29GiB写成实际结果。

正式入口`./scripts/run.sh carbon_ct_minibeam`以2×100k运行成功：首批79.526s、第二批29.337s，缓存命中1次。与冻结旧版前两批逐体素/审计对比通过，最大相对剂量差5.8259e-7，质量accepted且零队列溢出。证据`official_default_check.json`和`integration.json`；主目录未commit或push。

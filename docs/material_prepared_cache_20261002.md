# 材料电子包离线运行缓存

本文记录 C12 全图 prepared v1：host 检查完成，但原始 compact 基线被系统 OOM 杀掉，GPU 验证和完整生成已停止，部分派生表保留；它仍未通过全包 GPU 验收，不能把离线缓存等同于解决约94GiB内存需求。

后续独立的 **proton EM-only packed v2** 已通过全包 host、GPU 配对剂量及吞吐检查：先移除仅用于建索引的 primary 行，再保存预校验结果，减少驻留量和每进程准备成本。它与本文 v1 的格式和适用范围不同；使用 `--packed` 生成；proton CT minibeam 现通过 YAML 默认选择，无需环境变量，详见 [proton 试验与使用说明](proton_electron_model_trials_20261002.md)。不要把 proton 的通过外推为 C12 v1 的通过。

## 问题和边界

C12 肺 CT minibeam 的原始 combined 1,000-primary 运行整批墙钟 954.58887 s，GPU primary/electron/secondary 内核合计 2.29025 s。运行库累计分配 100,453,630,192 bytes（约 93.6 GiB）。原始路径每个进程都重新解码状态、排序轨迹、构建父子关系、能量 crossing 索引及出生 CDF。

本次只改变准备与加载方式。保存所有状态行、原行号、顺序、double 数值、64-bit 家谱索引、既有无损 32-bit crossing 行号和出生 CDF。GPU 剂量始终 FP32；不改变电子横向扩散、峰谷贡献或输运近似。不删除提取数据，不用 proton 响应替代 C12。

准备后的数据仍很大；本轮不宣称降低常驻内存。原始响应/轨迹是离线生成和审计依赖，缓存命中时不再重新读取它们。原始 index 仍需 pin 校验，派生物绑定其中的响应和状态 SHA，保持完整来源链。

## 使用

先离线生成（每张表原子发布，中断后可重用已完成的表）：

```bash
python3 scripts/prepare_material_electron.py \
  --index data/ct_electron/carbon/index.json \
  --cache out/material_prepared_cache/carbon
```

然后在兼容的 decoded CT packet 配置中显式使用：

```bash
CARBON_MATERIAL_PREPARED_DIR="$PWD/out/material_prepared_cache/carbon" \
  bash scripts/run.sh carbon_ct_minibeam \
  --histories-per-batch 100000 --batch-count 2 --batch-output out/carbon_prepared_example
```

目录是派生缓存根，支持多个 index 的表，表身份由 pinned 内容决定。首次调用不会隐式现场生成；缓存缺失、校验失败、版本不匹配、非 decoded/non-packet 配置或 short-range cache 开启均明确报错。

`scripts/material_prepared_sources.txt` 列出生成算法和格式相关源文件，CMake 与离线工具共同计算算法 SHA，代码修改会使旧缓存失效。格式只支持兼容的本机 ABI（endianness/结构尺寸进入 identity；本轮验证 GCC 写、项目 Clang 读），不是通用跨平台交换格式。修改数据/算法后应重新运行离线工具，不手改清单或跳过 SHA。

加载器按原来的分层显存策略分配，同一块数据的尺寸、顺序和 hot 标志保持一致。主机块直接读入最终 USM，不保留第二份巨大 host vector；显存小块经过有界临时缓冲区。读入时计算每块 SHA，清单和块均验证后才发布完整表视图。

## 检查与证据

- C12 小表 227,563 行及 proton 小表 14,804 行，所有 decoded 字段、家谱、crossing 和 birth 数组与原始路径重建结果一致；GCC 和项目 Clang 交叉读取 C12 均通过。
- 数据位损坏、截断、清单损坏、旧身份和额外清单字段均被拒绝。证据 `out/material_prepared_negative/results.json`。
- FP32 完整编译成功：`out/build_material_prepared.log`。
- `benchmark/carbonCT/minibeam_validation/validate_prepared.py` 顺序执行原始 combined/compact 检查、C12 缓存 smoke、proton 缓存及 carbon water 回归，检查质量、零溢出、离散 ledger、EM 审计及剂量。通过后继续原任务 GPU 两批 100k 与 TOPAS 初始比较，不自动标记剂量验收。
- 原始 compact 加载曾与离线编译及 GPU 编译重叠，此次时间不用于速度比较，剂量保留。已将后续重负载准备和 GPU 检查串行化，并登记 `failed.md`。
- 缓存验证状态：`out/material_prepared_validation/status.json`；自动流程日志：`out/material_prepared_validation_driver.log`。

测量应分别报告离线生成成本、新进程启动/加载、首批和复用批次墙钟、内核时间及内存。未清空系统文件缓存，不能将新进程测试称为严格的冷磁盘 I/O 测试。

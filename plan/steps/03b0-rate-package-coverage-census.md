# 03B-0：18 isotope rate/package coverage census

## 目标

在修改 runtime rate、event sampler 或 package 之前，确定 18 个显式 runtime isotope 在 H/O 两种靶上的：

- rate group 是否存在，以及是否有正 macroscopic rate；
- CINPKG03 global energy-node 是否存在；
- `±0.51 MeV/u` replay window 的 support interval；
- rate 正样本是否落在 package support 外；
- package support 是否完全没有正 rate overlap。

本步骤是只读诊断，不修改 sampler、yield、rate、stopping、MCS、cascade generation 或归一化。

## 实施

新增工具：`startup/package_tools/census_cinel02_rate_package_coverage.py`

工具只扫描 rate CSV 和 CINPKG03 的 persisted global energy-node index，不加载 interaction/product stream，也不进行 Monte Carlo 抽样。报告固定覆盖 18 个 isotope × H1/O16 共 36 个 group，输出 JSON、CSV 和 Markdown。

新增轻量回归：`tests/test_cinel02_rate_package_coverage.py`，覆盖 replay window union、support membership 和 zero-rate sample 过滤。环境没有 pytest，因此使用等价 Python 断言执行并通过；脚本和测试文件均通过 `py_compile`。

## canonical 输入

- Rate：`/mnt/sda/wuwei/cinel02-cascade-h1o16/e400MeVu_5000000h_bin1_cascade_5001_h1o16_cascade/compiled/cascade_e400_rates.csv`
- Package：`/mnt/sda/wuwei/cinel02-lightion-production-5m-v2/research107/research_hybrid_e200light107.cinpkg`
- Replay tolerance：`0.51 MeV/u`
- Rate SHA256：`aa811ff18684a6a8f81f79fa38f1130b4103555ea2932abb77f7163e1c160ed7`
- Package SHA256：`8a54b8544aea484fa3ff649fc372c22d4b48deff4c2fe37dbd31b2f7f6adb25d`

执行命令：

```bash
python3 startup/package_tools/census_cinel02_rate_package_coverage.py \
  --rate /mnt/sda/wuwei/cinel02-cascade-h1o16/e400MeVu_5000000h_bin1_cascade_5001_h1o16_cascade/compiled/cascade_e400_rates.csv \
  --package /mnt/sda/wuwei/cinel02-lightion-production-5m-v2/research107/research_hybrid_e200light107.cinpkg \
  --output plan/artifacts/cinel02-rate-package-census-e200-g1/census.json \
  --csv-output plan/artifacts/cinel02-rate-package-census-e200-g1/census.csv \
  --markdown-output plan/artifacts/cinel02-rate-package-census-e200-g1/RESULTS.md
```

## 结果

输出目录：`plan/artifacts/cinel02-rate-package-census-e200-g1/`

- 36/36 rate groups 存在；34/36 有正 rate sample。
- 34/36 package groups 有 global event nodes。
- `6Be+H1`：rate group 存在但 445 个 sample 全为 0；package node 数为 0。
- `6Be+O16`：rate group 存在但 445 个 sample 全为 0；package node 数为 0。
- 因此当前 package/rate 对 Be-6 同时缺失 secondary reaction coverage；这与 GPU 100k 中 `6Be` replay candidate 为 0 一致。
- `7Be/9Be/10Be`、`6Li/7Li` 均有正 rate 和 package nodes，不支持“所有 Be/Li isotope 都没有 secondary coverage”的解释。
- 全局有 3249 个正 rate sample 落在 package support 外，主要来自 rate 表高能/低能范围宽于当前 light-ion package；这不是本步骤直接修改的对象，需要由后续 occupancy-aware support audit 判断是否会被 GPU 实际访问。
- `9Be+O16` 有 1 个 package support interval 没有正 rate overlap；这是局部 rate/package 边界不一致，留给 03B-2 处理。

## 结论与下一步

[x] 03B-0 已完成：Be-6 coverage 缺口已被确定性确认。

下一步为 **03B-1 Be-6 rate/package coverage root-cause gate**：

1. 从 package raw/contract/summary 追溯是否存在 `6Be` projectile exposure campaign；
2. 区分“源数据从未生成 6Be 二次 projectile”与“compiler 过滤/unsupported mapping 丢失”；
3. 不允许把 Be-7 rate 或任何其他 isotope alias 给 Be-6；
4. 在获得独立 `6Be+H1`、`6Be+O16` exposure/rate 和 event package 之前，不修改 runtime 物理参数；
5. 若 6Be 仅作为 primary CINEL02 product 出现而没有可复用的 secondary event 数据，应明确将其标为 unsupported secondary cascade，而不是静默使用零 rate。

阶段 B 尚未完成：support-aware rate segmentation、runtime support mask 和 miss 下降门槛仍待实现。

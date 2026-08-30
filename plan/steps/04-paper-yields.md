# 步骤 04：恢复论文 Table-1 固定产额

## 目标

将论文模式与手调的 200/300/400 MeV/u isotope-yield knots 分离。`fred_paper` 在所有能量使用 95 MeV/u Table 1 probabilities，仅缩放能量和角度分布。

## 实施任务

1. 增加清晰模式边界：`fred_paper` 与 `topas_inclxx_matched` 不共享手调 yield 行为。
2. paper 模式按 H/O target 选择固定 Table-1 概率，不访问 `kFredProb*ByE`。
3. 保留 energy-dependent knots 只作为单独的 legacy/experimental ablation（若代码仍需兼容），默认关闭并明确标记非论文模型。
4. 配置和输出元数据必须打印当前 yield model。

## A/B 矩阵

四种能量分别运行：固定 Table 1/energy knots × kerma off/on。kerma-on 仅用于诊断历史偏差。比较 species composition、plateau、distal tail 和总能量。

## 验收

- 100 MeV/u 两种 yield 模式结果应近似一致。
- paper 模式不含 200/300/400 MeV/u 人工 knot。
- yield-knot bias 与 kerma bias 可由矩阵分别量化。
- 不以 TOPAS 单 bin 一致来反向调整 paper 模式。

## 建议提交

```text
revert(fred): disable unsupported energy-dependent yield knots in paper mode
```


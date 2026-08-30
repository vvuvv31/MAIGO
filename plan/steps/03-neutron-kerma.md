# 步骤 03：关闭固定 neutron vertex kerma

## 目标

删除“完整 neutron KE 计入 untracked，同时再在顶点沉积 8%”的双重计数。论文复现模式不保留无依据的固定 vertex kerma。

## 实施任务

1. `fred_paper` 默认且强制 `inelastic_neutron_kerma_fraction: 0.0`。
2. neutron 未输运时，其 KE 只计入 escaped/untracked neutral energy，dose 为零。
3. 若保留实验性 kerma 开关，必须满足 `local_deposit + escaped_neutron = original_neutron_energy`，并显式标记为非论文模式。
4. 后续 neutron dose 只能来自显式输运、经过验证的空间 kernel，或与 TOPAS physics list 一致的响应模型。

## A/B/C 定位运行

| Run | bookkeeping | dose |
| --- | --- | --- |
| A | neutron KE 全部 escaped | 0 |
| B | 显式输运或已验证 kernel | 正确空间分布 |
| C | 原固定 8% vertex kerma | 仅作历史定位，不得作为最终模式 |

## 验收

- A 模式 neutron-associated dose 严格为零，能量账本闭合。
- C 仅能由明确的 diagnostic/legacy 配置启用。
- 记录 A/C 对 200–400 MeV/u plateau 的影响，作为偏差归因证据。

## 建议提交

```text
fix(fred): remove fixed neutron vertex kerma
```


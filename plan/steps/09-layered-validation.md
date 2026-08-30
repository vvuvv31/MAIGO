# 步骤 09：分层水模体回归与最终验收

## 启用顺序

| 阶段 | 开启内容 | 核心检查 |
| --- | --- | --- |
| A | EM + elastic | 电磁基线 |
| B | + attenuation only | primary survival、reaction depth |
| C | + projectile fragments | distal tail、forward fragments |
| D | + target fragments | 低能宽角 halo |
| E | + explicit remnant | A/Z、能量闭合 |
| F | + neutron model | neutron dose 空间分布 |
| G | + secondary MCS | core/halo σ |
| H–K | 100/200/300/400 MeV/u full | 能量趋势与最终指标 |

每一级保存 primary/projectile/target/species/remnant/neutron dose、escaped energy、反应深度、A/Z/energy residual。任一级失败都回退到该物理层定位，不在更高层调参。

## 运行规模

```text
100k histories  smoke test
1M histories    每个物理提交
10M histories   halo 与 distal tail
100M histories  最终论文级验证
```

GPU jobs 始终在本地 WSL RTX 2080Ti 运行。TOPAS 经 localhost `sbatch` 执行。所有 dose 比较来自 3D scorer；IDD 由横向求和得到。

## 100–300 MeV/u 工程验收

```text
whole-depth integrated dose error <= 2.5%
Bragg peak position difference    <= 0.5 mm
plateau IDD difference            <= 3%
distal-tail integral difference   <= 5%
core sigma difference             <= 3%
halo sigma difference             <= 10%
```

## 400 MeV/u 外推验收

```text
integrated dose difference <= 5%
无随深度单调放大的平台偏差
无人工 distal spike
无能量相关手工 normalization
```

## 最终产物

- 可复现配置、commit、随机种子、history 和运行环境清单。
- 每层 ablation 表格及 100–400 MeV/u IDD/lateral/halo 图。
- 守恒与异常计数报告。
- 论文模型与 TOPAS-matched 扩展模型的边界说明。
- 在主 README 将所有步骤标为“已完成”并链接证据。

## 建议提交

```text
test(fred): add staged 100-400 MeV-u water regressions
```


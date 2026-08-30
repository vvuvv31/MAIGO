# 步骤 08：标定次级碎片 MCS

## 前置门槛

只有步骤 01–07 全部通过，species yield、birth angle、energy spectrum 和 range 已可信时才执行。本步骤不能补偿 event generator 错误。

## 验证顺序

1. Birth-angle-only：关闭 secondary MCS，检查生成点后短距离 angular profile，验证 Eq. 12/21。
2. MCS-only ion beams：分别模拟 p、d、t、He、Li/Be、B/C；关闭 nuclear interactions，只比较 lateral broadening。
3. 建立 `fmcs[species_group][energy_bin][range_fraction_bin]` LUT，至少分 H、He、Li/Be、B/C 四组。

论文示例 1.29–1.43 只作初值范围参考，不能全 species 固定为 1.35。每个 LUT 值须注明数据源、拟合目标和适用范围。

## TOPAS 基准

TOPAS 必须在 `wuwei@127.0.0.1` 用 `sbatch` 运行，数据位于 `/mnt/sda/wuwei`，总资源不超过 192 CPU/128G。使用 3D scorer 获取横向分布。

## 验收

- 关闭 nuclear interactions 时，各 species 的 lateral width 达到预定误差。
- 加入 MCS 后 halo σ 明显改善，IDD 积分不出现显著变化。
- 若 IDD 明显改变，返回检查 secondary stopping/range/scoring，不继续调 `fmcs`。

## 建议提交

```text
fix(fred): add species-energy-depth secondary fmcs tables
```


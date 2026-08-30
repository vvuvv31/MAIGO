# 步骤 01：修复 GPU 非弹性截面与 target-H 查表网格

## 问题

kernel 用 stopping-power 的 `Emin=0.01 MeV/u`、`ΔE=0.1 MeV/u` 索引原始 1 MeV/u 截面表，导致约 40 MeV/u 以上全部 clamp 到 400 MeV/u；`target_h_fraction` 同样错误。

## 实施方案

优先在 host 侧把 macroscopic XS 与 target-H fraction 重采样到 stopping-power transport grid：

```cpp
for (std::size_t i = 0; i < transport_energy.size(); ++i) {
    const double e = transport_energy[i];
    xs_on_transport_grid[i] = cross_section.interpolate(e);
    h_on_transport_grid[i] =
        cross_section.interpolate_target_h_fraction(e);
}
```

kernel 继续使用 transport-grid metadata，但只能访问重采样数组。若采用独立 XS metadata，必须同时传递 `xs_min_energy`、`xs_inverse_step`、`xs_table_size`，并证明边界行为一致。

## 测试

1. CPU/GPU 在 1、5、10、20、40、95、100、200、300、400 MeV/u 比较 XS 和 H fraction。
2. 测试低端、精确网格点、网格中点、400 MeV/u 及超界 clamp。
3. 只开启 attenuation：发生非弹性时杀死 primary，不生成 secondary。
4. 输出 primary survival、reaction-depth histogram、H/O reaction ratio。

## 硬门槛

```text
relative XS difference       < 1e-5
target-H fraction difference < 1e-5
invalid/out-of-bounds access = 0
```

## 预期而非强制拟合目标

低能末端 reaction count 增加，primary survival 降低，高能平台正偏差缩小，低能 H reaction fraction 上升。若趋势不符，先检查单位和表边界，不进入步骤 02。

## 建议提交

```text
fix(sycl): use correct energy grid for inelastic XS and target-H lookup
test(sycl): compare CPU/GPU XS interpolation at 1-400 MeV/u
```


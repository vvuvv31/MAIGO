# 步骤 07：显式处理 remnant、Q 与 excitation

## 目标

删除按剩余 A 经验沉积、上限约 8 MeV 的 target remnant dose；用显式 A/Z、kinetic energy、excitation 和 Q-value 进入同一守恒账本。

## 数据结构与输运

```cpp
struct NuclearRemnant {
    int A;
    int Z;
    float kinetic_energy_MeV;
    float excitation_energy_MeV;
};
```

- 带电 remnant 的 CSDA range 大于 cutoff：加入 secondary queue。
- range 小于物理 cutoff：局域沉积其 kinetic energy，并记录终止原因。
- excitation 独立记录，不得默认等同局域 dose。
- queue 容量、overflow 策略与诊断必须明确。

## 两种模式

1. Paper-minimal：`Q=0`，available remainder 记为 excitation/untracked；只有明确释放模型时才沉积。
2. Extended physics：使用核质量表计算 Q；质量表版本、同位素覆盖和异常处理必须测试。

## 验收

- 删除经验 `Arem × constant` 和 8 MeV cap 路径。
- remnant 的 A/Z 与能量均进入 closure。
- remnant transport/local-stop 两条路径都有单元测试。
- remnant 局域 dose 与 halo 的变化可分别解释，queue overflow 为零。

## 建议提交

```text
fix(fred): introduce explicit target remnant and excitation ledger
```


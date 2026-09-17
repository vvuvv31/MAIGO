# GPU Monte Carlo Scorer 建议

如果目标只是**复现 TOPAS 的 3D 物理剂量和 LET\(_d\)**，那么生产输出层面，`DoseToMedium` + LET scorer 基本已经够了。

当前模型中：

- 剂量按体素沉积能除以体素质量得到；
- LET 按
  \[
  \mathrm{LET}_d
  =
  \frac{\sum_i L_i\Delta E_i}{\sum_i\Delta E_i}
  \]
  计算。

但如果是为了做 **GPU Monte Carlo 的物理验证和调试**，建议额外保留若干辅助 scorer / diagnostic。否则一旦与 TOPAS 有偏差，很难判断问题来自 stopping power、核反应、碎片输运还是计分器本身。

---

## 1. Primary / secondary dose 分开计分

建议至少区分：

\[
D
=
D_{\mathrm{primary}}
+
D_{\mathrm{secondary}}
\]

例如分别记录：

- primary \(^{12}\mathrm C\) dose；
- fragment dose；
- reaction-local deposit。

这样如果总剂量不对，可以快速判断问题主要来自主碳离子还是碎片部分。

---

## 2. 粒子种类 fluence / dose scorer

至少建议统计：

\[
{}^{12}\mathrm C,\quad
B,\quad
Be,\quad
Li,\quad
He,\quad
p
\]

因为核反应模型采用 package-driven 方法，真正需要验证的不只是总剂量，还包括碎片组成是否合理。

否则可能出现：

> 总剂量看起来与 TOPAS 很接近，但碎片谱实际上不正确。

对于 CRPKG / CCAS 的验证，species fluence 会很有价值。

---

## 3. Primary survival / reaction count

建议沿深度统计 primary carbon survival，例如：

\[
N_C(z)
\]

以及非弹性反应次数：

\[
N_{\mathrm{inelastic}}(z)
\]

还可以按照 projectile species、material / Schneider section、energy bin 分别统计反应次数。

这可以直接验证：

\[
P_{\mathrm{int}}
=
1-\exp[-\Sigma(E,m)\Delta s]
\]

以及 Geant4 cross-section table 的加载、插值和材料选择是否正确。

---

## 4. Energy ledger

Energy balance 建议作为**必须保留的 diagnostic**。

模型应满足：

\[
E_{\mathrm{in}}
=
E_{\mathrm{dep}}
+
E_{\mathrm{esc}}
+
E_{\mathrm{beamline}}
+
E_{\mathrm{untracked}}
\]

建议进一步分别统计：

- continuous electromagnetic deposit；
- reaction-local deposit；
- secondary-particle deposit；
- residual nuclear heat；
- escaped energy；
- beamline loss；
- queue-lost / untracked energy。

这样一旦 reaction package、cascade package 或 secondary queue 出现问题，可以快速定位。

---

## 5. LET numerator / denominator 分开保存

最终 LET 为：

\[
\mathrm{LET}_d
=
\frac{\sum_i L_i\Delta E_i}
{\sum_i\Delta E_i}
\]

建议内部不要只保存最终 LET，而是分别保存：

\[
N_{\mathrm{LET}}
=
\sum_i L_i\Delta E_i
\]

以及：

\[
D_{\mathrm{LET}}
=
\sum_i\Delta E_i
\]

这样可以正确进行多 shard 合并。LET 不能简单做算术平均，而应按剂量权重组合。

---

## 6. Primary LET 和 all-hadron LET 分开

建议至少保留：

\[
\mathrm{LET}_{d,\mathrm{primary}\ {}^{12}\mathrm C}
\]

以及：

\[
\mathrm{LET}_{d,\mathrm{all\ charged\ hadrons}}
\]

不要只输出一个总 LET。

---

# 推荐的 scorer / diagnostic 分类

| 项目 | 建议 |
|---|---|
| DoseToMedium | **必须** |
| LET\(_d\), all charged hadrons | **必须** |
| LET\(_d\), primary carbon | **强烈建议** |
| Primary / secondary dose split | **验证时强烈建议开启** |
| Species fluence | **验证核反应时非常有用** |
| Reaction count / primary survival | **验证 cross section 时非常有用** |
| Energy balance components | **必须作为 diagnostic** |
| Residual nuclear heat | **必须监控** |
| Queue overflow count | **必须监控，验证时必须为 0** |
| Gamma analysis | **后处理，不属于 MC scorer** |

---

# Production 与 Validation 的建议配置

## Production 模式

生产计算建议尽量简化输出：

\[
\boxed{
\mathrm{Production}
=
D_{\mathrm{medium}}
+
\mathrm{LET}_d
}
\]

必要时再加 primary-carbon LET。

这样可以减少 GPU atomic 操作、显存占用、输出文件大小和 I/O 开销。

## Validation / Debug 模式

验证模式建议额外开启：

\[
\boxed{
D_{\mathrm{primary}},
\quad
D_{\mathrm{secondary}},
\quad
\Phi_{\mathrm{species}},
\quad
N_{\mathrm{reaction}},
\quad
E_{\mathrm{ledger}}
}
\]

其中尤其推荐：

### Primary carbon fluence / survival curve

\[
\Phi_{{}^{12}C}(z)
\]

可以直接验证 primary attenuation 和 nuclear cross section。

### Fragment fluence / fragment dose

例如：

\[
\Phi_p(z),\quad
\Phi_{\alpha}(z),\quad
\Phi_B(z)
\]

以及：

\[
D_{\mathrm{fragment}}(z)
\]

可以用于验证 CRPKG / CCAS 的碎片产生和后续输运。

---

# 总结

如果目标是最终临床式输出：

\[
\boxed{
\mathrm{DoseToMedium}
+
\mathrm{LET}_d
}
\]

基本足够。

但如果目标是证明 GPU Monte Carlo 的物理模型与 TOPAS 一致，则建议至少增加：

\[
\boxed{
\mathrm{Primary\ survival}
+
\mathrm{Fragment\ fluence}
+
\mathrm{Primary/Secondary\ dose}
+
\mathrm{Energy\ ledger}
}
\]

其中 **primary carbon fluence / survival curve** 和 **fragment dose / fluence scorer** 对验证 CRPKG / CCAS 核反应模型尤其重要。


默认编译选项为production,只有手动指定validation mode 的时候才启用上面的scorer
# 当前下一步

## 2026-07-15：高能坪区 / 积分差已改善

诊断：高能前端坪区偏低主因是 **中性粒子动能未沉积**（带电路径在 200 MeV 已闭合）。  
对策：`neutral_local_kerma_fraction=0.298`（用 200 MeV TOPAS 中性剂量 / GPU 中性出生动能标定），配合 400 MeV/u 反应末态包。

| E | 积分差 | 中段坪区 mean rel | 尾积分差 | NRMSE |
|---|--------|-------------------|----------|-------|
| 100 | **−0.07%** | ~0 | −1.1% | 0.06% |
| 200 | **+0.34%** | +1.2% | −4.4% | 1.04% |
| 300 | **−0.27%**（was −2.55%） | **+0.25%**（was −2.6%） | +0.95% | 1.62% |
| 400 | **−1.01%**（was −4.92%） | **−0.50%**（was −5.1%） | −3.6% | 2.12% |

配置：`config/beam_*MeVu_multi_energy.yaml`  
汇总：`validation/results/windows_b580_vs_topas_multi_energy_summary.metrics.json`  
诊断图：`validation/results/plateau_deficit_diagnosis.png`

### 中间能量 150/250/350 MeV/u（已完成）

| E | 积分差 | 中段坪区 | ΔR80 | NRMSE | 峰高差 |
|---|--------|----------|------|-------|--------|
| 150 | +0.07% | +0.23% | +0.02 mm | 0.46% | +2.1% |
| 250 | +0.28% | +0.46% | +0.30 mm | 1.39% | −0.46% |
| 350 | −0.96% | −0.07% | +0.51 mm | 1.90% | −2.7% |

图：`validation/results/windows_b580_vs_topas_mid_energy_overview.png`  
全套 100–400：`windows_b580_vs_topas_energy_suite_curves.png`  
汇总：`windows_b580_vs_topas_mid_energy_summary.metrics.json`

### Emittance BiGaussian + σ(z)（200 MeV/u，已完成）

TOPAS / GPU 均用同一 BiGaussian 参数：  
`σx=σy=0.2 mm`，`σx'=σy'=0.032`，`ρx=−0.9411`，`ρy=+0.9411`。

剂量加权 σ_rms(z) 对比（100k）：

| 量 | 值 |
|----|----|
| 入射端 σ_rms TOPAS / GPU | 0.434 / 0.366 mm |
| 全深 mean rel σ | **−1.8%** |
| 峰区 (70–100 mm) mean rel | **−1.5%** |
| mean \|Δσ\| | 0.20 mm |

图：`validation/results/emittance_200_sigma/sigma_vs_depth.png`  
GPU 配置：`config/beam_200MeVu_emittance_sigma.yaml`  
TOPAS：`validation/topas/carbon_200MeVu_water_emittance*.txt`

### 再下一步（按优先级）

1. 用正式 neutron package + mode D/full 替换 interim local kerma  
2. 峰高仍约 −3–4%（300/400）：查带电碎片/限制能损语义  
3. 可选：RNG 键修复；论文图  
4. （以后）异质体/CT  

无 MCS scale、无全局 dose scale。local kerma 不是对带电 IDD 做全局缩放。

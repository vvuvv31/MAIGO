# 当前下一步

## 2026-07-15：多能量 TOPAS 已补齐

远程 100/300/400 MeV/u 总 IDD（100k）已跑完并与 GPU 对比。

| E | 积分差 | ΔR80 | NRMSE | 备注 |
|---|--------|------|-------|------|
| 100 | −0.34% | +0.001 mm | 0.056% | 优秀 |
| 200 | −0.72% | +0.10 mm | 1.03% | 良好 |
| 300 | −4.5% | +0.40 mm | 1.8% | 末态包顶箱限制 |
| 400 | −6.9% | +0.64 mm | 3.5% | 末态包顶箱限制，尾差大 |

报告：`validation/results/status_report_2026-07-15.md`  
汇总：`validation/results/windows_b580_vs_topas_multi_energy_summary.metrics.json`

### 再下一步（按优先级）

1. **远程补 300/400 MeV/u 反应末态 package**（或 400 MeV 束流 reaction n-tuple）以解除顶箱限制  
2. 可选：RNG 键修复 → GPU bit-reproducible  
3. 可选：论文图（多能量、消融、性能）  
4. （以后）neutron/gamma；异质体/CT  

无 MCS scale、无全局 dose scale。

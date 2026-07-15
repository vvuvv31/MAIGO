# 当前下一步

## 2026-07-15：均匀水模体 charged 主线收敛完成

报告：`validation/results/status_report_2026-07-15.md`

| 项 | 状态 |
|----|------|
| neutron/gamma | **暂缓** |
| 步长 / 横向网格 10k | **完成** |
| 1e6 统计 + 复现 | **完成** |
| **100–400 MeV/u 多能量 100k** | **完成** |

### 多能量摘要（固定模型，无逐能量 scale）

| E | R80 mm | vs 期望 |
|---|--------|---------|
| 100 | 25.9 | 上升 |
| 200 | 87.0 | vs TOPAS ΔR80 +0.10 mm，积分 −0.72% |
| 300 | 172.5 | 上升 |
| 400 | 275.6 | 上升 |

已知限制：末态 reaction package 顶箱 200 MeV/u。

### 再下一步（论文/扩展）

1. 远程补做 100/300/400 MeV/u TOPAS 总 IDD（及可选 reaction package）  
2. 可选：RNG 键修复 → GPU bit-reproducible  
3. 可选：论文图（消融、多能量、性能）  
4. （以后）neutron/gamma；异质体/CT  

无 MCS scale、无全局 dose scale。TOPAS 仅 `v@192.168.31.5`。

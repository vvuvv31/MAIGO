# 当前下一步

## 2026-07-15：收敛阶段收束

报告：`validation/results/status_report_2026-07-15.md`

| 项 | 状态 |
|----|------|
| neutron/gamma | **暂缓** |
| maximum_step_mm 10k | **完成** → 默认 0.5 mm |
| 横向 voxel 10k | **完成** → 对比 5 mm / 绝对束宽 2.5 mm |
| **1e4–1e6 统计收敛 + 同 seed 复跑** | **完成** |

### 1e6 摘要

- 吞吐 **~4.15×10⁴ histories/s**（B580，IDD-only charged）
- NRMSE·√N ≈ 常数 → MC 噪声缩放正常
- 同 seed 两次：非 bit 相同，NRMSE ~3e-5（原子队列索引影响 cascade RNG）
- overflow = 0

### 再下一步

1. **100–400 MeV/u 多能量**（至少 100 / 200 / 300 / 400，与 TOPAS 对比）  
2. 可选：论文前 100k 步长/横向复核  
3. 可选：RNG 键改为 history 局部序号以实现 GPU bit-reproducible  
4. （以后）neutron/gamma  

无 MCS scale、无全局 dose scale。TOPAS 仅远程主机。

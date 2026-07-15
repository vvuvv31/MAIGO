# 当前下一步

## 2026-07-15：neutron 暂缓；步长与横向网格收敛阶段性完成

报告：`validation/results/status_report_2026-07-15.md`

| 项 | 状态 |
|----|------|
| neutron/gamma 闭合 | **暂缓**（全深度 ~0.7%） |
| `maximum_step_mm` 10k 扫描 | **完成** → 默认 0.5 mm |
| 横向 voxel 10/5/2.5 mm 10k | **完成** → 对比用 5 mm，绝对束宽用 2.5 mm |

### 再下一步

1. **1e6 histories 统计收敛 + 可复现性**（同 seed 两次、吞吐）  
2. 可选：步长/横向 100k 复核（论文前）  
3. **100–400 MeV/u 多能量**  
4. （以后）neutron/gamma  

不要调 MCS scale。无全局 scale。TOPAS 仅远程主机。

### 横向收敛要点

- IDD 对 10/5/2.5 mm 不敏感  
- 峰区 σx：10 mm → ~5.2 mm；5 mm → ~2.9 mm；2.5 mm → ~2.0 mm  
- 与 TOPAS 3D 比较保持 **5 mm** 网格  

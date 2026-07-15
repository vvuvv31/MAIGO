# 当前下一步

## 2026-07-15 更新：neutron 暂缓，进入 charged 收敛

**决策**：忽略 neutron/gamma 剂量闭合问题，仅写入状态报告备案，不阻塞主线。

报告：`validation/results/status_report_2026-07-15.md`

- charged-origin + MCS 100k 已完成（全深度 −0.199%，尾 +3.07%）
- neutron 方案 D 脚手架保留；1k smoke dose-like ~33% TOPAS 中性剂量，**当前不追**
- 已完成 **maximum_step_mm** 10k B580 收敛扫描（1.0–0.1 mm）

### 当前任务顺序

1. ~~`maximum_step_mm` 收敛（10k smoke）~~ **已完成阶段性**  
2. **横向 voxel 尺寸收敛**（例如 10 / 5 / 2.5 mm，固定 0.5 mm 步长）  
3. 100k 步长复核（可选，论文前）  
4. 1e6 histories 统计收敛 + 可复现性  
5. 100–400 MeV/u 多能量  
6. （以后）neutron/gamma 正式闭合  

不要调 MCS scale。新 TOPAS 仅 `v@192.168.31.5`。无全局 scale。

### 步长 10k 摘要（相对 0.1 mm）

| step | NRMSE | ΔR80 mm | 吞吐 hist/s |
|------|-------|---------|-------------|
| 1.0 | 4.8e-3 | +0.059 | 41k |
| 0.5 | 4.4e-3 | +0.053 | 28k |
| 0.25 | 2.5e-3 | +0.024 | 21k |
| 0.1 | ref | 0 | 11k |

生产默认保持 `maximum_step_mm: 0.5`。

---

## 历史：中性与 charged-origin 记录

见 `validation/results/status_report_2026-07-15.md` 与 git 历史  
`a7f09e7` / `d9aa5ee`（charged-origin）、`9f6d116`–`4ec8a41`（neutral 脚手架与方案 D）。

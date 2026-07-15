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

### 远程 TOPAS 多能量 IDD（进行中）

参数与 runner 已入库；**本机无法 SSH 到 `v@192.168.31.5`（公钥未授权）**，需在可登录主机执行：

见 `validation/topas/REMOTE_MULTI_ENERGY_IDD.md`

1. `scp` 参数文件与 `run_multi_energy_idd_remote.sh`  
2. smoke：`bash validation/topas/run_multi_energy_idd_remote.sh all smoke`  
3. development 100k ×3（56 线程，`nohup`）  
4. 拉回 CSV → `validation\scripts\postprocess_multi_energy_topas.cmd`  
5. 与 `windows_b580_multi_energy_*MeVu_idd_100k.csv` 对比  

### 再下一步（结果落地后）

1. 完成上述 TOPAS 标准化并更新多能量 metrics  
2. 可选：RNG 键修复 → GPU bit-reproducible  
3. 可选：论文图  
4. （以后）neutron/gamma；异质体/CT  

无 MCS scale、无全局 dose scale。TOPAS 仅 `v@192.168.31.5`。

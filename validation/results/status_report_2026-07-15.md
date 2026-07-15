# 项目状态报告（2026-07-15）

## 1. 已完成（charged 路径，当前论文主线）

| 里程碑 | 状态 | 关键指标 |
|--------|------|----------|
| C-12 能量相关核反应截面 | 完成 | 直接 Geant4 表，非 IDD 拟合 |
| 带电碎片生成/输运/级联 | 完成 | 两代 cascade，overflow = 0 |
| 3D voxel + MCS | 完成 | Highland/Lynch–Dahl，无散射 scale |
| charged-origin 8 类 scorer | 完成 | 100k 最大逐 voxel 闭合 `6.70e-11` MeV/primary |
| GPU vs TOPAS charged 祖先和 | 完成 | 全深度 **−0.199%**，90 mm 后 **+3.07%** |

比较语义固定为：**绝对 MeV/primary，无全局 scale**；GPU charged 总量对比 TOPAS  
`primary_c12 + secondary C/B/Be/Li/He + proton + other_charged`。

正式配置与结果入口：

- 配置：`config/beam_200MeVu_mcs_voxel_100k.yaml`
- 闭合：`validation/results/windows_b580_mcs_3d_100k_charged_origin_closure.metrics.json`
- 对比：`validation/results/windows_b580_mcs_3d_100k_vs_topas.metrics.json`

## 2. Neutron / gamma：已记录，当前忽略

### 2.1 为何暂缓

1. **对总量影响有限**：TOPAS 祖先归属中性来源约 **16.4 MeV/primary（~0.71% 全深度）**；  
   charged 路径已到 **0.2%** 量级，继续抠 neutron 不是当前最高 ROI。
2. **尾部更敏感但可后置**：90 mm 后 neutron-origin 约占尾部 **~5.6%**，应在 charged 收敛做完后再系统处理。
3. **实现已有脚手架但未闭合**：TOPAS n-tuple、binary package、方案 D（first interaction）已接入；  
   1k smoke 显示 dose-like 仅约 TOPAS 中性剂量的 **33%**，continuation residual 仍大。

### 2.2 方案 D 诊断摘要（不继续追）

- 模式：`neutral_transport_mode: first_interaction`（一次自由程 + 一次包）
- 明确 **禁止** 把 neutron 动能当场 kerma 沉在产生 voxel
- 1k SYCL CPU（无 cascade）：

| 量 | MeV/primary |
|----|-------------|
| charged-from-neutral | 5.33 |
| residual continuation/nested | 25.93 |
| free-path escape | 16.49 |
| local deposit | ~0 |
| TOPAS neutral-origin | 16.35 |

- 元数据：`validation/results/windows_neutral_mode_d_1k.metadata.json`
- 已知风险：重 cascade+GPU 配置曾出现 Level Zero `DEVICE_LOST`；lean CPU 路径可复现

### 2.3 以后再做时的顺序（备忘，非当前任务）

1. 100k development neutral package（远程 TOPAS）
2. cascade 源 + 方案 D / 有限代 continuation
3. 与祖先 neutron/gamma IDD、3D 比较（绝对量，无 scale）

**当前决策：neutron/gamma 剂量闭合从主线 backlog 移出，报告中备案，不阻塞 charged 收敛。**

## 3. 当前下一步（忽略 neutron 后）

按 guide 路线图，charged-origin 与 MCS 已完成后应执行：

1. **`maximum_step_mm` 收敛**（1.0 / 0.5 / 0.25 / 0.1 mm）  
   观察 R80、peak、FWHM、NRMSE、runtime；确认 Highland 逐步近似稳定区  
2. **横向 voxel 尺寸收敛**  
3. **1e6 histories 统计收敛 + 可复现性**  
4. **100–400 MeV/u 多能量**  

## 4. 步长收敛（本报告后已开跑）

B580，10k histories，charged cascade + MCS，**neutral 关闭**，步长 1.0 / 0.5 / 0.25 / 0.1 mm，  
参考步长 0.1 mm。脚本：`validation/scripts/analyze_step_convergence.py`。

| maximum_step_mm | NRMSE→0.1 mm | 积分差 | ΔR80 (mm) | ΔFWHM (mm) | 耗时 (s) | 吞吐 (hist/s) |
|-----------------|--------------|--------|-----------|------------|----------|---------------|
| 1.0 | 4.78e-3 | −0.16% | +0.059 | +0.043 | 0.24 | 41k |
| 0.5 | 4.41e-3 | −0.13% | +0.053 | +0.079 | 0.36 | 28k |
| 0.25 | 2.52e-3 | −0.21% | +0.024 | −0.008 | 0.49 | 21k |
| 0.1 (ref) | 0 | 0 | 0 | 0 | 0.92 | 11k |

**阶段性结论**：生产默认 **0.5 mm** 仍合理；0.25 vs 0.1 的 R80/FWHM 差在 10k 统计下已很小。  
正式写进论文前应用 **100k** 同 seed 族复核。overflow 均为 0。

产物：

- `validation/results/windows_b580_step_convergence_10k.metrics.json`
- `validation/results/windows_b580_step_convergence_10k.png`
- `validation/results/windows_b580_step_*_idd_10k.csv`

## 5. 横向 voxel 收敛（10k B580，FOV 300 mm）

固定 `maximum_step_mm=0.5`，neutral 关闭，横向体素 **10 / 5 / 2.5 mm**（bin 30 / 60 / 120）。

| voxel (mm) | IDD NRMSE→2.5 | 积分差 | σx@86.75 mm | Δσx vs 2.5 |
|------------|---------------|--------|-------------|------------|
| 10 | 2.4e-4 | −0.007% | 5.16 | +3.20 |
| **5** | **3.9e-4** | **−0.09%** | **2.88** | **+0.91** |
| 2.5 (ref) | 0 | 0 | 1.97 | 0 |

**结论**：

- **IDD 几乎不敏感**横向体素（NRMSE < 5e-4）
- **横向宽度受分箱限制**：10 mm 不可用；5 mm 与 TOPAS 祖先网格一致，适合对比；引用“绝对 MCS 束宽”时宜用 **2.5 mm**
- 生产 3D 比较保持 **5 mm**（与现有 TOPAS 100k 基准同网格）

产物：`validation/results/windows_b580_lateral_voxel_convergence_10k.metrics.json` / `.png`

## 6. 统计收敛与可复现性（1e4 / 1e5 / 1e6，B580）

IDD-only charged 路径（MCS+cascade，neutral 关，seed=20260715）：

| N | NRMSE→1e6 | 积分差 | ΔR80 (mm) | 耗时 (s) | 吞吐 (hist/s) |
|---|-----------|--------|-----------|----------|---------------|
| 1e4 | 8.76e-4 | −0.21% | +0.001 | 0.61 | 16k |
| 1e5 | 2.78e-4 | +0.025% | +0.0002 | 2.81 | 36k |
| **1e6** | 0 | 0 | 0 | **24.1** | **41.5k** |

- `NRMSE · √N ≈ 0.0878` 近似常数 → **统计噪声主导，收敛正常**
- 全部 secondary/cascade **overflow = 0**
- 同 seed 两次 1e6：**非 bit-identical**（GPU 原子队列下标进入 cascade RNG），但  
  NRMSE **2.9e-5**，最大 bin 差 **0.014 MeV/primary**，积分差 **−0.0013%**  
  → **实用可复现性极好**；严格位级复现需改 RNG 键（history + secondary 序号）

产物：`validation/results/windows_b580_stats_convergence_1e6.metrics.json` / `.png`

## 7. 多能量 100–400 MeV/u（100k B580）

固定参数：`straggling_scale=1.2`、`maximum_step_mm=0.5`、同一 SP/XS 表与 cascade 包，**无逐能量调参**，neutron 关。

| E (MeV/u) | R80 (mm) | peak z (mm) | FWHM (mm) | 积分 (MeV/p) | 核反应/primary | 吞吐 |
|-----------|----------|-------------|-----------|--------------|----------------|------|
| 100 | 25.86 | 25.75 | 1.21 | 1180 | 0.164 | 62k/s |
| 200 | 86.99 | 86.75 | 2.36 | 2280 | 0.384 | 36k/s |
| 300 | 172.48 | 171.75 | 4.86 | 3104 | 0.586 | 23k/s |
| 400 | 275.58 | 274.75 | 8.00 | 3522 | 0.742 | 21k/s |

- **R80 严格随能量上升**
- **GPU 200 vs TOPAS development**：积分 **−0.72%**，ΔR80 **+0.10 mm**，NRMSE **1.03%**，峰深一致
- overflow = 0
- **限制**：反应末态包仅 0–200 MeV/u 分箱；300/400 MeV/u 的高能反应夹到顶箱。SP/XS 仍为 1–400 全表。正式多能量末态需后续 TOPAS 反应包。

产物：`validation/results/windows_b580_multi_energy_100_400.metrics.json` / `.png`

## 8. 远程 TOPAS 100/300/400 MeV/u 总 IDD（已完成）

在 `v@192.168.31.5`（TOPAS 4.1.p1 / Geant4 11.1.p3，56 线程）完成 smoke + **100k development**，
本地标准化后与 GPU 100k 多能量曲线对比：

| E (MeV/u) | 积分差 | ΔR80 (mm) | NRMSE | 峰高差 | 尾积分差 | 2%/2 mm γ |
|-----------|--------|-----------|-------|--------|----------|-----------|
| **100** | **−0.34%** | **+0.001** | **0.056%** | **+0.06%** | −1.8% | **100%** |
| 200 | −0.72% | +0.10 | 1.03% | +0.78% | — | — |
| 300 | −4.47% | +0.40 | 1.76% | −4.3% | −20.8% | 99.1% |
| 400 | −6.88% | +0.64 | 3.53% | −6.1% | −38.4% | 65.4% |

- **100 MeV/u 极好**（在末态包能量覆盖内）  
- **200 可用**  
- **300/400 尾部与峰高明显变差**，与「反应末态包仅至 200 MeV/u 分箱」一致  

产物：`topas_{100,300,400}MeVu_development.csv`、`windows_b580_vs_topas_*MeVu.metrics.json`、  
`windows_b580_vs_topas_multi_energy_summary.metrics.json`

## 9. 约束（不变）

- 新 TOPAS 作业仅 `v@192.168.31.5`，≤56 线程；禁止 WSL TOPAS  
- Windows 原生 oneAPI + Arc B580 为 GPU 执行环境  
- 禁止用全局 scale 贴合 TOPAS  
- 不调 MCS scale 追逐差异  

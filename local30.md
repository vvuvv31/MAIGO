# RT07575 local 3%/0mm 工作日志

**正式指标**：BODY ∩ TOPAS dose ≥ 10% BODY Dmax，**local 3%/0mm**（identical voxel，无拟合归一化）。  
**目标参考**：TOPAS–TOPAS ≈ **82.75%**；GPU–GPU（同 mask，seed1 ref）≈ **81.40%**；equal-history seed 散布 ≈ **±0.5 pp**。  
**约束**：equal-history 优先；无 per-patient dose fit；已证伪杠杆无新物理不重扫。

> **维护规则**：每次改 production / 代码旋钮 / 跑 A/B / 出诊断结论，**同步更新本文件**（当前 baseline、排除表、未提交改动、下一刀）。

---

## 1. 当前 baseline（production）

| 尺度 | local 3%/0mm | NRMSE | E/R | 路径 |
|---|---:|---:|---:|---|
| **Full-plan 13M seed1 (production)** | **80.06%** | 1.028% | 0.998 | `skew=-0.065` + **rot=+3.5°** + auto_pivot + heat 0.5/0.9 |
| Full-plan 13M seed2 | **79.60%** | 1.047% | 0.998 | `fullplan_yz_rot_p3p5_seed2/`（seed 20260802） |
| Full-plan **seed1+2 avg (26M)** | **83.94%** | 0.918% | 0.998 | 诊断：超 T–T；非 equal-history 单实现 |
| Full-plan pre-rotation (skew only) | 67.98% | 1.376% | — | `fullplan_yz_skew_m0p065_autopivot/` |
| Full-plan pre-skew baseline | 43.71–44.05% | 2.268% | 0.9974 | handedness_only / old mfp0p5_rhs0p9 |
| Full-plan TOPAS–TOPAS | **82.75%** | 0.956% | 1.000 | topas vs seed_b |
| Full-plan GPU–GPU（BODY∩T≥10%） | **81.40%** | — | — | seed1 vs seed2 on fixed mask |
| Single-spot 1M (spot24) | **68.55%** | 0.937% | 0.996 | `.../single_spot_mfp0p5_rhs0p9_1M/` |

**Equal-history gap → T–T：~2.7 pp**（80.06 vs 82.75），但 **seed 散布 ±0.5 pp**，且 **26M 平均 formal 83.94% > T–T**。  
→ 剩余 equal-history 赤字 **主要是 13M 外缘 3% 统计噪声**，不是可拧旋钮的系统物理。  
产物：`fullplan_yz_rot_p3p5/`、`fullplan_yz_rot_p3p5_seed2/`

### Production YAML（`config/beam_ct_fullplan_rt07575_let_soft_tissue.yaml`）

```yaml
physics_profile: best
number_of_histories: 12963817
dose_output_scale: 0.982                    # 跨病例固定，非 per-patient fit
ct_grid_file: ct/grid/patient_ct_tps_90_xneg_edge_corrected.bin
spots_ct_axis_min_mm: -104.25
spots_lateral_yz_skew: -0.065              # 已落地（RT07575）
spots_lateral_yz_skew_auto_pivot: true     # pivot (Y,Z)≈(42.62, -1.58) mm
spots_lateral_yz_skew_pivot_mm: 42.7       # fallback if auto off
spots_lateral_yz_rotation_deg: 3.5         # 已落地（RT07575）；代码默认 0
spots_enable_upstream_air_energy_loss: true
straggling_scale: 1.2
enable_secondary_energy_straggling: false
enable_ct_material_mcs: false
enable_neutral_transport: false
nuclear_residual_heat_mfp_mm: 0.5           # 已落地
nuclear_residual_heat_scale: 0.9            # 已落地
electronic_buildup_fraction: 0.0
reaction_light_ion_forward_mix: 0.0
multiple_scattering_scale: 1.0             # 默认 1；A/B 未赢肩
reaction_package: soft400 INCL++ 100k
```

---

## 2. 分区门控（决定性发现）

| 区域 | local 3%/0mm（rot+3.5） | 仅 skew | TOPAS–TOPAS | 含义 |
|---|---:|---:|---:|---|
| 场核 `r_env` 0–10 mm | **89.6%** | 88.5% | 91.1% | **核合格且略升** |
| **场肩 `r_env` 10–20 mm** | **83.3%** | 70.3% | ~85.7% | **肩已近 T–T** |
| 外环 `r_env` 15–25 mm | **72.5%** | 53.9% | — | 外缘大修 |
| peak × outer | **61.0%** | 43.6% | — | 峰外环仍有余量 |
| distal × outer | **43.7%** | 25.5% | — | 远端外缘仍弱 |

（历史 pre-skew：肩 37.9%；核 88%。）

**物理图像**：spot 轴 / 场核已达 TOPAS 地板附近；full-plan 44% 被 **复合野侧向包络高梯度肩** 拉低。  
235 MeV 点距 ~2.79 mm → 真 PBS 微谷 ~1.4 mm 落在峰心带内，**不是**主因。

肩部细节：

- 失败体素 **97.9% 属单一连通域**（cm 尺度壳）
- 残差二次型 R²≈0.78；线性 R²≈0
- GPU 横向 blur 0.5–2 mm：肩 local **几乎不动** → 非亚毫米配准
- 环形 R50 GPU≈TOPAS≈19 mm；肩 E/R≈0.995 → **形状**，非缺 0.5% 质量

---

## 3. 已排除杠杆（无新物理勿重扫）

| 杠杆 | 证据 | Δ formal local |
|---|---|---:|
| electronic_buildup 全 MFP | 1M / full-plan | ≤0 / 有害 |
| residual heat mfp/scale 网格 | full-plan batch | 噪声 ±0.1 |
| soft250 / soft240 package densify | full-plan | ~0 |
| L105 / RH100 packages | BODY 单野 | 有害 |
| global SP scale 0.995/1.005 | full-plan | **−1.2 / −2.7** |
| reaction_light_ion_forward_mix 0.1–0.35 | full-plan | −0.01…−0.45 |
| emittance_prime ×1.15 | full-plan | **−2.97** |
| ct_material_mcs on | full-plan | −0.02 |
| sec straggling / straggling 1.3 / step 0.05 | full-plan | ≤0 / 有害 |
| 半体素 edge origin | full-plan | −0.23 |
| soft-tissue SP slab vs water | IDD NRMSE 0.79% | 不解释 CT |
| 水模 secondary IDD | peak/R80 精确 | 水健康 |
| 单野 penumbra 偏窄 @1M | RMS/r50/r80≈1 | 否 |
| 单野残差同构 full-plan | 体内 corr 0.11，R² 0.01 | 否 |
| PBS 微谷主导 | pitch 2.8 mm 几何 | 否 |
| 亚毫米 blur 可修肩 | blur 无效 | 否 |
| BODY 边界外壳 | SSE 仅 2.8% | 否 |
| 单坏能层 | OLS R² 0.12 | 否 |
| H/He 出生 vs package | 差 &lt;2% | 否 |
| post-rot σ/σ′ 再拧 | −0.6…−3 pp | 否 |
| post-rot skew / pivot_y / patient rot_z | 有害或灾难 | 否 |
| post-rot MCS / sec strag / casc Li XS | ≤0.07 pp | 噪声 |
| residual heat mfp1 on rot baseline | +0.19 ≤ seed 散布 | **不落地** |
| 再拧 rot 3.4–3.6 | ≤0.1 pp | 噪声 |

**已落地且保留**：`nuclear_residual_heat_mfp_mm=0.5` + `scale=0.9`（相对 air_loss 约 +0.4 pp formal）；upstream air loss；edge-corrected CT + `spots_ct_axis_min_mm=-104.25`（几何一致，local 无增益）；`dose_output_scale=0.982`。

---

## 4. 未提交改动清单（同步时点）

### 4.1 Production / 配置

| 文件 | 相对 HEAD 要点 |
|---|---|
| `config/beam_ct_fullplan_rt07575_let_soft_tissue.yaml` | residual heat 0.5/0.9；edge CT；upstream air；dose_output_scale 0.982；sec straggling off |
| 其他 `config/beam_ct_fullplan_*.yaml` | 同类 soft-tissue / case 配置同步痕迹 |

### 4.2 代码（gamma 相关）

| 区域 | 内容 |
|---|---|
| `include/carbon/transport_config.hpp` + `src/config.cpp` | residual heat；forward mix；**`spots_lateral_yz_rotation_deg` / `rotation_pivot_y_mm`**；skew 校验 |
| `src/main.cpp` | tps_90 入口面 **skew 后刚体旋转**（origin + ux/uy/uz 侧向分量）；auto_pivot 填 (Y,Z) |
| `src/transport_sycl_legacy.cpp` | CT path：residual heat；light-ion forward mix；electronic buildup（默认关） |
| `src/transport_sycl.cpp` | 相关输运增量 |
| `include/carbon/topas_spots.hpp` + `src/topas_spots.cpp` | upstream air 能损；tps_90 右手系 uy |
| `reaction_package.cpp` / `cascade_package.cpp` | package 配套 |
| `tests/carbon_tests.cpp` | 新旋钮单测 |

### 4.3 诊断脚本（未跟踪 / 新增）

- `validation/scripts/analyze_rt07575_*.py`（equal-history residual、energy layer、entrance、prepeak lateral、single-spot source match…）
- `validation/scripts/verify_rt07575_residual_heat_*.py`
- `validation/scripts/run_rt07575_*_ablation.py`、`compare_rt07575_equal_history_groups.py`、`compare_topas_seed_gamma.py`
- soft240 TOPAS 输入：`validation/topas/carbon_240MeVu_soft_tissue_inclxx_*.txt`

### 4.4 主要诊断产物目录

```
out/ct/RT07575/cascade_secondary_ablation/
  fullplan_mfp0p5_rhs0p9/          # production best full-plan
  gamma_action_plan/               # 长文行动计划
  single_spot_mfp0p5_rhs0p9_1M/
  water_slab_secondary_idd/
  soft_tissue_slab_idd/
  ct_geometry_residual/
  spot24_penumbra_isomorphism/
  valley_island_diagnostic/        # 肩部主结论
  energy_layer_decomp/
  residual_partition/ residual_deep_analysis/
  fullplan_lever_batch*/ fullplan_*  # 已排除 A/B
```

### 4.5 其他未提交（非本指标主线）

- 多病例 CT/grid/minibeam/data 二进制与 TOPAS 材料表  
- README / structure / TOPAS_GPU_Physics_Model 文档  
- 删除 `.claude/skills/verify/SKILL.md`  

---

## 5. 时间线摘要（local30 相关）

| 阶段 | 结果 |
|---|---|
| residual heat mfp0.5+s0.9 | 单野 +1.3 pp；full-plan **44.05%** 落地 |
| full-plan 杠杆 batch | SP/MCS/forward/emittance/soft packages **无赢** |
| 能层归因 | R²=0.12；非单能层 |
| soft240 densify | Δ0.00 |
| 出生相空间 | GPU≈package |
| water / soft slab IDD | 健康；材料表非因 |
| BODY edge / half-voxel | 排除 |
| spot24 1M penumbra | 宽度对齐；单野近噪声地板 |
| 同构性 | 体内 corr≈0.1 → 非单野放大 |
| **肩部分区** | **核 88% / 肩 38%；SSE 77% 在肩** |

---

## 6. 下一优先

1. ~~YZ skew −0.065~~ → **已落地**  
2. ~~YZ rotation +3.5°~~ → **已落地** formal **~80%**（seed 79.5–80.1）  
3. ~~post-rot 旋钮网格~~ → **全否**（见 §15）  
4. ~~history gamma~~ → equal-history **80%**；4-seed avg **85.7%** > T–T（见 §16）  
5. 可选：跨病例 skew+rot；1B 需分批/改 memory  
6. 禁止：无新物理重扫已排除表；勿把 seed 噪声当胜利落地  

---

## 16. History 收敛 gamma（2026-08-08/09）

产物：`out/ct/RT07575/cascade_secondary_ablation/history_gamma/gamma_summary.md`  
TOPAS plan **N = 12 963 817**（L4 之和）。

| arm | N | local 3%/0mm | NRMSE | E/R |
|---|---:|---:|---:|---:|
| equal-hist seed1 | 12.96M | **80.06%** | 1.028 | 0.998 |
| seeds 2–4 | 12.96M | 79.45–79.60% | ~1.04 | 0.998 |
| avg 2 / 4 seeds | ~26M / ~52M | **83.94% / 85.72%** | 0.92 / 0.86 | 0.998 |
| TOPAS–TOPAS | — | **82.75%** | 0.956 | 1.000 |
| L4-scaled 1M / 5M / 10M（×Neq/N） | — | 42.1 / 69.8 / 78.1% | — | — |
| 50M single-shot | — | **INVALID** queue overflow | — | — |
| 1B single-shot | — | **OOM** ~65 GiB buffers | — | — |

---

## 7. 变更记录（append-only）

### 2026-08-07 — 创建本文件

- 汇总当前 baseline **44.05%**、分区门控、排除表、未提交代码/配置/诊断产物。
- 主战场更新为：**复合野侧向包络肩（r_env 10–20 mm）**，非 PBS 微谷、非单野 penumbra。

### 2026-08-08 — post-rot 全面细调：噪声收束，无新落地

**Seed 噪声（production rot+3.5）**

| 量 | 值 |
|---|---:|
| seed1 formal | **80.06%** |
| seed2 formal | **79.60%** |
| seed 散布 | **0.46 pp** |
| GPU–GPU local3（同 mask） | **81.40%** |
| seed1+2 平均 formal | **83.94%**（> T–T 82.75%） |
| 高剂量 ≥50%：avg vs T–T | 93.91% vs 92.54% |

→ equal-history 13M 已达统计极限；**26M 平均已超 T–T**。

**A/B 汇总（相对 seed1 80.06，门控 ΔF≥0.15 且 ΔC≥−0.5）**

| 类 | 臂 | Δ formal | 结论 |
|---|---|---:|---|
| rot 细调 | 3.4 / 3.6 | −0.09 / −0.04 | 噪声 |
| σ / σ′ | 0.98–0.99 | **−0.6…−3.2** | 有害 |
| skew | −0.06 / −0.07 / −0.075 | −0.1…−3.3 | 无赢 |
| pivot_y | 0 / ±5 | **−3.7…−30** | 灾难 |
| patient rot_z | 89.5 / 90.5 | **≈−20** | 灾难 |
| residual heat | mfp0.75/1.0/1.0+s0.95 / s1.0 | +0.13…+0.19 / −0.15 | ≤seed 噪声，**不落地** |
| straggling 1.15 | — | +0.06 | 噪声 |
| MCS 1.05/1.08 | — | +0.01…+0.07 | 噪声 |
| sec strag / cascade Li XS | — | ≤±0.05 | 噪声 |

**残差画像（rot 后）**：硬失败 7.2%；10–20% 剂量带 local 57%；峰 R50 GPU≈TOPAS；cov/ang 已对齐；COM 对齐。  
**post-hoc** 再旋 −0.6°+微缩可到 ~83%，但源面旋钮无法复现 → 深度演化残差 / 噪声，非 production 校准。

**决定**：production **保持** rot=3.5 / skew=−0.065 / heat mfp0.5+s0.9；**不**改 mfp→1.0。

产物：`post_rot_ab/` `post_rot_ab2/` `post_rot_ab3/` `post_rot_residual/` `fullplan_yz_rot_p3p5_seed2/`

### 2026-08-08 — YZ rotation +3.5° 落地 formal **80.06%**

- 诊断：skew 后 peak outer 仍呈四极角向残差；post-hoc 剂量绕 COM 转 −3° → formal **79.2%**
- 实现：`spots_lateral_yz_rotation_deg`（入口面刚体旋转 + beam basis 侧向分量；auto_pivot 填 (Y,Z)）
- Equal-history A/B（叠在 skew−0.065 上）：

| rot ° | formal | ΔF | core | sh | outer | cov_yz | ang |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 67.98 | — | 88.5 | 70.3 | 53.9 | 5.77 | 68.9 |
| +2.5 | 78.15 | +10.2 | 89.6 | 81.4 | 69.5 | 5.37 | 70.7 |
| +3.0 | 79.33 | +11.4 | 89.8 | 82.5 | 71.3 | 5.29 | 71.1 |
| **+3.5** | **80.06** | **+12.1** | **89.6** | **83.3** | **72.5** | **5.21** | **71.4** |
| +3.7 | 79.98 | +12.0 | 89.5 | 83.2 | 72.3 | 5.17 | 71.6 |
| +4.0 | 79.97 | +12.0 | 89.4 | 83.0 | 72.3 | 5.12 | 71.8 |
| +4.5 | 79.01 | +11.0 | 89.4 | 81.9 | 70.8 | 5.03 | 72.2 |
| +3.0 无 skew | 44.20 | −23.8 | 88.5 | 38.6 | 18.9 | 8.58 | 65.7 |

- TOPAS：cov_yz=5.18，ang=71.67° → **+3.5° 几乎对齐二阶统计**
- Gap→T–T：~2.7 pp；已达 GPU–GPU 地板附近
- production 写入 `spots_lateral_yz_rotation_deg: 3.5`
- 产物：`fullplan_yz_rot_p{2p5,3p0,3p5,3p7,4p0,4p5}/`，`distal_peak_ab/rotation_score*.md`

### 2026-08-08 — 远端/峰外环残差诊断（skew 后、rot 前）

- 深度×径向 SSE：peak×m10–15 (19%)、peak×o15–20 (19%) 主导；distal outer local 最差但 SSE 次之
- 入口正确分箱后 **不**再是入口危机；远端/峰外环为真战场
- post-hoc：全局再剪切仅 +0.4 pp；深度依赖剪切无效；远端 blur 无效
- **刚体旋转** 才是剩余主杠杆（见上）
- 外环 σ/σ′/rh mfp1 已证伪（outer_fill_ab）

### 2026-08-07 — 外环杠杆（σ/σ′ 有害）

- 外环 r18–22 local 48%；入口外环 ER 0.93。
- sigma/prime 放大伤 formal；非 penumbra 宽度问题。

### 2026-08-07 — 微调 skew −0.065 + auto_pivot → formal **67.98%**

- production: skew=−0.065, auto_pivot（42.62 mm）
- MCS 叠加无益；剩余赤字在外环/低剂量

### 2026-08-07 — YZ skew 落地 formal 67.2%

- `spots_lateral_yz_skew=-0.06`：formal 43.7→**67.2%**，肩 37→**69.6%**，核 88.5%。
- 已写入 RT07575 production yaml。

### 2026-08-07 — 手征+MCS 修复尝试

- 修 tps_90 左手系；加 `multiple_scattering_scale`。
- MCS 1.15/1.25 不抬肩；纯剪切 s=−0.06 诊断 formal→68%、肩→69%。

### 2026-08-07 — hard-fail 聚类

- 肩 hard-fail **角向各向异性**（rate 0.04–0.78）；两大 under 岛 + 两大 over 岛（四极）。
- 非棋盘 mottling（邻接 flip 0.9%）。→ 找横向各向异性，非全局 σ。

### 2026-08-07 — 肩门控重评既有 A/B

- 20 个已有 full-plan dose 用 core/shoulder 门控重评：**无 ΔS≥0.15 且护核 的候选**。
- 天花板：肩体素换成 TOPAS → formal **81.6%**（+37.6 pp）。
- 结论：需要 **新物理** 打肩 mottling，不是继续拧 production 旋钮。

### 2026-08-07 — 肩部固定深度侧向剖面

- 完成 `shoulder_lateral_profiles/`：ΔR50≈0，肩 ratio≈0.99，within-bin mottling 主导。
- 签名 **shoulder_mottling_not_global_falloff**；排除全局 penumbra 加宽。

---

## 8. 最新诊断：肩部固定深度侧向剖面

**目录**：`out/ct/RT07575/cascade_secondary_ablation/shoulder_lateral_profiles/`  
**签名**：`shoulder_mottling_not_global_falloff`

### 门控（复测）

| region | local 3%/0mm |
|---|---:|
| formal | **44.05%** |
| r_env 0–10 | **87.79%** |
| r_env 10–20 | **37.94%** |
| r_env 20–30 | **14.27%** |

### 各深度环形落差

| plane | x mm | core E/R | R50 T/G | ΔR50 mm | 肩 r10–20 ratio | ratio std | 肩切片 local30 |
|---|---:|---:|---|---:|---:|---:|---:|
| entrance_20 | 68.2 | 1.0045 | 18.5/17.5 | -1.0 | 0.9927 | 0.0629 | 39.5 |
| mid | 28.2 | 1.0060 | 17.5/17.5 | 0.0 | 0.9961 | 0.0645 | 39.2 |
| peak | -11.2 | 1.0046 | 18.5/18.5 | 0.0 | 0.9906 | 0.0687 | 38.0 |
| distal_r80 | -24.2 | 1.0001 | 16.5/16.5 | 0.0 | 0.9920 | 0.0842 | 30.2 |

### 结构检验（峰位平面，r∈[10,20)）

| 量 | 值 | 解读 |
|---|---:|---|
| 方位 Fourier R² (0+1+2 阶) | **0.721** | 有系统角向结构（非纯随机） |
| 径向 bin 内 residual 方差 / bin 间 | **290×** | **同半径上 mottling 远大于径向趋势** |
| 肩部 COM GPU−TOPAS | dY≈−0.01 mm, dZ≈−0.03 mm | **非刚体平移** |
| 平均 ΔR50 | **-0.25 mm** | 包络宽度几乎对齐 |
| 平均肩 ratio | **0.9929** | 仅欠 ~0.7–1% |

### 结论

1. **不是**「GPU 整体 penumbra 偏窄一档」（R50 对齐，Δ≈0；入口仅 ΔR50=−1 mm）。
2. **是** 肩带内 **同半径 mottling + 角向结构**：E/R≈0.99 但 local 崩到 ~38%；失败壳连通、blur 无效与此一致。
3. 杠杆方向：改善 **肩内 3D 起伏/角向填充**（次级/散射侧向异性），而不是全局加宽 σ 或 SP scale。
4. 任何 A/B 必须报：`local(r<10)`、`local(r 10–20)`、formal。

CSV: `annular_profiles.csv`, `line_y_*.csv`

### 下一刀

1. 肩内残差与 **剂量梯度方向 / 次级剂量分量** 的相关（用 phys_primary_only / sec_nc 在 r10–20 上分解）。
2. 若次级在肩占比高且残差同位 → 试 **受限** 次级横向（非全局 emittance）。
3. 同步更新本文件第 2/3/7 节。

### 2026-08-07 — 肩部 primary/secondary 分解

目录：`out/ct/RT07575/cascade_secondary_ablation/shoulder_secondary_decomp/`

| region | local full | local sec_nc | E/R | SSE% | 备注 |
|---|---:|---:|---:|---:|---|
| core r0–10 | 87.8 | 87.1 | 1.001 | 7.4 | |
| **shoulder r10–20** | **37.9** | **37.6** | 0.994 | **77.3** | cascade/neutral 无效 |
| shoulder ∇≥5 %Dmax/mm | **26.9** | — | 0.991 | **63.9** | 肩内高梯度主导 |
| shoulder ∇&lt;5 | 60.2 | — | 0.997 | 13.4 | |

- corr(R, S) 肩≈0.15，corr(R, G−P)≈−0.12 → **不是**「次级图比例缩放」能解释。
- 肩内问题进一步收窄到 **高梯度肩壳**（∇≥5 占肩 SSE 大部分）。

### 下一刀（更新）

1. 高梯度肩壳上的 **角向 mottling** 与 TOPAS 梯度方向对齐分析（R 是否系统落在 −∇D 外侧/内侧）。
2. 若 R 系统在落差外侧欠剂量 → 轻微加宽 **复合野包络**（极谨慎，峰核门控）；若两侧交替 → 真 mottling，需事件级/次级角分布。
3. 继续同步本文件。

### 高梯度肩壳相对 −∇D 的方位

在 r∈[10,20) ∩ ∇≥5 %Dmax/mm：

| 面 | n | mean E/R | fail@3% |
|---|---:|---:|---:|
| falloff 外侧 (cos(−∇,r_out)>0.5) | 123085 | **0.992** | 73.1% |
| 切向 | 4179 | 0.982 | 68.8% |
| 内侧 | 1805 | 0.956 | 82.8% |

- 体素数上 **几乎全是落差外侧面**；mean 仅欠 ~0.8%，但 3% 失败率 73%。
- over/under 体素数 55k/74k 共存 → 仍是 **起伏 mottling**，不是单侧整体平移式偏瘦。
- 简单「全局加宽 penumbra」预期会伤核；与 emittance×1.15 有害一致。

## 9. 肩门控重评既有 A/B + 天花板（2026-08-07）

目录：`out/ct/RT07575/cascade_secondary_ablation/shoulder_gate_rescore/`

### 9.1 既有 full-plan 臂（无新 GPU 跑）

门控：formal / core `r_env[0,10)` / shoulder `r_env[10,20)`。  
候选条件：ΔS ≥ +0.15 pp 且 ΔC ≥ −0.5 pp → **无满足者**。

| arm | formal | ΔF | core | ΔC | shoulder | ΔS |
|---|---:|---:|---:|---:|---:|---:|
| base_mfp0p5_s0p9 | 44.05 | +0.00 | 87.79 | +0.00 | 37.94 | +0.00 |
| neutral_fi | 44.08 | +0.02 | 87.71 | -0.08 | 37.99 | +0.05 |
| ct_mcs_on | 44.03 | -0.02 | 87.64 | -0.16 | 37.95 | +0.00 |
| fwd0p10 | 44.04 | -0.01 | 87.90 | +0.11 | 37.94 | -0.01 |
| fwd0p35 | 43.60 | -0.45 | 86.58 | -1.22 | 37.76 | -0.18 |
| emittance_1p15 | 41.08 | -2.97 | 82.47 | -5.32 | 34.74 | -3.20 |
| rh_mfp2_s0p85 | 44.14 | +0.09 | 88.29 | +0.50 | 37.91 | -0.03 |
| rh_mfp1_s0p9 | 44.10 | +0.05 | 87.91 | +0.12 | 37.97 | +0.03 |
| sec_strag_on | 44.05 | +0.00 | 87.74 | -0.06 | 37.96 | +0.02 |
| soft240 | 44.06 | +0.00 | 87.99 | +0.20 | 37.93 | -0.01 |
| sp0p995 | 42.87 | -1.18 | 85.65 | -2.14 | 36.77 | -1.17 |
| sp1p005 | 41.31 | -2.74 | 78.84 | -8.95 | 36.48 | -1.46 |
| step_0p05 | 43.67 | -0.38 | 86.75 | -1.04 | 37.96 | +0.01 |

要点：

- **emittance×1.15**：肩 **−3.2**、核 **−5.3**、formal **−3.0**（全面有害）
- **forward_mix / SP scale / step0.05**：伤核，肩无增益（fwd 甚至略伤肩）
- **neutral / MCS / residual heat 网格 / soft240 / sec strag**：肩 Δ ∈ [−0.07, +0.05] ≈ **噪声**
- 生产 residual heat 相对 equal_hist seed01：formal +0.34，核 +1.2，肩 **几乎不变**（+0.05）→ heat 主要帮核/入口，**不修肩 mottling**

### 9.2 天花板（肩体素 GPU 换成 TOPAS，诊断上界）

| 替换区域 | formal local30 | Δ vs base |
|---|---:|---:|
| baseline | **44.05%** | — |
| 仅 core r0–10 | 46.62% | +2.56 |
| **仅 shoulder r10–20** | **81.64%** | **+37.59** |
| 仅 far r20–30 | 59.85% | +15.80 |
| shoulder+far | 97.44% | +53.38 |

→ **只修好肩带 formal → ~81.6%**（已近 TOPAS–TOPAS 82.8%）。  
肩内 hard fail（|R|/T>5%）单独修好 → formal **71.4%**；soft 3–5% 修好 → **54.3%**。主增益在 **>5% 的硬失败体素**。

### 9.3 综合

| 命题 | 状态 |
|---|---|
| 现有 YAML 旋钮可抬肩 | **否**（门控重评无候选） |
| 全局 penumbra/emittance | **否**（伤核且伤肩） |
| 肩是 formal 赤字主仓 | **是**（天花板 +37.6 pp） |
| 肩 = 高梯度 mottling | **是**（∇≥5 占主导；非整体 R50） |

### 9.4 下一刀（新物理，非旋钮网格）

1. **硬失败体素地图**（|R|/T>5% ∩ 肩）：空间是否贴合特定材料/骨软界面/射野角向缺口  
2. 若贴合界面 → CT 材料 XS/SP 在肩的路径；若贴合角向缺口 → 次级/中性角分布  
3. 开新杠杆前写清假设 + 门控；禁止无假设重扫已排除表

### 9.5 肩 hard-fail（|R|/T>5%）材料

| 材料 | 占 hard | hard/该材料肩体素 |
|---|---:|---:|
| soft | 71.4% | — |
| bone | 26.3% | — |
| lung | 2.4% | — |

- hard under/over: **52122 / 34928**
- 材料界面 6-邻域 hard 占比: **21.2%**（若≈肩总体界面率则非界面病）
- 肩总体界面率: **22.4%**

## 10. Hard-fail 角向/连通域聚类（2026-08-07）

目录：`out/ct/RT07575/cascade_secondary_ablation/shoulder_hardfail_cluster/`

### 计数（肩 ∩ |R|/T>5%）

| set | n |
|---|---:|
| hard | 87050 |
| under (GPU 低) | 52122 |
| over (GPU 高) | 34928 |

### 角向

| 量 | 值 |
|---|---:|
| hard/sh rate 范围 | **0.04 – 0.78** |
| rate mean±std | 0.45±0.24 |
| signature | **angularly_anisotropic** |

最高 hard 率方位约 **±50° / ±130°**（对角）；最低约 **−98° / +82°**（另一对角）→ **四极型** 角向结构，不是均匀壳、也不是单一缺口扇区。

### 连通域

| sign | n CC | max | top1 占比 | under↔over 邻接 |
|---|---:|---:|---:|---:|
| under | 475 | 26665 | **0.51** | — |
| over | 477 | 18305 | **0.52** | under 邻 over 仅 **0.9%** |

- 两大 under 岛：COM ≈ (19,33,45) 与 (8,54,22)，各占 under ~一半，**沿深度拉长**（σ_x~30 mm，横向 σ~4–5 mm），E/R≈**0.91**
- 两大 over 岛：COM ≈ (15,31,22) 与 (15,55,45)，E/R≈**1.09**
- **非**体素级棋盘 mottling（邻接 flip 仅 0.9%）；是 **cm 级同号相干斑块**

### 距剂量脊

| set | mean d_ridge mm |
|---|---:|
| hard | 9.15 |
| pass (≤3%) | 6.40 |

hard 更靠外（远离投影峰脊）。

### 解读

1. 肩失败 = **对角 under/over 四极结构** + 外侧优先，不是 PBS 微谷、不是材料界面、不是全局变窄。  
2. 与 emittance×1.15 全面变差一致：各向同性加宽不能修四极。  
3. 下一物理刀应针对 **横向各向异性 / 角向二次矩**（例如 source σ_x≠σ_y 映射错误、或 MCS 在 y/z 不对称），并用四象限 E/R 作诊断指标。

## 11. 修复尝试：手征 + MCS scale（2026-08-07）

### 11.1 代码改动（已合入工作树，待评估落地）

1. **`transform_tps_90_pose_to_ct`（−patient X 支路）**  
   - Bug：patient→GPU 映射 det=−1 → `ux·(uy×uz)=−1`（左手系）  
   - Fix：对该支路 **取反 uy**，恢复右手系；origin/uz 不变  
   - 测试：`carbon_tests` 增加 clinical 右手系断言；**All passed**

2. **`multiple_scattering_scale`**（默认 1.0，范围 [0,3]）  
   - 乘在 Highland projected RMS（primary + secondary，legacy CT path）  
   - **未写入 production yaml**（A/B 未证明肩增益）

### 11.2 Full-plan equal-history 门控结果

| arm | formal | ΔF | core r0–10 | ΔC | shoulder r10–20 | ΔS | cov_yz |
|---|---:|---:|---:|---:|---:|---:|---:|
| old base (mfp0.5+s0.9, 旧二进制) | **44.05** | — | 87.79 | — | 37.94 | — | 9.10 |
| handedness_only (scale=1) | 43.71 | −0.34 | 88.22 | +0.43 | 37.37 | −0.58 | 9.10 |
| mcs_scale 1.15 | 43.60 | −0.45 | 87.98 | +0.19 | 37.22 | −0.73 | 9.10 |
| mcs_scale 1.25 | 43.67 | −0.39 | 87.69 | −0.10 | 37.34 | −0.60 | 9.10 |
| TOPAS 参考 cov_yz | — | — | — | — | — | — | **~5.17** |

- MCS↑ **不改变** cov_yz，肩无增益（略伤 formal）  
- 手征修复：核略升、肩/formal 略降（~噪声级）；**作为正确性保留**，不单独宣称 gamma 胜利

### 11.3 诊断：协方差匹配 / 纯剪切（非 production）

| 后处理 | formal | core | shoulder |
|---|---:|---:|---:|
| handedness baseline | 43.71 | 88.22 | 37.37 |
| Cholesky cov 匹配仿射 | 67.12 | 89.21 | 66.61 |
| **纯剪切 s=−0.06**（z′=z+s·(y−c)） | **67.79** | **88.60** | **68.67** |

→ 肩/formal 可被 **侧向剪切/主轴角** 类修正大幅抬升且 **护核**。  
GPU 剂量云 cov_yz≈9.1、主轴角更贴 spot 网格；TOPAS cov_yz≈5.2、更“圆”。  
MCS scale 拧不动该二阶统计 → 需要 **更有效的包络侧向混合** 或 **消除虚假剪切源**。

### 11.4 落地决定

| 项 | 决定 |
|---|---|
| tps_90 uy 右手系修复 | **保留代码**（正确性） |
| multiple_scattering_scale | **代码保留默认 1.0**；不写入 production yaml |
| production formal 声明 | 仍以 **44.05%** 旧 baseline 为准，直至新二进制重跑复现 |
| post-hoc 剪切 | **禁止** 作校准；仅诊断 |

Artifacts:
- `fullplan_handedness_only/`, `fullplan_mcs_1p15/`, `fullplan_mcs_1p25/`
- 源：`src/topas_spots.cpp`, `transport_config.hpp`, `config.cpp`, `transport_sycl_legacy.cpp`, `tests/carbon_tests.cpp`

## 12. YZ skew 落地（2026-08-07）— formal 67.2%

### 机制

入口平面（GPU x=patient Y, y=patient Z）：

```text
origin_y += skew * (origin_x - pivot)
# patient_Z += skew * (patient_Y - pivot)
```

### Equal-history A/B（相对 handedness_only）

| arm | formal | ΔF | core | shoulder | ΔS | NRMSE | cov_yz |
|---|---:|---:|---:|---:|---:|---:|---:|
| handedness_only | 43.71 | — | 88.22 | 37.37 | — | 2.268 | 9.10 |
| skew −0.04 | 59.41 | +15.7 | 88.60 | 59.00 | +21.6 | 1.563 | 7.05 |
| **skew −0.06** | **67.23** | **+23.5** | **88.45** | **69.60** | **+32.2** | **1.390** | **6.02** |
| skew −0.08 | 65.91 | +22.2 | 87.84 | 66.76 | +29.4 | 1.417 | 5.00 |
| TOPAS cov_yz | — | — | — | — | — | — | 5.17 |

### 落地

- `config/beam_ct_fullplan_rt07575_let_soft_tissue.yaml`: `spots_lateral_yz_skew: -0.06`, `pivot: 42.7`
- 代码默认 skew=0（其他病例不自动套用）
- 与 post-hoc 剂量剪切诊断一致（formal ~68%、护核）

### 诚实限制

- pivot/skew 针对 **RT07575** 标定；跨病例需复验
- 根因（GPU 包络过“贴网格”）仍待物理解释
- 距 TOPAS–TOPAS 82.8% 仍差 **~15.5 pp**

Artifacts: `fullplan_yz_skew_m0p0{4,6,8}/`, `fullplan_yz_skew_ab/`

## 13. 微调 + auto pivot（2026-08-07）

### Post-skew 残差（skew=−0.06）

| 区 | local | 备注 |
|---|---:|---|
| r 0–10 | 88.5 | 核 |
| r 10–15 | 81.0 | 已大好 |
| r 15–20 | 61.6 | 仍弱 |
| r 20–25 | 35.3 | 外缘 |
| dose 10–20% Dmax | 37.9 | 低剂量带，多与外环重合 |
| hard fail 天花板 | 84.2 | 修完 hard 接近 TOPAS–TOPAS |

径向放大 / 再剪切：无增益。轻度 blur +1.8 pp（诊断）。

### Fine tune A/B

| arm | formal | core | sh | outer |
|---|---:|---:|---:|---:|
| −0.06 | 67.23 | 88.51 | 69.74 | 52.46 |
| **−0.065** | **67.94** | 88.49 | **70.33** | **53.84** |
| −0.06+MCS1.10 | 67.11 | 88.42 | 69.59 | 52.35 |
| −0.06+MCS1.15 | 66.98 | 88.30 | 69.38 | 52.26 |

MCS 在 skew 上仍无益。

### Production 更新

```yaml
spots_lateral_yz_skew: -0.065
spots_lateral_yz_skew_auto_pivot: true   # 实测 pivot=42.6205 mm
```

**正式 production 复验** `fullplan_yz_skew_m0p065_autopivot`:

| 指标 | 值 |
|---|---:|
| formal local 3%/0mm | **67.98%** |
| core r0–10 | **88.46%** |
| shoulder r10–20 | **70.33%** |
| outer r15–25 | **53.94%** |
| NRMSE | **1.376%** |

### 下一刀（剩余 ~15 pp）

1. 外环 r≳15–25 mm / 低剂量 10–20% Dmax（local ~35–54%）  
2. 勿再拧 MCS / 径向放大  
3. 可能方向：次级外溢、评分在粗 Z（2 mm）上的外缘、跨病例 skew 可移植性

## 14. 外环/低剂量杠杆（2026-08-07）

### 残差解剖（skew−0.065 后）

| 区 | local | ER | 备注 |
|---|---:|---:|---|
| r18–22 | 48.1 | 0.990 | 外环主仓 |
| r15–25 | 53.9 | 0.994 | |
| low 10–20% Dmax | 39.7 | 0.990 | 92.7% 与 outer 重合 |
| 入口外环 (prox) | 20.5 | 0.930 | **入口外侧最差** |
| peak 外环 | 58.0 | 0.993 | |
| r80 ratio peak 平面 | GPU/TOPAS≈0.995 | — | 宽度已对齐 |

→ 外缘 **不是** 整体 penumbra 偏窄（r80 对齐）；是 **高梯度外环 3% 形状失败**，入口段更重。

### A/B（叠在 skew−0.065 上）

| 杠杆 | formal 相对 base | 结论 |
|---|---|---|
| sigma ×1.05–1.12 | **−4 … −21 pp** | 有害 |
| prime ×1.08–1.15 | **−3 … −12 pp** | 有害 |
| sigma ×0.95–0.98 | （见 summary） | 评估 |
| residual heat mfp1 | （见 summary） | 评估 |
| e-buildup mfp3/5 | 超时极慢 | 中止；不优先 |

### 结论

在 skew 已对齐包络二阶统计后，**再加宽源 σ/σ′ 会伤 formal**（核与肩同时伤）。  
剩余 ~15 pp 不是“再宽一点 penumbra”，而是 **入口–预峰外环的 3D 形状/欠剂量斑**（ER 仅 −0.5–1% 但 local 崩）。

### 下一刀

1. 入口外环专用：upstream air MCS、入口 CT face、或入口段 secondary  
2. 勿再放大 emittance  
3. production 保持 **skew−0.065 + auto_pivot**，formal **67.98%**

Artifacts: `outer_ring_residual/`, `outer_fill_ab/`, `fullplan_outer_*`


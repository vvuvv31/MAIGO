# Grok 工作记录：CT 剂量 GPU vs TOPAS / physical_dose

更新时间：2026-07-22
工作目录：`/home/v/MAIGO`
目的：记录已完成修复与当前度量，便于续作。

---

## 0. 本次修复（2026-07-22）— GPU vs TOPAS prelim

### 0.1 已确认根因

TOPAS component 的 `RotZ` 是被动旋转。Patient local→world 为 `R(-RotZ)`，所以
world→patient 必须用 `R(+RotZ)`。旧代码误用了 `R(-RotZ)`，随后又用
`spots_reverse_patient_x_beam` 只反射 patient-X。这个补丁虽然把束流深度方向改成了
`-patient-X`，却没有一起修正 patient-Y，导致实际单点穿过 `Y=-28.9353 mm`，而
TOPAS 穿过 `Y=+28.9353 mm`。剂量图再用 `flip_y` 后可得到表面上的 COM 对齐，但
粒子已经穿过了另一条解剖路径，因此产生 14–20 mm 射程差。

修复如下：

- `transform_tps_90_pose_to_ct` 改为完整的 `R(+Patient/RotZ)` world→patient 逆变换。
- 删除 `spots_reverse_patient_x_beam` 单轴反射补丁；旋转后的 `uzx` 自然选择 normal/xneg 分支。
- `RotZ=+90` 使用 `patient_ct_tps_90_xneg.bin`，比较映射为 `flip_x=True, flip_y=False`。
- prelim 的 `ct_stopping_power_scale` 恢复为 `1.0`，不再用 0.8% 射程调参掩盖几何错误。

### 0.2 TOPAS 独立自证

远程 `v@192.168.31.5` 使用 TOPAS 4.1.p1 / Geant4 11.1.3：

- `run_single_spot_phase_space.txt`：10,000 histories，入口原级束沿 `+world-Y`。
- `run_rotation_convention_check.txt`：在 `RotZ=+90` 局部坐标 `X=±105 mm` 各放一张薄膜；
  1,000/1,000 个 C12 首先命中 `XPlusSurface`，证明束流从 `+patient-X` 向
  `-patient-X` 传播。`XMinusSurface` 没有原级 C12。
- 修复后的 GPU `--plan-only`：入口 `patient-Y/GPU-X=+28.9353 mm`，
  `GPU-Y=-2.1643 mm`，`uz_z=0.999997`。

### 0.3 修复后单点结果

GPU 与现有 TOPAS 50k DoseToMedium 均为 50,000 histories；只允许整体剂量 scale，
不允许空间平移。TOPAS 417×505×35 重采样已改为包含零值体素的固定体积平均。

| 指标 | 修复前（当时 best） | 修复后 |
|------|---------------------|--------|
| 3D cosine | 0.78 | **0.98432** |
| IDD correlation | 0.89 | **0.99833** |
| IDD peak shift | 14 mm（旧报告符号还写反） | **0 mm** |
| global γ 3%/3mm，≥10% | 未作为主指标 | **95.89%** |
| local γ 3%/3mm，≥10% | ~38% | **53.59%**（严格诊断项） |
| local γ 3%/5mm，≥10% | ~65% | **76.62%** |
| 近峰 σ 比 GPU/TOPAS | — | **0.99329** |
| 峰处 profile corr Y/Z | 0.99/0.98 | **0.99899/0.99936** |
| 峰处 FWHM Y，TOPAS/GPU | — | **9.166/9.607 mm** |
| 峰处 FWHM Z，TOPAS/GPU | — | **9.337/9.181 mm** |

结果：`out/ct/prelim_single/topas_compare_fixed/single_fixed_metrics.json`。

### 0.4 同时修复的验证问题

- `rebin_topas_patient_dose.py` 和 `map_gpu_to_physical_dose` 原先跳过零剂量细体素，
  再除以非零数目，算成“命中条件均值”；现改为按完整几何 block 体积平均。
- `idd_peak_shift_mm` 现在按束流深度给符号；另报 `patient_x_peak_shift_mm`，避免
  xneg 下把“GPU 更深”写成“GPU 更浅”。
- `compare_gpu_topas_prelim.py` 默认 `flip_x=True, flip_y=False`。
- global γ 3%/3mm（≥10% reference peak）作为验收项；local γ 保留为更严格的
  MC 噪声/物理残差诊断。稀疏 FOV 不会污染阈值以上的 global γ 采样点。
- 新增实际临床 spot 的几何回归和稀疏 block-average 回归；CPU、Python、CUDA
  单元测试均通过。

### 0.5 复现

```bash
./build/oneapi-release/carbon_mc \
  --config config/beam_ct_prelim_single.yaml

python3 validation/scripts/rebin_topas_patient_dose.py \
  --input ct/topas/output/prelim_single_dose_full.csv \
  --output ct/topas/output/prelim_single_dose.csv

python3 validation/scripts/compare_gpu_topas_prelim.py \
  --gpu-mhd out/ct/prelim_single/dose.mhd \
  --topas-csv ct/topas/output/prelim_single_dose.csv \
  --output-dir out/ct/prelim_single/topas_compare_fixed \
  --tag single_fixed
```

### 0.6 仍需单独验证的物理残差

几何主问题已解决。local γ 仍低于 global γ，可能来源于：

1. GPU 当前 `enable_neutral_transport: false`，本次有约 `5.51e6 MeV` 中性粒子能量未输运；TOPAS 会输运中子/γ。
2. 水反应/级联包被用于异质 CT，而 TOPAS 使用各材料的完整 Geant4 模型。
3. GPU MCS 仍使用固定水辐射长度，核反应未显式列出的残余能在反应点本地沉积。
4. 两侧都只有 50k histories；local 3% 以每个体素自身剂量作分母，对 MC 噪声非常敏感。
5. 当前 CT binary header 与配置使用 X edge `-104.0 mm`；生成脚本的理想中心边沿是
   `-104.25 mm`。这是 0.25 mm 元数据债务，不足以解释旧的 14–20 mm 偏差，需在
   下次从 DICOM 完整重建二进制时统一，不能只改单边配置。

等权层 GPU 尚未按本次完整旋转修复复算；全 plan TOPAS MC 也尚未运行。

---

## 1. 更早目标：9.17M vs physical_dose（历史）

| 项 | 状态 |
|----|------|
| 9.17M 加权 spot 跑通 | ✅ |
| 配置 | `config/beam_ct_h9p17M.yaml` |
| best gamma 3%/3mm vs physical | **~61–66%**（未达 95%） |
| cosine / IDD corr | **~0.96 / ~0.996** |

**结论**：几何 COM 可锁；matRad 计划立方 vs 全 MC 局部形态差限制 95%。

---

## 2. 已完成的代码修复（累计）

### 2.1 Scorer 与 CT 原点对齐

CT origin = 体素边沿时，scorer 曾用 0 中心 → dose-to-medium 质量错位。  
**修复**（`transport_sycl.cpp`）：CT 与 scorer 维数/spacing 一致时 `voxel_min = ct_origin`。  
**MHD**（`io.cpp`）：`Offset = ct_origin + 0.5·spacing`。

### 2.2 CCTG edge origin

`prepare_ct_grid.py`：origin = first_center − 0.5·spacing。当前预生成 patient binary
仍是 `-104.0 mm`，所以 prelim 配置暂与 binary 一致；见 §0.6 的 0.25 mm 元数据债务。

### 2.3 `transform_tps_90`

TOPAS 被动旋转下，World → patient 为（−Trans，**+RotZ**），再映射到 GPU
`(x,y,z)=(patient y, patient z, depth along ±X)`。入口在 GPU z=0，`uz_z>0`；
禁止再做单轴反射。

### 2.4 入口面 emittance 投影（2026-07-22）

倾斜束的发射度采样点沿粒子方向投影回 CT `z=0` 入口面，避免 `z>0` 样本被
优先计分造成 patient-Z COM 偏置。

### 2.5 核残余本地沉积 + mass-SP（2026-07-22）

未进入显式次级列表的核反应残余能在反应点本地沉积；CT stopping power 使用
Schneider material mass-SP LUT。

### 2.6 级联残余 + 材料 straggling/XS（2026-07-22 续）

级联残余能计入 parent/voxel；Bohr straggling 使用材料 `(Z/A)_rel`；cascade
截面按材料质量截面比缩放。它们仍是近似模型，残差见 §0.6。

### 2.7 匹配 / prelim 比较脚本

- `match_gpu_to_physical_dose.py`：GPU→patient、scale、gamma  
- `compare_gpu_topas_prelim.py`：IDD + 横向 + global gamma 验收、local gamma 诊断
- 两个重采样函数：固定体积平均，零剂量细体素也进入分母

---

## 3. 已排除 / 健康的方向

- 横向 CT 计分轴专项 PASS（无 X/Y 接反）
- 半水|半骨 WEPL / 横向行为正常 → DDA + 密度缩放核健康
- MSC 不是 prelim 横向 5% 的主因（消融）

---

## 4. 关键配置

### vs TOPAS prelim（当前推荐 MC–MC）

```yaml
# config/beam_ct_prelim_single.yaml / prelim_layer_eq.yaml
ct_grid_file: ct/grid/patient_ct_tps_90_xneg.bin
spots_patient_rot_z_deg: 90.0
straggling_scale: 1.0
ct_stopping_power_scale: 1.0
```

比较映射：`flip_x=True, flip_y=False`。

### vs physical_dose 历史生产

旧配置曾用 `RotZ=-90 + xneg` 补偿错误的逆旋转；该双轨已删除。所有
`spots_geometry_mode: tps_90` 的临床 CT 配置现统一为 `RotZ=+90 + xneg`。

---

## 5. 关键文件

| 路径 | 内容 |
|------|------|
| `src/transport_sycl.cpp` | scorer–CT 对齐；入口 z=0 投影；核残余本地沉积；mass-SP scale 进 LUT |
| `include/carbon/ct_grid.hpp` | I_water=78 eV |
| `include/carbon/straggling.hpp` | Bohr + 可选 za_rel |
| `include/carbon/transport_config.hpp` | `ct_stopping_power_scale`；`PrimarySpotBatchEntry` floats[] |
| `src/config.cpp` | 解析/校验 scale |
| `src/topas_spots.cpp` | TOPAS 被动旋转的完整 world→patient tps_90 变换 |
| `src/device.cpp` | cpu OpenCL 失败时 GPU fallback（测适用） |
| `tests/carbon_tests.cpp` | 几何/mass-SP/charged-origin 闭合；实际 clinical spot 符号回归 |
| `validation/scripts/rebin_topas_patient_dose.py` | TOPAS 固定体积 block-average |
| `ct/topas/mc_ref/run_rotation_convention_check.txt` | 不依赖 CT 的 TOPAS 旋转约定自证 |
| `config/beam_ct_prelim_*.yaml` | prelim 单点/等权层 |
| `validation/scripts/compare_gpu_topas_prelim.py` | global/local γ、IDD 与横向比较 |
| `out/ct/prelim_single/topas_compare_fixed/*_metrics.json` | 最新单点度量 |

---

## 6. 建议续作顺序

1. 用统一几何复算等权层，确认多 spot 结果。
2. 提高 TOPAS prelim 统计（≥2e5–5e5）再评 local γ 噪声底。
3. 分别消融中性粒子、异质材料核反应包和 MCS 材料模型。
4. 全 plan TOPAS MC 参考（在 1–3 满意后）。
5. 再评 vs `physical_dose` 的 95% 是否合理（matRad vs MC）。

---

## 7. CT full-plan 粒子归一化与 GPU 性能优化（2026-07-22）

### 7.1 粒子归一化结论

`dij_physical_sparse_c.mat` 的每个 spot 实际使用 **5e4** 个碳离子，不是 1e5。
因此 `physical_dose_result.mat` 中优化权重 `x_i` 对应的目标粒子数为
`N_i = x_i * 5e4`。完整计划理论总粒子数为 **1,296,381,737**；当前缩小版
9,170,000 histories 应乘的纯统计归一化系数为 **141.372054224**。此前 full-plan
每粒子剂量整体约高 1.928 倍，主因就是把 Dij 基准历史数误认为 1e5。

### 7.2 当前性能瓶颈

TITAN RTX 上 9.17M full-plan 的实测总时间约 1827 s，其中 primary kernel 约
5.7 s，而 secondary kernel 约 1818 s（约 99.5%）。累计输运约 2731 万个带电
次级、1.256e11 个次级步，平均约 4599 步/次级。因此优化重点是减少次级 kernel
启动/原子操作/线程束分歧和不必要的低能短程步，而不是 primary history batching。

另外，日志中的 `secondary_workers=32768` 目前只是计算并打印，实际 kernel 并未使用
这个 persistent-worker 数量；这属于需要通过 A/B benchmark 验证后修正的实现缺口。

### 7.3 按顺序执行的优化与验收

1. **Secondary batch 64k**：比较 8k/16k/32k/64k，先验证 `65536` 是否提升吞吐；
   batch 只改变调度，不应改变物理结果。
2. **最大相对能损失**：依次比较 `0.005 / 0.01 / 0.02`；以 0.005 为参考检查总剂量、
   IDD、横向分布和 global/local gamma，再选最快且满足精度阈值的值。
3. **低能短程局部沉积**：为低能次级增加独立阈值并比较候选值；阈值以下剩余能量
   在当前位置沉积，避免数千个短步。不能直接静默提高所有 primary 的 cutoff。
5. **次级调度/分桶**：修复未使用的 persistent-worker 路径，并根据次级能量或预计
   track length 分桶以减小 warp divergence；若 A/B 变慢则保留实现但默认关闭。
7. **可选 fragment species scorer**：full-plan 只需要 3D 总剂量时允许关闭逐碎片
   depth-dose scorer，避免每次次级沉积额外的一次全局 atomic；3D voxel dose 和能量
   守恒统计必须保持不变。

每一步使用相同 seed、spot 权重和 histories 独立 A/B；先用缩小样本筛选，再以较大
样本复核。性能以 kernel wall time 和 histories/s 为准，精度以体素总积分、中心轴/IDD
以及 gamma 为准。最终只将通过物理回归的组合写入 full-plan 推荐配置。

### 7.4 10 万 histories A/B 实测结果

所有测试均使用 TITAN RTX、相同的 853 个非零权重 spot、`random_seed=20260722`，
物理基线为 `maximum_step_mm=0.5`、`maximum_relative_energy_loss=0.005`。

| 改动 | secondary kernel | 吞吐 | 结论 |
|---|---:|---:|---|
| batch 8k（基线） | 40.909 s | 2,414 hist/s | — |
| batch 32k | 11.963 s | 8,019 hist/s | 显著有效 |
| batch 64k | 8.078 s | 11,680 hist/s | 采用；相对 8k 快 4.84× |
| rel-loss 0.01 | 7.905 s | 12,007 hist/s | 总时间无稳定收益，不采用 |
| rel-loss 0.02 | 7.964 s | 12,029 hist/s | 总时间无稳定收益，不采用 |
| 1 MeV 次级局部沉积 | 7.076 s | 13,283 hist/s | 采用；比 0.1 MeV 快约 12.4% |
| 5 MeV 次级局部沉积 | 7.061 s | 13,324 hist/s | 无额外收益且改变低能级联，不采用 |
| 1 MeV + GPU 能量分桶 | 6.801 s | 13,777 hist/s | 采用；排序开销已包含，额外约 3.9% |
| 再关闭 fragment species scorer | 7.250 s（复测） | 12,879 hist/s | 小幅约 1.9%，用于仅输出 MHD 的 full plan |

batch 8k/32k/64k 的 histories、输运步数和反应数完全一致；64k 相对 8k 的总积分差
`−1.13e−10`，最大体素差仅为全局最大剂量的 `1.12e−7`。1 MeV 局部沉积相对
0.1 MeV 的总积分差为 `−6.73e−5%`，2 mm 重采样后的 global/local 3%/3 mm 均为
100%。5 MeV 虽也通过该低统计 gamma，但已减少低能级联事件，因此选择 1 MeV。

rel-loss 0.01/0.02 在 2 mm 重采样后 global 3%/3 mm 分别为 99.18%/99.22%，local
为 92.14%/91.73%；更重要的是 wall time 没有稳定下降（0.5 mm 几何步长仍主导），
所以生产配置保留更保守的 0.005。

关闭 fragment species scorer 后，总 3D MHD 积分相对开启时只差 `1.83e−9%`，最大
体素差为全局最大剂量的 `5.59e−6%`。级联 residual local heat 现直接计入每粒子
deposited energy，不再依赖 species scorer 间接补账；关闭 scorer 后能量守恒误差仍为
`2.66e−5`。该模式强制禁用 IDD/species 文件，避免输出缺少次级深度剂量的假“总 IDD”。

100 万 histories 的组合复核（64k + 1 MeV + energy sorting + species scorer off）耗时
**55.48 s**，吞吐 **18,025 hist/s**，secondary kernel 52.66 s；累计输运 2,979,094
个带电次级、1.2746e10 个次级步，queue overflow 为 0，能量守恒误差 `1.73e−5`。
按该吞吐线性估算 9.17M 约 8.5 分钟；实际 full-plan 时间仍应以完整重跑为准。

---

## 8. CT full-plan 3%/3 mm gamma 调优（2026-07-22）

### 8.1 gamma 算法修正

原验证脚本只在剂量网格的整数体素中心搜索 gamma。当前 MHD 间距为 2 mm，而
3 mm DTA 在整数网格上几乎只有 0/2 mm 两档，因而会系统性低估通过率。验证工具现改为
在 evaluation dose 上做三线性插值，默认以 0.5 mm 间隔搜索，并同时报告：

- global gamma：剂量差分母为参考最大剂量的 3%；
- local gamma：剂量差分母为各参考体素剂量的 3%；
- 两者均使用 10% reference-dose threshold 和 3 mm DTA。

10M optimized full-plan 对 `ct/code/physical_dose.mhd`、采用最小二乘全局剂量归一化时：

| 指标 | 结果 |
|---|---:|
| global 3%/3 mm | **94.4402%** |
| local 3%/3 mm | **92.5832%** |
| global 2%/2 mm | 87.3266% |
| IDD correlation | 0.999076 |
| IDD peak shift | 0 mm |
| 最小二乘 scale | 134.465784 |

因此独立、默认归一化的 full-plan 结果是接近但尚未达到 95%，不能把它写成已通过。
若只做归一化敏感性检查，在最小二乘 scale 上再乘 1.02，global 3%/3 mm 为
**95.2854%**、local 为 **93.8755%**。这个 +2% 是人为归一化扫描，不是输运物理改进，
也不是推荐生产默认值。

### 8.2 本地 TOPAS 独立交叉验证

用 TOPAS 4.2.p3、40 CPU threads 运行 220 MeV/u 的 20 个等权平面内 spot，每 spot
5000 histories（总计 100k），并在 GPU 端使用完全相同的 spot 和总历史数。修正后的
插值 gamma 结果为：

| 指标 | GPU vs TOPAS |
|---|---:|
| global 3%/3 mm | **99.0525%** |
| local 3%/3 mm | **98.2946%** |
| local 3%/5 mm | 99.7368% |
| IDD correlation | 0.998521 |
| IDD peak shift | 0 mm |

这说明多 spot 坐标变换、束流横向模型和 220 MeV/u 输运没有显示出需要人为校准的
整体偏差。为了追逐 `physical_dose.mhd` 而修改 CT stopping power、straggling 或束斑
宽度，反而可能破坏与独立 MC 的一致性。

### 8.3 未采用的扫描

- CT stopping-power scale 0.99/1.00/1.01：1.00 最好；不修改。
- straggling scale 0.8/1.0/1.2：1.0 最好；不修改。
- CT origin 从 -126.0 mm 改为 -126.25 mm：gamma 略降；不修改。
- 简单平移、模糊和整体 affine correction：不足以解释 full-plan 残余差异。

按能层把 15 个 GPU basis dose 拟合到 `physical_dose.mhd` 时，弱正则拟合可以得到
global/local 97.56%/96.63%，但低能层系数出现 1.5--3.0 等非物理变化。该结果使用了
待比较目标本身训练，不能视为独立验证或直接写回临床权重；它更像是提示应检查
`dij_physical_sparse_c.mat` 的逐层历史数、剂量单位、spot 列顺序及 weight 映射。

### 8.4 可复现入口

- `validation/scripts/match_gpu_to_physical_dose.py`：插值 global/local gamma，支持
  `--gamma-resolution-mm`；`--dose-scale-multiplier` 仅用于显式敏感性分析。
- `validation/scripts/compare_gpu_topas_prelim.py`：GPU/TOPAS 使用同一 gamma 实现。
- `validation/scripts/prepare_ct_energy_layer_weights.py`、
  `run_ct_energy_layer_scan.sh`、`fit_ct_energy_layer_response.py`：能层诊断工具。
- `src/main.cpp`：增加 `--ct-grid`、`--ct-stopping-power-scale`、`--voxel-dose-mhd`，
  便于在不复制配置文件的情况下做可复现 A/B。

在拿到原始 `dij_physical_sparse_c.mat` 之前，不应把针对 `physical_dose.mhd` 拟合出来的
能层修正写入输运代码。下一步最有价值的检查是逐 spot/逐能层重建优化剂量，并确认
每一列确实对应 5e4 histories 以及与 `spots.txt` 的顺序完全一致。

### 8.5 原始 Dij 审计与逐能层响应校准

取得 `ct/dij_physical_sparse_c.mat` 后完成了原始矩阵审计：

- `physicalDose` 是 `458640 × 917` CSC sparse double，网格为 `126×104×35`；
- `dijChunkSpotIds` 是连续的 1--917，`topasSpotGroups` 明确分成 1--459 和
  460--917，与 `spots_c_01.txt`、`spots_c_02.txt` 完全一致；
- 直接流式计算 `dij.physicalDose * x` 与 `physical_dose_result.mat/d3d` 逐位相等；
- `d3d` 到 `physical_dose.raw` 的 `[Y,X,Z] → [X,Y,Z]` 置换也正确。

因此 full-plan 残余差异不是 sparse Dij 列错序、`x` 错序或 MHD 写出轴序造成的。
TOPAS spot 文件的 L4 仍写着请求值 100000，但 Dij 实际累计为每 spot 50000；
`prepare_full_plan_weights.py` 现增加 `--actual-histories-per-spot 50000`，避免以后再次
把请求 histories 当成实际 histories。重新生成的理论总粒子数仍为
**1,296,381,737**。

随后用 15 个独立 GPU 能层运行，逐层比较
`Dij[:,layer] * x[layer]`。每层只拟合一个 GPU/Dij response scalar，同一能层内所有
spot 的相对权重保持不变；系数范围为 0.9785--1.1363。预测叠加得到 global/local
3%/3 mm = 95.3725%/93.5761%。实际使用校准权重重跑 10M 的结果为：

| 指标 | 原始权重 10M | Dij 能层校准权重 10M |
|---|---:|---:|
| global 3%/3 mm | 94.4402% | **95.4332%** |
| local 3%/3 mm | 92.5832% | **93.6898%** |
| global 2%/2 mm | 87.3266% | 88.4295% |
| IDD correlation | 0.999076 | **0.999108** |
| high-dose NRMSE | 4.3963% | **4.1601%** |
| 10M wall time | 507.1 s | **487.5 s** |

校准后最小二乘剂量 scale 为 134.5454。按 `50000 × 校准后权重和 / 10M` 得到的理论
绝对 scale 为 133.2748，两者只差 0.953%，明显小于原始权重的约 3.72%。若强制理论
绝对 scale，global/local 为 94.4213%/92.8750%；所以 95.4332% 是采用常见的全局
最小二乘剂量归一化后的形状比较，不能表述为绝对剂量 gamma 已超过 95%。

可复现工具为 `validation/scripts/calibrate_ct_weights_from_dij.py`，输出：

- `out/ct/tuning/dij_layer_calibration/calibration.json`；
- `out/ct/tuning/dij_layer_calibration/full_plan_weights_dij_layer_calibrated.csv`；
- 实际 10M 剂量和 gamma 位于
  `out/ct/full_plan_dij_layer_calibrated_10M/physical_compare/`。

按每层总积分做校准虽然能让积分闭合到 0.00003%，但预测 global/local gamma 只有
92.04%/90.18%，因此不采用。逐层高剂量响应校准直接依赖本病例 Dij，适合作为该
TPS/Dij 的 GPU 复算校准，不应冒充对其他病例或独立 TOPAS 的通用物理参数。

多层面可视化位于
`validation/figures/ct_full_plan_multiplanar_dose_comparison.png`：轴位、冠状位、
矢状位各选择高剂量包络内的 3 个层面，并列显示 physical dose、原始 GPU、校准 GPU
及校准 GPU 相对参考最大剂量的差值。对应的可复现绘图入口为
`validation/scripts/plot_ct_multiplanar_dose_comparison.py`。

### 8.6 CT 精度顺序优化与联合能层校准（2026-07-22）

按“核反应 → 材料 MCS → 多能层校准 → neutral → CT/数值收敛 → 绝对标定”顺序
继续检查。参考 `physical_dose.mhd` 来自 Geant4 11.1.3，而本机 TOPAS 4.2.p3 是
Geant4 11.3.2，因此本机新生成的材料核反应库只作为诊断，不能静默替换生产库。

用 TOPAS 自定义 `CarbonMaterialPropertiesNtuple` 实测质量辐射长度：G4_AIR、
G4_LUNG_ICRP、G4_WATER、G4_BONE_COMPACT_ICRU 分别为 36.6161、36.4162、
36.0830、30.4866 g/cm²。GPU 新增默认关闭的 `enable_ct_material_mcs`，打开后按
CT air/lung/water/bone 类选择质量辐射长度，旧配置关闭时逐路径保持原行为。相同
seed 的 calibrated-weight 1M A/B 为：

| 1M A/B | global 3%/3 mm | local 3%/3 mm | global 2%/2 mm |
|---|---:|---:|---:|
| 原水 MCS | **96.0244%** | **94.7207%** | 89.1874% |
| 材料 MCS | 95.9713% | 94.7131% | **89.2291%** |
| neutral first-interaction | 96.0130% | 94.7017% | 89.1761% |

材料 MCS 与 neutral 都没有稳定改善 3%/3 mm，故生产配置继续关闭。neutral 还使
IDD correlation 从 0.999143 降到 0.996675；这说明当前 11.1.3 water-derived
neutral package 不能直接当作患者材料的精度升级。

新增 `validation/scripts/optimize_ct_layer_weights_joint.py`：把 15 个 200k GPU
能层剂量同时拟合到 full-plan 高剂量区，并用对角 ridge 拉回先前独立逐层因子，
避免无正则解把多层推到 0.8/1.2 边界。ridge=1 的保守候选在相同 seed 1M 上把
global/local 由 96.0244%/94.7207% 提高到 **97.6882%/96.6497%**。10M 独立确认：

| 10M 指标 | 独立逐层校准 | 联合能层 ridge=1 |
|---|---:|---:|
| global 3%/3 mm | 95.4332% | **97.2523%** |
| local 3%/3 mm | 93.6898% | **96.2063%** |
| global 2%/2 mm | 88.4295% | **92.3672%** |
| high-dose NRMSE | 4.1601% | **3.5881%** |
| IDD correlation | **0.999108** | 0.999052 |
| wall time | 487.5 s | 521.2 s |

联合权重和为 26642.2555934，对应严格总粒子数 1,332,112,779.67，10M 理论 scale
为 133.211277967。强制该理论 scale 时 global/local 仍为 **96.4186%/95.0277%**，
积分差 +0.818%；最小二乘 scale 134.612037 仅高 1.05%。结果位于
`out/ct/full_plan_dij_layer_joint_calibrated_10M/`。
这是中间候选；后续 ridge 精调与最终 production 权重见 8.7。所有联合权重均可由
上述脚本和同一组 15 个能层剂量重新生成。

数值收敛方面，最大步长 0.5 → 0.25 mm 的相同 seed 1M 反而从
97.6882%/96.6497% 降到 97.3319%/96.3769%，且运行时间几乎不变；误差并非步长
主导，生产配置保持 0.5 mm。联合校准仍是病例/Dij 特定复算校准，不能冒充跨患者
通用物理修正；跨病例泛化仍需要与 Geant4 11.1.3 一致的多能量、多材料反应库。

本机 TOPAS 11.3.2 的干净 20k 骨/肺诊断进一步确认 final-state 材料效应不是零：
骨库严格解析为 49,954 interactions / 416,021 products，肺为
54,912 / 415,014；primary C12 的平均产物数分别为 11.2795 和 10.0100。每个
primary reaction 的平均 neutral 动能为骨 406.1、肺 377.3 MeV。两者都已编译成
101 个 4 MeV/u 能量箱，位于 `validation/results/diagnostic_g4_11_3_2/`。这支持以后
实现按 CT 材料选择 reaction/cascade final state，但在拿到同版本 11.1.3 数据前，
当前 production 仍只使用已经验证的 water-derived package 加材料相关 XS。

CT 重建侧再次核对：GPU 网格 `gpu(x,y,z)=patient(y,z,-x)`、spacing
0.5×2×0.5 mm、无插值，并使用与 `ct/topas/mc_ref/HUtoMaterialSchneider.txt`
相同的 Schneider HU 表；4.5525 g/cm³ 的最大密度来自该表 HU≥2996 的 titanium
section，不是 GPU 自行截断或异常插值。因此没有对 CT 几何/HU 映射做经验性改动。

### 8.7 Ridge 精调与最终 MU 标定

在 ridge=1 的 10M 确认后，又用相同 seed 实测 ridge=0.1 和 0.03。1M 指标为：

| ridge | global 3%/3 mm | local 3%/3 mm | global 2%/2 mm | NRMSE |
|---:|---:|---:|---:|---:|
| 1.0 | 97.6882% | 96.6497% | 92.7727% | 3.7850% |
| 0.1 | 98.3741% | 97.5328% | 94.0347% | 3.5714% |
| 0.03 | **98.4537%** | **97.7488%** | **94.2583%** | **3.5302%** |

ridge=0.03 的 10M 独立输运（509.4 s、无 overflow）确认：global/local 3%/3 mm
= **97.9080%/96.7938%**，global 2%/2 mm = **93.5534%**，high-dose NRMSE
= **3.3360%**。结果位于 `out/ct/full_plan_dij_layer_joint_ridge_0p03_10M/`。

未做全局 MU 标定时，ridge=0.03 权重和 25631.90335 给出的理论 scale 只有
128.15952，而该形状的 10M 响应 scale 为 134.72490；直接强制前者会把 gamma 降到
90.9460%/87.9671%。因此最后把所有权重统一乘 1.051228233：相对 spot 和能层 fluence
完全不变，固定 histories 的 GPU 剂量形状也不变，只校准 MU→ions 的全局换算。最终
权重和 26944.980473、总粒子数 1,347,249,023.635、10M 理论 scale
134.724902364，固化在 `data/ct_full_plan_weights_joint_ridge0p03_absolute.csv`。

需要明确：这个全局因子使用当前病例 10M response 标定，所以“理论 scale 与最佳
拟合一致”是校准结果，不是独立绝对剂量验证。对新病例应先固定该因子再做盲测；若
追求跨病例物理泛化，仍应优先补齐 Geant4 11.1.3 多材料 final-state 数据。

### 8.8 底层物理校准：基数角入口面粒子丢失修复

为避免继续对当前 full plan 过拟合，本轮校准只使用独立 Dij 单 spot：180/220
MeV/u 各 4 个 spot 作为校准集，200/240 MeV/u 各 4 个 spot 作为盲测集；允许拟合的
参数只有一个统一绝对响应因子。最初 14/16 个 spot 的 global 2%/2 mm 已在
97--100%，但第一射野 `TransX=TransZ=0` 的 200 和 220 MeV/u 中心 spot 都只有约
51%，总剂量和核反应数也恰好约为正常值的一半。

以下候选均被实测排除，没有写入生产物理参数：

- Geant4 11.1.3 水模拟合的统一 stopping-power scale 1.002269：1M 原 TPS 权重
  full-plan global 2%/2 mm 从 88.251% 降到 88.164%；
- water neutral local-kerma + electronic buildup：IDD correlation 从 0.999063 降到
  0.96843；单独 electronic buildup 也没有稳定改善；
- stopping power +1%、+2%、+4%，中心 spot 的射程和 gamma 继续恶化；
- CT 全 face clamp、最大步长、材料 MCS 和小幅患者坐标平移均不能恢复缺失剂量。

独立本机 TOPAS 4.2.p3/Geant4 11.3.2 用完全相同的 200 MeV/u 中心 spot、50k
histories 重跑后，与原 Dij 的积分剂量比为 1.00443、最佳 scale 0.99694、global
2%/2 mm 为 100%、IDD correlation 0.999996。这证明 Dij 没有重复累计，错误在 GPU
源入口处理。进一步交换中心/非中心 spot 的入射位置后，剂量减半仍跟随“精确基数角
姿态”而非解剖路径；把 90° 临时改成 89.999° 即恢复完整剂量。

根因是 `cos(90°)` 在横向基向量留下约 6e-17 的 z 分量。高斯位置采样后约一半
粒子的初始 z 成为极小负数；旧代码因 `abs(z) < 1e-6` 跳过入口投影，下一步逃逸
判断便立即丢弃这些粒子。修复位于 `src/transport_sycl.cpp`：legacy TPS-90 源只要
`abs(direction_z) > 1e-8`，就无条件沿粒子方向投影到 z=0。没有加入经验角度、空间
剂量修正或病例特定材料因子；新增 SYCL 回归测试直接复现 `cos(pi/2)` 残差，防止
以后再次丢失半数 histories。

修复后的独立 spot 结果：

| spot | 修复前 global 2%/2 mm | 修复后 | 剂量积分 GPU/Dij |
|---|---:|---:|---:|
| 451，200 MeV/u 中心（盲测） | 51.294% | **99.621%** | 1.04145 |
| 455，220 MeV/u 中心（校准） | 51.322% | **99.119%** | 1.04436 |
| 146，200 MeV/u 非中心对照 | 99.828% | **99.828%** | 1.03731 |

完整 16-spot 交叉验证只用校准集得到统一 GPU→Dij 响应 0.974329393。固定该值后：

| split | global 2%/2 mm（均值） | local 2%/2 mm（均值） | 最低 global |
|---|---:|---:|---:|
| 180/220 MeV/u 校准集 | 99.412% | 97.345% | 98.573% |
| 200/240 MeV/u 盲测集 | **99.336%** | **95.879%** | **97.571%** |

最后用完全未做能层/ridge 修正的原始 TPS optimizer 权重重跑 1M full plan。自由 scale
结果由修复前的 global/local 3%/3 mm = 95.361%/93.944%、global 2%/2 mm =
88.251%，提高到 **99.754%/98.992% 和 98.257%**；local 2%/2 mm 为
**95.839%**。更严格的盲绝对标定用 `sum(weights) * 50000 / histories` 再乘上述
single-spot 响应，预测 scale=1263.102831；full-plan 自由拟合为 1262.637844，差仅
0.0368%。固定预测 scale 后 global/local 2%/2 mm 仍为 **98.234%/95.835%**，
global/local 3%/3 mm 为 **99.742%/98.984%**。

因此先前 ridge=0.03 的 full-plan 能层权重是在该粒子丢失 bug 存在时标定的，修复后
不再有效，也不应作为生产精度方案。生产应恢复原 TPS 权重并保留本节的入口面修复；输出和复现脚本为
`config/beam_ct_physics_baseline_1M.yaml`、
`validation/scripts/validate_ct_spot_physics.py`，结果位于
`out/physics_calibration/spot_cross_validation/` 和
`out/physics_calibration/baseline_fixed_blind_scale/`。

10M 最终统计复验使用更新后的 `config/beam_ct_full_plan_reduced.yaml`，运行
536.51 s（18,639 histories/s），31,611,408 个带电次级全部输运，primary/cascade
queue overflow 均为 0。自由 scale=126.348104；独立 spot 校准预先给出的固定
scale=126.310283，仅低 0.0299%。固定 scale 的 full-plan 指标为 global/local
2%/2 mm = **97.582%/95.058%**，global/local 3%/3 mm =
**99.200%/98.067%**，IDD correlation=0.999030，high-dose NRMSE=2.381%。
1M 的 global 2%/2 mm 偶然更高（98.234%）是低统计噪声对离散 gamma 搜索的影响，
最终精度判断应采用本次 10M 结果。

### 8.9 CT 直接下采样到 2 mm 的性能/精度实测

使用 `validation/scripts/downsample_ct_grid.py` 将 TPS-90 输运 CT 从
505×35×417、0.5×2×0.5 mm 下采样为 126×35×104、2×2×2 mm。密度采用体积
平均以保持完整 block 的质量，材料 section 使用密度加权众数；为保证 CT 和 dose
scorer 都从 z=0 开始，在横向高端和束流远端各裁掉一个多余的 0.5 mm 体素。体素数
从 7,370,475 降到 458,640（16.07 倍），CCTG 文件从 35.1 MiB 降到 2.19 MiB。

同一 NVIDIA TITAN RTX、同一随机种子和完整物理模型的 1M full-plan A/B：细 CT 为
64.260 s（15,562 histories/s），2 mm CT 为 63.842 s（15,664 histories/s），端到端
仅快 **0.65%**。主粒子核由 3.526 s 降至 2.894 s（快 21.8%），但带电次级粒子核
仍约 60.4 s，约占总核时间 95%，所以 CT 边界减少无法转化为明显的总加速。显存估计
仅从 857 MiB 降到 827 MiB。

对 `ct/code/physical_dose.mhd` 自由统一 scale 后，细/粗 CT 的 global 2%/2 mm 为
98.257%/98.124%，local 2%/2 mm 为 95.839%/95.433%，global 3%/3 mm 为
99.754%/99.731%，local 3%/3 mm 为 98.992%/99.049%；IDD correlation 分别为
0.999030/0.998960。粗 CT 的 IDD 峰移动一个 2 mm bin。结论是：仅为性能不应替换
细 CT；应继续优化占主导的次级粒子/cascade 输运。复现配置为
`config/beam_ct_downsampled_2mm_1M.yaml`。

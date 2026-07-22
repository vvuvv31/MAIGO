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

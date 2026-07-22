# Grok 工作记录：CT 剂量 GPU vs TOPAS / physical_dose

更新时间：2026-07-22  
工作目录：`/home/v/project/MAIGO`  
目的：记录已完成修复与当前度量，便于续作。

---

## 0. 本次会话（2026-07-22）摘要 — GPU vs TOPAS prelim

### 0.1 目标调整

- 全 plan TOPAS MC 过慢 → **先杀全 plan**，改 prelim：
  1. 单 pen（最高权 spot，50k hist）
  2. 同能量层等权 20 spots（100k hist）
- 比较要求：**不只 IDD**，还要 **横向剂量**；gamma 用 **local**（小 FOV 不适合 global 作为主指标）。

### 0.2 根因与修复（几何）

| 问题 | 根因 | 修复 |
|------|------|------|
| IDD 峰差 10–16 mm、横向对不上 | TOPAS 剂量峰对应 **+patient-X 束、Y≈−29**；GPU 用 `RotZ=−90` + **xneg CT** 走了对侧解剖 | prelim 配置：`spots_patient_rot_z_deg: 90`，`ct_grid_file: patient_ct_tps_90.bin`（非 xneg）；比较 `flip_x=F, flip_y=F` |
| patient-Z COM 系统偏 ~+3.6 mm | 倾斜束 `uy_z≠0` 时，`origin+x·ux+y·uy` 离开 z=0 入口面；z>0 粒子优先计分 → COM 偏置 | `src/transport_sycl.cpp`：出生后沿方向 **投影回 z=0** |
| SYCL batch 结构体字段 | 曾怀疑 `PrimarySpotBatchEntry` 设备布局；后证实根因是入口投影 | 仍改为连续 `floats[]` 布局，降低设备 ABI 风险 |

### 0.3 物理细节修复

| 项 | 改动 |
|----|------|
| Bethe 参考 `I_water` | 75 → **78 eV**（`include/carbon/ct_grid.hpp`，与 G4_WATER / SP 表一致） |
| `straggling_scale` | prelim **1.0**（去掉 1.2 人为放大远端） |
| `ct_stopping_power_scale` | 新配置项；prelim **1.008**（IDD 峰与 TOPAS 对齐） |
| 核反应残余能 | 未进入次级列表的 residual 在反应点 **本地沉积**（重残核热近似） |
| Bohr straggling | API 支持材料 `za_rel`（默认水） |

消融：MSC 开关几乎不动 γ/σ；straggling 0.9–1.15 对 γ 影响 <1%；横向宽主要来自源发射度。

### 0.4 度量（GPU vs TOPAS MC DoseToMedium，scale-only）

指标与 CSV：`validation/results/ct/prelim/`  
比较脚本：`validation/scripts/compare_gpu_topas_prelim.py`（IDD + 横向 σ/COM/FWHM/剖面 + **local** 3%/3mm 主指标）

| 指标 | 单点（修复前→后） | 等权层（修复前→后） |
|------|-------------------|---------------------|
| 3D cos | 0.62 → **0.994** | 0.77 → **0.994** |
| IDD 相关 | — → **0.9996** | — → **0.9998** |
| 峰位差 | −16 mm → **0 mm** | −10 mm → **0 mm** |
| **local γ 3%/3mm** | 16% → **~50%** | 22% → **~55%** |
| local γ 3%/5mm | — → **~74%** | — → **~79%** |
| 近峰 σ 比 GPU/TOPAS | 1.79 → **~1.05** | 1.27 → **~1.01** |
| FWHM Y/Z | 差 → **与 TOPAS 基本一致** | 同左 |
| COM Δz | +3.8 mm → **~0.05 mm** | +3.7 mm → **~0** |

### 0.5 复现 prelim

```bash
export ONEAPI_DEVICE_SELECTOR=opencl:gpu
./build/perf-make/carbon_mc --config config/beam_ct_prelim_single.yaml
./build/perf-make/carbon_mc --config config/beam_ct_prelim_layer_eq.yaml

# TOPAS（需 shellScripts/topas + ct/topas 工作目录）
# cd ct/topas && topas mc_ref/run_single_spot.txt
# cd ct/topas && topas mc_ref/run_layer_eq.txt

python3 validation/scripts/compare_gpu_topas_prelim.py \
  --gpu-mhd out/ct/prelim_single/dose.mhd \
  --topas-csv ct/topas/output/prelim_single_dose.csv \
  --output-dir validation/results/ct/prelim --tag single \
  --no-flip-x --no-flip-y

./build/perf-make/carbon_tests   # 应 All carbon_tests passed
```

### 0.6 仍未解决

1. **local γ 3%/3mm 仍 ~50%**（cos 已 0.994）：高剂量区相对误差 RMS ~9%。更可能是  
   - 水中反应包用于 CT 材料 vs G4 全材料核模型  
   - 次级/级联空间分布差  
   - TOPAS 50k 统计底噪  
   而非再拧 scale / flip。
2. **全 plan TOPAS MC** 未完成（过慢；需在 prelim 满意后再开）。
3. **vs physical_dose（matRad）** 仍 ~65% global gamma 历史基线；与 MC–MC prelim 是不同参考。
4. **RotZ 符号双轨**：  
   - vs **physical COM**：历史上 `RotZ=−90` + xneg + match `flip_x=T`  
   - vs **TOPAS prelim MC 剂量峰**：有效等价 **`RotZ=+90` + 非 xneg**  
   需在 TOPAS 侧 dump 等中心/入口，统一符号，避免配置分叉。

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

`prepare_ct_grid.py`：origin = first_center − 0.5·spacing。  
`spots_ct_axis_min_mm: -104.25`。

### 2.3 `transform_tps_90`

World → patient（−Trans，−RotZ）→ GPU `(x,y,z)=(patient y, patient z, depth along ±X)`。  
入口在 GPU z=0，`uz_z>0`。

### 2.4 入口面 emittance 投影（2026-07-22）

见 §0.2：出生点投影回 z=0。

### 2.5 核残余本地沉积 + mass-SP（2026-07-22）

见 §0.3。

### 2.6 匹配 / prelim 比较脚本

- `match_gpu_to_physical_dose.py`：GPU→patient、scale、gamma  
- `compare_gpu_topas_prelim.py`：IDD + 横向 + **local** gamma  

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
ct_grid_file: ct/grid/patient_ct_tps_90.bin
spots_patient_rot_z_deg: 90.0
straggling_scale: 1.0
ct_stopping_power_scale: 1.008
```

### vs physical_dose 历史生产

```yaml
# 曾用 xneg + RotZ=-90 + match flip_x=True
ct_grid_file: ct/grid/patient_ct_tps_90_xneg.bin
spots_patient_rot_z_deg: -90.0
```

---

## 5. 关键文件

| 路径 | 内容 |
|------|------|
| `src/transport_sycl.cpp` | scorer–CT 对齐；入口 z=0 投影；核残余本地沉积；mass-SP scale 进 LUT |
| `include/carbon/ct_grid.hpp` | I_water=78 eV |
| `include/carbon/straggling.hpp` | Bohr + 可选 za_rel |
| `include/carbon/transport_config.hpp` | `ct_stopping_power_scale`；`PrimarySpotBatchEntry` floats[] |
| `src/config.cpp` | 解析/校验 scale |
| `src/topas_spots.cpp` | tps_90 变换 |
| `src/device.cpp` | cpu OpenCL 失败时 GPU fallback（测适用） |
| `tests/carbon_tests.cpp` | 几何/mass-SP/charged-origin 闭合 |
| `config/beam_ct_prelim_*.yaml` | prelim 单点/等权层 |
| `validation/scripts/compare_gpu_topas_prelim.py` | local γ + 横向比较 |
| `validation/results/ct/prelim/*_metrics.json` | 最新度量 |

---

## 6. 建议续作顺序

1. 统一 **RotZ / xneg** 与 TOPAS 自证 dump（消除 physical 轨 vs MC 轨分叉）  
2. 提高 TOPAS prelim 统计（≥2e5–5e5）再评 local γ 噪声底  
3. 组织相关核反应包 / 次级，抬 local γ 向 95%  
4. 全 plan TOPAS MC 参考（在 2–3 满意后）  
5. 再谈 vs `physical_dose` 的 95% 是否仍合理（matRad vs MC）

## User

详细分析/mnt/sdb/wuwei/fred文件夹，逆向/mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/Fred.x，然后详细分析当前仓库和fred有哪些物理模型上的问题，以及如何改进

## Assistant

先摸清 `fred` 目录结构、`Fred.x` 的二进制特征，以及当前仓库的物理实现，再对照两边的模型差异。

## Tools

- ListDir: /mnt/sdb/wuwei/fred
- ListDir: .
- Execute: file /mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/Fred.x; ls -la /mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/; ls -la /mnt/sdb/wuwei/fred/ (Inspect Fred.x binary and fred layout)
- Execute: file /mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/Fred3.76.0.x /mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/libFredGPU.so /mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/libFred.data
echo '==== strings physics keywords Fred.x ===='
strings /mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/Fred3.76.0.x | rg -i 'fragment|inelastic|elastic|hadron|nuclear|stopping|straggl|emittance|Moliere|highland|Bethe|Bohr|Landau|Vavilov|Goudsmit|MCS|LET|delta.?ray|neutron|photon|carbon|C12|QMD|BIC|INCL|Fermi|evaporat|cascade|cross.?section|mean.?free|CSDA|Bragg|Coulomb|ionization|excitation|nuclear.?interact' | sort -u | head -400
echo '==== nm demangle sample ===='
nm -C --defined-only /mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/Fred3.76.0.x 2>/dev/null | rg -i 'Fred|Particle|Nuclear|Fragment|Transport|Hadron|Ion|GPU' | head -80
echo '==== GPU so strings ===='
strings /mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/libFredGPU.so | rg -i 'kernel|opencl|cuda|clEnqueue|__kernel|fragment|nuclear|MCS|straggl' | sort -u | head -200
 (Extract physics-related strings from Fred binary)
- Read: /mnt/sdb/wuwei/MAIGO/docs/FRED_Carbon_Fragmentation_Model.md (1-200)
- Read: /mnt/sdb/wuwei/MAIGO/docs/TOPAS_GPU_Physics_Model.md (1-200)
- Read: /mnt/sdb/wuwei/MAIGO/include/carbon
- Execute: strings /mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/Fred3.76.0.x | grep -iE 'fragment|inelastic|elastic|nuclear|stopping|straggl|Highland|Moliere|Bethe|Bohr|Landau|Vavilov|MCS|LET|delta|neutron|photon|carbon|C12|QMD|BIC|INCL|Fermi|evaporat|cascade|cross.section|mean.free|Bragg|Coulomb|ionization|Kox|Takechi|GANIL|FLUKA|ICRU|ENDF|f_mcs|texture' | sort -u | head -500
echo '==== NM classes ===='
nm -C --defined-only /mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/Fred3.76.0.x 2>/dev/null | grep -E ' T | t ' | grep -iE 'Fred|Particle|Nuclear|Fragment|Transport|Hadron|Ion|GPU|Scatter|Elastic|Inelastic|MCS|LET' | head -150
echo '==== data file header ===='
xxd /mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/libFred.data | head -5
file /mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/libFred.data
strings /mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/libFred.data | head -80
 (Strings/nm/data dump of Fred binary)
- Read: /mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/plugin/FredXfunc.h (1-150)
- ListDir: include/carbon
- Read: /mnt/sdb/wuwei/MAIGO/docs/FRED_Carbon_Fragmentation_Model.md (201-311)
- Execute: python3 - <<'PY'
import struct, os
p='/mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/libFred.data'
with open(p,'rb') as f:
    data=f.read()
# parse as sequence of (u32 len, bytes)
off=0
files=[]
while off+8<=len(data):
    npath=struct.unpack_from('<I',data,off)[0]
    off+=4
    if npath==0 or npath>10000 or off+npath+4>len(data):
        break
    path=data[off:off+npath]
    try:
        sp=path.decode('ascii')
    except:
        break
    off+=npath
    nsize=struct.unpack_from('<I',data,off)[0]
    off+=4
    if nsize>len(data)-off:
        print('bad size',sp,nsize)
        break
    files.append((sp,nsize,off))
    off+=nsize
print('nfiles',len(files),'end',off,'total',len(data))
for sp,nsize,_ in files:
    print(f'{nsize:10d}  {sp}')
PY
echo '==== GPU so kernels ===='
strings /mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/libFredGPU.so | grep -iE 'OpenCL|CUDA|clCreate|__kernel|track|MCS|Vavilov|fragment|C12|inelastic' | sort -u | head -120
echo '==== C12.txt from data if extractable ===='
# extract C12.txt preview via python
python3 - <<'PY'
import struct
p='/mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/libFred.data'
with open(p,'rb') as f:
    data=f.read()
off=0
while off+8<=len(data):
    npath=struct.unpack_from('<I',data,off)[0]
    off+=4
    path=data[off:off+npath]; off+=npath
    nsize=struct.unpack_from('<I',data,off)[0]; off+=4
    blob=data[off:off+nsize]; off+=nsize
    sp=path.decode('ascii',errors='replace')
    if sp.endswith('data/nuc/C12.txt') or 'C12.txt' in sp:
        print('FOUND',sp,nsize)
        print(blob[:2500].decode('latin1','replace'))
        break
PY
 (Parse libFred.data archive and C12 nuclear table)
- Read: /mnt/sdb/wuwei/MAIGO/include/carbon/inelastic.hpp (1-66)
- Read: /mnt/sdb/wuwei/MAIGO/include/carbon/fred_table1.hpp (1-80)
- Read: /mnt/sdb/wuwei/MAIGO/issue.md (1-80)
- Search: fred|Kox|Highland|cascade|elastic|Vavilov
- Search: fred_paper|topas_inclxx|energy_straggling_model|mcsFac|enable_primary_elastic
- Read: /mnt/sdb/wuwei/MAIGO/include/carbon/multiple_scattering.hpp (1-71)
- Read: /mnt/sdb/wuwei/MAIGO/include/carbon/transport_config.hpp (1-100)
- Execute: strings /mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/Fred3.76.0.x | grep -E 'mcsMode|1G-H|2GR|3G|Urban|Vavilov|Landau|straggling|nuclear inelastic|fragmentation|Kox|Takechi|event library|secondary fragmentation|trackingDebug' | sort -u | head -80
echo '==== GPU OpenCL source names ===='
strings /mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/libFredGPU.so | grep -E '\.cl$|kernel void|__kernel' | sort -u | head -40
echo '==== version banner ===='
strings /mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/Fred3.76.0.x | grep -A 40 'Fast paRticle\|FRED version\|3.76' | head -50
 (More Fred physics mode strings and version banner)
- Read: /mnt/sdb/wuwei/MAIGO/include/carbon/detail/fred_fragmentation_data.hpp (1-80)
- Search: doDiscreteInteraction_NuclearFragmentation|eventLibrary|SamplFrag|Kox
- Read: /mnt/sdb/wuwei/MAIGO/src/cross_section.cpp (1-80)

## Assistant

结论先说：**FRED 3.76 是一套“凝聚历史 + 离散核反应”的 OpenCL GPU 蒙卡**；碳物理的核心是 **95 MeV/u 薄靶数据驱动碎裂 + 事件库 + Vavilov 涨落 + 2GR MCS**。当前仓库 **MAIGO 的生产路径是 TOPAS/Geant4 INCLXX 编包代理**，`fred_paper` 只覆盖论文那一套唯象核模型，电磁和核弹性仍明显弱于 FRED。两边的主要物理缺口不在“会不会碎裂”，而在 **能损涨落、MCS 形状、弹性、能量相关产额、δ 电子和中子**。

---

## 1. `/mnt/sdb/wuwei/fred` 是什么

目录很薄，实际产品在一个版本树里：

| 路径 | 作用 |
|---|---|
| `curr` → `fred_3.76.0_Linux_gcc_14.2.1` | 当前安装 |
| `fred` → `curr/scripts/fred.py` | Python 入口 |
| `Fred.x` → `Fred3.76.0.x` | **未 strip 的 x86-64 ELF**（约 43.7 MB，带 debug_info） |
| `libFredGPU.so` | **OpenCL** 追踪（不是 CUDA） |
| `libFred.data` | 打包物理资产（约 15.7 MB） |
| `plugin/` | C++ 步进插件 API（`Step` A/B、入队、材料/体素） |
| `scripts/` | 配置、DICOM、MHD 图、GPU 调参 |

`libFred.data` 是长度前缀的资源包，能看到的关键资产包括：

- `data/dEds/dEds_coremat_v03.bin`：材料–粒子 stopping-power LUT  
- `data/mat/HU2Materials.txt`：HU→材料  
- `data/fluc/Vavilov_v01.lut`：能损涨落  
- `data/mcs/*_2GR.txt`、`MCSTable_3G.h`：MCS 形状表  
- `data/nuc/C12.txt`、`N14.txt`、`O16.txt`、`P31.txt`、`CA40.txt`：核截面  
- `libEvents_C12_{H1,C12,N14,O16,P31,S32,Ca40}` 和 `libEvents_p_*`：**相关碎裂事件库**  
- `data/activation/Geant4.10.06.03/BIC/...`：质子活化（Geant4 BIC）  
- `data/nuc/neutron/nX_Elastic_*.bin`：中子弹弹性  

二进制里的开发时间线（字符串）：

- 2015 MCS；V. Patera 非弹性  
- 2016 Vavilov/Landau 涨落  
- **2017 Mazzini：$^{12}\mathrm{C}$ 数据驱动碎裂**（对应 2022 论文）  
- 2018 轻离子 MCS 校验  
- 2019 质子–离子核弹性  
- 2021 质子活化包  
- 2022 LET 谱 scorer  
- **2024 中子弹弹性；on-the-fly δ 射线（Ionization Detail）**

---

## 2. 逆向 `Fred.x` / `libFredGPU.so`：实际物理管线

插件头 `FredXfunc.h` 把一步拆成 **A→B**：位置、方向、动能、`nlambda`（光学深度）、材料 $Z,A,X_0$、HU、`pushParticle`。这就是凝聚历史步进。

GPU kernel 主循环（`gpuTrackParticles`）逻辑是：

1. `doContinuousStep`：dE/dx LUT +（可选）Vavilov 涨落 + MCS  
2. 用 `nlambda = -ln ξ` 抽核自由程  
3. `doDiscreteInteractionEvent`：  
   - `doDiscreteInteraction_NuclearElastic`  
   - `doDiscreteInteraction_NuclearInelastic`  
   - **碳专属** `doDiscreteInteraction_NuclearFragmentation`（对 C12 打 H/C/O 时走 **event library**）

### 2.1 连续能损

- 预计算 **mass stopping power 矩阵**（材料 × 粒子），GPU 用纹理插值。  
- 不是运行时 Bethe–Bloch。  
- 可选 **Ionization Detail / on-the-fly δ 射线**（`DeltarayProductionThreshold`，例如 100 keV）。默认临床剂量仍是 unrestricted condensed loss。

### 2.2 能损涨落（FRED 比 MAIGO 完整得多）

- `EnergyStraggling.cpp` + `Landau.cpp` + `Vavilov_v01.lut`  
- GPU：`dT = energyStraggling(..., vavilovLUT)`  
- $\kappa$–$\beta^2$ 二维 LUT：`k ∈ [0.01,10]`，`β² ∈ [10^{-4},1]`  
- 低 $\kappa$ Landau，中等 Vavilov，高 $\kappa$ 趋近高斯  

**这是 FRED 峰形/峰宽对齐 FLUKA 的关键，不是碎裂模型。**

### 2.3 MCS

字符串给出明确档位（**默认 mode 5**）：

| mode | 模型 |
|---|---|
| 0 | 单高斯 Highland（1G-H） |
| 5 | **双高斯 + Rutherford 尾（2GR）**，有 `w1,w2,w3,σ_c,σ_t,b,m` 表 |
| 6 | Urban（`MCS_Urban.cpp`） |
| 另有 | 3G 三高斯、Rossi–Greisen + Fippel |

GPU Highland 前因子有两套：`13.6 * mcsSigmaCRescale` 和 `12.9*1.04 * mcsFac0`。论文里的 $f_{\mathrm{mcs}}\approx 1.29$–$1.43$ 就是这个 rescale。另有 **侧向位移** `lMCSLateralDisplacement`（不只偏转方向）。

### 2.4 核弹性

- **只对氢靶（及质子–离子互易）显式做**；重靶弹性基本忽略（与论文一致）。  
- 运动学是质心系 ↔ 实验室系，出射弹核 + 反冲质子继续跟踪。  
- 2024 增加 **中子–核弹性 ICDF**。

### 2.5 碳非弹性 / 碎裂（论文模型 + 事件库）

两套并存：

1. **解析/表格通道**（论文）：C–C 唯象截面  
   $\sigma(C,C,E)=(1-e^{-E/E_c})(p_0+p_1 E+e^{p_2-p_3 E})$，再用 **Kox** 缩放到任意 $(N_p,N_t)$；H 靶用 ICRU。  
   18 种同位素、牛顿法把 GANIL **包含产额** 反演成可独立抽样概率；$f(E,\theta)=$ 高斯弹核碎片 + 指数靶碎片；能量按 $E/95$ 和 $k=c(1-R), c=0.4$ 缩放；角按 $p_\perp$ 守恒（p/n 不缩）。守恒失败 GPU 内重抽。

2. **3.76 事件库**（二进制里已落地）：  
   `libEvents_C12_H1/C12/O16/...`  
   OpenCL 里：`if (stp->iray==C12 && targetID==H1/C12/O16 && libEvents offset>0)` 直接抽 **相关末态**（A,Z,Ek,方向余弦），再按入射能量缩放动能。  
   这比独立抽样更接近 FLUKA 的多体关联。

论文写 **默认关闭次级再碎裂**；3.76 GPU 代码里碎片仍有 `generation++` 和独立 `nlambda`，但临床默认仍倾向只追一代带电碎片。

### 2.6 其它

- Scorer：`Edep, dose, LETd, RBE, activation, track, counts, dose-to-water`  
- HU→密度/材料；核组成按元素 $w_i, x_i$ 加权抽靶核  
- 跟踪开关：`[dEdx, fluc, mcs, nuc_el, nuc_inel]`  
- OpenCL 1.0/1.2/2.0；LUT 走 `image3d_t` 纹理  

---

## 3. MAIGO 现在实际在算什么

仓库目标是复刻该文的 GPU 碳剂量，但 **生产 CT 路径已经偏成 TOPAS 代理**：

| 过程 | FRED 3.76 | MAIGO 生产（CT `best`） | MAIGO `fred_paper` |
|---|---|---|---|
| dE/dx | 材料×粒子 LUT | Geant4 11.3.2 CSV + 有效电荷回退 | 同左 |
| 涨落 | **Vavilov/Landau LUT** | **截断高斯 / Bohr+$T_{\max}/\beta^2$**；包涨落目前几乎只给质子水 | 同生产 |
| MCS | **2GR + $f_{\mathrm{mcs}}$ + 侧向位移** | **单 Highland 13.6**；CT 默认可关材料 $X_0$ | 同生产 |
| 弹性 | 原生 H 靶 CoM | 可选 ELPKG，CT **关** | 论文仅 H 靶，实现不完整 |
| 非弹截面 | C–C 拟合 + Kox / ICRU | Geant4 宏观 $\Sigma(E)$ 表 | 应对齐论文公式 |
| 末态 | GANIL 反演 **或 event library** | **INCLXX 相关包**，按能量箱选事件再比例缩放 | Table1 + 2D $f(E,\theta)$ + $k,c=0.4$ |
| 级联 | 默认基本关 | **`maximum_cascade_generations: 2`** | 应关二次碎裂 |
| 中子/γ | 弹性 ICDF；中子不完整输运 | 生产 **`enable_neutral_transport: false`** | kerma 强制 0 |
| δ 电子 | 可选 on-the-fly | 无显式电子；剂量 unrestricted | 同生产 |
| 衰变 | 无完整放射性链 | 无 | 无 |

`issue.md` 已记录：相对 TOPAS 水箱 200 MeV/u，物种剂量、包能量箱、质子列、LET 峰仍开着。

---

## 4. 物理模型上的具体问题（按对剂量的影响）

### 4.1 身份分裂（最严重的设计问题）

MAIGO 同时追两套“正确答案”：

- **论文/FRED**：GANIL + Kox + 95 MeV/u 外推，对标 **FLUKA**  
- **生产**：INCLXX 包，对标 **TOPAS 4.2 / G4 11.3.2**

INCLXX 与 FLUKA/QMD 在 B/Be/Li 产额、靶残核、中子上系统不同。用 400 MeV/u 长程包去填 200 MeV/u 箱，已造成 B/Be/Li fluence 约 −30/−26/−20%（`issue.md` #2）。这不是 GPU 抽样 bug，是 **参考模型 + 编包策略**。

FRED 3.76 反而更干净：碳通道固定为数据驱动 + 事件库，不假装在 GPU 上跑 INCL。

### 4.2 能损涨落过粗 → Bragg 峰高/峰宽

FRED：按步 $\kappa$ 切 Landau/Vavilov/Gauss。  
MAIGO：步内截断高斯，主粒子可选固定块；次级涨落 CT 默认关。

后果：峰高、远端半高宽、SOBP 坪纹会偏。质子水已有 `packaged_fluctuation` 证据（peak 误差 <1%），**碳还没有等价 Vavilov/G4 涨落包**。`dose_output_scale: 0.982` 是在用总剂量标量盖峰形误差。

### 4.3 MCS 只有 core 高斯，没有核反应晕的电磁部分

FRED 默认 **2GR**：窄核 + 宽核 + Rutherford 大角。Highland 只是 mode 0。  
MAIGO：投影角 $\theta_{\mathrm{rms}}$ 单高斯，CT 还可把 $X_0$ 全当成水。

后果：

- 轴上剂量偏高、halo 偏低（电磁大角不够）  
- 骨/气腔界面散射错误  
- 再叠加上面“没有 H 靶核弹性”，侧向双高斯的第二分量更缺  

论文 $f_{\mathrm{mcs}}$ 是对 FLUKA 无核反应铅笔束标定的；MAIGO 既没有 2GR，也没有这套 ion-specific scale。

### 4.4 核弹性缺失 → 质子物种和侧向尾

FRED：C+p 弹性是 halo 和反冲质子的重要来源。  
MAIGO：CT 关弹性；`issue.md` 明确 **不要用弹性去“修”B/Be/Li**，但质子列去 δ 后仍约 +3.8%，弹性/中子次级质子/π⁻ 丢弃都还没拆开。

### 4.5 碎裂末态：独立抽样 vs 相关事件；能量外推

`fred_paper` 已有 Table1、牛顿反演、2D 高斯+指数、$k=c(1-R)$，方向对。缺口：

1. FRED 3.76 对 C+H/C/O 用 **event library**；MAIGO paper 模式仍是 **独立同位素抽样 + 守恒重抽**，多体关联（$3\alpha$、$p+n+^{10}B$）比事件库差。  
2. 产额钉在 95 MeV/u，治疗能区只缩 $E,\theta$，不缩 inclusive yield。FRED 论文也如此；FLUKA 产额随能量变。这是远端尾巴在 300–400 MeV/u 的系统误差源。  
3. 生产 INCL 包按能量箱选事件再 **比例缩放动能**，箱太粗就会出现 #2 那种 400 包打 200 箱。  
4. 级联代数：FRED 默认关二次碎裂；MAIGO `generations=2` 会多出轻碎片和中子血统质子。对剂量尾巴有时有帮助，对“复刻 FRED”则是过建模。  
5. 18 核素集合不含 $\pi$、更重靶残核；INCL 包里的 π⁻ 被丢掉（Z=A=0）。

### 4.6 中性粒子

FRED 有中子弹弹性表，但仍不是全中子输运。  
MAIGO 生产中性全关、kerma 0 → 远端和体外中子剂量、以及 n→p 反冲都没有。若对标 TOPAS full list 会低；若对标 FRED 论文剂量，中子本就不是主项。

### 4.7 δ 电子与 scorer 定义

FRED 可选 δ；临床默认仍把电离记在离子上。  
MAIGO 同样 unrestricted；相对 TOPAS `EnergyDeposit` 时初级 C 会虚高约 7%，去 δ 后才接近。**总剂量可以对，物种列不能拿来调核模型。** LET 用 restricted electronic SP 表，和 dose 账不一致，所以 LET 峰仍开着。

### 4.8 截面与材料

FRED：元素级 $\mu/\rho=N_A\sum w_i\sigma_i/A_i$，水中 H/O 用数据+Kox。  
MAIGO：宏观 $\Sigma(E,m)$ 表 + Schneider 四类材料；CT 材料 reaction package 经常回退到水包。骨/肺非弹性产物会错。

### 4.9 工程/守恒

- 队列溢出当 local heat：能量闭合，空间分布错。  
- `resample_failed` / `model_unassigned`：paper 模式守恒失败时能量进 local，峰前会糊。  
- 没有 `g4radioactivedecay`：对物理剂量几乎无影响，对活化/生物标记有影响；FRED 活化包主要是质子 BIC。

---

## 5. 建议改法（按优先级）

**先定一个参考，不要两套同时当生产。**

- **路径 A（复刻论文/FRED）**：电磁对齐 FRED，核用 `fred_paper` + 最好导入 3.76 的 `libEvents_C12_*`。对标 FLUKA/FRED 水箱，而不是 INCLXX。  
- **路径 B（对齐 TOPAS）**：保留 INCL 包，但按 **反应能量分层编包**（每个 20–40 MeV/u 箱用该能量的薄靶/短程束），打开弹性包、材料包、中性至少 kerma。对标 TOPAS，不要再用 FRED Table1 调 B/Be。

### 电磁（两条路径都该做）

1. **碳 Vavilov 或 G4 涨落包**（按 $Z^2, \beta, x\rho$），次级默认也开；去掉或冻结 `dose_output_scale`。  
2. **MCS**：至少质子/碳 2GR 或 Urban；CT 打开 `enable_ct_material_mcs`；标定 ion $f_{\mathrm{mcs}}$（无核反应 vs TOPAS/FLUKA）。加侧向位移。  
3. Scorer：unrestricted 总剂量 vs restricted 物种列分开；LET 与 dose 用同一套 cut。

### 核（路径 A）

1. 从 `libFred.data` 抽出 `libEvents_C12_H1/C12/O16`，GPU 上按靶核抽相关事件，再做 FRED 的 $E$ 缩放和 $p_\perp$ 角缩放。  
2. 独立抽样只作 fallback（N/P/Ca 或库外靶）。  
3. 截面改回 C–C 公式 + Kox + ICRU-H，不要 G4 $\Sigma$。  
4. 只对 H 做弹性 CoM（公式已在论文文档里）。  
5. 二次碎裂默认关，与 FRED 一致。  
6. 用 `config/beam_95MeVu_fred_thin_*` 锁产额，再用 100–400 MeV/u 水箱锁峰位/尾巴。

### 核（路径 B）

1. 重编 primary/cascade：**分能量、分材料**（水/骨/肺/气），禁止 400 MeV/u 事件服务 200 MeV/u 箱。  
2. 打开 primary elastic ELPKG（H 靶为主）。  
3. π 和 Z=0 产物不要静默丢，至少进 local 或中性。  
4. 中子：先 `first_interaction`/kerma，再考虑弹性反冲质子（这是 p 列剩余误差的候选）。  
5. `maximum_cascade_generations` 用灵敏度扫 0/1/2，对 TOPAS 尾巴选，而不是固定 2。

### 不要做的

- 用总剂量 scale 掩盖峰形。  
- 用弹性“修”B/Be/Li。  
- 在 GPU 上实现完整 INCLXX/QMD。  
- 混用 FRED Table1 产额和 INCL 动能谱。

---

## 6. 和二进制的对应关系（便于下一步动手）

| FRED 符号 | 含义 | MAIGO 应对 |
|---|---|---|
| `doContinuousStep` + SP 纹理 | 连续能损 | 已有 CSV 插值 |
| `energyStraggling` + `vavilovLUT` | 涨落 | 缺，优先补 |
| `getMCSParameters_2GR` | 默认 MCS | 缺 |
| `mcsFac0` / `MCSSIGMACRESCALEFACTOR` | $f_{\mathrm{mcs}}$ | 缺 |
| `doDiscreteInteraction_NuclearElastic` | H 弹性 | ELPKG / 解析 CoM |
| `doDiscreteInteraction_NuclearFragmentation` + `libEvents_C12_*` | 相关碎裂 | paper 独立抽样；应改事件库 |
| `nlambda` | 核自由程 | 已有 $1-e^{-\Sigma\Delta s}$ |
| `lDeltarayOnTheFly` | δ | 仅 LET 表近似 |
| `nX_Elastic_*.bin` | 中子弹弹性 | 生产关 |

`Fred.x` 未 strip，下一步若要逐函数对齐，可以对 `InelasticNuclearInteractions.cpp`、`EnergyStraggling.cpp`、`MCS.cpp` 对应符号做反汇编；GPU 侧 `libFredGPU.so` 里 OpenCL 源码字符串是完整的，比 CPU 更适合对照实现。

如果你指定走路径 A 还是 B，我可以按该路径列出具体改哪些 `include/carbon/*`、编包脚本和验证配置，并直接改代码。

## User

 路径 A（复刻论文/FRED）：电磁对齐 FRED，核用 fred_paper + 最好导入 3.76 的 libEvents_C12_*。对标 FLUKA/FRED 水箱，而不是 INCLXX。

## Assistant

按路径 A 推进：先摸清 `fred_paper` 与输运接入点，再从 FRED 3.76 抽出 `libEvents_C12_*`，把电磁和核通道对齐到 FRED/FLUKA。

## Tools

- Search: fred_paper|FredFragment|sample_fred|inelastic_neutron
- ListDir: src/detail
- Read: /mnt/sdb/wuwei/MAIGO/plan.md (1-80)
- Read: /mnt/sdb/wuwei/MAIGO/include/carbon/detail/fred_fragmentation_data.hpp (171-570)
- Read: /mnt/sdb/wuwei/MAIGO/src/detail/sycl_inelastic_device.inc (1-150)
- Search: paper|event_lib|inelastic_model|physics_model|fred
- Read: /mnt/sdb/wuwei/MAIGO/config/beam_95MeVu_fred_thin_smoke.yaml (1-26)
- Execute: python3 << 'PY'
import struct
p='/mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/libFred.data'
data=open(p,'rb').read()
print('len',len(data))
print(data[:64])
# try nfiles then records
n=struct.unpack_from('<I',data,0)[0]
print('first u32',n)
# maybe nfiles=105
off=4
files=[]
for i in range(min(n,300)):
    if off+4>len(data):
        print('eof at',i,off)
        break
    plen=struct.unpack_from('<I',data,off)[0]
    off+=4
    if plen==0 or plen>512:
        print('bad plen',plen,'i',i,'off',off-4)
        break
    path=data[off:off+plen]
    off+=plen
    if off+4>len(data):
        print('no size')
        break
    sz=struct.unpack_from('<I',data,off)[0]
    off+=4
    try:
        sp=path.decode('ascii')
    except Exception as e:
        print('decode fail',e,path[:40]); break
    if sz>len(data)-off:
        print('size too big',sp,sz)
        break
    files.append((sp,sz,off))
    off+=sz
print('parsed',len(files),'end',off,'remain',len(data)-off)
for sp,sz,_ in files:
    print(f'{sz:10d} {sp}')
PY (Parse libFred.data as nfiles + path/size records)
- Read: /mnt/sdb/wuwei/MAIGO/include/carbon/transport_config.hpp (221-340)
- Search: sample_carbon_inelastic|enable_inelastic|p_target_h|prob_proj
- Read: /mnt/sdb/wuwei/MAIGO/CMakeLists.txt (1-80)
- Execute: python3 << 'PY'
import struct,re
p='/mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/libFred.data'
data=open(p,'rb').read()
# TOC: u32 n, then n*(u32 plen, path)
n=struct.unpack_from('<I',data,0)[0]
off=4
paths=[]
for i in range(n):
    plen=struct.unpack_from('<I',data,off)[0]
    off+=4
    path=data[off:off+plen].decode('ascii','replace')
    off+=plen
    paths.append(path)
print('n TOC',n,'toc_end',off)
for path in paths:
    print(path)

# find libEvents in blob
for m in re.finditer(b'libEvents_[A-Za-z0-9_]+', data):
    print('at',m.start(), m.group())
print('C12.txt at', data.find(b'data/nuc/C12.txt'))
# search typical headers
for key in [b'Fragmentation Event', b'Nevents', b'nEvents', b'EVENTLIB', b'libEvents']:
    i=data.find(key)
    print('find',key,i)
PY
# strings around event library loading in Fred.x
strings -n 8 /mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/Fred3.76.0.x | grep -E 'libEvents|Event library|nFrag|Afrag|Efrag|SamplFrag|numEvents|Nevents' | head -60
 (List libFred.data TOC and event-library strings)
- Execute: # search for .dat files and event lib loading
find /mnt/sdb/wuwei/fred -name '*libEvents*' -o -name '*.dat' 2>/dev/null | head
strings /mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/Fred3.76.0.x | grep -iE 'cannot find.*libEvents|loading.*event|event library|libEvents_C12|\.dat' | head -40
echo '==== GPU offsets struct ===='
strings /mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/libFredGPU.so | grep -iE 'eventLibrary|SamplFrag|SamplETheta|nFragMax|MAXFRAG' | sort -u | head -40
# nm event related
nm -C /mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/Fred3.76.0.x 2>/dev/null | grep -iE 'EventLib|libEvents|Fragmentation|loadEvent' | head -40
# size of rodata looking for large blobs
echo '==== binary size sections ===='
readelf -S /mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/Fred3.76.0.x | head -40
 (Find where FRED stores C12 event libraries)
- Read: /mnt/sdb/wuwei/MAIGO/src/config.cpp (161-250)
- Read: /mnt/sdb/wuwei/MAIGO/src/detail/sycl_inelastic_device.inc (151-350)
- Read: /mnt/sdb/wuwei/MAIGO/include/carbon/energy_loss_fluctuation.hpp (1-80)
- Read: /mnt/sdb/wuwei/MAIGO/include/carbon/straggling.hpp (1-80)
- Execute: python3 << 'PY'
import struct,zlib
p='/mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/libFred.data'
data=open(p,'rb').read()
n=struct.unpack_from('<I',data,0)[0]
off=4
paths=[]
for i in range(n):
    plen=struct.unpack_from('<I',data,off)[0]; off+=4
    paths.append(data[off:off+plen]); off+=plen
print('toc_end',off,'next 64', data[off:off+64])
print('as u32s', [struct.unpack_from('<I',data,off+4*i)[0] for i in range(8)])
print('as u64s', [struct.unpack_from('<Q',data,off+8*i)[0] for i in range(4)])
# try n sizes
sizes=[struct.unpack_from('<I',data,off+4*i)[0] for i in range(n)]
print('sum sizes if u32', sum(sizes), 'filelen-off-4n', len(data)-off-4*n)
sizes64=[struct.unpack_from('<Q',data,off+8*i)[0] for i in range(n)]
print('first 10 u64 sizes', sizes64[:10], 'sum', sum(sizes64) if all(s<2e8 for s in sizes64) else 'bad')
# maybe zlib stream
for wbits in (15, -15, 31):
    try:
        d=zlib.decompress(data[off:], wbits)
        print('zlib',wbits,'ok',len(d))
    except Exception as e:
        pass
# maybe just concatenated with size prefix per file
pos=off
ok=0
for i,path in enumerate(paths[:15]):
    if pos+4>len(data): break
    sz=struct.unpack_from('<I',data,pos)[0]
    print(i, path.decode(), 'sz',sz, 'pos',pos)
    if sz>50_000_000: break
    pos+=4+sz
    ok+=1
print('parsed with size prefix', ok, 'pos',pos)
PY (Decode libFred.data payload after TOC)
- Read: /mnt/sdb/wuwei/MAIGO/src/config.cpp (621-700)
- Read: /mnt/sdb/wuwei/MAIGO/src/transport_sycl.cpp (1089-1138)
- Search: allowed_keys|known_keys|parse_bool
- Execute: python3 << 'PY'
import struct, os
p='/mnt/sdb/wuwei/fred/fred_3.76.0_Linux_gcc_14.2.1/libFred.data'
data=open(p,'rb').read()
n=struct.unpack_from('<I',data,0)[0]
off=4
paths=[]
for i in range(n):
    plen=struct.unpack_from('<I',data,off)[0]; off+=4
    paths.append(data[off:off+plen].decode()); off+=plen
sizes=[struct.unpack_from('<I',data,off+4*i)[0] for i in range(n)]
off += 4*n
print('data_start',off,'sum',sum(sizes),'remain',len(data)-off)
outdir='/mnt/sdb/wuwei/MAIGO/data/fred_376'
os.makedirs(outdir, exist_ok=True)
pos=off
for path,sz in zip(paths,sizes):
    blob=data[pos:pos+sz]; pos+=sz
    rel=path.replace('/','_')
    if sz<2_000_000 or 'Vavilov' in path or 'C12' in path or 'dEds' in path or '2GR' in path or 'HU2' in path:
        out=os.path.join(outdir, rel)
        open(out,'wb').write(blob)
        print(f'{sz:10d} {path} -> {rel[:60]} head={blob[:80]!r}')
print('end pos',pos,'file',len(data))
PY (Extract FRED packed tables into data/fred_376)
- Search: unknown key|allowed_config|valid_keys|ConfigValues
- Read: /mnt/sdb/wuwei/MAIGO/src/config.cpp (2-161)
- Read: /mnt/sdb/wuwei/MAIGO/include/carbon/cross_section.hpp (1-80)
- Read: /mnt/sdb/wuwei/MAIGO/tests/carbon_tests.cpp (3211-3260)
- Read: /mnt/sdb/wuwei/MAIGO/src/config.cpp (321-360)
- Search: sample_straggling|gaussian_clamped|enable_energy_straggling
- Read: /mnt/sdb/wuwei/MAIGO/include/carbon/straggling.hpp (81-160)
- Read: /mnt/sdb/wuwei/MAIGO/tests/carbon_tests.cpp (3361-3410)
- Read: /mnt/sdb/wuwei/MAIGO/include/carbon/transport_config.hpp (601-632)
- Search: test_kox_icru
- Read: /mnt/sdb/wuwei/MAIGO/src/cross_section.cpp (71-150)
- Read: /mnt/sdb/wuwei/MAIGO/config/beam_200MeVu_inelastic.yaml (1-40)
- Read: /mnt/sdb/wuwei/MAIGO/src/transport_sycl.cpp (1361-1480)
- Search: enable_fragment_cascade|maximum_cascade
- Read: /mnt/sdb/wuwei/MAIGO/src/transport_sycl.cpp (1477-1556)
- Search: secondary_queue|enqueue_secondary
- Read: /mnt/sdb/wuwei/MAIGO/src/transport_sycl.cpp (1556-1645)
- Read: /mnt/sdb/wuwei/MAIGO/include/carbon/transport_config.hpp (261-285)
- Search: straggling_sampler
- Search: clamp_sampled_energy_loss|sample_moment_matched
- Read: /mnt/sdb/wuwei/MAIGO/include/carbon/straggling.hpp (156-235)
- Read: /mnt/sdb/wuwei/MAIGO/src/config.cpp (1466-1490)

## Assistant

事件库 `.dat` 不在 3.76 的 `libFred.data` 里；先把论文截面、H 弹性、Vavilov 式涨落和 FRED 向 MCS 接到现有 `fred_paper` 抽运上。

## Tools

- Execute: rm -rf /mnt/sdb/wuwei/MAIGO/data/fred_376 (Remove misaligned FRED dump directory)
- Read: /mnt/sdb/wuwei/MAIGO/include/carbon/detail/fred_fragmentation_data.hpp (184-213)
- Read: /mnt/sdb/wuwei/MAIGO/src/cross_section.cpp (201-280)
- Edit: /mnt/sdb/wuwei/MAIGO/include/carbon/detail/fred_fragmentation_data.hpp
- Edit: /mnt/sdb/wuwei/MAIGO/include/carbon/detail/fred_fragmentation_data.hpp
- Edit: /mnt/sdb/wuwei/MAIGO/include/carbon/cross_section.hpp
- Edit: /mnt/sdb/wuwei/MAIGO/src/cross_section.cpp
- Edit: /mnt/sdb/wuwei/MAIGO/src/cross_section.cpp
- Edit: /mnt/sdb/wuwei/MAIGO/include/carbon/straggling.hpp
- Edit: /mnt/sdb/wuwei/MAIGO/include/carbon/straggling.hpp
- Edit: /mnt/sdb/wuwei/MAIGO/include/carbon/straggling.hpp
- Edit: /mnt/sdb/wuwei/MAIGO/include/carbon/transport_config.hpp
- Edit: /mnt/sdb/wuwei/MAIGO/include/carbon/transport_config.hpp
- Search: from_csv|primary_inelastic
- Read: /mnt/sdb/wuwei/MAIGO/src/main.cpp (71-120)
- Read: /mnt/sdb/wuwei/MAIGO/src/transport_sycl.cpp (401-430)
- Edit: /mnt/sdb/wuwei/MAIGO/src/config.cpp
- Edit: /mnt/sdb/wuwei/MAIGO/src/config.cpp
- Edit: /mnt/sdb/wuwei/MAIGO/src/main.cpp
- Edit: /mnt/sdb/wuwei/MAIGO/src/transport_sycl.cpp
- Edit: /mnt/sdb/wuwei/MAIGO/src/transport_sycl.cpp
- Search: water_density
- Read: /mnt/sdb/wuwei/MAIGO/src/transport_sycl.cpp (641-680)
- Edit: /mnt/sdb/wuwei/MAIGO/src/transport_sycl.cpp
- Edit: /mnt/sdb/wuwei/MAIGO/src/transport_sycl.cpp
- Edit: /mnt/sdb/wuwei/MAIGO/src/transport_sycl.cpp
- Read: /mnt/sdb/wuwei/MAIGO/src/transport_sycl.cpp (431-450)
- Read: /mnt/sdb/wuwei/MAIGO/src/transport_sycl.cpp (444-483)
- Edit: /mnt/sdb/wuwei/MAIGO/src/transport_sycl.cpp
- Search: water_density_g_per_cm3
- Read: /mnt/sdb/wuwei/MAIGO/tests/carbon_tests.cpp (1371-1400)
- Read: /mnt/sdb/wuwei/MAIGO/tests/carbon_tests.cpp (1-40)
- Search: test_csv_target_h
- Edit: /mnt/sdb/wuwei/MAIGO/tests/carbon_tests.cpp
- Edit: /mnt/sdb/wuwei/MAIGO/tests/carbon_tests.cpp
- Edit: /mnt/sdb/wuwei/MAIGO/config/beam_200MeVu_fred_paper.yaml
- Edit: /mnt/sdb/wuwei/MAIGO/docs/PATH_A_FRED.md
- Execute: cmake --build /mnt/sdb/wuwei/MAIGO/build/cpu-debug --target carbon_tests -j 8 && /mnt/sdb/wuwei/MAIGO/build/cpu-debug/carbon_tests (Build and run carbon_tests on CPU debug)
- Execute: python3 - <<'PY'
import math
e=95
Ec=30; p0=762; p1=14e-4; p2=6.7; p3=13.4e-3
s=(1-math.exp(-e/Ec))*(p0+p1*e+math.exp(p2-p3*e))
print(s)
PY (Evaluate paper C-C cross section at 95 MeV/u)
- Edit: /mnt/sdb/wuwei/MAIGO/tests/carbon_tests.cpp
- Execute: python3 - <<'PY'
import math
NA=6.02214076e23
M=18.01528
def sigH(e):
    if e>=250: return 250
    return 250*(1+0.8*math.exp(-e/60))
def kox(ap,zp,at,zt,e,inc):
    a_kox=1.85
    ap13=ap**(1/3); at13=at**(1/3)
    ce=0.80*(1-math.exp(-e/40))
    r=ap13+at13+a_kox*ap13*at13/(ap13+at13)-ce
    bc=(1.44*zp*zt)/(1.3*(ap13+at13))
    ecm=(at/(ap+at))*inc
    cb=(1-bc/ecm) if ecm>bc else 0
    return 38.013*r*r*cb
def sigcc(e):
    return (1-math.exp(-e/30))*(762+14e-4*e+math.exp(6.7-13.4e-3*e))
e=200; inc=2400
sH=sigH(e)
kcc=kox(12,6,12,6,e,inc)
ko=kox(12,6,16,8,e,inc)
sO=sigcc(e)*ko/kcc
n=1*NA/M
macro_cm=n*(2*sH+sO)*1e-27
print('sH',sH,'sCC',sigcc(e),'sO',sO,'per_mm',macro_cm/10)
PY
cd /mnt/sdb/wuwei/MAIGO/build/cpu-debug && ./carbon_tests (Recompute water XS and rerun tests)
- Edit: /mnt/sdb/wuwei/MAIGO/tests/carbon_tests.cpp
- Execute: cmake --build /mnt/sdb/wuwei/MAIGO/build/cpu-debug --target carbon_tests -j 8 && /mnt/sdb/wuwei/MAIGO/build/cpu-debug/carbon_tests (Rebuild and rerun carbon_tests)

## Assistant

路径 A 已经接到当前仓库里。`libEvents_C12_*.dat` **不在** FRED 3.76 的 `libFred.data` 里（那包只有 dE/dx、Vavilov LUT、2GR MCS、活化、核文本）；3.76 运行时从搜索路径另载事件库，缺文件就退回论文的 2D PDF。因此核末态继续用现有 `fred_paper` 抽样，没有假装导入不存在的 `.dat`。

### 已落地

1. **非弹性截面（论文，不再用 Geant4 水表）**  
   C–C 拟合 × Kox(Np,Nt)/Kox(C,C)；氢用 ICRU。  
   `nuclear_model: fred_paper` 时 `from_fred_paper_water()` 现场建宏观 $\Sigma$。

2. **H 靶核弹性（论文 §4）**  
   质心系各向同性 + 实验室系变换；反冲质子入 secondary queue。  
   $\sigma_{\mathrm{el}}^{\mathrm{H}}$ 目前是治疗能区量级的占位（~220–400 mb），等 ENDF 再换。

3. **电磁向 FRED 靠**  
   - `energy_straggling_model: vavilov_landau`：薄步 Landau/Moyal，厚步高斯（还不是 FRED 那张打包 LUT）。  
   - MCS：`multiple_scattering_scale: 1.40` ≈ (14.1/13.6)×$f_{\mathrm{mcs}}$。

4. **配置与文档**  
   `config/beam_200MeVu_fred_paper.yaml`  
   `docs/PATH_A_FRED.md`

5. **测试**  
   `carbon_tests` 已通过（含 C–C 截面、弹性动能守恒、Vavilov sampler）。

### 使用

```yaml
nuclear_model: fred_paper
enable_nuclear_elastic: true
energy_straggling_model: vavilov_landau
multiple_scattering_scale: 1.40
```

不要再挂 INCLXX reaction package，也不要用 `dose_output_scale` 去盖峰形。

### 相对完整 FRED 3.76 仍缺

- `libEvents_C12_{H1,C12,O16}.dat` 相关事件库  
- 真正的 `Vavilov_v01.lut` 抽样  
- 2GR MCS + 侧向位移  
- ENDF C+p 弹性表  
- 与 FRED/FLUKA 水箱 100–400 MeV/u 的 IDD/halo 对标  

下一步若有完整 FRED data 目录（含那些 `.dat`），可以把事件库接到 H/C/O 靶上，Table 1 只留给其余核。

## User

使用topas提取上面缺少的数据

## Assistant

先摸清仓库里现有的 TOPAS 抽表管线，再针对路径 A 缺的弹性、涨落、MCS 和碎裂事件库补提取。

## Tools

- ListDir: startup
- Read: /mnt/sdb/wuwei/MAIGO/startup/README.md (1-120)
- Search: elastic|fluctuation|fragment|event
- Read: /mnt/sdb/wuwei/MAIGO/startup/extensions/CarbonElasticNtuple.cc (1-180)
- Read: /mnt/sdb/wuwei/MAIGO/startup/proton_water/generate_fluctuation_campaign.py (1-80)
- ListDir: startup/package_tools
- Read: /mnt/sdb/wuwei/MAIGO/startup/run_topas_database_extraction.sh (1-80)
- Read: /mnt/sdb/wuwei/MAIGO/startup/proton_water/energy_loss_fluctuation_thin_slab.txt.in (1-56)
- Read: /mnt/sdb/wuwei/MAIGO/startup/extensions/ElasticCrossSectionQueryNtuple.cc (1-107)
- Read: /mnt/sdb/wuwei/MAIGO/startup/extensions/EnergyLossFluctuationNtuple.cc (1-80)
- Read: /mnt/sdb/wuwei/MAIGO/startup/templates/database_extraction.txt.in (1-118)
- Read: /mnt/sdb/wuwei/MAIGO/startup/proton_water/run_topas_single_cpup6.slurm (1-43)
- Execute: ssh -o BatchMode=yes -o ConnectTimeout=10 wuwei@127.0.0.1 'hostname; ls /home/wuwei/topas 2>/dev/null | head; ls /mnt/sda/wuwei 2>/dev/null | head; which topas 2>/dev/null; find /home/wuwei /mnt/sda/wuwei /opt -name topas -type f 2>/dev/null | head; ls /home/wuwei/Applications 2>/dev/null | head; echo --- sinfo ---; sinfo -o "%P %a %l %D %N" 2>/dev/null | head' (Find TOPAS install and slurm on local cluster host)
- Execute: ssh -o BatchMode=yes wuwei@127.0.0.1 'ls /home/wuwei/topas | head -80; echo ==== build extensions ====; ls /home/wuwei/topas/topas-build 2>/dev/null | head; ls /mnt/sda/wuwei | head -40; echo ==== g4data ====; ls /home/wuwei/G4DATA 2>/dev/null | head; ls /usr/local/share/Geant4 2>/dev/null | head; echo ==== elastic jobs ====; ls -d /home/wuwei/topas/c12_h1o16_elastic* /mnt/sda/wuwei/*elastic* 2>/dev/null | head; echo ==== topas version ====; /home/wuwei/topas/topas-build/topas --version 2>&1 | head -20; echo ==== env ====; ls /home/wuwei/topas/*setup* /home/wuwei/topas/*.sh 2>/dev/null; grep -l G4LEDATA /home/wuwei/topas/* 2>/dev/null | head' (Inspect remote TOPAS build, G4 data, existing elastic jobs)
- Execute: ssh -o BatchMode=yes wuwei@127.0.0.1 'ls /home/wuwei/topas/extensions | sort; echo ==== cmake cache G4 ====; grep -E "G4_DIR|GEANT4|G4DATA|TOPAS_EXTENSION" /home/wuwei/topas/topas-build/CMakeCache.txt | head -30; echo ==== existing elastic in ====; head -40 /mnt/sda/wuwei/c12_h1o16_elastic_100.txt.in; echo ==== slurm example ====; cat /mnt/sda/wuwei/c12_h1o16_elastic_100.slurm | head -40' (List remote TOPAS extensions and G4 cmake paths)
- Read: /mnt/sdb/wuwei/MAIGO/startup/proton_water/generate_fluctuation_campaign.py (81-230)
- Read: /mnt/sdb/wuwei/MAIGO/startup/proton_water/finalize_fluctuation_campaign.py (1-80)
- ListDir: scripts
- Execute: ssh -o BatchMode=yes wuwei@127.0.0.1 'squeue -u wuwei; echo ==== elastic 100 status ====; ls -la /mnt/sda/wuwei/c12_h1o16_elastic_100/ 2>/dev/null | head; ls /mnt/sda/wuwei/c12_h1o16_elastic_100/H1 2>/dev/null | head; ls /mnt/sda/wuwei/c12_h1o16_elastic_100/H1/event 2>/dev/null | head; ls /mnt/sda/wuwei/c12_h1o16_elastic_100/H1/xs 2>/dev/null | head; echo ==== fluctuation existing ====; ls /mnt/sda/wuwei | grep -i fluc; echo ==== cinel thin ====; ls /mnt/sda/wuwei/cinel02-cascade-h1o16-thin 2>/dev/null | head' (Check running jobs and existing extraction outputs)
- Read: /mnt/sdb/wuwei/MAIGO/data/stopping_power_water_geant4_11_3_2.csv (1-5)
- Search: compile_energy_loss|packaged_fluctuation
- Execute: ls /mnt/sdb/wuwei/MAIGO/startup/package_tools 2>/dev/null; ls /mnt/sdb/wuwei/MAIGO/startup/proton_water/*.py | head
# check elastic phsp sizes and xs
ssh -o BatchMode=yes wuwei@127.0.0.1 'echo H1 event; wc -l /mnt/sda/wuwei/c12_h1o16_elastic_100/H1/event/*.header; ls -lh /mnt/sda/wuwei/c12_h1o16_elastic_100/H1/event/*.phsp; echo O16; ls -lh /mnt/sda/wuwei/c12_h1o16_elastic_100/O16/event/*.phsp 2>/dev/null; echo xs header; head -5 /mnt/sda/wuwei/c12_h1o16_elastic_100/H1/xs/*.header; echo msc; ls /mnt/sda/wuwei/c12_primary_msc_summary_400_emonly 2>/dev/null | head'
 (Inspect existing elastic ntuple sizes and package_tools)
- Read: /mnt/sdb/wuwei/MAIGO/startup/proton_water/run_fluctuation_campaign.py (1-80)
- Read: /mnt/sdb/wuwei/MAIGO/tests/test_compile_energy_loss_fluctuation_grid.py (1-40)
- Read: /mnt/sdb/wuwei/MAIGO/startup/proton_water/generate_fluctuation_campaign.py (211-360)
- Read: /mnt/sdb/wuwei/MAIGO/startup/proton_water/run_fluctuation_campaign.py (81-200)
- Execute: ssh -o BatchMode=yes wuwei@127.0.0.1 'cat /mnt/sda/wuwei/c12_h1o16_elastic_100/run.slurm' (Read full remote elastic extraction slurm)
- Read: /mnt/sdb/wuwei/MAIGO/startup/proton_water/run_fluctuation_campaign.py (251-342)
- Execute: ssh -o BatchMode=yes wuwei@127.0.0.1 'tail -n 80 /mnt/sda/wuwei/c12_h1o16_elastic_100/elastic_target_only.txt.in' (Read rest of elastic TOPAS template)

## Assistant

开始生成 C-12 涨落战役，并写弹性/薄靶碎裂的 TOPAS 提取任务，提交到本机 `sbatch`。

## Tools

- Execute: mkdir -p /mnt/sdb/wuwei/MAIGO/startup/work /mnt/sdb/wuwei/MAIGO/startup/path_a_extract
python3 /mnt/sdb/wuwei/MAIGO/startup/proton_water/generate_fluctuation_campaign.py \
  --output-dir /mnt/sdb/wuwei/MAIGO/startup/work/c12_water_fluctuation \
  --projectile-z 6 --projectile-a 12 \
  --energies-mevu 50,100,150,200,250,300,350,400 \
  --stopping-power-file /mnt/sdb/wuwei/MAIGO/data/stopping_power_water_geant4_11_3_2.csv \
  --maximum-step-mm 0.1 \
  --maximum-relative-energy-loss 0.01 \
  --density-fractions 0.125,0.25,0.5,1 \
  --histories 50000 \
  --threads 50 \
  --seed-base 2026083000 \
  --topas-version 4.2.p3 \
  --geant4-version 11.3.2
python3 - <<'PY'
import json
from pathlib import Path
p=Path('/mnt/sdb/wuwei/MAIGO/startup/work/c12_water_fluctuation/campaign_manifest.json')
c=json.loads(p.read_text())
print('points', c['grid']['point_count'])
print('first', c['points'][0]['label'], c['points'][0]['areal_density_g_per_cm2'])
PY (Generate C-12 water fluctuation TOPAS campaign)
- Execute: python3 - <<'PY'
from pathlib import Path
src=Path('/mnt/sdb/wuwei/MAIGO/data/stopping_power_water_geant4_11_3_2.csv')
out=Path('/mnt/sdb/wuwei/MAIGO/startup/path_a_extract/stopping_power_water_campaign.csv')
lines=[ln for ln in src.read_text().splitlines() if ln.strip() and not ln.startswith('#')]
out.write_text('\n'.join(lines)+'\n')
print(out, 'lines', len(lines), 'head', lines[0])
PY
python3 /mnt/sdb/wuwei/MAIGO/startup/proton_water/generate_fluctuation_campaign.py \
  --output-dir /mnt/sdb/wuwei/MAIGO/startup/work/c12_water_fluctuation \
  --projectile-z 6 --projectile-a 12 \
  --energies-mevu 50,100,150,200,250,300,350,400 \
  --stopping-power-file /mnt/sdb/wuwei/MAIGO/startup/path_a_extract/stopping_power_water_campaign.csv \
  --maximum-step-mm 0.1 \
  --maximum-relative-energy-loss 0.01 \
  --density-fractions 0.125,0.25,0.5,1 \
  --histories 50000 \
  --threads 50 \
  --seed-base 2026083000 \
  --topas-version 4.2.p3 \
  --geant4-version 11.3.2
python3 - <<'PY'
import json
from pathlib import Path
c=json.loads(Path('/mnt/sdb/wuwei/MAIGO/startup/work/c12_water_fluctuation/campaign_manifest.json').read_text())
print('points', c['grid']['point_count'])
print('sample', [(p['energy_MeV_per_u'], p['areal_density_g_per_cm2']) for p in c['points'][:4]])
PY (Strip SP CSV comments and generate fluctuation campaign)
- Edit: /mnt/sdb/wuwei/MAIGO/startup/path_a_extract/elastic_target_only.txt.in
- Edit: /mnt/sdb/wuwei/MAIGO/startup/path_a_extract/thin_fragment_target.txt.in
- Edit: /mnt/sdb/wuwei/MAIGO/startup/path_a_extract/run_fluctuation.slurm
- Edit: /mnt/sdb/wuwei/MAIGO/startup/path_a_extract/run_elastic_events.slurm
- Edit: /mnt/sdb/wuwei/MAIGO/startup/path_a_extract/run_thin_fragment.slurm
- Edit: /mnt/sdb/wuwei/MAIGO/startup/path_a_extract/README.md
- Edit: /mnt/sdb/wuwei/MAIGO/startup/path_a_extract/run_fluctuation.slurm
- Execute: chmod +x /mnt/sdb/wuwei/MAIGO/startup/path_a_extract/*.slurm
ssh -o BatchMode=yes wuwei@127.0.0.1 'mkdir -p /mnt/sda/wuwei/path_a_topas_extract'
rsync -a --delete \
  /mnt/sdb/wuwei/MAIGO/startup/path_a_extract/elastic_target_only.txt.in \
  /mnt/sdb/wuwei/MAIGO/startup/path_a_extract/thin_fragment_target.txt.in \
  /mnt/sdb/wuwei/MAIGO/startup/path_a_extract/run_fluctuation.slurm \
  /mnt/sdb/wuwei/MAIGO/startup/path_a_extract/run_elastic_events.slurm \
  /mnt/sdb/wuwei/MAIGO/startup/path_a_extract/run_thin_fragment.slurm \
  /mnt/sdb/wuwei/MAIGO/startup/path_a_extract/README.md \
  wuwei@127.0.0.1:/mnt/sda/wuwei/path_a_topas_extract/
rsync -a /mnt/sdb/wuwei/MAIGO/startup/work/c12_water_fluctuation/ \
  wuwei@127.0.0.1:/mnt/sda/wuwei/path_a_topas_extract/c12_water_fluctuation/
ssh -o BatchMode=yes wuwei@127.0.0.1 'ls /mnt/sda/wuwei/path_a_topas_extract; echo points; python3 -c "import json; print(len(json.load(open(\"/mnt/sda/wuwei/path_a_topas_extract/c12_water_fluctuation/campaign_manifest.json\"))[\"points\"]))"; cd /mnt/sda/wuwei/path_a_topas_extract; sbatch run_fluctuation.slurm; sbatch run_elastic_events.slurm; sbatch run_thin_fragment.slurm; squeue -u wuwei' (Rsync Path A extract campaign and submit three sbatch jobs)
- Execute: ssh -o BatchMode=yes wuwei@127.0.0.1 'squeue -u wuwei; echo ==== latest job log ====; ls -lt /mnt/sda/wuwei/job_*.log 2>/dev/null | head -5; echo ==== fluc slurm head ====; head -20 /mnt/sda/wuwei/path_a_topas_extract/run_fluctuation.slurm' (Check slurm queue and whether jobs started)
- Execute: ssh -o BatchMode=yes wuwei@127.0.0.1 'echo ==== 206 ====; cat /mnt/sda/wuwei/job_206.log; echo ==== 207 ====; cat /mnt/sda/wuwei/job_207.log; echo ==== 208 ====; cat /mnt/sda/wuwei/job_208.log; echo ==== err ====; cat /mnt/sda/wuwei/job_206.err /mnt/sda/wuwei/job_207.err /mnt/sda/wuwei/job_208.err 2>/dev/null | head -40' (Read initial slurm logs for the three extract jobs)
- Execute: ssh -o BatchMode=yes wuwei@127.0.0.1 'echo thin H1 95; grep -E "Number of|ERROR|Fatal" /mnt/sda/wuwei/path_a_topas_extract/thin_fragment/H1/e95/*.{header,log} 2>/dev/null | head -40; echo ==== thin log tail ====; tail -30 /mnt/sda/wuwei/path_a_topas_extract/thin_fragment/H1/e95/topas.log; echo ==== histories in run.txt ====; grep Histories /mnt/sda/wuwei/path_a_topas_extract/thin_fragment/H1/e95/run.txt' (Verify thin-fragment TOPAS actually scored 1M histories)
- Edit: /mnt/sdb/wuwei/MAIGO/docs/PATH_A_FRED.md

## Assistant

已经用本机 TOPAS 4.2.p3 / Geant4 11.3.2 开抽路径 A 缺的三套表。FRED 自带的 `libEvents_*.dat` 和 Vavilov LUT 无法从 3.76 数据包解开，所以用 TOPAS 做可复现代理。

数据写在 `wuwei@127.0.0.1:/mnt/sda/wuwei/path_a_topas_extract/`。说明见 `startup/path_a_extract/README.md`。

| 作业 | 名称 | 抽什么 | 规模 |
|---|---|---|---|
| **206** `c12_fluc` | C-12 水 **EM-only 能损涨落** | 8 个能量 × 4 个面密度，50k/点 | 替代 Vavilov LUT |
| **207** `c12_el_lib` | **H1 / C12 / O16 核弹性** | 0–400 MeV/u 宏观截面 + 95/200/300/400 事件（50 万/点） | 替代 ENDF C+p 占位 |
| **208** `c12_thin_frag` | **1 mm 薄靶碎裂事件库** | H1/C12/O16 × 95/200/300/400，1M/点，INCLXX 相关末态 | 替代 `libEvents_C12_*` |

三个作业都在 `compute` 上跑（各 50 核）。涨落点已经在快速过 50–250 MeV/u；薄靶 95 MeV/u **H1 已完成**：1e6 历史、约 **1.15×10⁵** 条 `CarbonReaction` 记录。100 MeV/u 的 H1/O16 弹性此前已有一份：`/mnt/sda/wuwei/c12_h1o16_elastic_100/`。

**2GR MCS** 不走 TOPAS：仍用 FRED `libFred.data` 里的 `data/mcs/*_2GR.txt`。远端还有 `c12_primary_msc_summary_400_emonly` 可标定 Highland $f_{\mathrm{mcs}}$。

作业结束后需要再编进 GPU 的：

1. 涨落 ntuple → `packaged_fluctuation` CSV（C-12 / G4_WATER）  
2. 弹性 XS phsp → 替换 `calculate_sigma_el_H_mb`  
3. 薄靶 reaction/cascade → H/C/O 相关事件库  

跑完后可以说一声，我直接编包并接到 `fred_paper` 路径上。

## User

已完成

## Assistant

先核对三套 TOPAS 提取是否都写全，再编成 `fred_paper` 能用的涨落包、弹性截面和薄靶事件库。

## Tools

- Execute: ssh -o BatchMode=yes wuwei@127.0.0.1 'squeue -u wuwei; echo ==== job status ====
for j in 206 207 208; do echo -- $j --; tail -5 /mnt/sda/wuwei/job_$j.log; echo ERR; tail -8 /mnt/sda/wuwei/job_$j.err; done
echo ==== fluc headers ====
find /mnt/sda/wuwei/path_a_topas_extract/c12_water_fluctuation/runs -name "*.header" | wc -l
echo ==== elastic ====
find /mnt/sda/wuwei/path_a_topas_extract/elastic -name "*elastic.header" | sort
echo ==== thin ====
find /mnt/sda/wuwei/path_a_topas_extract/thin_fragment -name "*reactions.header" | sort
' (Verify all three TOPAS extract jobs completed)
- Search: prepare_topas_fluctuation|compile_energy_loss|compile_elastic|prepare_topas_elastic
- ListDir: data
- Read: /mnt/sdb/wuwei/MAIGO/tests/test_prepare_topas_fluctuation.py (1-159)
- Read: /mnt/sdb/wuwei/MAIGO/tests/test_compile_energy_loss_fluctuation_grid.py (1-150)
- Read: /mnt/sdb/wuwei/MAIGO/include/carbon/energy_loss_fluctuation.hpp (81-120)
- Read: /mnt/sdb/wuwei/MAIGO/startup/extensions/CarbonReactionNtuple.cc (1-120)
- Execute: ls /mnt/sda/wuwei/path_a_topas_extract/c12_water_fluctuation/runs 2>/dev/null | head
head -20 /mnt/sda/wuwei/path_a_topas_extract/c12_water_fluctuation/runs/e50mevu_ad0p00125gcm2/energy_loss_fluctuation.header 2>/dev/null
head -3 /mnt/sda/wuwei/path_a_topas_extract/elastic/H1/e400/xs/c12_H1_400mevu_xs.phsp 2>/dev/null
head -5 /mnt/sda/wuwei/path_a_topas_extract/elastic/H1/e400/xs/c12_H1_400mevu_xs.header 2>/dev/null
# elastic xs files
ls /mnt/sda/wuwei/path_a_topas_extract/elastic/H1/e400/xs/
grep -E "Scored Entries" /mnt/sda/wuwei/path_a_topas_extract/thin_fragment/*/e95/*reactions.header
 (Read local extract samples: fluctuation header, elastic XS, thin counts)
- Read: /mnt/sdb/wuwei/MAIGO/src/energy_loss_fluctuation.cpp (1-120)
- Read: /mnt/sdb/wuwei/MAIGO/tests/test_compile_energy_loss_fluctuation_grid.py (149-190)
- Search: packaged_fluctuation|energy_straggling_package
- Search: EnergyLossFluctuationTable|energy_straggling_package
- Read: /mnt/sdb/wuwei/MAIGO/src/energy_loss_fluctuation.cpp (111-210)
- Read: /mnt/sdb/wuwei/MAIGO/startup/proton_water/finalize_fluctuation_campaign.py (81-260)
- Read: /mnt/sda/wuwei/path_a_topas_extract/elastic/H1/e400/xs/c12_H1_400mevu_xs_xs.header (1-8)
- Read: /mnt/sda/wuwei/path_a_topas_extract/thin_fragment/H1/e95/c12_H1_95mevu_thin_reactions.header (1-29)
- Execute: find /mnt/sdb/wuwei/MAIGO /mnt/sda/wuwei -name '*fluctuation*.csv' 2>/dev/null | head
find /mnt/sdb/wuwei/MAIGO -name 'prepare_topas_fluctuation.py' 2>/dev/null
# sample elastic xs
head -5 /mnt/sda/wuwei/path_a_topas_extract/elastic/H1/e400/xs/c12_H1_400mevu_xs_xs.phsp
# sample reaction line
head -3 /mnt/sda/wuwei/path_a_topas_extract/thin_fragment/H1/e95/c12_H1_95mevu_thin_reactions.phsp
# fluctuation phsp sample
head -2 /mnt/sda/wuwei/path_a_topas_extract/c12_water_fluctuation/runs/e50mevu_ad0p00125gcm2/energy_loss_fluctuation.phsp
# elastic event header scored
grep "Scored Entries" /mnt/sda/wuwei/path_a_topas_extract/elastic/H1/e95/event/*.header
 (Sample ntuple rows and find any existing fluctuation compiler)
- Read: /mnt/sdb/wuwei/MAIGO/tests/test_run_fluctuation_campaign.py (1-80)
- Read: /mnt/sdb/wuwei/MAIGO/src/energy_loss_fluctuation.cpp (201-280)
- Search: EnergyLossFluctuation|sample_loss_ratio
- Read: /mnt/sdb/wuwei/MAIGO/src/transport_sycl.cpp (1186-1225)

## Assistant

三套 TOPAS 数据都齐了。接下来写编包脚本，把涨落、弹性和薄靶事件接到 `fred_paper`。

## Tools

- Edit: /mnt/sdb/wuwei/MAIGO/startup/package_tools/prepare_topas_fluctuation.py
- Edit: /mnt/sdb/wuwei/MAIGO/startup/package_tools/compile_energy_loss_fluctuation_grid.py
- Execute: cd /mnt/sdb/wuwei/MAIGO && python3 -m unittest tests.test_prepare_topas_fluctuation tests.test_compile_energy_loss_fluctuation_grid -v (Run fluctuation package-tool unit tests)
- Edit: /mnt/sdb/wuwei/MAIGO/startup/package_tools/prepare_topas_fluctuation.py
- Edit: /mnt/sdb/wuwei/MAIGO/startup/package_tools/compile_energy_loss_fluctuation_grid.py
- Execute: kill 2>/dev/null; python3 - <<'PY'
from prepare_topas_fluctuation import default_probabilities
import sys
sys.path.insert(0,'/mnt/sdb/wuwei/MAIGO/startup/package_tools')
from prepare_topas_fluctuation import default_probabilities, empirical_quantile, exact_type7_quantile_integral, trapezoid_integral
import math
p=default_probabilities()
print(len(p), p[-3], p[-2], p[-1], p[-2]-p[-3], 0.99999 in p)
sample_count=100000
samples=[-math.log1p(-(i+0.5)/sample_count) for i in range(sample_count)]
mean=sum(samples)/sample_count
norm=[v/mean for v in samples]
q=[empirical_quantile(norm, x) for x in p]
print(abs(trapezoid_integral(p,q)/exact_type7_quantile_integral(norm)-1))
PY
cd /mnt/sdb/wuwei/MAIGO && python3 -m unittest tests.test_prepare_topas_fluctuation tests.test_compile_energy_loss_fluctuation_grid -v (Fix probability grid and re-run package-tool tests)
- Execute: python3 - <<'PY'
import json, subprocess, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

root = Path('/mnt/sda/wuwei/path_a_topas_extract/c12_water_fluctuation')
prep = Path('/mnt/sdb/wuwei/MAIGO/startup/package_tools/prepare_topas_fluctuation.py')
campaign = json.loads((root/'campaign_manifest.json').read_text())
points = campaign['points']

def one(point):
    label = point['label']
    run = root/'runs'/label
    out = run/'fluctuation_point.json'
    if out.is_file():
        return label, 'skip', ''
    cmd = [
        sys.executable, str(prep),
        '--input', str(run/'energy_loss_fluctuation'),
        '--output', str(out),
        '--projectile-z', '6', '--projectile-a', '12',
        '--material', 'G4_WATER',
        '--energy-mev-per-u', str(point['energy_MeV_per_u']),
        '--areal-density-g-per-cm2', str(point['areal_density_g_per_cm2']),
        '--expected-histories', '50000',
    ]
    r = subprocess.run(cmd, text=True, capture_output=True)
    return label, 'ok' if r.returncode==0 else 'fail', r.stderr

fails=[]
with ThreadPoolExecutor(max_workers=8) as ex:
    futs=[ex.submit(one,p) for p in points]
    for fut in as_completed(futs):
        label, status, err = fut.result()
        print(status, label)
        if status=='fail':
            fails.append((label, err[-400:]))
if fails:
    print('FAILURES', len(fails))
    print(fails[0])
    sys.exit(1)
print('prepared', len(points))
PY (Prepare all 32 C-12 fluctuation inverse-CDF points)
- Read: /mnt/sda/wuwei/path_a_topas_extract/c12_water_fluctuation/runs/e50mevu_ad0p00125gcm2/energy_loss_fluctuation.header
- Edit: /mnt/sdb/wuwei/MAIGO/startup/package_tools/prepare_topas_fluctuation.py
- Execute: cd /mnt/sdb/wuwei/MAIGO && python3 -m unittest tests.test_prepare_topas_fluctuation -q && python3 - <<'PY'
import json, subprocess, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

root = Path('/mnt/sda/wuwei/path_a_topas_extract/c12_water_fluctuation')
prep = Path('/mnt/sdb/wuwei/MAIGO/startup/package_tools/prepare_topas_fluctuation.py')
campaign = json.loads((root/'campaign_manifest.json').read_text())

def one(point):
    label = point['label']
    run = root/'runs'/label
    out = run/'fluctuation_point.json'
    cmd = [
        sys.executable, str(prep),
        '--input', str(run/'energy_loss_fluctuation'),
        '--output', str(out),
        '--projectile-z', '6', '--projectile-a', '12',
        '--material', 'G4_WATER',
        '--energy-mev-per-u', str(point['energy_MeV_per_u']),
        '--areal-density-g-per-cm2', str(point['areal_density_g_per_cm2']),
        '--expected-histories', '50000',
    ]
    r = subprocess.run(cmd, text=True, capture_output=True)
    return label, r.returncode, r.stderr[-500:]

fails=[]
with ThreadPoolExecutor(max_workers=8) as ex:
    futs=[ex.submit(one,p) for p in campaign['points']]
    for fut in as_completed(futs):
        label, rc, err = fut.result()
        print(('ok' if rc==0 else 'fail'), label)
        if rc!=0:
            fails.append((label, err))
if fails:
    print(fails[0])
    sys.exit(1)
print('all prepared')
PY (Re-run fluctuation point preparation after header parse fix)
- Execute: mkdir -p /mnt/sdb/wuwei/MAIGO/data/packages
python3 /mnt/sdb/wuwei/MAIGO/startup/package_tools/compile_energy_loss_fluctuation_grid.py \
  --points /mnt/sda/wuwei/path_a_topas_extract/c12_water_fluctuation/runs/*/fluctuation_point.json \
  --output /mnt/sdb/wuwei/MAIGO/data/packages/c12_G4_WATER_fluctuation_50k.csv \
  --output-metadata /mnt/sdb/wuwei/MAIGO/data/packages/c12_G4_WATER_fluctuation_50k.metadata.json
python3 - <<'PY'
import json
from pathlib import Path
m=json.loads(Path('/mnt/sdb/wuwei/MAIGO/data/packages/c12_G4_WATER_fluctuation_50k.metadata.json').read_text())
print('points', m['grid']['point_count'], 'hist', m['histories']['total'], 'cols', m['grid']['probability_count'])
print('csv bytes', Path('/mnt/sdb/wuwei/MAIGO/data/packages/c12_G4_WATER_fluctuation_50k.csv').stat().st_size)
PY
# elastic XS
python3 - <<'PY'
from pathlib import Path
import math
NA=6.02214076e23
A_H=1.00782503223
M_W=18.01528
n_pure=NA/A_H
n_water_h=2*NA/M_W
phsp=Path('/mnt/sda/wuwei/path_a_topas_extract/elastic/H1/e400/xs/c12_H1_400mevu_xs_xs.phsp')
rows=[]
for line in phsp.read_text().splitlines():
    line=line.strip()
    if not line: continue
    e, macro=line.split()
    e=float(e); macro=float(macro)
    macro_w=macro * n_water_h / n_pure
    # sigma mb: macro_cm-1 = macro_mm * 10; sigma = macro_cm / n * 1e27
    sigma_mb = (macro*10.0)/n_pure * 1e27 if n_pure else 0
    rows.append((e, sigma_mb, macro_w))
out=Path('/mnt/sdb/wuwei/MAIGO/data/c12_elastic_h1_water_topas_11_3_2.csv')
out.write_text('energy_MeV_per_u,sigma_el_H_mb,water_macroscopic_elastic_per_mm\n'+
               '\n'.join(f'{e:.8g},{s:.8g},{m:.8g}' for e,s,m in rows)+'\n')
print('elastic rows', len(rows), 'sigma(100)', next(s for e,s,m in rows if abs(e-100)<1e-9),
      'sigma(400)', rows[-1][1], 'macro_w(100)', next(m for e,s,m in rows if abs(e-100)<1e-9))
PY
 (Compile fluctuation CSV and C-12+H elastic XS table)
- Execute: python3 - <<'PY'
from pathlib import Path
rows=[]
for line in Path('/mnt/sdb/wuwei/MAIGO/data/c12_elastic_h1_water_topas_11_3_2.csv').read_text().splitlines()[1:]:
    if not line.strip(): continue
    e,s,m=line.split(',')
    rows.append((float(e), float(s), float(m)))
assert len(rows)==401
hdr=Path('/mnt/sdb/wuwei/MAIGO/include/carbon/detail/c12_elastic_h_xs.hpp')
lines=[
'#pragma once',
'#include <cstddef>',
'namespace carbon {',
'inline constexpr int kC12ElasticHGridSize = 401;',
'inline constexpr float kC12ElasticHEnergyMeVu[] = {',
]
lines.append(','.join(f'{e:.8g}F' for e,_,__ in rows) + '};')
lines.append('inline constexpr float kC12ElasticHSigmaMb[] = {')
lines.append(','.join(f'{s:.8g}F' for _,s,__ in rows) + '};')
lines.append('inline constexpr float kC12ElasticHWaterMacroPerMm[] = {')
lines.append(','.join(f'{m:.8g}F' for *_,m in rows) + '};')
lines.append('inline float interpolate_c12_elastic_h_water_macro_per_mm(float e_per_u) noexcept {')
lines.append('  if (!(e_per_u > 0.0F)) return 0.0F;')
lines.append('  if (e_per_u <= kC12ElasticHEnergyMeVu[0]) return kC12ElasticHWaterMacroPerMm[0];')
lines.append('  if (e_per_u >= kC12ElasticHEnergyMeVu[kC12ElasticHGridSize-1]) return kC12ElasticHWaterMacroPerMm[kC12ElasticHGridSize-1];')
lines.append('  int lo=0, hi=kC12ElasticHGridSize-1;')
lines.append('  while (hi-lo>1) { int mid=(lo+hi)/2; if (kC12ElasticHEnergyMeVu[mid]<=e_per_u) lo=mid; else hi=mid; }')
lines.append('  const float e0=kC12ElasticHEnergyMeVu[lo], e1=kC12ElasticHEnergyMeVu[hi];')
lines.append('  const float t=(e_per_u-e0)/(e1-e0);')
lines.append('  return kC12ElasticHWaterMacroPerMm[lo]*(1.0F-t)+kC12ElasticHWaterMacroPerMm[hi]*t;')
lines.append('}')
lines.append('} // namespace carbon')
hdr.write_text('\n'.join(lines)+'\n')
print('wrote', hdr, 'bytes', hdr.stat().st_size)
PY
# compile event libraries 95 MeV/u
python3 - <<'PY'
from pathlib import Path
from collections import defaultdict
import struct, json, math

MAXF=8
NKEEP=8192

def parse_phsp(path):
    events=defaultdict(list)
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if not line.strip(): continue
        # record kind may have spaces: first token reaction/secondary
        parts=line.split()
        # find numeric run id: after kind
        # kind is first field but "reaction" / "secondary"
        kind=parts[0]
        # remaining: run event track parent pdg name? name can have no spaces
        # From header: Record Kind, Run, Event, Track, Parent, PDG, Name, Z, A, charge, ke, inc, vx,vy,vz, dx,dy,dz, w, process, type, subtype, model
        # Name is C12, alpha, gamma, proton, neutron, primary_continuation
        if kind not in ('reaction','secondary'):
            continue
        # After kind: 5 ints then name then z a ...
        run, event, track, parent, pdg = map(int, parts[1:6])
        # name is parts[6] unless continuation? primary_continuation is one token
        name=parts[6]
        z=int(parts[7]); a=int(parts[8]); charge=float(parts[9]); ke=float(parts[10]); inc=float(parts[11])
        dx,dy,dz=map(float, parts[15:18])
        events[event].append((kind, z, a, ke, dx, dy, dz, name, inc))
    return events

def pack(target, z, a, energy, src):
    events=parse_phsp(src)
    keys=sorted(events)
    kept=[]
    stride=max(1, len(keys)//NKEEP)
    for key in keys[::stride]:
        frags=[]
        nke=0.0
        inc=energy*12
        for rec in events[key]:
            kind, zz, aa, ke, dx, dy, dz, name, inc_e = rec
            inc=inc_e
            if kind=='reaction':
                continue
            nrm=math.sqrt(dx*dx+dy*dy+dz*dz) or 1.0
            dx,dy,dz=dx/nrm, dy/nrm, dz/nrm
            if zz==0 and aa==1:
                nke += ke
            elif zz>=1 and aa>=1:
                frags.append((zz,aa,ke,dx,dy,dz))
            # skip gamma
        if not frags:
            continue
        if len(frags)>MAXF:
            # keep highest KE
            frags.sort(key=lambda t: t[2], reverse=True)
            extra=sum(t[2] for t in frags[MAXF:])
            frags=frags[:MAXF]
            nke += extra
        kept.append((frags, nke, inc))
        if len(kept)>=NKEEP:
            break
    out=Path(f'/mnt/sdb/wuwei/MAIGO/data/packages/c12_{target}_{energy}MeVu_events.bin')
    payload=bytearray()
    payload += b'FELB'
    payload += struct.pack('<IIIfHH', 1, len(kept), MAXF, float(energy), z, a)
    for frags, nke, inc in kept:
        payload += struct.pack('<Bf', len(frags), nke)
        for i in range(MAXF):
            if i < len(frags):
                zz,aa,ke,dx,dy,dz=frags[i]
                payload += struct.pack('<bbffff', zz, aa, ke, dx, dy, dz)
            else:
                payload += struct.pack('<bbffff', 0, 0, 0.0, 0.0, 0.0, 1.0)
    out.write_bytes(payload)
    meta={'target':target,'Z':z,'A':a,'energy_MeVu':energy,'events':len(kept),'source_events':len(keys),'bytes':len(payload)}
    out.with_suffix('.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(target, energy, 'kept', len(kept), 'from', len(keys), 'bytes', len(payload))

base=Path('/mnt/sda/wuwei/path_a_topas_extract/thin_fragment')
for tgt,z,a in (('H1',1,1),('C12',6,12),('O16',8,16)):
    pack(tgt,z,a,95, base/tgt/'e95'/f'c12_{tgt}_95mevu_thin_reactions.phsp')
PY (Generate elastic XS header and 95 MeV/u event-library binaries)
- Read: /mnt/sdb/wuwei/MAIGO/src/energy_loss_fluctuation.cpp (281-330)
- Edit: /mnt/sdb/wuwei/MAIGO/include/carbon/detail/fred_fragmentation_data.hpp
- Edit: /mnt/sdb/wuwei/MAIGO/include/carbon/detail/fred_fragmentation_data.hpp
- Edit: /mnt/sdb/wuwei/MAIGO/include/carbon/detail/fred_fragmentation_data.hpp
- Edit: /mnt/sdb/wuwei/MAIGO/include/carbon/fred_event_library.hpp
- Edit: /mnt/sdb/wuwei/MAIGO/src/fred_event_library.cpp
- Edit: /mnt/sdb/wuwei/MAIGO/CMakeLists.txt
- Edit: /mnt/sdb/wuwei/MAIGO/include/carbon/inelastic.hpp
- Read: /mnt/sdb/wuwei/MAIGO/src/detail/sycl_inelastic_device.inc (2-46)
- Search: sample_carbon_inelastic_products_device
- Edit: /mnt/sdb/wuwei/MAIGO/src/detail/sycl_inelastic_device.inc
- Edit: /mnt/sdb/wuwei/MAIGO/src/detail/sycl_inelastic_device.inc
- Search: #include
- Edit: /mnt/sdb/wuwei/MAIGO/include/carbon/transport_config.hpp
- Edit: /mnt/sdb/wuwei/MAIGO/src/config.cpp
- Edit: /mnt/sdb/wuwei/MAIGO/src/config.cpp
- Edit: /mnt/sdb/wuwei/MAIGO/src/transport_sycl.cpp
- Read: /mnt/sdb/wuwei/MAIGO/src/transport_sycl.cpp (501-550)
- Edit: /mnt/sdb/wuwei/MAIGO/src/transport_sycl.cpp
- Edit: /mnt/sdb/wuwei/MAIGO/src/transport_sycl.cpp
- Search: free_device\(fred_prob
- Edit: /mnt/sdb/wuwei/MAIGO/src/transport_sycl.cpp
- Edit: /mnt/sdb/wuwei/MAIGO/src/transport_sycl.cpp
- Edit: /mnt/sdb/wuwei/MAIGO/tests/carbon_tests.cpp
- Edit: /mnt/sdb/wuwei/MAIGO/tests/carbon_tests.cpp
- Edit: /mnt/sdb/wuwei/MAIGO/config/beam_200MeVu_fred_paper.yaml
- Edit: /mnt/sdb/wuwei/MAIGO/config/beam_200MeVu_fred_paper.yaml
- Edit: /mnt/sdb/wuwei/MAIGO/tests/carbon_tests.cpp
- Edit: /mnt/sdb/wuwei/MAIGO/tests/carbon_tests.cpp
- Edit: /mnt/sdb/wuwei/MAIGO/tests/carbon_tests.cpp
- Edit: /mnt/sdb/wuwei/MAIGO/docs/PATH_A_FRED.md
- Execute: cmake --build /mnt/sdb/wuwei/MAIGO/build/cpu-debug --target carbon_tests -j 8 && cd /mnt/sdb/wuwei/MAIGO && /mnt/sdb/wuwei/MAIGO/build/cpu-debug/carbon_tests (Rebuild and run carbon_tests including new Path A assets)
- Execute: python3 - <<'PY'
from pathlib import Path
rows=[]
for line in Path('/mnt/sdb/wuwei/MAIGO/data/c12_elastic_h1_water_topas_11_3_2.csv').read_text().splitlines()[1:]:
    if not line.strip(): continue
    e,s,m=line.split(',')
    rows.append((float(e), float(s), float(m)))
hdr=Path('/mnt/sdb/wuwei/MAIGO/include/carbon/detail/c12_elastic_h_xs.hpp')
def arr(name, vals, fmt):
    body=','.join(fmt(v) for v in vals)
    return f'inline constexpr float {name}[] = {{{body}}};'
lines=[
'#pragma once',
'namespace carbon {',
'inline constexpr int kC12ElasticHGridSize = 401;',
arr('kC12ElasticHEnergyMeVu', [r[0] for r in rows], lambda v: f'{v:.8g}f' if v!=int(v) else f'{int(v)}.0f'),
arr('kC12ElasticHSigmaMb', [r[1] for r in rows], lambda v: f'{v:.8g}f'),
arr('kC12ElasticHWaterMacroPerMm', [r[2] for r in rows], lambda v: f'{v:.8g}f'),
'''inline float interpolate_c12_elastic_h_water_macro_per_mm(float e_per_u) noexcept {
  if (!(e_per_u > 0.0f)) return 0.0f;
  if (e_per_u <= kC12ElasticHEnergyMeVu[0]) return kC12ElasticHWaterMacroPerMm[0];
  if (e_per_u >= kC12ElasticHEnergyMeVu[kC12ElasticHGridSize-1]) return kC12ElasticHWaterMacroPerMm[kC12ElasticHGridSize-1];
  int lo=0, hi=kC12ElasticHGridSize-1;
  while (hi-lo>1) { int mid=(lo+hi)/2; if (kC12ElasticHEnergyMeVu[mid]<=e_per_u) lo=mid; else hi=mid; }
  const float e0=kC12ElasticHEnergyMeVu[lo], e1=kC12ElasticHEnergyMeVu[hi];
  const float t=(e_per_u-e0)/(e1-e0);
  return kC12ElasticHWaterMacroPerMm[lo]*(1.0f-t)+kC12ElasticHWaterMacroPerMm[hi]*t;
}
} // namespace carbon
''']
hdr.write_text('\n'.join(lines))
print('ok', hdr.stat().st_size)
PY
cmake --build /mnt/sdb/wuwei/MAIGO/build/cpu-debug --target carbon_tests -j 8 && cd /mnt/sdb/wuwei/MAIGO && ./build/cpu-debug/carbon_tests (Fix float suffixes and rebuild carbon_tests)
- Edit: /mnt/sdb/wuwei/MAIGO/startup/path_a_extract/compile_thin_events.py

## Assistant

三套 TOPAS 提取都已编进仓库，并接到 `fred_paper` 路径。

### 编出来的数据

| 资产 | 内容 |
|---|---|
| `data/packages/c12_G4_WATER_fluctuation_50k.csv` | C-12 水 inverse-CDF，32 个 (E, xρ) 点 × 50k（共 160 万历史） |
| `data/c12_elastic_h1_water_topas_11_3_2.csv` | C-12+¹H 弹性：0–400 MeV/u，**100 MeV/u ≈ 101 mb**（替换原来的 ~270 mb 占位） |
| `data/packages/c12_H1_95MeVu_events.bin` | 8192 个薄靶 H 相关末态 |
| `data/packages/c12_C12_95MeVu_events.bin` | 8192 个 C 末态 |
| `data/packages/c12_O16_95MeVu_events.bin` | 4068 个 O 末态（源事件更少） |

工具：`startup/package_tools/prepare_topas_fluctuation.py`、`compile_energy_loss_fluctuation_grid.py`（单测已过）。

### 已接到运行时

- **弹性**：GPU 上的水中 H 弹性宏观截面改为 TOPAS 表插值。  
- **碎裂**：`nuclear_model: fred_paper` 时加载 H/O 事件库；抽到事件后按 $E/95$ 缩放动能。抽不到或能量超限才回退 Table 1。  
- **配置**：`config/beam_200MeVu_fred_paper.yaml` 已写上事件库路径。

`carbon_tests` 已通过（含弹性表和事件库加载）。

### 还没接到 GPU 步进里的

涨落 CSV 已编好，但 kernel 仍用 `vavilov_landau`。下一步是把 `EnergyLossFluctuationTable` 拷进 device，做成 `packaged_fluctuation`。需要的话我可以接着接。

那次后台网格搜索可以忽略：它没找到 335 点组合，而涨落包已经用另一套概率网格编好，相关单测也过了。当前状态仍是弹性表、H/O 事件库已接入，`packaged_fluctuation` 还没进 GPU kernel。

## User

当前的仓库里有自己实现了fluctuation吗

## Tools

- Search: energy_straggling|sample_condensed_energy_loss|vavilov_landau|packaged_fluctuation|EnergyLossFluctuation
- Read: /mnt/sdb/wuwei/MAIGO/include/carbon/transport_config.hpp (245-274)
- Read: /mnt/sdb/wuwei/MAIGO/include/carbon/straggling.hpp (97-186)

## Assistant

有。仓库里自己实现了能损涨落，而且有两层：一套已经进输运，一套只编了表、还没进 GPU kernel。

### 已经在跑的（自己写的 condensed-history 抽样）

实现在 `include/carbon/straggling.hpp`，CPU/GPU 都走 `sample_condensed_energy_loss`。开关是 `enable_energy_straggling`，模型名是 `energy_straggling_model`。

| 模型 | 做什么 | 是否接到步进 |
|---|---|---|
| `gaussian_clamped`（默认，别名 `legacy_calibrated`） | 高斯 $μ+σz$，截到 $[0,\min(2μ,E)]$ | 是 |
| `moment_matched` | 厚步高斯、薄步 Gamma，只截剩余动能 | 是 |
| `vavilov_landau` | 薄步 Landau/Moyal、厚步高斯（解析近似，不是 FRED 那张 LUT） | 是 |

方差用 Bohr / $T_{\max}/β^2$ 相对论项（`condensed_total_loss_variance_MeV2`），不是 Geant4 逐步调用 `G4UniversalFluctuation`。还可以开 `enable_step_stable_straggling`（按固定物理块抽，避免步长切开改变方差）和 `enable_secondary_energy_straggling`（次级默认关）。

`config/beam_200MeVu_fred_paper.yaml` 现在用的是 **`vavilov_landau`**。

### 编好了但还没接到 kernel 的

- `EnergyLossFluctuationTable`（`include/carbon/energy_loss_fluctuation.hpp`）：从 CSV 读 inverse-CDF，按 $E$、面密度、均匀随机数抽 `loss/mean`。
- TOPAS 抽出来的 C-12 包：`data/packages/c12_G4_WATER_fluctuation_50k.csv`。
- 配置里 **没有** 活的 `packaged_fluctuation` / `energy_straggling_package_file` 接到 `transport_sycl.cpp`。文档里提过这套，当前分支 kernel 不会去读那张表。

所以：有自己的 fluctuation，生产路径是解析/高斯模型；TOPAS 数据驱动的包只是数据资产，还没替代步进里的抽样。

## User

当前的inelastic的进度怎么样了

## Tools

- Search: nuclear_model|fred_paper|sample_carbon_inelastic|enable_inelastic|fred_event_library
- Read: /mnt/sdb/wuwei/MAIGO/src/detail/sycl_inelastic_device.inc (1-90)
- Read: /mnt/sdb/wuwei/MAIGO/config/beam_200MeVu_fred_paper.yaml (1-41)
- Read: /mnt/sdb/wuwei/MAIGO/docs/PATH_A_FRED.md (1-72)

## Assistant

当前 inelastic 已经能在 GPU 上跑完整一条链，但**还没有对上 FRED/FLUKA 水箱剂量**；核模型身份是「论文公式 + TOPAS 薄靶事件库」，不是 FRED 原装 `libEvents`。

### 已经接到运行时的

| 环节 | 状态 |
|---|---|
| 步内碰撞 | $1-e^{-\Sigma s}$，碰撞点在步内 | 完成 |
| 水中宏观非弹截面 | `nuclear_model: fred_paper` 时用论文 C–C 拟合 × Kox + ICRU-H，不再读 Geant4 CSV | 完成 |
| 靶核 H/O | $P(\mathrm{H})=2\sigma_H/(2\sigma_H+\sigma_O)$ | 完成 |
| 末态（优先） | 加载 `c12_H1_95MeVu_events.bin` / `c12_O16_95MeVu_events.bin`，按 $E/95$ 缩放动能 | 完成（INCLXX 薄靶，不是 GANIL/FRED 库） |
| 末态（回退） | Table 1 独立抽样 + Eq.12 高斯/指数 + $k=c(1-R)$，守恒失败重抽 | 完成 |
| 中子 | 不进剂量，只进 untracked；顶点 kerma = 0 | 完成 |
| 次级带电 | 入 secondary queue 继续 stopping + MCS | 完成 |
| 反应后初级 | 杀掉（`energy_MeV = 0`） | 完成 |
| 弹性（并行离散过程） | TOPAS C+p 截面表 + 论文 CoM 运动学，反冲质子入队 | 完成（这是 elastic，但和 inelastic 抢 $\Sigma_{\mathrm{tot}}$） |

配置入口：`config/beam_200MeVu_fred_paper.yaml`（`enable_inelastic: true`）。

###  intentionally 没做 / 还差的

1. **没有 FRED 3.76 的 `libEvents_C12_*.dat`**。发行包里没有；现在用的是 TOPAS INCLXX 1 mm 薄靶相关末态。产额/关联更像 Geant4，不像 GANIL Table 1。  
2. **产额钉在 95 MeV/u 再按能量缩放**，没有治疗能区随 $E$ 变的 inclusive yield（论文也如此）。  
3. **二次碎裂/级联**：这条 `fred_paper` 路径没有 INCL cascade package，碎片默认不再核反应。  
4. **CPU serial 输运不跑 inelastic**（`transport_cpu.cpp` 里截面参数还是 unused）。  
5. **没有 100–400 MeV/u 水箱 vs FRED/FLUKA 的 IDD/尾巴验收**。代码能跑，物理对不对还没关门。  
6. 文档 `PATH_A_FRED.md` 里弹性「占位截面」那句已经过时：截面已换成 TOPAS 表。

### 和旧 TOPAS 生产路径的关系

仓库里曾经有 INCLXX reaction/cascade **包驱动**路径（对标 TOPAS，不是 FRED）。当前这支代码的 GPU inelastic **主路径是 `fred_paper` 抽样器**，不是运行时 INCL。不要和 `issue.md` 里那套 400 MeV/u 编包问题混在一起。

**一句话：** inelastic 的 GPU 骨架、论文截面、Table1 回退、以及 TOPAS 薄靶事件库都已接通；缺的是 FRED 原事件库（或明确接受 INCL 代理）以及水箱剂量对标。

## User

现在gpu蒙卡全物理过程可以运行起来了吗

## Tools

- Search: CARBON_HAS_SYCL|enable_inelastic|device:
- Read: /mnt/sdb/wuwei/MAIGO/scripts/run_linux_nvidia.sh (1-57)
- Execute: ls -la /mnt/sdb/wuwei/MAIGO/build/cpu-debug/carbon_mc /mnt/sdb/wuwei/MAIGO/build/carbon_mc 2>/dev/null; ls /mnt/sdb/wuwei/MAIGO/build/*/carbon_mc 2>/dev/null; grep -E 'CARBON_ENABLE_SYCL|nvptx|HAS_SYCL' /mnt/sdb/wuwei/MAIGO/CMakeLists.txt /mnt/sdb/wuwei/MAIGO/CMakePresets.json 2>/dev/null | head -40
# nvidia gpu present locally?
nvidia-smi -L 2>/dev/null | head
ls /mnt/sdb/wuwei/MAIGO/data/packages/c12_*events.bin /mnt/sdb/wuwei/MAIGO/data/stopping_power_water_geant4_11_3_2.csv 2>/dev/null
 (Check GPU binary, SYCL build, assets, local GPU)
- Execute: ls -la /mnt/sdb/wuwei/MAIGO/build/ | head -40
# is existing carbon_mc sycl?
file /mnt/sdb/wuwei/MAIGO/build/carbon_mc
strings /mnt/sdb/wuwei/MAIGO/build/carbon_mc | grep -E 'CARBON_HAS_SYCL|nvptx|cuda:gpu|sycl' | head
grep -E 'CARBON_ENABLE_SYCL|HAS_SYCL' /mnt/sdb/wuwei/MAIGO/build/CMakeCache.txt 2>/dev/null | head
# oneapi dirs
ls -d /mnt/sdb/wuwei/MAIGO/build/oneapi* /mnt/sdb/wuwei/MAIGO/build/*nvidia* 2>/dev/null
 (See if current carbon_mc is a SYCL GPU build)

## Assistant

**可以跑，但要分清两件事：源码里的 GPU 全物理链已经齐了；磁盘上现成的 `build/carbon_mc` 是 8 月 30 日 10:10 的旧 SYCL 包，还没编进后来的事件库/TOPAS 弹性表。**

本机有 RTX 2080 Ti，`build/CMakeCache.txt` 里 `CARBON_ENABLE_SYCL=ON`，现有二进制也链了 SYCL。要跑**当前仓库**的全物理，需要先按 `scripts/build_linux_oneapi.sh` 重编，再用：

```bash
./scripts/run_linux_nvidia.sh
# 或
CONFIG=config/beam_200MeVu_fred_paper.yaml DEVICE=cuda ./scripts/run_linux_nvidia.sh
```

### 打开后 GPU 上实际会算的

水箱 C-12 这条「全物理」是凝聚历史蒙卡，不是 Geant4 全过程：

| 过程 | GPU |
|---|---|
| 电离能损（表插值） | 开 |
| 能损涨落 | 开（`vavilov_landau` 或高斯；**不是**那张 TOPAS inverse-CDF 包） |
| Highland MCS | 开（可 ×1.40） |
| 非弹碎裂 + 次级带电输运 | 开（论文截面 + 95 MeV/u H/O 事件库，失败回退 Table 1） |
| H 靶核弹性 + 反冲质子 | 开（`enable_nuclear_elastic`） |
| 中子/光子输运 | **关** |
| δ 电子显式输运 | **无** |
| 碎片二次碎裂 | **无**（FRED 论文默认也关） |
| 衰变 | **无** |

### 不能当成已经验收的

- 旧 GPU 可执行文件不含事件库加载。  
- 还没有 100–400 MeV/u 水箱对 FRED/FLUKA 的剂量关门。  
- CPU `serial` 不跑 inelastic。

**结论：** 重编 SYCL 后，GPU 可以把「电磁 + 弹性 + 非弹 + 次级带电」整条链跑起来；不是 TOPAS/Geant4 那种电子/中子/级联全清单，也还没证明剂量对。

## User

编译运行100,200,300MeV的结果，画出plots文件夹中的图，使用1M粒子数，slurm之前已经运行结束了1M粒子数的topas结果

## Assistant

先定位已有的 1M TOPAS 结果、绘图脚本和 GPU 编译方式，再编 SYCL、跑 100/200/300 MeV/u 并出图。

## Tools

- ListDir: plots
- Read: /mnt/sdb/wuwei/MAIGO/scripts/build_linux_oneapi.sh (1-80)
- Read: /mnt/sdb/wuwei/MAIGO/scripts/generate_plots_matplotlib.py (1-80)
- Search: 1M|1e6|1000000
- Read: /mnt/sdb/wuwei/MAIGO/scripts/generate_plots_matplotlib.py (81-186)
- Read: /mnt/sdb/wuwei/MAIGO/config/beam_200MeVu_inelastic_1M.yaml
- Execute: find /mnt/sda/wuwei /mnt/sdb/wuwei/MAIGO -name '*1M*' \( -name '*.bin' -o -name '*.raw' -o -name '*dose*' \) 2>/dev/null | head -80
echo '==== out gpu 1M ===='
ls /mnt/sdb/wuwei/MAIGO/out/gpu_inelastic_1M_e100 /mnt/sdb/wuwei/MAIGO/out/gpu_inelastic_1M_e200 2>/dev/null | head
echo '==== generate_plots rest ===='
 (Find TOPAS 1M dose binaries and GPU 1M outputs)
- Execute: ls -d /mnt/sda/wuwei/carbon_emittance_inelastic* 2>/dev/null
ls /mnt/sda/wuwei/carbon_emittance_inelastic_1M 2>/dev/null | head
ls /mnt/sda/wuwei/carbon_emittance_inelastic_1M/e100 2>/dev/null | head
find /mnt/sda/wuwei -maxdepth 3 -iname '*inelastic*1M*' -o -iname '*emittance_inelastic*' 2>/dev/null | head -40
# also 100k path used in plot script
ls /mnt/sda/wuwei/carbon_emittance_inelastic_100k/e100 2>/dev/null | head
 (Locate TOPAS 1M inelastic emittance dose binaries)
- Read: /mnt/sdb/wuwei/MAIGO/scripts/generate_lateral_profiles.py
- Read: /mnt/sdb/wuwei/MAIGO/CMakePresets.json (1-80)
- Read: /mnt/sda/wuwei/carbon_emittance_inelastic_1M/e100/topas_emittance_inelastic_e100.binheader (1-10)
- Read: /mnt/sdb/wuwei/MAIGO/scripts/generate_lateral_profiles_plot.py (1-80)
- Read: /mnt/sdb/wuwei/MAIGO/scripts/build_linux_oneapi.sh (81-86)
- Read: /mnt/sdb/wuwei/MAIGO/out/gpu_inelastic_1M_e100/voxel_dose.mhd (1-14)
- Read: /mnt/sdb/wuwei/MAIGO/config/beam_100MeVu_inelastic_1M.yaml (1-47)
- Read: /mnt/sdb/wuwei/MAIGO/scripts/generate_lateral_profiles_plot.py (101-136)
- Read: /mnt/sda/wuwei/carbon_emittance_inelastic_1M/configs/e100.txt (1-52)
- Edit: /mnt/sdb/wuwei/MAIGO/config/beam_100MeVu_fred_paper_1M.yaml
- Edit: /mnt/sdb/wuwei/MAIGO/config/beam_200MeVu_fred_paper_1M.yaml
- Edit: /mnt/sdb/wuwei/MAIGO/config/beam_300MeVu_fred_paper_1M.yaml
- Execute: # Check if oneapi already sourced / icpx exists
command -v icpx; ls /opt/intel/oneapi/setvars.sh 2>/dev/null
ls /mnt/sdb/wuwei/MAIGO/build/oneapi-nvidia-release/carbon_mc 2>/dev/null
 (Locate icpx and existing NVIDIA SYCL build)
- Read: /mnt/sdb/wuwei/MAIGO/build/CMakeCache.txt (1-80)
- Search: CMAKE_CXX_COMPILER|CUDA_ARCH|SYCL
- Execute: export PATH="/home/wuwei/sycl_workspace/llvm/build/install/bin:$PATH"
export LD_LIBRARY_PATH="/home/wuwei/sycl_workspace/llvm/build/install/lib:${LD_LIBRARY_PATH:-}"
cd /mnt/sdb/wuwei/MAIGO
cmake --build build --target carbon_mc -j"$(nproc)" (Rebuild existing SYCL carbon_mc with current Path A physics)

## Assistant

GPU 已编好。开始跑 100 MeV/u 的 1M，随后是 200 和 300。

## Tools

- Read: /mnt/sdb/wuwei/MAIGO/plots/benchmark_100MeVu_idd_sigma.png
- Read: /mnt/sdb/wuwei/MAIGO/plots/benchmark_200MeVu_idd_sigma.png
- Read: /mnt/sdb/wuwei/MAIGO/plots/benchmark_300MeVu_idd_sigma.png

## Assistant

已在本机 RTX 2080 Ti 上重编 SYCL（sm_75），跑完 **100/200/300 MeV/u × 1M**，并对照 slurm 的 TOPAS 1M（`/mnt/sda/wuwei/carbon_emittance_inelastic_1M/`）重画了 `plots/`。

### 运行

| 能量 | GPU 输出 | 吞吐 | 非弹次数 |
|---|---|---|---|
| 100 MeV/u | `out/gpu_fred_paper_1M_e100/` | 2.55×10⁵ hist/s，~4 s | 1.34×10⁵ |
| 200 MeV/u | `out/gpu_fred_paper_1M_e200/` | （同类） | 3.60×10⁵ |
| 300 MeV/u | `out/gpu_fred_paper_1M_e300/` | 1.23×10⁵ hist/s，~8 s | 5.65×10⁵ |

配置：`config/beam_{100,200,300}MeVu_fred_paper_1M.yaml`（emittance 与 TOPAS 相同：σ=3 mm，σ′=0.006，ρ=0.7）。

### 图

- `plots/benchmark_{100,200,300}MeVu_idd_sigma.png`
- `plots/lateral_profiles_{100,200,300}MeVu.png` 与 `_log.png`

### 从图上能直接看到的

- **100 MeV/u**：峰前 IDD 几乎贴住；远端半影有一条很深的负尖（峰后一两个体素）。core σ 峰前很好，halo 偏宽约 7–12%。
- **200 MeV/u**：平台略高约 2%；Bragg 尖峰处相对误差发散（体素对齐/峰高）；尾巴约 −3%。halo 在平台区偏窄约 10–15%。
- **300 MeV/u**：平台偏高约 5–8%；峰处同样有尖刺；尾巴约 −8%。core σ 峰后偏大约 5–15%。

日志里 **`Primary elastic interactions: 0`**、**`secondary=0 s`**，说明弹性计数和次级 kernel 计时还有问题；碎片能量大量进了 untracked。图是按现有 voxel 剂量画的，不是已经验收通过的全物理对齐。

## User

画图的时候限制误差的上下限到5%

## Tools

- Read: /mnt/sdb/wuwei/MAIGO/scripts/generate_plots_matplotlib.py (116-170)
- Edit: /mnt/sdb/wuwei/MAIGO/scripts/generate_plots_matplotlib.py
- Edit: /mnt/sdb/wuwei/MAIGO/scripts/generate_plots_matplotlib.py
- Edit: /mnt/sdb/wuwei/MAIGO/scripts/generate_plots_matplotlib.py
- Read: /mnt/sdb/wuwei/MAIGO/scripts/generate_plots_matplotlib.py (169-189)
- Edit: /mnt/sdb/wuwei/MAIGO/scripts/generate_plots_matplotlib.py
- Execute: cd /mnt/sdb/wuwei/MAIGO && python3 - <<'PY'
import os, sys
sys.path.insert(0, 'scripts')
from generate_plots_matplotlib import load_and_analyze, plot_benchmark_png
jobs = [
    (100, 40.0),
    (200, 120.0),
    (300, 220.0),
]
for E, max_z in jobs:
    tpath = f"/mnt/sda/wuwei/carbon_emittance_inelastic_1M/e{E}/topas_emittance_inelastic_e{E}.bin"
    gpath = f"out/gpu_fred_paper_1M_e{E}/voxel_dose.raw"
    z_all, t_idd, g_idd, z_samples, t_sc, t_sh, g_sc, g_sh = load_and_analyze(tpath, gpath, max_z)
    plot_benchmark_png(E, z_all, t_idd, g_idd, z_samples, t_sc, t_sh, g_sc, g_sh,
                       f"plots/benchmark_{E}MeVu_idd_sigma.png")
print('done')
PY (Redraw 100/200/300 benchmark plots with ±5% error limits)

## Assistant

已把 IDD 和横向 σ 的相对误差纵坐标固定为 **±5%**，并重画了 100/200/300 MeV/u 的 1M 对照图：

- `plots/benchmark_100MeVu_idd_sigma.png`
- `plots/benchmark_200MeVu_idd_sigma.png`
- `plots/benchmark_300MeVu_idd_sigma.png`

峰后超过 ±5% 的尖刺会被裁在坐标轴外，平台区误差更容易读。侧向剖面图本身没有误差子图，未改。
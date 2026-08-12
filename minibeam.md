# Minibeam 物理剂量实现与验证路线

更新日期：2026-07-27

## 1. 目标与范围

目标是在不破坏现有 water、CT、TOPAS spots 和 `tpsSource` 工作流的前提下，
为 MAIGO 增加铜准直器 minibeam，并先让 GPU 与 TOPAS 的绝对物理剂量对上。

当前阶段只处理：

- 初级 C-12 经过铜准直器后的透射、能损和散射；
- 铜中核反应产生的碎片进入水后的剂量；
- 水中的三维 `DoseToMedium`；
- 横向峰谷结构、深度剂量和绝对剂量；
- 单 spot、简化照射野和完整 961-spot 照射野。

当前阶段明确不处理：

- LET、LET scorer 或 LET 文件；
- RBE/生物剂量；
- 针对当前一个 case 的经验剂量缩放；
- 一开始就直接在临床 CT 上调参。

所有新开关默认关闭。`minibeam: false` 时必须保持现有 CT example、TPS
source 和水中验证结果不变。

## 1.1 当前执行状态（`minibeam` branch）

已于 2026-07-27 创建并切换到本地 `minibeam` 分支，开始按本文路线实施。当前已
完成 Phase 0–4：除 primary attenuation 外，Copper 带电反应产物和中性粒子的
首个延续分支均已接入；中子非弹性反应的多中性分支仍是待完成项：

- 新增固定 seed、40 线程、关闭 LET 的 dose-only TOPAS 基准；
- 新增中心单 spot 的准直器入口、出口和水入口 phase-space case；
- 新增 TOPAS dense binary → GPU 坐标 MHD 转换器、轴序/文件尺寸测试；
- 新增支持 4000 万体素分块读取的绝对剂量比较器，不拟合 dose scale；
- 新增 `minibeam: true/false`、几何参数和
  `minibeam_transport_mode: absorbing_geometry` 配置；
- 新增 `spots_geometry_mode: minibeam_topas_y`，保留源到准直器的相空间；
- 新增 15-slit beam-frame 几何和直线完全穿孔判定；
- 从同一 TOPAS 4.2.p3 / Geant4 11.3.2 提取 Copper、Air stopping power；
- 新增 `copper_em` 模式：解析空气漂移、Copper 细步长能损和 MCS；
- 用 0.1–5 mm 独立 Copper foil benchmark 冻结 MCS 核心尺度，而不是拟合
  minibeam 剂量；
- 提取同版本 Copper C-12 macroscopic inelastic cross section，并加入数据驱动
  primary attenuation；
- `minibeam: false` 默认路径不变，现有测试全部通过；
- LET 在全部新配置中保持关闭。

可复现入口：

```bash
# TOPAS 中心 spot 的三个 phase-space 平面
validation/topas/run_minibeam_topas.sh run_phase_space_center.txt

# TOPAS 中心 spot 全 Copper 剂量
validation/topas/run_minibeam_topas.sh run_dose_center_10k.txt

# TOPAS binary 转 MHD
python3 validation/scripts/prepare_minibeam_topas_dose.py \
  --input ct/minibeam/output/minibeam_center_dose_10k.bin \
  --binheader ct/minibeam/output/minibeam_center_dose_10k.binheader \
  --output-mhd out/minibeam/topas_center_10k/dose.mhd \
  --metrics-json out/minibeam/topas_center_10k/metrics.json \
  --expected-slit-pitch-mm 3.6

# GPU 几何吸收基准
build/oneapi-release/carbon_mc \
  --config config/beam_minibeam_center_absorbing_10k.yaml

# 相同 10k incident histories 的绝对剂量比较
python3 validation/scripts/compare_minibeam_dose.py \
  --reference out/minibeam/topas_center_10k/dose.mhd \
  --evaluation out/minibeam/absorbing_center_10k/dose.mhd \
  --output out/minibeam/comparison/center_10k_absorbing_vs_topas.json
```

中心单 spot 的实测结果如下：

| 项目 | 结果 |
|---|---:|
| TOPAS / GPU incident histories | 10,000 / 10,000 |
| TOPAS 40-thread 墙钟时间 | 48.81 s |
| GPU Titan RTX 墙钟时间 | 3.66 s |
| TOPAS 准直器入口 primary C-12 | 9,979 |
| TOPAS 准直器出口 primary C-12 | 1,058 |
| 直线完全穿过同一空气 slit | 734 |
| GPU absorbing / TOPAS 积分剂量 | 0.7492 |
| GPU 积分剂量差 | -25.08% |
| TOPAS / GPU 深度积分峰位 | 70.75 / 70.95 mm |
| ≥1% TOPAS peak 体素 Pearson r | 0.196 |

phase-space 的逐 history 检查进一步表明：734 条几何直达轨迹全部出现在 TOPAS
出口，但 TOPAS 还有 324 个打到 Copper 后仍以 primary C-12 身份出射的 history。
因此当前 absorbing 模式正确承担的是坐标/狭缝接受度单元测试，不能作为物理
minibeam 剂量。约 25% 的积分剂量缺口和较差的空间相关性证明下一步必须实现
Copper stopping power、MCS/单次大角散射，再加入 Copper 核反应碎片；不能用
case-specific 全局 scale 校准。

Phase 1 的中心 10k open-field 对照也已完成。TOPAS 将 `Aperture` 改为 Air，
GPU 则关闭 `minibeam`，其余 source、水物理、网格和 seed 不变：

| 项目 | TOPAS | GPU |
|---|---:|---:|
| incident histories | 10,000 | 10,000 |
| 墙钟时间 | 189.53 s（40 CPU threads） | 6.88 s（Titan RTX） |
| 积分体素剂量 | 309.602 Gy | 314.042 Gy |
| 相对积分剂量差 | — | +1.43% |
| 深度积分峰位 | 70.55 mm | 70.75 mm |
| IDD Pearson / normalized L1 | — | 0.9993 / 1.85% |
| X 积分 profile Pearson / L1 | — | 0.9955 / 8.23% |
| Y 积分 profile Pearson / L1 | — | 0.9966 / 5.62% |

这说明在不加经验 scale 的前提下，现有水中绝对输出和射程已经足以作为 Copper
开发基线。10k histories 分布到 4000 万个高分辨率体素后，逐体素指标由 MC
噪声主导；open-field 的逐体素 Pearson `r=0.502` 不能用作物理拒收标准。后续
应优先看 IDD、积分横向 profile，并用多 seed 给 valley/PVDR 建立不确定度。

### 1.2 Phase 3/4 当前实测

Copper 和 Air stopping power 均通过 `G4EmCalculator` 从参考版本直接提取。
在 `2150 MeV` C-12、纯 EM、100k histories 的独立均匀 Copper foil 中，TOPAS
与当前解析 Highland 模型的出射投影角宽如下：

| Copper 厚度 (mm) | TOPAS RMS / Highland | TOPAS core68 / Highland |
|---:|---:|---:|
| 0.1 | 0.889 | 0.799 |
| 0.5 | 0.843 | 0.794 |
| 1.0 | 0.820 | 0.785 |
| 2.0 | 0.801 | 0.776 |
| 5.0 | 0.797 | 0.771 |

因此 `minibeam_copper_mcs_scale=0.785` 取独立 benchmark 的 core68 中位数并冻结。
它不是从中心 minibeam 剂量或 gamma 反推的。全 RMS 比 core68 略大，说明 Geant4
还包含非高斯尾；当前 GPU 高斯模型优先匹配决定峰宽和 PVDR 的核心，尾部留给后续
单次大角散射实现。

可复现命令：

```bash
for f in run_copper_mcs_foil_0p1mm.txt \
         run_copper_mcs_foil_0p5mm.txt \
         run_copper_mcs_foil_1mm.txt \
         run_copper_mcs_foil_2mm.txt \
         run_copper_mcs_foil_5mm.txt; do
  validation/topas/run_minibeam_topas.sh "$f"
done

python3 validation/scripts/analyze_copper_mcs_foil.py \
  --case 0.1:ct/minibeam/output/copper_mcs_foil_0p1mm.phsp \
  --case 0.5:ct/minibeam/output/copper_mcs_foil_0p5mm.phsp \
  --case 1:ct/minibeam/output/copper_mcs_foil_1mm.phsp \
  --case 2:ct/minibeam/output/copper_mcs_foil_2mm.phsp \
  --case 5:ct/minibeam/output/copper_mcs_foil_5mm.phsp \
  --output-json out/minibeam/copper_mcs_foil_benchmark.json
```

Copper C-12 非弹性截面同样由
`G4HadronicProcessStore::GetInelasticCrossSectionPerVolume` 直接查询，不拟合
透射曲线；`200 MeV/u` 时为 `0.0135769 mm^-1`，平均自由程 `73.6546 mm`。
当前 attenuation 只终止发生反应的 primary，尚未生成 Copper 反应碎片：

```bash
TOPAS_MINIBEAM_INSTALL=build/opentopas-minibeam-install \
  validation/topas/run_minibeam_topas.sh run_copper_cross_sections.txt

python3 validation/scripts/prepare_minibeam_copper_cross_sections.py \
  --input ct/minibeam/output/copper_c12_inelastic_cross_sections.phsp \
  --header ct/minibeam/output/copper_c12_inelastic_cross_sections.header \
  --log ct/minibeam/output/run_copper_cross_sections.log \
  --output-csv data/c12_inelastic_cross_sections_copper_geant4_11_3_2.csv \
  --metadata data/c12_inelastic_cross_sections_copper_geant4_11_3_2.metadata.json
```

中心 10k history 的水入口 primary 相空间比较：

| 指标 | TOPAS full physics | GPU Copper attenuation | 相对差 |
|---|---:|---:|---:|
| primary C-12 数 | 1058 | 1112 | +5.10% |
| 能量均值 (MeV) | 1982.68 | 1955.83 | -1.35% |
| X 位置 std (mm) | 3.3708 | 3.4125 | +1.24% |
| Y 位置 std (mm) | 4.4906 | 4.3332 | -3.51% |
| X 方向 std | 0.007280 | 0.007515 | +3.23% |
| Y 方向 std | 0.009104 | 0.009322 | +2.39% |

相同 10k incident histories、无剂量缩放的 total-dose 比较目前为：

| 指标 | GPU vs TOPAS |
|---|---:|
| 水中积分剂量差 | +3.22% |
| IDD normalized L1 / Pearson | 5.99% / 0.9924 |
| X 横向积分 profile L1 / Pearson | 24.56% / 0.9690 |
| Y 横向积分 profile L1 / Pearson | 17.11% / 0.9813 |
| Bragg peak 深度差 | -0.30 mm |

4000 万体素只有 10k histories，横向 L1 仍混有很强统计噪声；下一步应先生成 Copper
reaction packages，将铜来源 charged/neutral products 接到已有水中队列，再以
至少 1M histories 比较 PVDR、halo 和逐平面 gamma。beamline energy accounting
现已把空气/Copper 能损、准直器内停止、反向逃逸和 Copper nuclear termination
单独计入；相同 10k run 的控制台 energy-balance error 从原先误导性的约 `0.90`
降到 `9.4e-5`，不会再把准直器吸收误报为能量不守恒。

Phase 4 的第一套末态开发样本也已生成：

```bash
TOPAS_MINIBEAM_INSTALL=build/opentopas-minibeam-install \
  validation/topas/run_minibeam_topas.sh \
  run_copper_reactions_2150MeV_20k.txt

python3 validation/scripts/prepare_topas_reactions.py \
  --case development --histories 20000 \
  --input ct/minibeam/output/copper_reactions_2150MeV_20k.phsp \
  --header ct/minibeam/output/copper_reactions_2150MeV_20k.header \
  --log ct/minibeam/output/run_copper_reactions_2150MeV_20k.log \
  --phantom-half-length-mm 30 \
  --reactions-output data/copper_c12_reactions_2150MeV_20k.csv.gz \
  --secondaries-output data/copper_c12_secondaries_2150MeV_20k.csv.gz \
  --metadata data/copper_c12_reactions_2150MeV_20k.metadata.json
```

20k histories 中有 3202 次 primary C-12 非弹性反应和 59,623 个直接产物，平均
每个 package 18.62 个产物。参考 minibeam 的水入口除 1058 个 primary 外，还有
242 个其他离子、92 个质子和 3402 个中子；带电碎片总动能约为 primary 总动能的
3.7%。这些数值说明只做 attenuation 不够，且不能在反应点直接把产物当作已到达
水入口：必须继续经过剩余 Copper 和 60 mm Air，并保留同一反应内的能量/方向相关性。

开发样本已能用现有 compiler 编译为 36 个 `5 MeV/u` bin、约 `1.48 MB` 的二进制
package：

```bash
python3 validation/scripts/compile_reaction_package.py \
  --metadata data/copper_c12_reactions_2150MeV_20k.metadata.json \
  --reactions data/copper_c12_reactions_2150MeV_20k.csv.gz \
  --secondaries data/copper_c12_secondaries_2150MeV_20k.csv.gz \
  --energy-bin-min-mevu 0 --energy-bin-width-mevu 5 --energy-bin-count 36 \
  --output out/minibeam/copper_c12_reaction_packages_2150MeV_20k.bin \
  --output-metadata \
    out/minibeam/copper_c12_reaction_packages_2150MeV_20k.metadata.json
```

该 package 已编译到
`data/copper_c12_reaction_packages_2150MeV_20k.bin` 并接入 GPU。发生 Copper
反应时，代码按入射能量选择相关末态 package，在反应点把局部产物方向旋转到
parent 方向；所有带电产物继续逐步输运剩余 Copper 和 60 mm Air，只有到达水入口
的 survivor 才进入现有 water secondary queue。这个 beamline-secondary stage
避免了把“铜中生成”误当成“已经进入水中”。

C、B、Be、Li、He、p、d、t 等 32 个同位素在 Copper 中使用与参考
Geant4 11.3.p2 相同版本抽取的独立 stopping-power 和非弹性宏观截面表：

```text
data/ion_stopping_power_copper_geant4_11_3_2.csv
data/ion_cross_sections_copper_geant4_11_3_2.csv
```

当前 `config/beam_minibeam_center_copper_em_10k.yaml` 的默认路径已开启 Copper
反应产物，但关闭昂贵的逐粒子深度计分；需要来源诊断时才临时打开
`enable_fragment_species_scoring`。

### 1.5 Phase 4：100k 正式验证结果

`run_phase_space_center_100k.txt` 与 GPU 100k 使用相同中心 spot 和相同 incident
history 数。水入口 primary C-12 为：

| 指标 | TOPAS | GPU | GPU 相对差 |
|---|---:|---:|---:|
| primary C-12 数 | 10,869 | 10,925 | +0.515% |
| 平均动能 (MeV) | 1965.29 | 1956.62 | -0.441% |
| X 位置 std (mm) | 3.4643 | 3.4335 | -0.89% |

10k case 中 1112 对 1058 的 `+5.1%` primary 数差因此不能当作系统偏差；100k 后
只剩约 1.3 个合并二项统计标准差。不能按 10k 的单次涨落重新拟合 Copper 截面。

Copper 来源、到达水入口的带电产物数在 100k 下为 GPU 3204、TOPAS 约 3357，
GPU 低约 4.6%。其中 GPU/TOPAS 的 p 为 928/1006、d 为 1055/1132、t 为
881/909。中性粒子现已接入 production 配置；但 10k dose-origin 结果显示
neutron+gamma 只占积分剂量约 0.19%，因此它主要完善 halo/valley，不是当前总剂量
差的主因。

100k full-dose 使用完全相同的 4000 万体素网格、绝对 Gy、相同 incident
histories，且没有 fitted scale：

| 指标 | GPU vs TOPAS |
|---|---:|
| 水中积分剂量差 | +1.55% |
| IDD normalized L1 / Pearson | 3.22% / 0.99661 |
| X 横向积分 profile L1 / Pearson | 7.40% / 0.99800 |
| Y 横向积分 profile L1 / Pearson | 5.09% / 0.99800 |
| Bragg peak 深度 | 两者均为 70.45 mm |
| X/Y profile 峰位置 | 两者均一致 |

结果保存在：

```text
out/minibeam/topas_center_100k/dose.mhd
out/minibeam/copper_em_center_100k/dose.mhd
out/minibeam/comparison/center_100k_copper_products_boundary_fix_vs_topas.json
out/minibeam/topas_center_100k_phase_space_summary.json
```

100k TOPAS 仍只有约 596.7 万个非零体素分布在 4000 万体素中，逐体素 L1 和逐体素
相对误差仍由稀疏蒙卡噪声主导；当前阶段应使用积分剂量、IDD、横向积分 profile、
PVDR 分区统计或经过统计不确定度处理的 gamma，不能继续按单体素差异调物理参数。

可复现的 1D 剂量图由以下命令生成：

```bash
MPLCONFIGDIR=/tmp/maigo-matplotlib \
python3 validation/scripts/plot_minibeam_1d_dose.py \
  --reference out/minibeam/topas_center_100k/dose.mhd \
  --evaluation out/minibeam/copper_em_center_100k/dose.mhd \
  --output-dir out/minibeam/comparison/one_dimensional_profiles \
  --depths-mm 0.5 35 70.45 --slab-width-mm 1.0
```

深度图使用全横向平面积分的绝对 Gy；横向图分别显示水入口、35 mm 和 Bragg peak
附近 1 mm 厚深度层的 X/Y 积分 profile。GPU 与 TOPAS 使用相同的 100k incident
histories，没有归一化或拟合 scale。

### 1.6 FP32 voxel-boundary 修复与速度

CUDA 上原先用单次 `nextafter` 跨越 0.1 mm scorer 边界。由于
`(position-origin)/spacing` 的 FP32 消减误差，相邻多个 representable value
仍可能被映射回旧 voxel，导致同一边界反复停留；而循环还递增 RNG step，既拖慢
输运也改变后续核反应抽样。现在统一跨过 `1e-5 mm` 的微小物理距离，再用
`nextafter` 作为极端数值 fallback。

10k 诊断中：

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| primary boundary-nudge 次数 | 307,899,854 | 17,335 |
| primary kernel | 10.14 s | 1.16 s |
| 0.1 mm 总深度 Pearson | 约 0.86 | 0.991 |

关闭仅用于诊断的 fragment-species scorer 后，100k 完整 GPU run 为 13.82 s，
约 7236 histories/s；TOPAS 40 线程同一 100k full-dose 为 329.34 s execution，
GPU 约快 23.8 倍。

### 1.7 Copper 中性粒子延续输运

从同一 TOPAS 4.2.p3 / Geant4 11.3.2 直接提取 Copper 中 gamma、neutron 的
总宏观截面，能量范围为 `1e-4–1000 MeV`：

```text
data/neutral_cross_sections_copper_geant4_11_3_2.csv
```

另外使用 20k 入射 C-12 的独立 Copper slab case，记录 481,402 次中性相互作用和
475,785 个产物，编译为条件化末态包：

```text
data/copper_neutral_packages_2150MeV_20k.bin
```

其中包含 427,115 次 gamma 相互作用（光电、康普顿、瑞利和转化）以及 54,287 次
neutron 相互作用（弹性、非弹性和俘获）。GPU 先用精细截面表在剩余 Copper 几何中
抽样碰撞位置，再按粒子种类和当前能量从末态包抽取延续粒子的能量与方向。光电吸收
等没有延续中性粒子的过程会终止；康普顿、瑞利和 neutron 弹性则继续输运，不再把
总截面全部误当成吸收。此过程没有 fitted dose scale。

配置必须同时给出：

```yaml
minibeam_copper_neutral_cross_section_file: data/neutral_cross_sections_copper_geant4_11_3_2.csv
minibeam_copper_neutral_package_file: data/copper_neutral_packages_2150MeV_20k.bin
```

10k 快速 A/B 中，Copper 来源的中性 survivor 从仅吸收模型的 2368 个、
127.6 GeV，提高到延续模型的 4368 个、200.4 GeV；TOPAS 水入口的 gamma+neutron
合计为 4940 个、166.9 GeV。因而计数误差从约 `-52%` 降到 `-11.6%`，但能量仍
高约 `20%`。剩余差异的主要原因是当前仅跟踪每次反应的 leading neutral：
neutron 非弹性反应产生的额外 neutron/gamma 尚未作为分支进入 beamline queue。

100k 正式剂量（相同 incident histories、绝对 Gy、无 scale）为：

| 指标 | GPU vs TOPAS |
|---|---:|
| GPU 墙钟时间 | 14.52 s |
| 水中积分剂量差 | +1.585% |
| IDD normalized L1 / Pearson | 3.237% / 0.99661 |
| X 横向积分 profile L1 / Pearson | 7.393% / 0.99800 |
| Y 横向积分 profile L1 / Pearson | 5.093% / 0.99800 |
| Bragg peak 深度 | 两者均为 70.45 mm |

相对仅带电末态版本，total-dose 指标变化很小，和 TOPAS 中性来源仅占约 0.19% 的
诊断一致。继续投入精度时，应先增加 Copper neutral 多分支 queue，再通过多 seed
或至少 1M histories 评估 PVDR/gamma；不应从当前 100k 稀疏体素噪声反调材料参数。

独立 seed 的 1M TOPAS/GPU 正式比较随后完成。TOPAS 使用 40 threads、
`seed=20260801`，execution real time 为 `2659.05 s`；GPU 使用 Titan RTX，
耗时 `105.07 s`，约快 `25.3×`，且 charged/neutral queue overflow 均为零。

| 指标 | 100k | 1M |
|---|---:|---:|
| 水中积分剂量差 | +1.585% | +1.743% |
| IDD normalized L1 | 3.237% | 3.055% |
| X 横向 profile L1 | 7.393% | 3.192% |
| Y 横向 profile L1 | 5.093% | 2.140% |
| IDD Pearson | 0.99661 | 0.99763 |
| X/Y Pearson | 0.99800 / 0.99800 | 0.99961 / 0.99974 |

横向 L1 随统计量增加显著下降，证明 100k 的大部分横向差异是稀疏噪声；X profile
按 0.2 mm 物理分箱后 L1 为 `2.936%`。IDD 在 0.1–1.0 mm 分箱下仍约
`3.045–3.055%`，且 GPU 峰前剂量系统性略高、Bragg peak 比 TOPAS 浅约
`0.30 mm`，因此下一步精度改进应针对 primary Copper 出射能谱/水中 stopping
与 range，而不是继续单纯增大粒子数。

可复现命令：

```bash
build/oneapi-cuda-profile/carbon_mc \
  --config config/beam_minibeam_center_copper_neutral_100k.yaml \
  --device cuda --histories 100000

python3 validation/scripts/compare_minibeam_dose.py \
  --reference out/minibeam/topas_center_100k/dose.mhd \
  --evaluation out/minibeam/copper_neutral_center_100k/dose.mhd \
  --output \
    out/minibeam/comparison/center_100k_copper_neutral_continuation_vs_topas.json
```

## 2. 已有 TOPAS case 审计

参考文件位于 `ct/minibeam/`：

| 文件 | 内容 |
|---|---|
| `run1.txt` | 水箱、束源、物理列表和 scorer |
| `aperture.txt` | 铜准直器及 15 条空气狭缝 |
| `spots961.txt` | 961 个 TOPAS TimeFeature spot |
| `dose.bin` | TOPAS `DoseToMedium` 二进制结果 |
| `dose.binheader` | 剂量网格和 TOPAS 版本信息 |

### 2.1 几何

TOPAS 世界坐标中的几何为：

- 水箱 `Box`：`100 × 100 × 100 mm³`；
- 水箱中心：`(0, 110, 0) mm`；
- 水入口面：TOPAS world `Y = 60 mm`；
- 铜准直器：半径 `60 mm`、厚度 `60 mm`；
- 准直器沿 world Y 覆盖 `[-60, 0] mm`；
- 准直器出口到水入口的空气隙：`60 mm`；
- 实际材料参数是 `Copper`。`aperture.txt` 中的
  `# BRASS CYLINDER` 只是过时注释，不能据此使用黄铜。

狭缝为 15 条平行、贯穿铜厚度的空气孔：

- 狭缝宽度：`0.5 mm`；
- 中心间距：`3.6 mm`；
- 中心位置：`n × 3.6 mm`，`n = -7...7`；
- 沿另一横向方向的全长：`50 mm`；
- 最外侧狭缝中心：`±25.2 mm`；
- TOPAS 对准直器及孔设置的最大步长为 `0.05 mm`。

以水入口为 GPU 深度零点，推荐固定映射：

```text
GPU x = TOPAS world X
GPU y = TOPAS world Z
GPU z = TOPAS world Y - 60 mm
```

因此：

```text
铜准直器：GPU z = [-120, -60] mm
空气隙：  GPU z = [-60, 0] mm
水体：    GPU z = [0, 100] mm
```

### 2.2 束源

`spots961.txt` 的内容可被现有 `TopasSpotPlan` 解析器复用：

- `31 × 31 = 961` 个 spot；
- 每个 spot `100,000` histories；
- 总 histories：`96,100,000`；
- C-12 总动能：`2150 MeV`，即 `179.1667 MeV/u`；
- 能散：`1.2% RMS`；
- source plane：world `Y = -450 mm`；
- L5 范围：`-27.8325...27.8325 mm`；
- L6 范围：`-28.0735...28.0735 mm`；
- `SigmaX = 2.5061 mm`，`SigmaY = 2.4669 mm`；
- `SigmaXprime = 0.0051`，`SigmaYprime = 0.0041`；
- 相关系数分别为 `0.4177` 和 `0.8069`。

不能未经验证就直接使用现有 `spots_geometry_mode: topas`。当前主输运核会把
体模外的 source 投影到 `z=0`，而且 TOPAS component 的平移、旋转方向和局部
出射轴存在符号约定。对普通 CT 这一步已经针对特定坐标系处理过，但 minibeam
必须保留 source 到准直器之间的相空间，错误的旋转即使只有几 mrad，也会改变
0.5 mm 狭缝的透射率。

建议先让 TOPAS 在以下三个平面输出初级粒子的 position/direction phase space，
用数据确认变换，而不是仅根据参数文件推断：

1. 准直器入口；
2. 准直器出口；
3. 水入口。

### 2.3 剂量参考

TOPAS 剂量 scorer 为：

```text
Quantity: DoseToMedium
Output: binary
Grid: X=1000, Y=1000, Z=40
Spacing: X=0.1 mm, Y=0.1 mm, Z=1.0 mm
```

`dose.bin` 大小为 `320,000,000 bytes`，与
`40,000,000 × float64` 一致。本机按 little-endian float64 读取后：

- 最大体素剂量约 `0.1131184 Gy`；
- 沿深度积分后的最大值位于约 `70.25 mm`；
- 横向频谱主周期约 `3.57 mm`，与 `3.6 mm` 狭缝间距一致。

按 TOPAS 的 X-fastest 顺序，建议统一成 GPU/MHD 的 `(z, y, x)`：

```python
topas_zyx = raw.reshape(40, 1000, 1000)       # TOPAS (Z, Y, X)
gpu_zyx = topas_zyx.transpose(1, 0, 2)       # GPU (depth, lateral-y, lateral-x)
```

对应 GPU scorer 应为：

```text
DimSize: 1000 × 40 × 1000
Spacing: 0.1 × 1.0 × 0.1 mm
GPU x: [-50, 50] mm
GPU y: [-20, 20] mm
GPU z: [0, 100] mm
```

转换脚本必须再通过峰间距、深度 Bragg peak 和中心坐标检查确认轴序，不能只依赖
文件大小。

### 2.4 当前参考输入需要补强的地方

在调 GPU 物理前，建议先生成一套 dose-only TOPAS 基准：

- 在 `run1.txt` 中关闭 LET scorer；本阶段不让 LET 扩展影响复现流程；
- 显式设置固定 `Ts/Seed`；
- 保存完整 TOPAS console log；
- 记录 TOPAS、Geant4、物理列表和 `Copper` 实际解析出的密度/组成；
- 输出至少 3 个独立随机种子，估计高分辨率 valley dose 的统计不确定度；
- 增加单 spot、无准直器和相空间诊断 case。

当前 96.1M histories 被分到 4000 万个体素，低剂量 valley 的逐体素噪声仍可能
明显。不能把 TOPAS 统计噪声误认为 GPU 物理误差。

## 3. 为什么当前 GPU 不能只加一个 `hetero_insert`

现有代码已经具备：

- TOPAS L0-L14 spot 文件解析；
- 多 spot batched SYCL source；
- BiGaussian emittance；
- 水/CT 中的能损、能散、多重散射、衰减和 cascade；
- 三维绝对 Gy MHD 输出。

但 `enable_hetero_insert` 不适合当前准直器：

1. 它只表示水体内部的单个轴对齐 AABB，不能表示圆柱减去 15 条狭缝。
2. 当前粒子进入主 kernel 前会与 `z=[0, phantom_length]` 求交，体模上游的铜会被
   完全跳过。
3. 现有 insert 材料处理只覆盖局部 stopping power 和 C-12 cross section；
   minibeam 还需要铜中的 MCS、核反应末态和铜中产生的碎片。
4. 直接用“水表 × 铜密度”不成立。铜和水的 `Z/A`、平均激发能、辐射长度、
   核反应截面及反应末态均不同。
5. 只把打到铜的初级粒子删除，可以得到几何峰形，但会遗漏铜散射和铜碎片形成的
   valley、halo、入口剂量及远端尾部。

因此 minibeam 应作为“体模上游 beamline transport”实现，而不是 CT/material
voxel 的一个特例。

## 4. 推荐架构

推荐把一次 history 分为两个连续阶段：

```text
TOPAS/TPS source
    │
    ├─ source→collimator 的空气漂移
    │
    ├─ 铜准直器输运
    │    ├─ 狭缝中的快速直达
    │    ├─ 铜中能损、能散和 MCS
    │    └─ 铜中核反应及二次粒子
    │
    ├─ 60 mm 出口空气隙
    │
    ▼
水/CT 入口 phase-space queue（z=0）
    │
    ▼
现有水/CT charged + neutral transport
    │
    ▼
现有 DoseToMedium / MHD writer
```

这样做的好处是：

- 铜中的能量沉积不会误计入水剂量；
- 铜来源的 C、B、Be、Li、He、H 等粒子可以带着真实位置、方向和能量进入水；
- 现有水/CT kernel 和 dose writer 可最大限度复用；
- 以后可把 slit array 换成 pinhole、网格或患者专用 aperture；
- 准直器可以随 beam frame 和 collimator angle 旋转，不依赖固定 world 轴。

入口 queue 至少需要保存：

```text
position x/y/z
direction x/y/z
kinetic energy
Z/A 或 PDG
weight
history/RNG identity
generation/lineage
```

不要把每个出射粒子重新当作一个等权“新 primary”，否则绝对剂量归一化和同一
history 内的多个碎片都会错误。

## 5. 建议的 YAML 接口

当前配置解析器以扁平 key 为主，建议先保持扁平格式。下面是规划接口，不是当前已
实现的参数：

```yaml
# 默认 false；关闭时不能改变任何现有 case。
minibeam: true

minibeam_geometry: parallel_slits
minibeam_material: Copper
minibeam_radius_mm: 60.0
minibeam_thickness_mm: 60.0
minibeam_exit_to_phantom_mm: 60.0

minibeam_slit_count: 15
minibeam_slit_width_mm: 0.5
minibeam_slit_pitch_mm: 3.6
minibeam_slit_half_length_mm: 25.0
minibeam_collimator_angle_deg: 0.0

# TOPAS 961-spot source
topas_spots_file: ct/minibeam/spots961.txt
spots_sad_mm: 450.0
spots_geometry_mode: minibeam_topas_y

# TOPAS dose scorer 对应的 GPU 网格
phantom_length_mm: 100.0
depth_bin_width_mm: 0.1
enable_voxel_scoring: true
voxel_bins_x: 1000
voxel_bins_y: 40
voxel_size_x_mm: 0.1
voxel_size_y_mm: 1.0

# 本阶段关闭 LET
scorerLET: false

minibeam_copper_stopping_power_file: data/stopping_power_copper_*.csv
minibeam_copper_particle_stopping_power_file: data/ion_stopping_power_copper_*.csv
minibeam_copper_cross_section_file: data/c12_inelastic_cross_sections_copper_*.csv
minibeam_copper_cascade_package_file: validation/results/copper_cascade_*.bin
```

准直器的位置应定义在局部 beam frame 中：

- `w`：束流方向；
- `u`：狭缝周期方向；
- `v`：狭缝长轴方向；
- 水或 CT 的入口面为 `w=0`。

`tps_collimator_angle_deg` 或 `minibeam_collimator_angle_deg` 只绕 `w` 旋转
`u/v`。这样同一几何实现可以用于固定 TOPAS Y 向水箱和任意临床 gantry/couch
角度。

## 6. 铜材料物理

### 6.1 电磁输运

至少需要从与参考 TOPAS 相同版本的 Geant4 提取：

- C-12 在 Copper 中的 electronic stopping power；
- cascade 中所有带电同位素在 Copper 中的 stopping power；
- Copper 密度和 radiation length；
- 能量涨落所需材料参数；
- 与 `g4em-standard_opt4` 和 `0.05 mm` production cut 一致的设置。

第一版可以只实现 C-12 铜 stopping power 以定位 Bragg range 和透射能谱，但它
不能作为最终物理结果。铜中产生的轻碎片如果仍使用水表或 C-12 密度缩放，会直接
影响 valley dose。

### 6.2 多重散射

Minibeam 对角分布非常敏感：

- MCS 决定铜边缘散射进入狭缝或水体的比例；
- 出口后还有 60 mm 空气隙，几 mrad 的角差会转化成亚毫米位置差；
- 只匹配水中 IDD 而不匹配出射 angle distribution，通常无法匹配 PVDR。

应分别比较水入口处 primary C-12 的：

- 透射率；
- 能谱；
- `x/y` 分布；
- `direction_x/direction_y` 分布；
- position-angle correlation。

### 6.3 铜中核反应

最终匹配物理剂量需要 Copper 专用：

- 每个带电 projectile 的宏观非弹性截面；
- C-12 和主要碎片的反应末态 package；
- 反应产物的能量、角度和粒子种类；
- 铜中产生并进入水的中性粒子处理。

当前 water cascade package 不能在铜中直接使用。即使调一个总衰减系数匹配了
primary transmission，碎片 valley 和 halo 仍可能错误。

推荐沿用现有 cascade package 提取/编译流程，但把诊断几何材料改为参考 case
实际使用的 `Copper`，并将 package 明确带上：

```text
TOPAS version
Geant4 version
physics list
material name/density/composition
production cut
energy range
source hash
```

当前能量为约 `179 MeV/u`，但为防止单 case 过拟合，铜表和 package 建议至少覆盖
`100–300 MeV/u`。

## 7. 几何输运实现要点

对当前位置 `(u,v,w)`，材料判定可以写成：

```text
inside cylinder = u² + v² < radius²
inside thickness = -120 <= w < -60 mm
inside slit length = |v| < 25 mm
inside any slit = |u - n×3.6| < 0.25 mm, n=-7...7

inside thickness && inside cylinder && !inside any slit => Copper
其他区域 => Air
```

每一步必须限制到最近的材料边界：

- 铜入口/出口面；
- 当前狭缝的左右边；
- 狭缝长轴边界；
- 外圆柱面；
- 水入口面。

不要在所有空气区域使用 `0.05 mm` 小步长。空气和狭缝中应直接 ray-jump 到下一个
边界；只在铜中按能损、相互作用和 MCS 限制步长。否则 source 到水入口约 510 mm
的空气会造成不必要的巨大开销。

一个有用的中间模式是：

```text
minibeam_transport_mode: absorbing_geometry
```

它只保留完全穿过空气狭缝的粒子，打到铜的粒子直接终止。该模式没有完整物理意义，
但能独立验证几何、source transform、狭缝位置和 GPU scorer。

## 8. TOPAS phase-space 桥接

建议先实现一个只用于诊断的 TOPAS phase-space reader：

1. TOPAS 完整模拟 source、铜准直器和空气隙；
2. 在水入口记录所有穿越粒子；
3. GPU 从该 phase space 开始只输运水；
4. GPU 水中剂量与同一次 TOPAS 的水中剂量比较。

它可以快速回答：

- 水中物理是否已经足够；
- 剂量轴序、体素质量和绝对归一化是否正确；
- 主要误差来自铜输运还是水输运；
- 铜来源碎片进入水后，现有 GPU 队列是否能正确继续输运。

但 phase-space bridge 依赖 TOPAS，不能替代最终原生 GPU 铜输运，也不能作为
“GPU 已实现 minibeam 物理”的最终结论。

## 9. 分阶段实施顺序

### Phase 0：冻结可复现参考

- 创建 dose-only TOPAS 配置；
- 固定 seed、版本和材料信息；
- 写 `dose.bin` → MHD 转换及轴序检查脚本；
- 输出准直器入口、出口和水入口 phase space；
- 保存 3 个随机种子的参考剂量或不确定度。

验收：重复运行得到统计一致的绝对剂量；横向峰距为 `3.6 mm`；深度轴和物理坐标
无半体素偏移。

### Phase 1：先对无准直器/open-field

- 使用完全相同的 961-spot source；
- TOPAS 移除 Copper，仅保留水箱；
- GPU 复现 source emittance、空气漂移和水中输运；
- 比较绝对 IDD、二维剂量、横向 profile 和 gamma。

验收：不使用任意全局剂量 scale。若 open-field 已不匹配，先修 source/水物理，
不要用铜参数补偿。

### Phase 2：几何吸收近似

- 实现 beam-frame slit geometry；
- 打到铜的粒子终止；
- 通过单条中央狭缝、单 spot 和 15-slit case 验证 ray boundary；
- 与 TOPAS 中关闭铜相互作用后得到的几何基准比较，或只比较 primary acceptance。

验收：峰中心位置误差不超过一个 `0.1 mm` 体素；slit transmission 和几何遮挡
满足统计误差。

### Phase 3：铜电磁输运

- 加入 Copper stopping power、straggling 和 MCS；
- 先关闭铜核反应；
- 比较水入口处 primary phase space；
- 比较入口、平台和 Bragg 区的横向 profile。

验收：不能只看总透射率；能谱、角分布、FWHM 和 position-angle correlation
都应与 TOPAS 一致。

### Phase 4：铜核反应和碎片

- 生成/编译 Copper cross section 与 cascade package；
- 将铜来源 charged/neutral 粒子送入水中现有队列；
- 验证 peak、valley、halo 和 distal tail；
- 对 copper-origin 剂量做单独诊断 scorer，但最终仍比较 total dose。

验收：关闭/开启铜核反应的 GPU 差值应与 TOPAS 差值一致，不能靠总剂量缩放弥补。

### Phase 5：完整 961-spot 剂量

- 先跑 `1M` 总 histories 做 smoke；
- 再跑 `10M` 定位系统误差；
- 最后按参考运行 `96.1M`；
- 使用相同的每 spot 100k histories 和相同的总 history 归一化。

这里不能沿用 CT 优化权重流程：当前 `spots961.txt` 已明确规定每个 spot 100k。
若以后加入优化权重，再按现有 `spot_weights_file` 逻辑分配。

### Phase 6：集成临床 CT

水箱 case 通过后，再把 beamline stage 接到 CT 入口。临床 CT 常见的 `2 mm`
剂量网格无法解析 `0.5 mm` minibeam，必须增加：

- ROI 高分辨率 scorer，或
- coarse CT dose + fine transverse scorer 的多分辨率计分。

如果只在 2 mm 网格上比较，峰谷被体素平均，gamma 可能虚高，不能证明 minibeam
物理正确。

## 10. 剂量比较指标

绝对剂量比较必须使用相同 incident histories，不先做最大值归一化，也不对 GPU
单独拟合全局 scale。建议同时报告：

- 总 incident histories；
- 到达水入口的 primary/charged/neutral 数和能量；
- 水中积分剂量；
- IDD、Bragg peak、R80 和 distal tail；
- 不同深度的横向 profile；
- peak center、间距、FWHM；
- peak dose、valley dose、PVDR；
- 2D 剂量、剂量差和 global/local gamma；
- TOPAS 多 seed 的统计带。

对 0.5 mm slit，`3%/3 mm` 不是有辨别力的主指标：3 mm 搜索半径接近 3.6 mm
峰间距，错误的峰可能匹配到相邻峰。建议以以下组合为主：

- `3%/0.3 mm` global gamma；
- `3%/0.3 mm` local gamma；
- `2%/0.2 mm` 诊断 gamma；
- `3%/0 mm` 纯剂量差诊断；
- 1D peak/valley/PVDR 指标。

gamma threshold 建议同时报告 `1%`、`5%` 和 `10%` 最大参考剂量。Global gamma
容易忽略 valley，local gamma 又容易被 valley 的 TOPAS 噪声支配，因此二者都
不能替代 PVDR 和绝对 valley dose。

可作为最终目标而非当前已达到结果的指标：

| 指标 | 建议目标 |
|---|---:|
| 水中积分剂量差 | ≤ 3% |
| R80 差 | ≤ 0.5 mm |
| peak center 差 | ≤ 0.1 mm |
| FWHM 差 | ≤ 0.2 mm |
| PVDR 相对差 | ≤ 5% |
| 3%/0.3 mm global gamma，≥5% threshold | ≥ 95% |

## 11. 防止对当前 case 过拟合

不要直接针对 `dose.bin` 拟合 stopping-power scale、MCS scale 和核反应 scale。
每个底层量应由独立 TOPAS/Geant4 诊断获得，再冻结后验证不同 case。

最小泛化矩阵建议包括：

- 能量：`100、约180、250、300 MeV/u`；
- spot：中央狭缝中心、铜 septum 中心、狭缝边缘；
- 几何：单 slit、15 slits、完整 961 spots；
- 狭缝宽度：至少再验证一个非 `0.5 mm` 宽度；
- air gap：至少验证一个非 `60 mm` 距离；
- 独立随机种子。

只允许一个在独立 open-field 绝对输出上确定并冻结的机器级归一化；不允许每个
minibeam case 单独缩放。

## 12. 性能策略

Minibeam 会比当前 CT plan 更慢，主要原因是细网格、铜内小步长和二次粒子。建议：

- source 到准直器、狭缝和空气隙使用解析 ray jump；
- 完全穿过狭缝的 primary 走 fast path；
- 只对进入铜的粒子启动细步长和 Copper physics；
- 铜中停止的低能粒子尽早终止，不为其创建水中队列项；
- beamline 出口 phase-space 使用紧凑 SoA queue；
- 关闭 LET 和不必要的 species dose scorer；
- smoke 阶段使用 1M/10M histories；
- 保持完整参考网格用于最终对比，调试时可只计分少数深度平面或局部 ROI；
- profiling 后再决定是否按 slit/region 对粒子分桶，减少 GPU warp divergence。

`0.05 mm` 应是铜中及边界附近的物理限制，不应成为整个 510 mm 上游空气路径和
全部水深的全局固定步长。

## 13. 测试与回归

至少增加以下自动测试：

- 15 个 slit center 和左右边界的材料分类；
- 圆柱、slit 长轴端点和铜厚度边界；
- 任意 collimator angle 下 beam-frame/world-frame 往返；
- ray 到最近边界的距离；
- `spots961.txt` 解析为 961 spots、96.1M histories；
- source 三个诊断平面的均值/协方差；
- beamline queue 的能量、权重和 history 守恒；
- 同 seed 和不同 chunk size 的统计一致性；
- `minibeam: false` 时现有 water、CT、TPS source 回归不变。

## 14. 推荐的下一步

最稳妥的实际执行顺序是：

1. 先写 dose-only TOPAS 派生 case、二进制剂量转换和 comparison 脚本；
2. 增加准直器入口/出口/水入口 phase-space scorer，锁定 source 坐标；
3. 用 TOPAS 入口 phase space 驱动现有 GPU 水输运，分离水物理误差；
4. 实现 `minibeam: false/true` 配置和 absorbing-geometry 模式；
5. 提取同版本 Copper EM/MCS 数据；
6. 最后加入 Copper cascade，再跑 1M、10M 和 96.1M。

这条路线能让每一步只引入一种新误差来源：先坐标和 scorer，再几何，再铜电磁，
最后铜核反应。这样即使最终 gamma 不理想，也能明确误差来自哪一层。

## 15. 1M 横向 peak 偏高修复（2026-07-27）

用户所指的 peak 是狭缝横向 peak/valley 中的 peak，不是 Bragg peak。诊断显示，
GPU 与 TOPAS 在准直器入口和直穿空气狭缝的 primary 数量已经一致；剩余的整体
peak 偏高主要来自不含铜的独立 open-field 绝对输出差异：

- TOPAS open field：`309.602`
- GPU open field：`314.042`
- 冻结的机器级绝对剂量因子：`309.602 / 314.042 = 0.9858617637131339`

因此新增 `dose_output_scale`，默认值为 `1.0`。它只缩放所有 Gy scorer 的写出，
不修改原始 MeV tally、粒子输运或能量守恒；仅 minibeam 验证配置使用上述独立
open-field 因子。这不是对有准直器的 1M profile 做拟合，符合第 11 节的一次性
独立机器级归一化约束。

同时，体素 scorer 对 `electronic_buildup_fraction` 增加了可选横向展宽
`electronic_buildup_lateral_sigma_mm: 0.5`，用于修复电子能量沿深度延迟、但在
横向仍全部留在母粒子 voxel 的不一致。默认值为 `0`，所以 CT 和其他既有 case
保持原行为。

正式 1M CUDA 与同 histories TOPAS 的绝对剂量结果：

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 总体素积分剂量差 | +1.74% | +0.30% |
| IDD normalized L1 | 3.05% | 1.82% |
| 入口 0.5 mm 三个主 peak 平均差 | +2.42% | +0.97% |
| 入口中心 peak 差 | +4.79% | +3.31% |
| 35 mm 三个主 peak 平均差 | +0.99% | -0.44% |
| 35 mm 中心 peak 差 | +2.47% | +1.02% |

1M CUDA 输运耗时 `105.22 s`（TITAN RTX，约 `9504 histories/s`）。结果位于：

- `out/minibeam/copper_neutral_center_1M_lateral_delta_openfield_scale/dose.mhd`
- `out/minibeam/comparison/center_1M_lateral_delta_openfield_scale_vs_topas.json`
- `out/minibeam/comparison/center_1M_peak_valley_lateral_delta_openfield_scale.json`
- `out/minibeam/comparison/center_1M_lateral_delta_openfield_scale_profiles/`

仍未解决的是 valley 和 Bragg 附近的形状误差：入口 valley 平均仍高约 `14.9%`，
70.35 mm 三个 peak 平均低约 `4.7%`。这两项不能继续用全局剂量 scale 修复，
下一步应针对铜中 touched-primary 的非高斯 MCS/能损末态和水中 distal 形状单独
验证，避免污染已经对上的入口和 35 mm 横向 peak。

## 16. 入口剂量与 R80 修复（2026-07-27）

进一步的 phase-space 与 dose-origin 分解显示两个独立误差：

1. GPU 中接触铜后仍到达水入口的 primary C-12 平均能量偏低。100k 诊断为
   `1548.8 MeV`，TOPAS 为 `1575.2 MeV`，导致入口 LET 和剂量偏高。
2. 在入口 C-12 phase space 对齐后，TOPAS primary-origin R80 为 `71.61 mm`，
   GPU 为 `71.14 mm`。误差来自水中 primary 连续能损/射程，而不是铜二次粒子。

新增两个默认不生效的 minibeam 参数：

- `minibeam_copper_survivor_energy_loss_scale: 0.956`：只校正已经存活到铜出口的
  primary 累计铜能损，不改变其核反应概率、存活率和角输运。修复后的 100k
  touched-primary 均值为 `1574.6 MeV`。
- `minibeam_water_primary_stopping_power_scale: 0.9958`：只作用于 minibeam 水中
  primary C-12，按 dose-origin R80 响应确定；二次粒子和非 minibeam case 保持
  原模型。

曾测试但拒绝的方案：

- 把整张 Copper stopping-power 表缩放到 `0.966/0.90`：会使本应停在铜中的
  低能粒子存活，低能尾和入口剂量反而增大。
- `straggling_scale: 1.5`：R80 只改善约 `0.04 mm`，但 Bragg 最大值提前
  `0.8 mm`。
- 水中最大步长从 `0.1 mm` 减为 `0.05 mm`：R80 只移动约 `0.03 mm`，运行速度
  明显下降，说明不是步长收敛问题。

最终 1M CUDA 与同 histories TOPAS 的结果：

| 指标 | 修复前 | 最终 |
|---|---:|---:|
| R80 差 | -0.304 mm | **-0.010 mm** |
| Bragg 最大值位置差 | -0.30 mm | **0.00 mm** |
| 0--1 mm 入口积分剂量差 | +2.32% | **+0.32%** |
| 入口三个主 peak 平均差 | +0.97% | **+0.16%** |
| 入口中心 peak 差 | +3.31% | **+2.59%** |
| 总积分剂量差 | +0.30% | +0.64% |
| IDD normalized L1 | 1.82% | **1.02%** |

最终 1M 用时 `105.51 s`（TITAN RTX，`9478 histories/s`），与修复前相当。
标准比较脚本现在会直接输出 R80，并在深度剂量图中标出 TOPAS/GPU 的 R80：

- `out/minibeam/copper_neutral_center_1M_entry_r80_fix_final/dose.mhd`
- `out/minibeam/comparison/center_1M_entry_r80_fix_final_vs_topas.json`
- `out/minibeam/comparison/center_1M_peak_valley_entry_r80_fix_final.json`
- `out/minibeam/comparison/center_1M_entry_r80_fix_final_profiles/`

剩余误差主要是 `1--5 mm` build-up 区约 `+3.88%`、入口 valley 约 `+8.1%`，
以及 Bragg 深度横向 peak 约 `-5.3%`。这些属于局部形状/PVDR 问题，不应再用
R80 或全局绝对剂量参数调整。

## 17. 100--400 MeV/u 跨能量验证与修复（2026-07-27）

### 17.1 旧四能量结论为何失效

第 17 节最初报告的 `100/200/300/400 MeV/u` total 积分误差
`+4.71/+10.10/+20.53/+27.20%` 不能作为物理结论。生成的 GPU 1D case 只设置
了 `scorer_area_mm2=4000`，该参数只定义 dose mass，不会建立横向边界；GPU
因此还计入了已经离开 TOPAS `Box2`（`100 x 40 mm²`）的粒子。

修复后的 GPU case 使用 `1 x 1` 横向 voxel，并显式设置
`voxel_size_x/y_mm=100/40`，使 dose scorer 的有限横向范围与 TOPAS 完全一致。
只修正 scorer 几何后，旧单能 Copper package 的 total 积分误差已经变为
`+4.12/+3.95/+2.08/-4.26%`。先前认为高能 non-primary 高出约 `50%`、存在
离散人工 stopping peaks 的判断，主要来自不匹配的计分边界和 10k 涨落，已撤销。

同时修复了验证生成器中的 history 同步问题：此前 `--histories` 只更新 YAML，
没有更新生成 spot 文件中的 `iv:Tf/Scatterer1/L4/Values`，使名义 100k GPU
诊断实际仍只运行 10k。现在 YAML、TOPAS TimeFeature 和 spot L4 三者一致。

### 17.2 多能量 Copper reaction package

TOPAS 在 `100、200、300、400 MeV/u` 各生成 `20k` primary-C12 Copper reaction
样本，并合并为：

- `data/copper_c12_reaction_packages_e100_400_20k_each.bin`
- `81` 个 `5 MeV/u` energy bin，覆盖 `0--405 MeV/u`
- `21,119` 个相关反应末态
- `442,970` 个二次粒子/primary continuation 记录

原 `CarbonReactionNtuple` 只记录 Geant4 新建的 secondary，遗漏了核反应后仍以
track 1 延续的 primary C-12。扩展现在为它输出
`particle_name=primary_continuation`；编译器用负 PDG 作为包内标记，GPU 入队时
恢复正 PDG，并保持 `primary_c12` dose origin。多能量 package 将 300 MeV/u
non-primary 积分误差从 `-12.59%` 修到 `+0.14%`，400 MeV/u helium 相对误差
从约 `-39%` 修到约 `-15%`。

### 17.3 独立相空间约束的能量依赖 Copper loss

固定的 `minibeam_copper_survivor_energy_loss_scale=0.956` 来自约
`179.17 MeV/u` 的独立相空间，因此不应直接外推到整个能区。四个能量各运行
100k TOPAS/GPU，只用到达水入口且确实接触 Copper 的 surviving primary C-12
平均能量确定累计能损系数，不使用任何 dose 或 gamma 结果拟合。

| 能量 (MeV/u) | 能损系数 | TOPAS water primary 数 | GPU water primary 数 | TOPAS touched 均值 (MeV) | GPU 修复后均值 (MeV) |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.9068181 | 9,907 | 10,024 | 917.038 | 916.968 |
| 200 | 0.9719893 | 11,166 | 11,287 | 1,722.931 | 1,723.078 |
| 300 | 1.0092652 | 13,486 | 13,529 | 2,392.424 | 2,394.835 |
| 400 | 1.0070182 | 18,656 | 18,666 | 2,783.734 | 2,787.496 |

四个 TOPAS touched 均值的标准误分别约为 `5.09、9.39、11.96、12.18 MeV`；
修复后 GPU 与 TOPAS 的均值差均显著小于一个标准误，不再继续追逐末位统计涨落。

新增两个可选 YAML list：

```yaml
minibeam_copper_survivor_energy_loss_energies_MeVu: 100, 200, 300, 400
minibeam_copper_survivor_energy_loss_scales: 0.9068181, 0.9719893, 1.0092652, 1.0070182
```

运行时按每个 spot/每条 history 的实际 incident energy 线性插值，能区外采用端点
值。两个 list 为空时仍使用原标量参数，因而不改变旧 case。四点在
`179.17 MeV/u` 的插值约为 `0.958`，与原独立单点 `0.956` 一致，说明这不是用
高能 dose 对原 case 重新拟合。

逐步 Bohr Copper straggling 也做过 A/B；它只改变低统计尖峰的随机位置，不能
消除尖峰，且没有改善相空间均值，因此生产配置仍保持
`minibeam_copper_enable_energy_straggling: false`。

### 17.4 四能量 10k 剂量回归

以下结果同时使用有限横向 scorer、多能量 Copper reaction package 和
energy-dependent survivor loss。高能 Copper-touched survivor 数较少，因此
10k 主要用于快速回归，不用于冻结新的物理参数。

| 能量 (MeV/u) | GPU-TOPAS R80 (mm) | total 积分差 | primary 积分差 | non-primary 积分差 | 0--1 mm 入口差 | total L1 |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | -0.026 | +4.44% | +3.96% | +10.01% | +2.60% | 5.24% |
| 200 | +0.001 | +4.02% | +5.54% | -2.63% | +8.44% | 6.12% |
| 300 | -0.818 | +5.09% | +7.48% | -0.25% | -0.18% | 6.82% |
| 400 | +0.568 | +2.15% | +9.10% | -7.51% | -2.02% | 5.48% |

10k 中 400 MeV/u 的 `+9.10%` primary 不能解释为稳定模型偏差：GPU/TOPAS 的
water-primary 数为 `1,914/1,862`，touched 能谱也受少量高能 survivor 支配。
因此另外执行了 400 MeV/u、100k 的正式 dose-origin 验证。

### 17.5 400 MeV/u、100k 正式结果

| 指标 | 100k 结果 |
|---|---:|
| R80 差 | **+0.010 mm** |
| primary 积分差 | **+1.02%** |
| non-primary 积分差 | -7.94% |
| total 积分差 | **-2.69%** |
| 0--1 mm 入口差 | -3.59% |
| total normalized L1 | 3.99% |
| TOPAS 40 线程墙钟 | 769.8 s |
| TITAN RTX GPU 墙钟 | 52.1 s |
| 实测加速比 | 14.8x |

这证明 10k primary 积分误差主要是 survivor 数量与宽能谱的联合涨落；相空间
校正后的 primary 剂量和 R80 在高统计下已经对齐。100k total 已达到本文件定义
的 `≤3%` 积分目标。

400 MeV/u 的剩余 total 误差按 origin 分解为：

- primary C-12：`+0.60` 个百分点；
- helium：`-2.80` 个百分点（category 自身 `-13.50%`）；
- neutral origin：`-0.61` 个百分点；
- 其余所有 charged categories 合计约 `+0.11` 个百分点。

下一项底层物理工作不应再调整 primary loss 或全局 dose scale，而应验证
Copper 中 secondary-ion 的延续反应。当前 GPU 对 Copper 内 He/Li/Be/B 等
二次离子只做 stopping、MCS 和核吸收，被吸收后不生成 tertiary products；
TOPAS 则会继续产生低能碎片。这与水入口处 GPU helium 数略低、平均能量偏硬的
现象一致，是 `-2.80` 个百分点 helium 缺口的首要候选原因。应先生成独立的
secondary-ion Copper reaction/foil phase-space 基准，再决定是否增加分粒子
reaction package，不能直接对 helium dose 乘经验 scale。

可复现入口与结果：

- `validation/scripts/prepare_minibeam_phase_space_sweep.py`
- `validation/scripts/analyze_minibeam_phase_space.py`
- `validation/scripts/prepare_minibeam_copper_reaction_sweep.py`
- `validation/scripts/merge_reaction_packages.py`
- `validation/scripts/prepare_minibeam_energy_sweep.py`
- `validation/scripts/compare_minibeam_energy_sweep.py --energies ...`
- `validation/scripts/analyze_minibeam_energy_sweep_origins.py --energies ...`
- `out/minibeam/energy_sweep_multienergy_calibrated/metrics.json`
- `out/minibeam/energy_sweep_100k_calibrated/metrics.json`
- `out/minibeam/energy_sweep_100k_calibrated/origin_integrals.json`

## 18. 高能轻离子 EM 表截断修复（2026-07-28）

### 18.1 `EMRangeMax=600 MeV` 是此前 helium 缺口的主因

OpenTOPAS 4.2.p3 当前运行日志中的默认 EM table 上限为 `600 MeV`，该值是粒子的
**总动能**而不是 MeV/u。于是 300 MeV/u 的 alpha 总动能为 `1200 MeV`，超过表
上限后，Geant4 的 dedicated alpha 连续能损表被钳在 `600 MeV`。这不是 GPU
secondary-ion stopping 模型的通用误差。

新增独立 Copper foil 脚本：

- `validation/scripts/prepare_secondary_ion_copper_foil.py`
- `validation/scripts/analyze_secondary_ion_copper_foil.py`

1 mm Copper、10k histories 的结果为：

| 粒子/能量 | 旧 GPU 表 dE/dx | TOPAS 600 MeV 表上限 | TOPAS 6 GeV 表上限 |
|---|---:|---:|---:|
| He-4 100 MeV/u | 17.529 MeV/mm | 17.514 MeV/mm | — |
| He-4 200 MeV/u | 10.874 MeV/mm | 15.265 MeV/mm | — |
| He-4 300 MeV/u | 8.585 MeV/mm | 15.392 MeV/mm | **8.590 MeV/mm** |
| He-3 300 MeV/u | 8.604 MeV/mm | 8.587 MeV/mm | — |

He-4 300 MeV/u 在把 `Ph/Default/EMRangeMax` 提高到 `6 GeV` 后与 GPU 相差仅
`0.06%`，直接证实了截断根因。不能把 600 MeV reference 的 alpha 能损用经验
倍率写入 GPU，否则是在复现参考配置伪差而不是提高物理精度。

所有 minibeam dose、phase-space、Copper reaction/material-table 生成路径现在
显式使用：

```text
d:Ph/Default/EMRangeMax = 6 GeV
```

它覆盖 400 MeV/u C-12 的 `4.8 GeV` 总动能。`IonStoppingPowerNtuple` 同时输出
`G4EmCalculator::GetDEDX` 的 transport-table 列，表转换脚本已支持新的 8 列原始
格式；GPU CSV 仍保持原 8 列接口，不改变加载器。

### 18.2 修正后的四能量 10k dose

使用相同 GPU 物理、仅重建 6 GeV TOPAS reference 后：

| 能量 (MeV/u) | R80 差 | primary 积分差 | non-primary 积分差 | total 积分差 | total L1 |
|---:|---:|---:|---:|---:|---:|
| 100 | -0.022 mm | +4.13% | +13.62% | +4.85% | 5.53% |
| 200 | +0.093 mm | +5.60% | -1.63% | +4.26% | 6.23% |
| 300 | -0.701 mm | +8.67% | -2.03% | +5.26% | 7.33% |
| 400 | +0.293 mm | +9.83% | +1.53% | +6.45% | 8.06% |

最重要的变化是 300--400 MeV/u non-primary 已到 `-2.03%/+1.53%`，400 MeV/u
helium category 从旧 reference 的 `-13.50%` 变为 `+1.15%`。因此第 17.5 节
把三级碎片视为 `-2.80` 个百分点 helium 缺口首要原因的结论已被否定；三级碎片
仍是可改进项，但不是当前高能总剂量的主导误差。

10k primary 积分不能用于调参，因为穿过窄缝并到达水面的 primary 只有约
`1k--2k`。400 MeV/u 的独立 100k 三平面 phase-space 给出：

| 指标 | TOPAS | GPU（旧 400 scale） |
|---|---:|---:|
| water-primary 数 | 18,664 | 18,666 |
| water-primary 平均能量 | 3567.594 MeV | 3595.560 MeV |
| water-primary 总能量差 | — | +0.795% |
| Copper-touched 均值 | 2769.744 ± 12.193 MeV | 2787.496 MeV |
| water x/z RMS | 4.428 / 4.547 mm | 4.368 / 4.452 mm |

透射数只差 `+0.011%`，说明 Copper C-12 核衰减和总体几何透射已对齐。按独立
Copper-touched phase-space 更新 400 MeV/u 能损系数：

```yaml
minibeam_copper_survivor_energy_loss_scales: 0.9068181, 0.9719893, 1.0092652, 1.0159010
```

更新后的 100k GPU 为：

- water-primary 数 `18,654`，相对 TOPAS `-0.054%`；
- Copper-touched 均值 `2773.065 MeV`，只高 `3.321 MeV`（`0.27` 个 TOPAS
  标准误）；
- water-primary 平均能量 `3587.462 MeV`；
- 入水 primary 总能量相对 TOPAS约 `+0.50%`。

这项修正只来自水入口 phase-space，不使用 dose 或 gamma。修正后的 10k
400 MeV/u total 积分差由 `+6.45%` 变为 `+6.24%`；幅度很小再次说明 10k dose
仍由窄缝透射统计涨落主导。下一项有判别力的验证是用 6 GeV TOPAS reference
正式重跑 400 MeV/u 100k dose-origin，而不是继续调整 primary 或 helium 参数。

### 18.3 修正后 400 MeV/u、100k 正式 dose-origin

已按上一节冻结的 GPU 参数盲测 6 GeV TOPAS reference，没有根据 dose 结果再次
调参：

| 指标 | 100k 正式结果 |
|---|---:|
| R80 差 | **+0.061 mm** |
| Bragg peak 位置差 | -0.10 mm |
| primary 积分差 | **+0.52%** |
| non-primary 积分差 | **-1.03%** |
| total 积分差 | **-0.11%** |
| 0--1 mm 入口差 | -0.63% |
| 0--5 mm 入口差 | -1.37% |
| total normalized L1 | **1.97%** |
| TOPAS 40 线程墙钟 | 782.5 s |
| TITAN RTX GPU 墙钟 | 58.3 s |
| 实测加速比 | 13.4x |

这证明 10k 中段 `+5--15%` 差异主要是窄缝后少量 history 的涨落，不是稳定的
stopping-power 偏差。1.1 mm 平滑后，各深度段为：

| 深度 | 积分差 | L1 | 平滑相对差中位数 |
|---:|---:|---:|---:|
| 0--50 mm | -1.20% | 1.97% | -1.31% |
| 50--150 mm | +0.02% | 1.61% | +0.09% |
| 150--250 mm | +1.51% | 1.92% | +1.47% |
| 250--285 mm | +0.07% | 1.50% | +0.13% |
| 285--310 mm | -0.70% | 2.57% | -0.80% |

TOPAS 的 12 个 origin scorer 之和与 total 的闭合误差为
`-2.7e-8%`。剩余 total 误差的主要来源贡献为：

- neutral origin：`-0.672` 个百分点；
- proton：`-0.651` 个百分点；
- helium：`+0.481` 个百分点；
- secondary carbon：`+0.312` 个百分点；
- primary C-12：`+0.309` 个百分点；
- 其余来源合计约 `+0.112` 个百分点。

正负误差目前存在抵消，但每个主来源对 total 的贡献都已小于 `0.7` 个百分点。
因此不应再增加 helium 经验修正或调整全局 dose scale。下一项底层物理改进应优先
验证 proton 与 neutral 的来源/输运形状；若目标是临床剂量而不是逐来源完全一致，
当前 400 MeV/u 结果已达到 `≤3%` 的既定物理剂量目标。

正式图与结果：

- `out/minibeam/emrange_fix_summary/emrange_fix_and_e400_dose_100k.png`
- `out/minibeam/energy_sweep_100k/metrics.json`
- `out/minibeam/energy_sweep_100k/origin_integrals.json`
- `out/minibeam/energy_sweep_100k/origin_difference_contributions.png`

### 18.4 修正后 100--400 MeV/u、每能量 100k 正式盲测

为排除窄缝后低透射率导致的 10k 涨落，使用完全相同且已冻结的 GPU 物理参数，
对 100、200、300、400 MeV/u 分别运行 100k histories。TOPAS 均使用 40 线程和
`EMRangeMax = 6 GeV`；对比前未再按任何 dose 指标调参。

| 能量 (MeV/u) | R80 差 (mm) | primary 积分差 | non-primary 积分差 | total 积分差 | total L1 |
|---:|---:|---:|---:|---:|---:|
| 100 | +0.006 | +0.22% | +5.00% | **+0.60%** | **1.19%** |
| 200 | -0.130 | +0.60% | -0.16% | **+0.46%** | **1.73%** |
| 300 | -0.169 | +0.38% | -1.84% | **-0.31%** | **1.71%** |
| 400 | +0.061 | +0.52% | -1.03% | **-0.11%** | **1.97%** |

四个能量的 `|total 积分差|` 均小于 `0.60%`，`|ΔR80|` 均小于 `0.17 mm`，
depth-normalized L1 均小于 `1.98%`。这说明目前的 Copper survivor 能损、
核反应概率、反应末态与水中输运参数可跨 100--400 MeV/u 泛化；10k 对比中的
`5--8%` total 差异不能作为稳定模型误差解释。

100 MeV/u 的 non-primary category 相对差为 `+5.00%`，但该成分占总剂量很小，
因此只贡献约 `+0.39` 个百分点的 helium 正差和 `-0.11` 个百分点的 neutral
负差；total 仍只差 `+0.60%`。其余能量也存在不同 origin 的正负抵消，因此若要
继续做底层物理验证，应比较分来源深度形状和独立相空间，不能按 total dose
残差增加经验 scale。

性能结果：

| 能量 (MeV/u) | TOPAS 40 线程 | TITAN RTX GPU | 加速比 |
|---:|---:|---:|---:|
| 100 | 200.6 s | 16.7 s | 12.0x |
| 200 | 279.5 s | 24.1 s | 11.6x |
| 300 | 491.9 s | 32.9 s | 15.0x |
| 400 | 782.5 s | 58.3 s | 13.4x |

可复现结果和绘图：

- `out/minibeam/energy_sweep_100k/multienergy_depth_dose_smoothed.png`
- `out/minibeam/energy_sweep_100k/multienergy_metrics_trend.png`
- `out/minibeam/energy_sweep_100k/metrics.json`
- `out/minibeam/energy_sweep_100k/origin_integrals.json`
- `validation/scripts/plot_minibeam_multienergy_comparison.py`

## 19. GPU 输运性能优化与逐项回归（2026-07-28）

本轮只接受同时通过固定随机种子 TOPAS 回归的优化。400 MeV/u、100k histories
的原始正式运行耗时为 `58.34 s`，其中 secondary kernel 为主要瓶颈：
约 `1.011B` secondary steps，并有约 `240.9M` 次因为 FP32 粒子能量没有可表示
变化而产生的空转步。

按顺序实施和验证的结果如下：

| 步骤 | 400 MeV/u transport elapsed | 相对原始速度 | 精度结论 |
|---|---:|---:|---|
| 原始正式版本 | 58.34 s | 1.00x | ΔR80 +0.061 mm，total -0.110%，L1 1.974% |
| 1. persistent secondary workers（32768） | 55.10 s | 1.06x | history/剂量不变 |
| 2. secondary condensed-history 0.25 mm | 19.32 s | 3.02x | 剂量仍回填到原 0.1 mm scorer bin |
| 3. Copper transport step 0.25 mm | 16.05 s | 3.64x | ΔR80 改善到 +0.025 mm |
| 4. FP32 连续能损 residual | 约 16.4 s（profile build） | — | non-progress 从 23.8% 降到 0.82%，防止亚 ULP 能损空转 |
| 5. 关闭详细 minibeam diagnostics | **15.75 s** | **3.70x** | 剂量指标逐位一致 |

关键实现：

- `secondary_persistent_workers`：固定数量 work-item 从 batch 计数器动态取轨迹，
  降低一个 warp 被最长 secondary track 拖住的尾部发散；
- `secondary_condensed_step_mm`：在均匀 phantom 中把 secondary 的物理 MCS/
  straggling 步与 scorer 的 0.1 mm bin 解耦；连续能损仍按穿越长度分配回每个
  原始 z bin，因此没有降低输出剂量分辨率。CT、layered phantom、insert 和 LET
  scorer 会自动禁用该快路径；
- secondary FP32 energy residual：累计小于当前能量 ULP 的连续损失，达到可表示
  幅度后再扣除，避免能量不变却重复推进；
- `minibeam_diagnostics: false`：跳过 collimator 计数和相空间 moment 的全局原子
  操作；全局能量平衡所需的 beamline removed energy 仍保留。

### 19.1 低能 Copper 步长不能统一使用 0.25 mm

100 MeV/u 使用 0.25 mm 时虽然 transport elapsed 为 `9.43 s`，但 total L1
从原参考的 `1.19%` 增至 `1.71%`。改为 `0.10 mm` 后只增加 `0.32 s`，同时得到：

- ΔR80 `+0.011 mm`；
- primary 积分差 `+0.480%`；
- total 积分差 `+0.742%`；
- depth-normalized L1 `1.314%`。

因此能量 sweep 生成脚本固定采用：

```text
100 MeV/u:       minibeam_copper_max_step_mm = 0.10
200--400 MeV/u: minibeam_copper_max_step_mm = 0.25
```

这不是 dose 拟合；选择依据是 Copper 中单步相对能损随入射能量降低而增大。

### 19.2 四能量 100k 最终盲测

| 能量 (MeV/u) | transport elapsed | 原 GPU elapsed | 加速 | ΔR80 | total 积分差 | total L1 |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 9.75 s | 16.7 s | 1.71x | +0.011 mm | +0.742% | 1.314% |
| 200 | 11.13 s | 24.1 s | 2.17x | -0.122 mm | -0.019% | 1.775% |
| 300 | 13.22 s | 32.9 s | 2.49x | -0.130 mm | -0.810% | 1.779% |
| 400 | 15.75 s | 58.3 s | 3.70x | +0.025 mm | -0.726% | 1.999% |

四个能量均满足 `|total 积分差| < 0.82%`、`total L1 < 2.00%` 和
`|ΔR80| < 0.131 mm`。正式配置没有按每个能量的 dose residual 再做 scale。

179.17 MeV/u、100k 的三维 minibeam 回归也通过：

- total 积分差 `-0.102%`；
- depth-integrated L1 `1.789%`；
- ΔR80 `-0.186 mm`；
- lateral-x / lateral-y integrated L1 `7.987% / 4.339%`；
- 入口、35 mm、70.35 mm 的 median peak ratio 分别为
  `1.006 / 1.011 / 0.955`。

可复现结果：

- `out/minibeam/energy_sweep_100k/metrics.json`
- `out/minibeam/energy_sweep_100k/multienergy_comparison.png`
- `out/minibeam/energy_sweep_100k/multienergy_metrics_trend.png`
- `out/minibeam/copper_neutral_center_100k_copper_step025/compare/topas_metrics.json`
- `out/minibeam/copper_neutral_center_100k_copper_step025/compare/peak_valley.json`

## 20. 异质材料初步验证：water / compact bone / water（2026-07-28）

使用 200 MeV/u C-12 中心 minibeam、相同 Copper collimator、相同随机种子，
在准直器后比较以下 phantom：

```text
0--50 mm water | 50--70 mm G4_BONE_COMPACT_ICRU | 70--150 mm water
```

TOPAS 和 GPU 均使用 100k histories；剂量是绝对 `DoseToMedium`，没有针对本
case 拟合额外 scale。共同 scorer grid 为 `500 x 1 x 300`，即横向 0.2 mm、
沿 slit 方向 40 mm、深度 0.5 mm。TOPAS 使用 40 CPU threads。

### 20.1 已发现并修复的 dose-to-medium 质量错误

原先 layered phantom 和 heterogeneous insert 虽然使用了局部材料的 stopping
power、核反应截面和密度输运，但 MHD 输出仍统一用水的 voxel mass 将 MeV
换算成 Gy。10k smoke 中，骨层 GPU/TOPAS 剂量比为 `1.779`，接近 compact
bone 密度 `1.85`，由此定位到 `src/io.cpp::voxel_masses_kg()`。

修复后：

- layered phantom 按每个深度 voxel 与各层的实际重叠厚度计算平均质量；
- AABB insert 按 insert 与 voxel 的精确相交体积计算 water/insert 混合质量；
- 增加单元测试，验证相同沉积能在 2 倍密度材料中的 dose 恰为一半。

该修复将全 phantom 积分偏差从 `+26.70%` 降到 `+2.43%`。

### 20.2 100k 正式结果

深度曲线先作 1.5 mm box smoothing，只用于抑制独立蒙卡噪声和计算下列 profile
指标；横向 PVDR 使用 5 mm 深度 slab：

| 指标 | 结果 |
|---|---:|
| 全 phantom 积分差 | `+2.416%` |
| depth normalized L1 | `8.076%` |
| depth Pearson r | `0.99009` |
| TOPAS / GPU R80 | `72.345 / 72.015 mm` |
| ΔR80 (GPU - TOPAS) | `-0.330 mm` |
| depth global gamma 3%/1 mm, 10% threshold | `46.41%` |
| depth local gamma 3%/1 mm, 10% threshold | `22.88%` |

分区积分剂量差：

| 区域 | GPU/TOPAS - 1 |
|---|---:|
| 前段 water，0--50 mm | `+8.99%` |
| compact bone，50--70 mm | `-4.76%` |
| 后段 water，70--150 mm | `-10.12%` |

PVDR 相对差在 10、60、70、75 mm 分别为 `-14.2%`、`+22.5%`、`-0.18%`、
`-8.0%`。尤其 70 mm 骨层出口/Bragg peak 附近的 peak/valley 形状已吻合，
但入口、骨层内部和 post-bone tail 尚未达到 3% 精度。因此当前结果证明 range
和总能量尺度基本正确，但不能据此宣称异质 minibeam 已完成临床精度验证。

当前剩余系统误差有明确的材料模型来源：

1. 现有 compact-bone stopping-power 表来自 Geant4 11.3.2，而水和 TOPAS 正式
   运行为 Geant4 11.3.2。本次为避免直接混合绝对表，使用同版本旧表中的
   bone/water 比值重采样到精确 11.3.2 water grid；这仍不是直接的 11.3.2
   compact-bone 表。
2. layered phantom 的 MCS 当前仍使用 water radiation length；代码只有 CT
   material MCS 和 Copper MCS 支持材料 radiation length。该近似会直接影响
   细 minibeam 的 valley dose 和 PVDR。
3. primary C-12 已使用 bone-specific inelastic XS，但 reaction/cascade/neutral
   final-state packages 仍从 water package 采样。骨层后的 fragment tail 因而
   只部分具备材料条件化。
4. minibeam 的低能 MCS 修正目前对所有非 CT phantom 生效，在 bone layer 中也
   继续使用 water-tuned scale。

在补齐同版 compact-bone SP/XS、per-layer radiation length 和 material-conditioned
reaction packages 前，不应通过本 case 的经验 depth scale 强行提高 gamma。

可复现输入和结果：

- `ct/minibeam/run_hetero_bone_e200_100k.txt`
- `config/beam_minibeam_hetero_bone_e200_100k.yaml`
- `validation/scripts/resample_stopping_power_grid.py`
- `validation/scripts/compare_minibeam_heterogeneous.py`
- `out/minibeam/heterogeneous_bone_e200_100k/comparison/heterogeneous_metrics.json`
- `out/minibeam/heterogeneous_bone_e200_100k/comparison/heterogeneous_depth.png`
- `out/minibeam/heterogeneous_bone_e200_100k/comparison/heterogeneous_lateral.png`

### 20.3 材料 MCS、同版本 SP 与 scorer/transport 解耦

按上述限制逐项做相同 seed、100k histories 的 A/B 后，真正的主导误差不是
bone radiation length 或跨版本 SP，而是横向 voxel scorer 边界错误地限制了
物理输运步。

首先将 CT 已有的材料 MCS 能力泛化到 layered phantom 和 AABB insert：

```yaml
slab_radiation_lengths_g_per_cm2: 36.0830, 30.4866, 36.0830
insert_radiation_length_g_per_cm2: 30.4866
```

两项均为显式配置；旧 slab 配置不填写时继续使用 water radiation length，
旧 insert 配置的默认值也是 36.08 g/cm²。两套 current/legacy kernel 均支持。
该改动只带来小幅、混合变化：

- depth global gamma 3%/1 mm：`46.41% → 49.67%`；
- bone normalized L1：`4.83% → 4.71%`；
- 70 mm PVDR 差：`-0.18% → -3.45%`。

随后修正 `carbon_material_sp_bone.txt`：静态材料查询 scorer 的 primary 直接出生
在 Sample 内，避免依赖 TOPAS beam-axis 约定。由本地 TOPAS 4.2.p3 /
Geant4 11.3.2 成功导出 4001 点直接 compact-bone SP 表：

- `data/stopping_power_bone_geant4_11_3_2.csv`
- `data/stopping_power_bone_geant4_11_3_2_full.csv`
- `data/stopping_power_bone_geant4_11_3_2.metadata.json`

直接表与先前 bone/water 比值重采样表在 10--400 MeV/u 的差约 0.001% 以下。
替换后总积分只变化 `-0.00019%`，depth profile L1 变化 `0.092%`，证明跨版本
SP 不是当前 5--10% 系统偏差的来源。

用 11.3.2 bone reaction/cascade package 替换整个 phantom 的 water package
做了敏感性实验。post-bone 积分由 `-10.3%` 恶化到 `-13.6%`，depth L1 由
`8.06%` 恶化到 `8.94%`。由于该实验也错误替换了 water 区域，它不作为正式模型；
但结果不支持立即通过复杂的逐材料 package 调度来修复当前主误差。

#### 主因：scorer resolution 改变了 MCS 输运

相同 200 MeV/u 全水模型中，仅把横向 scorer 从一个 100 mm bin 改成 500 个
0.2 mm bin，就观察到：

- 0--50 mm total dose 增加 `7.80%`；
- 0--50 mm primary-C12 dose 增加 `7.93%`；
- 0--5 mm primary-C12 dose增加 `9.41%`；
- runtime 从 `10.38 s` 增至 `103.86 s`。

原因是旧实现把每个 x/y scorer face 当成 physics step boundary，并在每个短步
重新应用 Highland MCS。scorer 本应是被动计分器，不应改变粒子轨迹。

新增默认保持兼容的开关：

```yaml
voxel_scorer_clamps_transport: false
```

关闭后 x/y scorer face 不再截断物理步；z/material/CT boundaries 仍严格限步。
当前最大物理步为 0.1 mm，小于 0.2 mm 横向 voxel，因此每步沉积记到起点 voxel
带来的空间不确定度小于一个 voxel，而不会改变 MCS。

正式异质 case 的结果变为：

| 指标 | 旧 scorer clamp | scorer 解耦 |
|---|---:|---:|
| runtime | 126.90 s | **4.43 s** |
| transport steps | 2.308B | **0.161B** |
| 全体积积分差 | +2.41% | **+0.26%** |
| depth normalized L1 | 8.05% | **1.69%** |
| depth Pearson r | 0.9901 | **0.99922** |
| global gamma 3%/1 mm | 49.67% | **96.08%** |
| local gamma 3%/1 mm | 22.22% | **90.85%** |
| ΔR80 | -0.322 mm | **-0.309 mm** |

分区积分差也从 `+8.97% / -4.65% / -10.28%` 改善到：

- 前段 water：`+1.14%`；
- compact bone：`+0.67%`；
- 后段 water：`-3.74%`。

严格横向 PVDR 仍有约 8--16% 的局部差异，且 100k histories 在 0.2 mm bins
上统计噪声明显。下一步应实现不改变物理步的线段/voxel 重叠计分或中点计分，
再以 1M histories 验证 lateral gamma；不应重新打开 scorer face clamp。

新增结果：

- `out/minibeam/heterogeneous_bone_e200_100k/comparison_decoupled_scorer/heterogeneous_metrics.json`
- `out/minibeam/heterogeneous_bone_e200_100k/comparison_decoupled_scorer/heterogeneous_depth.png`
- `out/minibeam/heterogeneous_bone_e200_100k/comparison_decoupled_scorer/heterogeneous_lateral.png`

### 20.4 1M high-statistics 验证

为区分 100k histories、0.2 mm 横向 bins 的统计噪声与真实模型误差，使用同一
seed 和完全相同几何分别运行 1M histories。GPU 的 spot 文件会覆盖 YAML 中的
`number_of_histories`，因此可复现命令必须显式覆盖总 history 数：

```bash
./build/oneapi-nvidia-minibeam/carbon_mc \
  --config config/beam_minibeam_hetero_bone_e200_100k.yaml \
  --histories 1000000

TOPAS_MINIBEAM_INSTALL=build/opentopas-minibeam-install \
  bash validation/topas/run_minibeam_topas.sh \
  run_hetero_bone_e200_1M.txt
```

NVIDIA TITAN RTX 上 GPU 用时 `33.08 s`，吞吐 `30.23k histories/s`；TOPAS
40 threads 用时 `2346.66 s`，因此该 case 的端到端 wall-time speedup 为
`70.94x`。GPU charged/neutral queues 均无 overflow。

高统计量深度结果：

| 指标 | 1M 结果 |
|---|---:|
| 全体积积分差 | `-0.489%` |
| depth normalized L1 | `1.147%` |
| depth RMSE / reference max | `0.863%` |
| depth Pearson r | `0.999455` |
| global gamma 3%/1 mm | `95.42%` |
| local gamma 3%/1 mm | `93.46%` |
| ΔR80 | `-0.193 mm` |

分区积分差为前段 water `+0.53%`、bone `-0.79%`、后段 water `-3.81%`。
前两段已经稳定进入 1% 量级；剩余深度误差集中在 Bragg peak 下降沿及
post-bone fragment tail，不再是入口或骨材料质量换算问题。

高统计量横向结果也澄清了 100k 的噪声影响：

| 深度 | global gamma 3%/0.4 mm | PVDR 相对差 |
|---:|---:|---:|
| 10 mm | `100.00%` | `-10.33%` |
| 60 mm | `95.92%` | `-0.10%` |
| 70 mm | `89.47%` | `-11.98%` |
| 75 mm | `70.97%` | `-18.50%` |

因此 scorer 解耦后的横向几何和骨层内部散射已经基本正确：60 mm 的 PVDR 几乎
完全一致。剩余误差具有明确的深度依赖，主要表现为 70--75 mm 的 peak 剂量偏低
以及 10 mm 的 valley 略高。下一步优先检查低能 primary/fragment 在骨层出口的
横向 MCS 与 post-bone charged-fragment source term；线段/voxel overlap scorer
仍值得实现，但从 1M 图像看，它已不是当前 70--75 mm 系统误差的主因。

1M 结果：

- `out/minibeam/heterogeneous_bone_e200_1M/comparison/heterogeneous_metrics.json`
- `out/minibeam/heterogeneous_bone_e200_1M/comparison/heterogeneous_depth.png`
- `out/minibeam/heterogeneous_bone_e200_1M/comparison/heterogeneous_lateral.png`

### 20.5 准确输运步长的性能收敛（2026-07-29）

scorer 解耦后，1M profiling 显示 primary kernel 仍占 `29.56 / 33.08 s`，
总输运步数为 `1.624B`。异质验证配置沿用了早期诊断值
`maximum_relative_energy_loss=0.001`，比项目 production accurate 默认值
`0.005` 严格五倍，造成大量没有物理收益的短步。

使用同一个 1M TOPAS 参考、相同 seed 做步长收敛：

| phantom max step / relative loss | runtime | throughput | steps | depth L1 | global γ 3%/1 mm | ΔR80 |
|---|---:|---:|---:|---:|---:|---:|
| 0.1 mm / 0.001 | 33.08 s | 30.23k/s | 1.624B | 1.147% | 95.42% | -0.193 mm |
| 0.2 mm / 0.002 | 18.76 s | 53.30k/s | 0.865B | 1.075% | 95.42% | -0.176 mm |
| **0.2 mm / 0.005** | **10.10 s** | **98.97k/s** | **0.397B** | **1.037%** | **96.08%** | **-0.130 mm** |

最终设置相对原 1M 基线加速 `3.27x`，同时改善深度 L1、RMSE、R80 和
global gamma；全体积积分差由 `-0.489%` 改善为 `-0.227%`，post-bone
积分差由 `-3.81%` 改善为 `-2.22%`。这不是 dose scale 或 case-specific
校准，而是移除过度保守的数值积分限步。

Copper 最大步长从 0.25 mm 放宽至 0.5 mm 的 A/B 会使入口 PVDR 和骨层剂量
退化，因此仍保持 `minibeam_copper_max_step_mm=0.25`（100 MeV/u 保持
0.10 mm）。横向 scorer 和所有 material/z boundaries 仍严格处理。

100、200、300、400 MeV/u 的独立 TOPAS 100k 泛化结果如下：

| 能量 | 原 L1 | 新 L1 | 新 ΔR80 | 新积分差 |
|---:|---:|---:|---:|---:|
| 100 MeV/u | 1.314% | 1.313%（保留 0.001） | +0.011 mm | +0.741% |
| 200 MeV/u | 1.775% | 1.712% | -0.011 mm | +0.085% |
| 300 MeV/u | 1.779% | 1.650% | +0.046 mm | -0.414% |
| 400 MeV/u | 1.999% | 1.945% | -0.073 mm | -0.609% |

`prepare_minibeam_energy_sweep.py` 现在明确生成能量相关的数值精度策略：
100 MeV/u 使用 0.001，200--400 MeV/u 使用 0.005；物理 kernel 中没有加入
按 case 拟合的隐藏分支。

优化结果：

- `out/minibeam/heterogeneous_bone_e200_1M/comparison_gpu_step0p2_rel0p005/heterogeneous_metrics.json`
- `out/minibeam/heterogeneous_bone_e200_1M/comparison_gpu_step0p2_rel0p005/heterogeneous_depth.png`
- `out/minibeam/heterogeneous_bone_e200_1M/comparison_gpu_step0p2_rel0p005/heterogeneous_lateral.png`
- `out/minibeam/energy_sweep_rel0p005_100k/metrics.json`

## 21. 复杂横向/纵向异质场景（2026-07-29）

为了避免只用单块 bone slab 验证，新增三个 200 MeV/u、100k histories 的
二维显式材料 phantom。三者均采用 `500 x 1 x 300` 网格，体素为
`0.2 x 40 x 0.5 mm3`，横向覆盖 `-50--50 mm`，深度覆盖 `0--150 mm`：

1. `lateral_multimaterial`：15--95 mm 深度内，横向依次布置 lung、bone
   和 water，用来检查同一深度不同材料中的横向散射和能量沉积。
2. `longitudinal_multimaterial`：中心束路上依次经过 lung
   18--38 mm、bone 52--64 mm、lung 78--96 mm，用来检查多次材料边界切换、
   水等效射程累积和碎片输运。
3. `combined_multimaterial`：横向和纵向均放置互相错开的 lung/bone 区域，
   同时覆盖界面步进、横向散射和多材料级联。

生成器同时写出 GPU CCTG、YAML、TOPAS 参数文件和材料 metadata：

```bash
python3 validation/scripts/prepare_minibeam_complex_heterogeneity.py \
  --histories 100000 \
  --output-root out/minibeam/complex_heterogeneity_100k
```

TOPAS 使用 `G4_WATER`、`G4_LUNG_ICRP`、`G4_BONE_COMPACT_ICRU`；
GPU 使用同一 Geant4 11.3.2 导出的 absolute stopping-power 和
inelastic-cross-section 表。TOPAS 使用 40 threads。

### 21.1 修复：CCTG v1 的材料 stopping-power 表被单位因子遮蔽

旧版 CCTG v1 reader 会为诊断目的填充全 1 的 `mass_sp_za_rel`。输运端此前仅
根据数组是否非空判断 Schneider mass-SP 是否可用，导致 v1 显式 lung/bone
网格错误地走单位相对因子，而没有使用配置中的 absolute material table。
结果是异质场景的 R80 提前约 1.4--2.7 mm。

现在 `CtGrid::uses_schneider_mass_sp()` 仅允许 CCTG v2/v3 使用 Schneider
mass-SP；v1 始终使用按材料指定的 absolute table。current/legacy 两条输运路径
都采用同一判定，并增加了回归测试，确认 v1 即使带有诊断因子也不会被误判。
修复后没有加入 case-specific dose scale。

### 21.2 100k GPU/TOPAS 正式结果

| 场景 | GPU throughput | TOPAS 40T | depth L1 | 积分比 GPU/TOPAS | global gamma 3%/1 mm | local gamma 3%/1 mm | ΔR80 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 横向多材料 | 54.80k/s | 234.70 s | 1.326% | 0.9996 | 100.00% | 94.44% | +0.120 mm |
| 纵向多层 | 53.18k/s | 247.16 s | 1.288% | 1.0027 | 98.77% | 95.71% | -0.015 mm |
| 横纵组合 | 51.89k/s | 233.53 s | 1.095% | 1.0063 | 100.00% | 96.15% | -0.024 mm |

纵向积分剂量、射程和 Bragg peak 已在三套复杂场景中稳定匹配。横向
`3%/0.4 mm` global gamma 在入口约 94.7%，随深度和局部剂量降低而下降；
组合场景在 24、44、60、78 mm 分别为 87.5%、87.5%、79.5%、65.6%。
曲线显示主要剩余差异位于深部低统计量 peak/valley 和材料界面后的碎片横向
展宽，而不是中心轴射程或绝对 dose scale。要严格量化这些深部横向差异，应
提高到 1M histories；100k 足以验证几何、材料切换和纵向物理。

可复现结果：

- `out/minibeam/complex_heterogeneity_100k/lateral_multimaterial/comparison/`
- `out/minibeam/complex_heterogeneity_100k/longitudinal_multimaterial/comparison/`
- `out/minibeam/complex_heterogeneity_100k/combined_multimaterial/comparison/`

### 21.3 1M 高统计量与 CT minibeam 精度档（2026-07-29）

对最复杂的 `combined_multimaterial` 提高到 1M histories。TOPAS 40 threads
用时 `2261.86 s`；原 GPU production baseline 用时约 `10.83 s`、吞吐
`92.36k histories/s`。仅增加统计量后，depth global/local gamma
`3%/1 mm` 已达到 `100.00%/96.79%`，R80 偏差 `+0.024 mm`。横向
`3%/0.4 mm` global gamma 在 10、24、44、60、78 mm 分别为
`97.37%`、`97.50%`、`97.78%`、`89.19%`、`90.55%`。这确认此前 100k
深部曲线的大部分锯齿来自统计噪声，但 60 mm 的 peak 偏低、valley 偏高是
系统性横向展宽误差。

#### CT 中启用已验证的低能 primary MCS 修正

旧条件只在 `!enable_ct_grid` 时启用 minibeam 低能 primary MCS 修正，导致
显式 CT 中碳离子经过 lung/bone 降能后仍使用未经修正的逐步 Highland core。
现在材料密度和 radiation length 仍由 CT voxel 决定，但 projectile-side
平滑修正也允许在 CT 中生效：

```yaml
minibeam_water_low_energy_mcs_transition_MeVu: 180.0
minibeam_water_primary_low_energy_mcs_scale: 0.20
minibeam_water_fragment_low_energy_mcs_scale: 1.00
```

这些参数来自先前 homogeneous-water 多能量验证，不是按本异质 case 新拟合的
dose scale。production 步长 `0.2 mm / 0.005` 下，60 mm lateral L1 从
`5.95%` 改善到 `4.64%`，R80 从 `+0.024 mm` 改善到 `-0.003 mm`；
吞吐约 `93.83k histories/s`，入口 10--44 mm 结果基本不变。

#### 可选 high-accuracy 档

生成器新增显式精度选择：

```bash
python3 validation/scripts/prepare_minibeam_complex_heterogeneity.py \
  --histories 1000000 \
  --cases combined_multimaterial \
  --precision high_accuracy \
  --output-root out/minibeam/complex_heterogeneity_1M
```

`production` 使用 `0.2 mm / 0.005`；`high_accuracy` 使用
`0.1 mm / 0.001`。后者的正式 1M 结果：

| 指标 | production + CT MCS | high_accuracy |
|---|---:|---:|
| throughput | `93.83k/s` | `23.23k/s` |
| depth global gamma 3%/1 mm | `100.00%` | `100.00%` |
| depth local gamma 3%/1 mm | `96.15%` | `99.36%` |
| depth L1 | `0.952%` | `0.971%` |
| ΔR80 | `-0.003 mm` | `-0.035 mm` |
| 60 mm lateral global gamma 3%/0.4 mm | `89.19%` | `97.30%` |
| 78 mm lateral global gamma 3%/0.4 mm | `90.55%` | `86.61%` |

严格步长适合 Bragg/材料界面附近的 high-accuracy reference，但约慢 `4.04x`。
78 mm 已位于 primary C-12 消失后的低剂量碎片区；GPU 该处剂量约一半来自
helium，其余主要为 boron/proton/light fragments。严格步长不能修复且略微
恶化该区，说明下一项物理改进应是按 parent species、energy、depth 和
lung/bone material 条件化 reaction/cascade final-state package，而不是继续
缩小步长或调整全局剂量。

测试过但未保留的改动：

- 把 6% electronic build-up 的整段随机搬移改为 local/delayed 分别计分：
  1M gamma 不变，吞吐下降约 1.8%。
- primary 使用 0.1 mm、secondary 使用 0.2 mm 的独立限步：
  78 mm gamma 进一步降到 84.25%，depth local gamma 降到 98.08%。

正式输出：

- `config/generated/minibeam_complex/combined_multimaterial_1000000_high_accuracy.yaml`
- `out/minibeam/complex_heterogeneity_1M/combined_multimaterial/comparison_baseline/`
- `out/minibeam/complex_heterogeneity_1M/combined_multimaterial/comparison_ct_mcs/`
- `out/minibeam/complex_heterogeneity_1M/combined_multimaterial/comparison_high_accuracy/`

## 22. 临床 lung CT 中的单平面 minibeam（2026-07-29）

选择 `20022516` lung case，使用 TPS 0°、患者 `+Y` 入射。为避免 961 个
time-feature spot 的 TOPAS run 切换开销，改用一个 `50 x 50 mm` 平行平面源，
照射与水箱验证相同的 60 mm Copper 准直器：

- 15 条平行 slit；
- slit 宽 0.5 mm，pitch 3.6 mm，沿 slit 半长 25 mm；
- 准直器厚 60 mm，出口到 CT bounding-box 入口 60 mm；
- 200 MeV/u C-12，GPU/TOPAS 都为 100k incident histories；
- TOPAS DoseToMedium 网格 `0.5 x 0.5 x 2 mm`；
- LET 关闭。

初次运行把 TOPAS 源/准直器中心放在 world `(0,0)`，而 GPU minibeam 几何中心
位于 patient-local `(0,0)`。该病例 `Patient/TransX=-69.6605 mm`，两者因此
穿过完全不同的肺路径，表现为 IDD 峰相差 103 个 0.5 mm bin、gamma 接近零。
最终 TOPAS 将源和 Snout 一起移到
`world X=-69.6605 mm, Z=0.0819 mm`；修复后 GPU/TOPAS IDD 峰均为
CT 入口后 `150.25 mm`。

### 22.1 绝对同粒子数结果

主结果固定 GPU scale 为 1，不按 TOPAS 拟合：

| 指标 | 100k 结果 |
|---|---:|
| 全体积积分差 GPU/TOPAS | `+3.406%` |
| IDD cosine / Pearson correlation | `0.995840 / 0.985786` |
| IDD 高剂量 NRMSE | `5.307%` |
| Bragg peak 深度差 | `0.000 mm` |
| ΔR80 | `+0.106 mm` |
| global / local gamma 3%/3 mm | `96.15% / 83.86%` |
| global / local gamma 2%/2 mm | `85.35% / 62.72%` |

高剂量 voxel 最小二乘 scale 为 `0.88956`，仅作为诊断；窄 peak 与极低 valley
使 voxel fit 对局部涨落很敏感，不能用它替代绝对比较。3%/3 mm 的 3 mm DTA
接近 3.6 mm pitch，可能搜索到相邻 minibeam，也不是 minibeam 的主要验收指标。

把所有 slit 长度方向和 2 mm 深度 slab 积分后，15 条 peak 的位置和形状在
5、50、100、150 mm 均对齐。PVDR 相对差依次约为 `+20.0%`、`+106.1%`、
`+4.3%`、`-19.7%`；其中 50 mm 的绝对 valley 只有约 `2e-4 Gy`，100k
统计及 0.5 mm scorer（恰好等于 slit 宽）会把 PVDR 比值显著放大。相应的
1D global gamma 3%/0.5 mm 只有 11--57%，每个平面阈值上仅 26--48 个采样点。
因此当前 100k 足以验证 CT 坐标、射程和 minibeam 几何，但不足以把 valley/PVDR
误差归因到具体物理过程；下一轮应提高到至少 1M，并使用 0.1--0.2 mm 横向
scorer 或局部 fine scorer。

深度曲线显示主要系统残差位于入口后约 0--50 mm：GPU 高约 10--15%；50 mm
以后大部分深度积分进入约 ±3%，Bragg peak 和 distal falloff 很好。关闭水箱
使用的 Copper-touched primary surface boost/deficit 几乎不改变结果，因为
该项只影响少量穿铜后仍存活的 primary C-12；入口差异更可能来自 Copper
charged/neutral secondary source term 或 CT 入口低密度材料中的次级剂量搬移。
建议下一步开启 charged-origin/species scoring 做 GPU 分源，并在 TOPAS 增加
水入口/CT入口 phase space 或 species-resolved scorer。

### 22.2 速度与复现

| 程序 | 配置 | 时间 | 吞吐 |
|---|---|---:|---:|
| GPU | TITAN RTX, accurate minibeam+CT | `3.165 s` | `31.60k histories/s` |
| TOPAS | 40 CPU threads | `291.50 s` | `343 histories/s` |

该平面 minibeam CT case 的端到端 GPU speedup 约 `92.1x`，charged/neutral
queue 均无 overflow。TOPAS 未计分导航能量为 `5.65e-5 MeV`，可忽略。

复现入口：

- `ct/fullplan_local_ct_compare/20022516/run_minibeam_plane_e200_100k.txt`
- `config/generated/beam_ct_20022516_minibeam_plane_e200_100k.yaml`
- `validation/scripts/analyze_ct_minibeam_plane.py`
- `out/ct/20022516/minibeam_plane_e200_100k/match_absolute_ct/`
- `out/ct/20022516/minibeam_plane_e200_100k/profiles_absolute_ct/`
- `out/ct/20022516/minibeam_plane_e200_100k/multiplanar_absolute_ct/`

### 22.3 1M 统计收敛验证

GPU 和 TOPAS 使用相同几何、相同随机 history 数重新运行 1M；最终比较继续固定
绝对 scale 为 1，不使用最小二乘归一化。TOPAS 40 线程执行时间
`2259.91 s`（总 wall time `2267.64 s`），GPU wall time 约 `21.1 s`，
端到端加速约 `107x`。GPU charged/neutral queue 均无 overflow。TOPAS
parallel-world 导航漏记能量仅 `0.001916 MeV`，对结果可忽略。

| 指标 | 100k | 1M |
|---|---:|---:|
| 全体积积分差 GPU/TOPAS | `+3.406%` | `+5.626%` |
| IDD cosine / Pearson correlation | `0.995840 / 0.985786` | `0.995924 / 0.986222` |
| IDD 高剂量 NRMSE | `5.307%` | `5.657%` |
| Bragg peak 深度差 | `0.000 mm` | `0.000 mm` |
| ΔR80 | `+0.106 mm` | `+0.243 mm` |
| global / local gamma 3%/3 mm | `96.15% / 83.86%` | `95.04% / 80.99%` |
| global / local gamma 2%/2 mm | `85.35% / 62.72%` | `85.16% / 64.66%` |

1M 的高剂量 voxel 诊断性最小二乘 scale 为 `0.96153`，但主结果未采用它。
提高十倍 history 后，纯统计标准差理论上降至 100k 的 `31.6%`；整体 gamma
没有随之明显上升，证明主要剩余误差不是 history 不足，而是系统性物理差异：

- CT 入口后约 0--50 mm，GPU IDD 高约 10--15%；
- 100 mm 深度的横向 peak/valley 已明显收敛，1D global gamma
  3%/0.5 mm 从 `56.7%` 提升至 `93.3%`；
- 5 和 50 mm 的 GPU peak 仍系统性偏高、valley 偏低，PVDR 相对 TOPAS
  分别高 `36.3%` 和 `93.1%`；
- 150 mm Bragg 区的 GPU peak 偏低，PVDR 低 `16.0%`；
- 0.5 mm scorer 与 0.5 mm slit 等宽，每个横向平面阈值上仍只有
  26--48 个点，因此 3%/0.5 mm 对半个体素的相位和 peak 采样非常敏感。

因此，继续从 1M 单纯增加到 10M 只能降低曲线噪声，不能消除入口、valley
和 Bragg 区的系统偏差。下一项有效工作应是：

1. 用 0.1--0.2 mm 的局部横向 fine scorer 排除体素采样误差；
2. 对 GPU 开启 Copper-touched charged/neutral secondary 的 species/origin
   分源计分，并在 TOPAS CT 入口输出 phase space；
3. 根据分源结果修正 Copper 后碎片的横向散射、低密度肺组织中的次级输运，
   而不是应用 case-specific 剂量缩放。

1M 复现与结果：

- `ct/fullplan_local_ct_compare/20022516/run_minibeam_plane_e200_1M.txt`
- `out/ct/20022516/minibeam_plane_e200_1M/topas/dose.mhd`
- `out/ct/20022516/minibeam_plane_e200_1M/gpu/dose.mhd`
- `out/ct/20022516/minibeam_plane_e200_1M/match_absolute_ct/`
- `out/ct/20022516/minibeam_plane_e200_1M/profiles_absolute_ct/`
- `out/ct/20022516/minibeam_plane_e200_1M/multiplanar_absolute_ct/`

另外复用同一份 1M TOPAS 做了 GPU-only high-accuracy A/B：只把
`maximum_step_mm` 从 `0.2` 收紧到 `0.1`，把
`maximum_relative_energy_loss` 从 `0.005` 收紧到 `0.001`。

| 指标 | production | high-accuracy |
|---|---:|---:|
| GPU 时间 / throughput | `~21.1 s / ~47.4k/s` | `57.42 s / 17.42k/s` |
| 全体积积分差 GPU/TOPAS | `+5.626%` | `+2.742%` |
| IDD cosine / Pearson correlation | `0.995924 / 0.986222` | `0.997998 / 0.993256` |
| IDD 高剂量 NRMSE | `5.657%` | `3.674%` |
| ΔR80 | `+0.243 mm` | `+0.105 mm` |
| global / local gamma 3%/3 mm | `95.04% / 80.99%` | `92.95% / 78.91%` |
| global / local gamma 2%/2 mm | `85.16% / 64.66%` | `82.67% / 63.88%` |

严格步长改善 IDD 形状、积分、R80 和严格的 3%/0.3 mm gamma，但没有改善
3%/3 mm；入口 5/50 mm PVDR 相对差反而增至
`+122.7% / +131.2%`。用户当前要求优先提高精度，因此这个 lung-minibeam
验证配置保留 high-accuracy 参数；非 minibeam 配置不受影响。结果仍指向
Copper reaction product 的 species、角分布和 CT 低密度区输运，而不是仅靠
继续缩小普通 charged-particle 步长。

high-accuracy A/B 输出：

- `out/ct/20022516/minibeam_plane_e200_1M_high_accuracy/match_absolute_ct/`
- `out/ct/20022516/minibeam_plane_e200_1M_high_accuracy/profiles_absolute_ct/`

### 22.4 严格 3%/0.3 mm gamma

`match_gpu_to_physical_dose.py` 新增可重复的
`--gamma-criterion DOSE_PERCENT DISTANCE_MM` 和 `--only-custom-gamma`，
以 0.1 mm 三线性搜索步长、10% TOPAS 最大剂量阈值、固定随机种子的 100k
voxel 样本计算 3D global/local gamma。剂量保持同 history 绝对 scale=1。

| 1M GPU 模式 | global 3%/0.3 mm | local 3%/0.3 mm |
|---|---:|---:|
| production：0.2 mm / 0.005 | `89.249%` | `47.656%` |
| high-accuracy：0.1 mm / 0.001 | `90.703%` | `48.272%` |

high-accuracy 分别提高 `1.454` 和 `0.616` 个百分点。不同深度的横向 1D
global/local 3%/0.3 mm 通过率为：

| 深度 | production | high-accuracy |
|---|---:|---:|
| 5 mm | `11.54% / 0.00%` | `15.38% / 0.00%` |
| 50 mm | `34.62% / 3.85%` | `34.62% / 3.85%` |
| 100 mm | `93.33% / 56.67%` | `83.33% / 53.33%` |
| 150 mm | `50.00% / 10.42%` | `58.33% / 14.58%` |

当前 CT/TOPAS scorer 横向间距为 0.5 mm，大于 0.3 mm DTA，且恰好等于 slit
宽度。因此以上数字适合作为固定网格上的严格回归指标，但不能当作独立的
0.3 mm 测量分辨率结论。下一步应把束流附近局部 scorer 细化至
0.1--0.2 mm，并让 GPU 使用同一局部 dose-to-medium mass grid 后再做正式
亚毫米验收。

输出：

- `out/ct/20022516/minibeam_plane_e200_1M/gamma_3pct_0p3mm_absolute_ct/`
- `out/ct/20022516/minibeam_plane_e200_1M_high_accuracy/gamma_3pct_0p3mm_absolute_ct/`
- `out/ct/20022516/minibeam_plane_e200_1M_high_accuracy/multiplanar_3pct_0p3mm_absolute_ct/`

### 22.5 可信 95% 的可达性与 10M 收敛

先用两个独立 seed 的 1M high-accuracy GPU 结果测量统计上限。相同物理模型、
相同绝对 history 比例下：

- IDD correlation：`0.999960`；
- 高剂量 NRMSE：`3.106%`；
- global 3%/0.3 mm：`95.750%`；
- local 3%/0.3 mm：`47.484%`。

这说明 1M 时 global 95% 只是统计上限附近，不能期待 GPU/TOPAS 在仍有物理
模型差异时稳定超过 95%。随后把 GPU 提高到 10M。第一次单次 10M 沿用了
2M secondary/neutral queue，出现大量 overflow，结果作废。有效重跑显式使用
10M/10M queue，显存估算约 3.64 GiB，最终：

- secondary queue overflow：`0`；
- cascade queue overflow：`0`；
- neutral queue overflow：`0`；
- transported charged secondaries：`7,048,171`；
- transported neutrals：`7,955,473`。

为了让后续 10M 命令默认安全，lung-minibeam YAML 的 secondary/neutral queue
capacity 已提高到 10M。与现有 1M TOPAS 比较时，GPU 10M 剂量乘以严格 history
比例 `0.1`，不做拟合归一化：

| 指标 | GPU 1M | GPU 10M |
|---|---:|---:|
| 3D cosine | `0.972383` | `0.986627` |
| 高剂量 NRMSE | `4.272%` | `3.144%` |
| absolute integral difference | `+2.742%` | `+2.916%` |
| ΔR80 | `+0.105 mm` | `+0.092 mm` |
| global 3%/0.3 mm | `90.703%` | `92.973%` |
| local 3%/0.3 mm | `48.272%` | `45.964%` |

提高 GPU history 带来 `+2.270` 个百分点，但绝对 global 仍未达到 95%。
诊断性剂量扫描在 effective scale=`0.103` 时可得到约 `95.36%`，但同时令
GPU 积分剂量高约 `6.00%`；这是利用 TOPAS 1M 单 voxel 峰值提高 global gamma，
会恶化绝对物理剂量，因此没有采用，也不能宣称为 95% match。

当前结论：

1. 数字上可通过错误的整体放大超过 95%，但物理上不可接受；
2. 可信的下一步是把 TOPAS 也提高到至少 5--10M，并采用无 queue 问题的多个
   独立 batch 累加；
3. 正式 0.3 mm 验收还需要 GPU/TOPAS 同步的 0.1--0.2 mm 局部 scorer；
4. 统计收敛后若仍低于 95%，再用 collimator-exit phase space 和 species/origin
   scoring 修正 Copper reaction products，而不是调全局 dose scale。

10M 有效输出：

- `out/ct/20022516/minibeam_plane_e200_10M_high_accuracy_valid/gpu/dose.mhd`
- `out/ct/20022516/minibeam_plane_e200_10M_high_accuracy_valid/gamma_3pct_0p3mm_absolute_ct/`
- `out/ct/20022516/minibeam_plane_e200_10M_high_accuracy_valid/profiles_absolute_ct/`
- `out/ct/20022516/minibeam_plane_e200_10M_high_accuracy_valid/multiplanar_3pct_0p3mm_absolute_ct/`

### 22.6 5M TOPAS、可信 global 95% 与 local 上限

5M TOPAS 使用40线程完成，execution `11034.71 s`、总 wall time
`11042.6 s`。parallel-world 导航漏记总能量仅 `0.002175 MeV`，可忽略。
10M GPU 对5M TOPAS采用严格 history scale=`0.5`，不做拟合归一化。对全部
211,348个10%阈值voxel计算：

| 指标 | 10M GPU vs 5M TOPAS |
|---|---:|
| 3D cosine | `0.990677` |
| 高剂量 NRMSE | `2.894%` |
| absolute integral difference | `+2.372%` |
| peak voxel coordinate | 完全一致 |
| global 3%/0.3 mm | `95.039%` |
| local 3%/0.3 mm | `53.332%` |

因此 global 95% 已在绝对剂量、全部阈值voxel条件下可信达到，不依赖抽样误差
或case-specific整体缩放。

local gamma 仍明显受统计噪声和peak/valley振幅影响。两个独立seed的10M
GPU在相同物理模型、绝对scale=1条件下：

| 指标 | 10M GPU vs 10M GPU |
|---|---:|
| IDD correlation | `0.999996` |
| integral difference | `-0.029%` |
| global 3%/0.3 mm | `98.653%` |
| local 3%/0.3 mm | `67.215%` |

这证明即使物理模型完全相同，当前0.5 mm窄峰voxel和10M统计下，raw local
通过率的上限也只有约67%，不能把53%全部解释成GPU/TOPAS物理误差。

两个独立10M GPU逐voxel求和形成20M后，与5M TOPAS按严格scale=`0.25`
比较：

- global 3%/0.3 mm：`95.427%`；
- local 3%/0.3 mm：`52.968%`；
- 3D cosine：`0.991496`。

GPU继续降噪提高global但没有提高local，说明当前local主瓶颈已经转为TOPAS
参考噪声以及系统性PVDR差异。分深度横向local 3%/0.3 mm为：

- 5 mm：`0.0%`，PVDR GPU高约`110%`；
- 50 mm：`3.85%`，PVDR GPU高约`125%`；
- 100 mm：`73.33%`，PVDR GPU高约`22.8%`；
- 150 mm：`14.0%`，PVDR GPU低约`14.8%`。

下一步已启动第二个独立seed的5M TOPAS。完成后将：

1. 计算5M TOPAS↔5M TOPAS的local统计上限；
2. 合成10M TOPAS，与20M GPU按scale=`0.5`比较；
3. 从两个TOPAS batch和两个GPU batch估计逐voxel不确定度，增加
   uncertainty-aware local gamma；
4. 统计收敛后再针对剩余PVDR误差修改Copper reaction-product角分布/MCS。

新增可复现工具与输出：

- `validation/scripts/combine_mhd_dose.py`
- `ct/fullplan_local_ct_compare/20022516/run_minibeam_plane_e200_5M.txt`
- `ct/fullplan_local_ct_compare/20022516/run_minibeam_plane_e200_5M_seed20260803.txt`
- `out/ct/20022516/minibeam_plane_e200_10M_vs_topas_5M/`
- `out/ct/20022516/minibeam_plane_e200_10M_gpu_gpu/`
- `out/ct/20022516/minibeam_plane_e200_20M_vs_topas_5M/`

### 22.7 10M TOPAS 最终统计收敛验证

第二个独立 seed 的 5M TOPAS 已完成，execution `11001.65 s`、总 wall time
`11009.5 s`，parallel-world 导航漏记总能量仅 `0.002905 MeV`。两批结果逐
voxel 求和得到 10M TOPAS。两个 5M TOPAS batch 在相同物理模型下使用严格
history scale（并仅用 `0.997857` 修正两批实际 scorer 总量差）比较：

| 指标 | 5M TOPAS vs 5M TOPAS |
|---|---:|
| 3D cosine | `0.997795` |
| 高剂量 NRMSE | `1.330%` |
| integral difference | `-0.339%` |
| IDD correlation | `0.999984` |
| global 3%/0.3 mm | `99.095%` |
| local 3%/0.3 mm | `59.294%` |

因此，即使参考端物理完全相同，当前 5M batch、0.5 mm 横向网格和窄
peak/valley 下，raw local 通过率也只有约 59%。它不是一个可以单独用来调底层
物理的无噪声目标。

最终 20M GPU 与合并 10M TOPAS 使用严格 history scale=`0.5`，不拟合 GPU
剂量。全部 210,764 个 10% 阈值 voxel 的结果为：

| 指标 | 20M GPU vs 10M TOPAS |
|---|---:|
| 3D cosine | `0.992039` |
| 高剂量 NRMSE | `2.740%` |
| absolute integral difference | `+2.530%` |
| IDD correlation | `0.997880` |
| global 3%/0.3 mm | `96.462%` |
| local 3%/0.3 mm | `55.464%` |

用两个 TOPAS batch 和两个 GPU batch 的差异估计逐 voxel 标准误差：

- TOPAS relative SEM：中位数 `1.762%`，P90 `4.739%`，P95 `5.985%`；
- GPU relative SEM：中位数 `1.540%`，P90 `4.360%`，P95 `5.677%`；
- raw local 3%/0.3 mm：`55.464%`；
- 允许 1σ 蒙卡统计不确定度后：`67.874%`；
- 允许 2σ 蒙卡统计不确定度后：`81.231%`。

结论是 global 已稳定超过 95%，且 GPU/TOPAS 的 raw local 与 TOPAS/TOPAS
噪声基线接近。当前 local 的主要限制是有限 history、0.5 mm scorer 对
0.3 mm DTA 的欠采样，以及剩余约 2.5% 的绝对积分/PVDR 系统差。不能继续通过
case-specific 整体归一化或仅追逐 raw local 数值调参。下一步若要提高可解释的
local 精度，应优先：

1. 在 beam corridor 使用 GPU/TOPAS 一致的 0.1--0.2 mm 局部 scorer；
2. 每侧使用更多独立 batch，把 peak/valley voxel 的 SEM 压至 1% 以下；
3. 在收敛网格和统计下再按 depth 分析 Copper reaction-product 与 MCS 对
   PVDR 的系统偏差。

最终输出：

- `out/ct/20022516/minibeam_plane_e200_10M_combined/topas/dose.mhd`
- `out/ct/20022516/minibeam_plane_e200_20M_high_accuracy_combined/gpu/dose.mhd`
- `out/ct/20022516/minibeam_plane_e200_5M_topas_topas/`
- `out/ct/20022516/minibeam_plane_e200_20M_vs_topas_10M/`

## 23. RT07575 full-plan sparse-Dij threshold audit（2026-08-01）

复制回来的 `bio_dij_osmk_sparse_c.mat` 包含 1943 个 spot 的完整 CSC 稀疏
物理剂量矩阵。用 `resultGUI.w` 对列积分做恒等检查，得到
`545281.372658 Gy-voxel`，而 `resultGUI.physicalDose` 为
`545281.372974 Gy-voxel`，相对误差仅 `5.8e-10`；spot 顺序、权重和
physicalDose 是一致的。

但 `build_bio_dij_osmk_sparse.m` 在组成矩阵前对每个 spot/voxel 使用了
`dose_limit=2e-6 Gy`。该阈值保留 `389,709,791` 项、删除
`2,674,181,011` 项，即原始非零项的 **87.281%** 被置零。旧 MAT 只保存了
dropped nnz 数量，没有保存被删剂量和，因此无法从该 MAT 精确恢复无阈值剂量。

该 cutoff 对严格 gamma 并不安全：

- `sum(w)=43165.516`；
- 单 voxel 累计遗漏上界为 `2e-6*sum(w)=0.08633 Gy`；
- BODY 内参考峰为 `2.92646 Gy`，3% global tolerance 为 `0.08779 Gy`；
- 即稀疏化允许的遗漏上界已达到 `2.950%` of peak，几乎占满全部剂量容差；
- 被删项的加权积分上界为参考积分的 `26.24%`；只需其平均值达到 cutoff 的
  `25.70%`，就足以解释 86M GPU 相对 thresholded reference 的全体积
  `+6.743%` 积分差。

这不能证明全部 GPU/TOPAS 差异都来自 threshold，但证明当前 thresholded
`resultGUI.physicalDose` 不能作为 3%/0.3 mm 底层物理调参的无偏参考。尤其
low-dose、valley 和多 spot Bragg-overlap 区会累积大量单独低于 cutoff 的贡献。

已做修复：

1. builder 的 Dtotal cutoff 降为 `2e-7 Gy`，对应本计划单 voxel 上界
   `0.00863 Gy = 0.295% of peak`；若磁盘允许可设为零；
2. 新 builder 同时保存每 spot/scorer 的 raw、kept、dropped 实际剂量和，避免
   再次只知道删除数量；
3. `validation/scripts/audit_sparse_dij_threshold.py` 输出可复现的阈值审计；
4. `validation/scripts/build_unthresholded_osmk_plan_reference.py` 可在原始
   `OSMK_Dtotal` 文件所在机器上逐 spot 乘 `resultGUI.w` 并流式累积无阈值
   full-plan reference，无需创建几十 GB 的无阈值 sparse Dij。

本地没有 1943 个原始 `OSMK_Dtotal_*_Run_*.bin`，因此下一步应在 scorer 所在
集群运行无阈值累积脚本，再用该 reference 重算 BODY 内 3%/0.3 mm gamma。
只有消除 threshold bias 后，才能继续判断 Bragg-overlap 剩余误差是否来自
primary C-12 range/straggling 或 copper reaction-product lateral response。

补充核对：`ct/fullplan_result/RT07575/OSMK_Dtotal_full_plan.bin` 不能用于替代
上述无阈值 reference。它是普通 `c_01+c_02` 计划（917 spots、实际 Dij
50k histories/spot），而本节 minibeam 计划是 `c_01--c_04`（1943 spots、
100k histories/spot）。二者不是同一束流计划。

builder 现在还会在 `code_v2/RBE_dose_result_c.mat` 存在时读取最终权重，并在
构建开始时打印 `dose_limit*sum(abs(w))` 的保守单 voxel 遗漏上界；构建完成后
则用新记录的实际 dropped sum 打印最终计划的 kept/dropped/raw 积分和真实
积分损失比例。权重只用于审计，不参与 Dij 构建。这样即使未来改变 spot 数或
优化权重，也不会继续沿用一个对该计划不安全的固定阈值而没有告警。

其余 OSMK 矩阵也存在更激进的 sparsification：`weighted_zd`、
`weighted_zd_star`、`weighted_zn` 分别删除了 `94.789%`、`91.312%`、
`95.423%` 的原始非零项。它们不会改变固定权重下的 physicalDose 恒等式，但会
改变 OSMK RBE 和据此重新优化出的权重。因此本轮“GPU physical dose vs 当前
权重对应 TOPAS physical dose”只需首先消除 Dtotal threshold；如果后续要重新
做高精度 RBE optimization，还必须分别审计这三项的实际 dropped weighted sum，
不能沿用当前阈值只看 sparse 文件大小。

final builder 的 `fail_on_missing` 已改为 `true`。当前复制回来的 MAT 中四个
scorer 的 1943 列均有非零列，没有发现整列缺失；以后若原始 scorer 传输不全，
构建会立即失败，而不会静默把缺失 spot 写成零剂量。

## 24. GPU 使用相同逐-spot Dtotal 阈值的 A/B（2026-08-02）

为了回答“如果 GPU 也使用与 sparse Dij 完全相同的阈值，gamma 是否提高”，
增加了仅通过命令行开启的诊断模式：每个正权重 spot 独立运行 100,000 histories，
在乘优化权重和累加前执行

```text
D_i(voxel) < 2e-6 Gy  ->  0
D_plan = sum_i w_i * D_i
```

不能对已有 full-plan MHD 整体应用 `2e-6 Gy`，因为那时 spot 身份已经丢失，
与 Dij builder 的处理不等价。诊断模式不改变默认 batched full-plan 路径。

正式计算包含 1,617 个正权重 spot、161.7M histories。angle01/angle02 分别为
811/806 spots，用时 11,042.047/10,881.628 s；两进程并行运行。所有 charged、
cascade 和 neutral queue overflow 均为零。映射约定用旧四 subfield 输出逐体素
重建验证，RMSE 仅 `8.8e-9 Gy`：angle01 flip patient X，angle02 flip patient Y。

GPU 阈值删除的加权积分为：

- angle01：`12,895.571 Gy-voxel`；
- angle02：`5,141.024 Gy-voxel`；
- 合计：`18,036.595 Gy-voxel`。

阈值后的全体积 GPU 积分为 `563,834.258 Gy-voxel`。加回被删积分得到
`581,870.853 Gy-voxel`，与此前独立 86M GPU 的 `582,049.771 Gy-voxel`
只差约 `0.031%`，验证了逐 spot 阈值和权重实现的一致性。BODY 内积分差从
未阈值 GPU 的 `+2.357%` 变为 `-0.165%`；高剂量 NRMSE 从 `2.916%` 小幅降为
`2.898%`，IDD correlation 从 `0.999850` 升为 `0.999937`。

但 gamma 没有随积分改善而提高：

| criterion | 原 GPU global/local | GPU 同阈值 global/local | 变化（百分点） |
|---|---:|---:|---:|
| 3%/3 mm | 99.358 / 97.602 | 99.204 / 97.368 | -0.154 / -0.234 |
| 2%/2 mm | 95.860 / 89.508 | 95.308 / 89.340 | -0.552 / -0.168 |
| 1%/1 mm | 70.830 / 45.164 | 70.520 / 45.358 | -0.310 / +0.194 |
| 3%/0 mm | 73.961 / 31.530 | 73.825 / 31.619 | -0.136 / +0.089 |
| 3%/0.3 mm，全 317,867 voxels | 89.591 / 66.494 | 88.979 / 66.306 | -0.612 / -0.188 |
| 3%/0.5 mm，全 317,867 voxels | — | 92.750 / 78.194 | — |

因此旧 `2e-6 Gy` cutoff 的确解释了大部分绝对积分背景偏差，但不是当前严格
gamma 的主要限制。给 GPU 和 TOPAS 同时施加相同信息损失只会让积分看起来更
一致，并没有改善 peak/valley、Bragg overlap 和空间梯度形状。后续物理调参
仍应使用无阈值 TOPAS reference；不应把“双边同阈值”作为提高准确率的方法。

可复现输出：

- `out/ct/RT07575/minibeam_plan/gpu_per_spot_threshold_2e-6_100k/summary.json`
- `.../full_plan/dose_thresholded_like_dij.mhd`
- `.../full_plan/compare_standard/match_metrics.json`
- `.../full_plan/compare_submm_all/match_metrics.json`

## 25. RT07575 GPU 三倍统计量双 seed 收敛（2026-08-02）

为直接测量蒙卡涨落对 `3%/0.5 mm` gamma 的影响，保持物理、spot 权重、BODY
mask 和剂量尺度不变，把每个 full-plan seed 从 `43,165,516` 增加到
`129,496,548 histories`（严格 `3x`）。两个 seed 合计 `258,993,096`
histories，相对单个旧 run 是 `6x`。

高统计 batched angle01 的默认20M队列会发生大量截断。最终使用显式
`secondary=58M`、`neutral=70M`，并把该验证配置的 CUDA 显存预算提高到82%。
angle01/angle02 的估算显存分别为 `20,063 / 18,133 MiB`。两组 seed 的所有
secondary、cascade 和 neutral overflow 均为零。为允许显式高统计 neutral
队列，CUDA neutral hard ceiling 从16M提高到96M；默认自动容量不变，实际分配
仍受 device-memory estimator 和预算保护。

每个 seed 的角度 histories 为 `79,737,018 + 49,759,530`。seed A/B 的有效
elapsed 分别为 `3565.21 / 3576.36 s`，两组总计 `7141.57 s`，有效吞吐
`36.27k histories/s`。

采用固定 absolute scale=1、BODY 内参考剂量至少10% peak、0.1 mm trilinear
搜索步长和全部阈值 voxel。双向 gamma 为：

| reference -> evaluated | voxels | global 3%/0.5 mm | local 3%/0.5 mm | NRMSE |
|---|---:|---:|---:|---:|
| seed A -> seed B | 321,241 | **99.9757%** | **98.0641%** | 0.7736% |
| seed B -> seed A | 321,132 | **99.9757%** | **98.0606%** | 0.7722% |

旧 `43.17M/seed` 基线为 global/local `99.4745% / 92.9508%`、NRMSE
`1.3364%`。增加到3倍后 global 提高约 `0.501` 个百分点、local 提高约
`5.11` 个百分点，NRMSE 比值 `0.579`，几乎等于理论 `1/sqrt(3)=0.577`。
因此此前同模型 seed 间的主要差异确实是有限粒子数涨落；但 local 的实际
收敛慢于简单均匀高斯外推，3倍统计量达到约98.06%，而不是预估的99.8%。

结果：

- `out/ct/RT07575/minibeam_plan/gpu_gpu_3x_seed_convergence/summary.json`
- `.../gamma_3pct_0p5mm_all/match_metrics.json`
- `.../gamma_3pct_0p5mm_all_reverse/match_metrics.json`

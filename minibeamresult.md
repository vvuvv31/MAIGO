# Copper minibeam 接入与验证结果

本文记录 `pristine` 分支从 `352fa95` 开始完成的 Copper minibeam GPU
输运、TOPAS 数据提取、多能量剂量验证和水中散射修正。本文中的大体积
TOPAS/GPU 输出保存在本机 `/mnt/sda/wuwei` 和仓库忽略的 `out/` 下，不随
Git 提交；生成配置、分析脚本和汇总数据随代码提交。

## 1. 实现范围

### 1.1 可选编译

minibeam 通过 CMake 选项 `CARBON_ENABLE_MINIBEAM` 编译，默认关闭，因此
普通 CT、水箱和 LET 构建不会装载 Copper 专用事件库，也不会在粒子热路径
增加运行时负担。

本次 NVIDIA 验证固定使用：

```text
CARBON_ENABLE_MINIBEAM=ON
CARBON_DOSE_FP32=ON
CARBON_DOSE_FP64=OFF
CARBON_DISABLE_INTEGRITY_CHECKS=ON
```

完整性检查当前由 CMake 强制关闭；运行时会打印醒目的
`[integrity-checks] DISABLED`。这满足当前开发需求，但正式分发前应重新评估
是否恢复 SHA256 内容验证。

### 1.2 几何和调用参数

GPU 实现使用矩形 Copper block。TOPAS example 的外形是圆柱体，但剂量相关
的 slit 几何保持一致。新增配置参数包括：

- collimator 几何中心到 isocenter 的距离；
- collimator 长、宽、厚；
- slit 长、宽、厚；
- slit 数量、间距和整体偏移；
- collimator 旋转角；
- water entrance 的世界坐标位置。

GPU 支持：

- `absorbing_geometry`：精确直线 aperture/吸收几何，用于 smoke test；
- `copper_em`：Copper 中连续能损、MCS、核衰减和产物输运；
- TOPAS/TPS spot plan、BiGaussian emittance、能散和 IEC 坐标变换；
- 水入口 primary phase-space 输出及诊断计数；
- dense primary track-length fluence MHD 输出。

### 1.3 Copper 和水中物理

实现和接入的主要物理项为：

- Geant4 11.3.2 提取的 C12/Copper stopping power 和 inelastic rate；
- 多离子 Copper stopping power 与 inelastic cross section 表；
- Copper condensed Highland MCS；
- 自建 General Ion Elastic 事件库接口；
- INCL++ Copper 反应事件包和带电产物输运；
- Copper 后 primary survivor 能损校正；
- 水中 Unified EM、energy straggling、primary/secondary inelastic；
- `cinel02_max_secondary_inelastic_generations=2`，允许碎片继续发生一次反应，
  同时保留末代 charged product 的 EM 输运；
- 水中 primary C12 的 step-size-aware Gaussian core/rare-wide-tail MCS。

正式 150/250/300 MeV/u 配置均关闭 GPU discrete Copper elastic。独立相空间
验证表明当前有限事件库会产生过强宽角尾；condensed Copper MCS 仍启用。

## 2. TOPAS reference 和统计口径

TOPAS example 位于 `benchmark/carbonminibeam/`，使用：

- 3 cm × 3 cm PBS field；
- 15 个 slit，宽 0.5 mm、pitch 3.6 mm；
- 250 mm 水箱；
- lateral/depth spacing = 0.1/0.25 mm；
- Geant4 11.3.2、OpenTOPAS 4.2.p3；
- `g4em-standard_opt4`、INCL++、自建 General Ion Elastic 及完整 hadronic 模块；
- 本地 Slurm 192 threads。

比较脚本从 MHD/TOPAS header 自动读取 grid spacing，峰谷曲线使用 1 mm
depth smoothing，中心九个 peak 和八个 valley。所有表中比值均为 GPU/TOPAS，
不做拟合归一化。

## 3. 完整 Copper+water 多能量结果

开启 fragment generation 2 后，原 10M、tail width 0.85 基线为：

| Energy (MeV/u) | Total dose | 2-D L1 | IDD L1 | Lateral L1 | Bragg peak | Bragg valley | Bragg PVDR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 150 | 0.9955 | 3.663% | 0.946% | 1.800% | 1.003 | 1.102 | 0.910 |
| 250 | 0.9902 | 3.919% | 1.001% | 1.798% | 0.988 | 1.056 | 0.936 |
| 300 | 0.9887 | 3.800% | 1.143% | 1.631% | 0.966 | 1.031 | 0.938 |

Bragg depth 为 51.375/124.375/168.625 mm；GPU 对应为
51.375/124.375/168.875 mm。总剂量、射程和 IDD 已接近，但 250/300 的
Bragg PVDR 仍系统偏低。

fragment generation 2 是必要修复。它将 250/300 MeV/u distal total ratio
从 1.146/1.131 改善到 0.992/0.989，将 charged-fragment distal ratio 从
1.192/1.170 改善到 1.032/1.023。该差距不是“secondary Unified EM 未开启”，
而是此前遗漏了 fragment reinteraction。

## 4. 水入口同源 replay 隔离

### 4.1 相空间筛选修正

最初使用 `GenericIon(6,12,6)` 会只保留 charge-state 6 的 C12，并遗漏低能、
宽角 primary。修正为 `GenericIon(6,12,*)` 后，10,000,128 个 incident
histories 得到：

- 1,202,596 个 C12 crossing；
- 1,202,572 个 parent-ID 0 C12 replay histories；
- surviving fraction 0.1202557；
- mean/std energy = 219.00/54.22 MeV/u；
- minimum energy = 0.0041 MeV/u。

GPU 和 TOPAS 均仅 replay 这 1,202,572 个 non-empty histories；若需要 incident
normalization，再共同乘 surviving fraction，不影响 GPU/TOPAS 比值。

### 4.2 tail width 0.85 的高统计 replay

| Metric | GPU/TOPAS or error |
|---|---:|
| Total dose | 0.9969 |
| 2-D L1 | 3.358% |
| IDD L1 | 0.499% |
| Lateral-integral L1 | 1.172% |
| Bragg peak | 0.978 |
| Bragg valley | 1.034 |
| Bragg PVDR | 0.946 |

旧 30,878-primary replay 的 Bragg valley/PVDR `0.808/1.258` 主要是统计噪声，
不能用于模型调参。高统计结果确认稳定误差为 GPU primary 随深度轻微过展宽。

## 5. Geant4 逐深度 primary 约束

TOPAS 在水深 40、80、100、115、124 mm 记录 parent C12 crossing；GPU 用
primary track-length fluence MHD。普通 RMS 会被完整 slit array 的宽度主导，
因此分析使用 3.6 mm 基频、二次谐波和折叠到单 pitch 的 profile。

tail width 0.85 的 primary 基频比为：

| Depth (mm) | 40 | 80 | 100 | 115 | 124 |
|---:|---:|---:|---:|---:|---:|
| GPU/TOPAS fundamental | 0.997 | 0.968 | 0.948 | 0.947 | 0.953 |

误差从约 80 mm 开始累积，105–115 mm 的剂量 peak 低约 3%、valley 高约 9%。
这排除了单纯入口相空间修正，并证明当前 Gaussian core 过宽。

基于上述 Geant4 profile，只测试了一个受约束候选：保持 tail strength 0.5，
将 width 从 0.85 调整为 0.90。该变化保持投影二阶矩，但把更多方差移入更稀有
的宽尾，使主 core 收窄约 3.5%。新基频比为：

| Depth (mm) | 40 | 80 | 100 | 115 | 124 |
|---:|---:|---:|---:|---:|---:|
| GPU/TOPAS fundamental | 0.997 | 0.972 | 0.959 | 0.965 | 0.978 |

对应 250 MeV/u 高统计 replay 改善为：

| Metric | width 0.85 | width 0.90 |
|---|---:|---:|
| 2-D L1 | 3.358% | 3.245% |
| Lateral-integral L1 | 1.172% | 1.002% |
| Bragg peak | 0.978 | 0.982 |
| Bragg valley | 1.034 | 1.009 |
| Bragg PVDR | 0.946 | 0.973 |

同 seed、完整 Copper+water 256k 检查在 250 和 300 MeV/u 均改善 2-D、lateral、
IDD 和 Bragg PVDR，因此正式 250/300 配置采用 width 0.90。150 MeV/u 保持
tail 关闭；把同一 tail 无条件应用到 150 会破坏已匹配的剂量。

注意：三能量完整 10M 表仍对应 width 0.85；width 0.90 已完成 1.20M 同源
water replay 和 250/300 完整 256k paired 验证，但尚未完成新的三能量完整
10M 复验。

## 6. 性能和审计

RTX 2080 Ti、FP32 dose：

| Workload | Histories | Elapsed | Throughput |
|---|---:|---:|---:|
| 250 MeV/u water replay, width 0.90 | 1,202,572 | 27.35 s | 43,970 histories/s |
| 250 MeV/u complete Copper+water | 256,000 | 9.41 s | 27,205 histories/s |
| 300 MeV/u complete Copper+water | 256,000 | 10.82 s | 23,652 histories/s |

对应运行满足：

- Unified-EM missing-domain count = 0；
- queue overflow = 0；
- energy balance residual < 1.8e-5；
- Bragg depth、总能量和 dose outputs 有限；
- FP32 dose scorer，未使用 FP64 fallback。

TOPAS 1,202,572-history replay 用时 29:51、峰值 RSS 4.05 GB；五深度
phase-space 诊断用时 32:35、峰值 RSS 4.51 GB。

## 7. 主要代码和数据改动

- `include/carbon/minibeam_collimator.hpp`：矩形 collimator/slit 几何；
- `include/carbon/transport_config.hpp`、`src/config.cpp`：minibeam 参数、校验和
  optional diagnostics；
- `src/transport_sycl.cpp`：Copper primary/fragment 输运、INCL++、elastic bank、
  水入口相空间、fragment audit 和 water MCS tail；
- `src/tps_source.cpp`、`src/plan_run.cpp`：显式 source pose/direction replay；
- `src/io.cpp`：phase-space CSV 和 primary fluence MHD；
- `data/ion_*_copper_geant4_11_3_2.csv`：Copper 多离子 stopping/rate 表；
- `benchmark/carbonminibeam/`：TOPAS configs、Slurm scripts、提取/比较/绘图脚本；
- `config/beam_minibeam_*`：smoke、full field、multi-energy 和 replay 配置；
- `failed.md`：已否决模型和允许重试条件。

## 8. 已否决路线

详细数值见 `failed.md`，主要包括：

- 单纯缩短 Copper step；
- Copper 段末一次性 Highland；
- 高频窄尾或 width 1.0 的过强 tail；
- 将 250/300 的 tail 无条件用于 150；
- delta aggregate 固定横向 Gaussian relocation；
- 单一位置条件 Copper 能损修正；
- 最后 10 mm 的低能 Highland recovery；
- 每步 Fermi–Eyges 位移叠加；
- 标准 Highland 全程方差差分；
- 单独精确截断 slit 侧壁。

这些路线不得原样重试，除非满足 `failed.md` 中记录的新证据条件。

## 9. 当前限制和后续工作

1. 当前仍是 research/non-production 模式。
2. Copper neutral products 主要作为 beamline energy sink，尚未建立完整中子/光子
   下游剂量模型。
3. Copper survivor energy-loss scale 和 water tail 是用独立相空间/剂量数据冻结的
   residual correction，不应外推到未验证材料和几何。
4. 需要补做 width 0.90 的 150/250/300 完整 10M 最终验收；150 预期保持不变。
5. 入口 valley 的主要剩余误差来自 Copper 联合位置—能量—角度分布；水中
   250 MeV/u Bragg PVDR 已由 0.946 改善到 0.973。
6. minibeam validation 配置引用本机 `/mnt/sda/wuwei` 下编译后的 Copper INCL++/
   elastic assets；仓库包含可复现的 TOPAS extraction/compile scripts，但不提交这些
   大体积运行产物。

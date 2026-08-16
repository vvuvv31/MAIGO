# Problem：机架角数字与当前照射逻辑不一致

新的 OpenTOPAS PencilBeamScanning / IEC 机架旋转，**不能**直接拿来替换本仓库已经对上的 CT 照射方式。对当前 case（尤其是 RT06423 这类 head、计划角写成 TPS 90° 的），GPU 蒙卡必须继续走**和现有 TOPAS 脚本相同的逻辑**，不要改成「固定 CT、按 IEC 数字转机架」。

## 问题是什么

真实 TPS：患者固定，机架角 θ 转束流。

本仓库已验证的 TOPAS 做法不是转机架，而是：

```text
固定 TPS 0° 束流基准（world 中从 y− 射向 y+）
+ Patient/RotZ = 计划角数字
```

也就是**转 CT**，不是转束流。RT06423 全计划脚本里是：

```text
d:Ge/Patient/RotZ = 90. deg
# 束流挂在 world / IEC_G，RotZ = 0
```

PBS extension 里若把 `IEC_G/RotZ` 设成 90 或 270，用的是 OpenTOPAS 的 IEC 约定：world = Rz(−RotZ) · local。在这个约定下：

| 脚本里的 `IEC_G/RotZ` | 束流在 world 里的主方向 | 和本 case「TPS 90°」对照 |
|---|---|---|
| 0 | +Y（y− → y+） | 与现脚本束流基准一致 |
| 90 | +X | 对应用户原先表里的 **270°** 照射 |
| 270 | −X | 对应用户原先表里的 **90°** 照射 |

所以：**数字 90 在 IEC 里不是本 case 口头说的 TPS 90°。**  
已经撤回过「把 90 改成真 90」的改动；比较剂量时不要再改这个符号。

## 本 case 必须保持的照射逻辑

GPU 侧不要切到任意角 TPS source / IEC 机架，除非单独做新验证。当前应对齐的是已闭环的 `tps_90` 路径：

1. CT 先按 `reorient_ct_grid_tps_90.py` 排成 **patient ±X 为深度**（RT06423 用 `patient_ct_tps_90_xneg.bin`）。
2. 束流仍按 TOPAS 的 **TPS 0° 基准**（源在 −SAD，主方向 +Y），再经 `Patient/RotZ` 进患者系。
3. 保留每个 spot 的扫描位移和 L7/L8 微倾角，不要压成平行束。
4. GPU 配置与 TOPAS 同一套摆位，例如 RT06423：

```text
spots_geometry_mode: tps_90
spots_patient_rot_z_deg: 90.0
spots_sad_mm: 450.0
ct_grid_file: benchmark/ct/RT06423/grid/patient_ct_tps_90_xneg.bin
```

对应 TOPAS：`Ge/Patient/RotZ = 90. deg`，束流 `IEC_G` / BeamPosition 的机架旋转保持 0。新编进集群的 PBS 全计划也是这样写的（`fullplan_tps/RT06423_s*`）。

## 不要做的事

- 不要为了「IEC 90 就是 90」去改 GPU 的 `tps_90` / `spots_patient_rot_z_deg`。
- 不要把本 case 改成固定原始 CT、只转 `tps_gantry_angle_deg`，除非另开一条验证、并先对上同一套 TOPAS。
- 不要用 PBS 的 `IEC_G/RotZ = 90` 去模拟本 case 的 TPS 90°；那样束流会走到 +X，和现有 RotZ=90 转 CT 的照射不是一回事。

## 相关入口

| 位置 | 内容 |
|---|---|
| `docs/ctplan.md` | 转 CT vs 转束流的约定 |
| `config/beam_ct_fullplan_rt06423_let.yaml` | 已验证 GPU 几何 |
| `benchmark/ct/RT06423/fullplan_mc/run_full_plan.txt` | 原 Time Feature TOPAS |
| `benchmark/ct/RT06423/fullplan_tps/` | 同几何的 PBS 脚本（机架仍为 0） |
| `src/topas_spots.cpp` | `tps_90` world→patient（含 RotZ） |

任意机架角的统一 GPU TPS source（`src/tps_source.cpp`）是另一条 opt-in 路径，不是本 case 对 TOPAS 的比较基准。

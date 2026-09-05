# Planning：spots、CT 几何与可复现运行

更新日期：2026-09-05。合并原 ctplan 和 geometry_rotation 的有效内容。
旧 DIJ / Time Feature 结果及历史建议保存在 [归档](archive/README.md)；
本文件不是任务进度表，也不取代 [plan](../plan/README.md) / [plan2](../plan2/README.md)。

## 1. 冻结输入，先区分参考类型

每次比较先固定 CT/DICOM/Schneider 文件、spots、beam model、TOPAS physics list、
患者 placement、scorer 网格、实际逐 spot histories、executable 与 physics 输入 SHA。

参考可能是逐 spot DIJ×优化权重，也可能是完整 TOPAS DoseToMedium Sum replicas。
两者 histories/归一口径不同。当前三病例使用后者，不再通过 LS scale 拟合 GPU 剂量。
旧 Hamilton/MU 转换的研究流程不是当前 histories 模式的默认算法。

TOPAS Time Feature 旧输入中 L2 常为 C12 总 KE，L3 为能散，L4 为 histories，
L5/L6 为源横向位置，L7/L8 为带扫描倾角的旋转，L9–L14 为 emittance。
当前 TPS CSV/beam-model 走 [tps_source.cpp](../src/tps_source.cpp)；
不要跨格式直接复制角度或把总 MeV 当 MeV/u。

## 2. 坐标约定

TOPAS 组件旋转采用被动约定。对于这里的 Patient RotZ：

```text
patient → world:  world = R(−RotZ) × patient + Trans
world → patient: patient = R(+RotZ) × (world − Trans)
```

点先减平移再逆旋转；方向仅旋转，不能加减平移。
见 [topas_spots.cpp](../src/topas_spots.cpp) 的
`transform_tps_90_pose_to_ct`。患者绝对 DICOM 坐标还涉及体积中心和 CT origin，
不能把上述组件局部坐标直接当作 DICOM LPS。

历史匹配基准使用 world 从 −Y 向 +Y 入射，加 Patient RotZ 转 CT。
它不等于“固定患者、给 IEC gantry 填同一个角度数字”。
保持 TOPAS 实际 placement 和 source basis；不要为追求数字上的 90° 修改旋转符号。

三种事情必须分开：

| 层 | 决定内容 |
|---|---|
| source | beam model、emittance、能散、扫描磁铁与方向 |
| placement | world/patient 的旋转和平移 |
| CT packing / output mapping | 数组轴置换、翻转、origin、spacing，不改物理 |

旧 `tps_90` / `tps_gantry_y` 是特定 packing/源路径；
`tpsSource` 可以与相应 CT packing 和显式 TOPAS placement 联合使用。
不能从 CT 文件名推断实际 source 路由。

## 3. 最新三病例的实际路径

2026-09-05 冻结配置启用 `tpsSource`、`enable_tps_coordinate_system`，
使用 `tps_angle_convention: dicom_lps`、CSV histories 模式，
并按病例配置 `tps_apply_topas_patient_placement`。
RT06423 虽仍用 packed `patient_ct_tps_90_xneg.bin`，源已不是仅靠旧
`spots_geometry_mode: tps_90` 的说明就能复现。
完整字段以 [冻结配置索引](../benchmark/topas10x/gpu_current_20260905.md) 为准。

- RT06423 / RT07575：患者几何映射后恢复到 native patient 剂量，再比较。
- RT07575 的 417 × 0.5 mm 半范围是 104.25 mm，不按 Gamma 拟合轴偏移。
- 20022516：CT 与源一起 +1 mm 重标记以满足原有 delta-tail origin 门禁，
  相对几何/payload 不变；比较输出已回到原患者坐标，不再次平移。

保留 spot 的位置–角度相关性、虚拟扫描磁铁距离、SAD 和 upstream air 能损。
束宽/发散/扫描角不是同一个量；不能把带扫描倾角的 PBS 简化成平行 pencil 后称同源。

## 4. Histories 与分片

当前目标是匹配每个 TOPAS replica 的逐 spot 整数 histories，而不是只匹配总权重。

1. 读取每个 replica 的权重、scale、实际整数化方式与过滤规则。
2. 对每个 spot 先得到参考整数 histories，再分配到 GPU shards。
3. 校验逐 spot 的 shard 之和等于参考总数；同时核对全计划总数。
4. 同一 shard 不重复计入；恢复已完成片必须验证输入/剂量/quality 哈希。
5. dose 为累计 Gy 时在相同网格直接求和；不要再乘“20 倍”。
6. 任一 overflow 或 accepted=false 不进入正式聚合；缩小片重跑，保留失败记录。

整数化误差使实际总数与简单 weight×20 略不同，不能拿理论小数替代日志计数。
过滤后的 FirstSpot 与原 CSV 下标也不一定相同；改变过滤条件必须重核对。

## 5. 新病例/新几何的最小检查

1. 核对 DimSize、spacing、origin、旋转、剂量/CT 数组顺序及 HU/density/section 对齐。
2. 比较中心与离轴 spots 在两个真空面上的位置、方向、能量、宽度和相关系数。
3. 检查三轴基向量、中心射线、等中心与入口；不能用剂量配准掩盖几何错误。
4. 先做小规模 CT 运行，检查质量、能量、lookup 和队列；通过后才扩大。
5. 冻结 mapping 和 Gamma 方法，不为更高分数更换阈值、mask 或平移。
6. source/CT 同步坐标重标记必须有 payload 和相对几何恒等审计。
7. GPU 仅本地 RTX 2080 Ti；TOPAS 本地 Slurm，总预算按 AGENTS。

入口工具：
[run_topas10x_gpu_benchmark.py](../tools/run_topas10x_gpu_benchmark.py)、
[prepare_topas10x_ct.cpp](../tools/prepare_topas10x_ct.cpp)、
[evaluate_topas10x_gpu_gamma.py](../tools/evaluate_topas10x_gpu_gamma.py)、
[dump_tps_phase_space.cpp](../tools/dump_tps_phase_space.cpp)。
执行前查看各工具 `--help`，创建新输出目录；本文不发起任何运行。

## 6. IDD、profile 与 Gamma

数组重排完成后，依据患者物理轴做横向求和得到 IDD；
过等中心 profile 是 3D 场中的一条线，不是 IDD。
MHD Offset 是体素中心，体积中心为 `Offset+(DimSize−1)×Spacing/2`。
当前 profile 对固定的另两轴等中心坐标做三线性插值，不吸附到最近体素。
详见 [scoring](scoring_validation.md) 和 [results](results.md)。

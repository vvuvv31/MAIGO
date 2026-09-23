# 物理体素边界修复与 TOPAS/GPU 比较

本轮确认的边界衔接问题已修复并通过运行测试，**总剂量验收仍为 FAIL**。在这组六个深度中心，全链 FWHM 差的绝对值平均由约 2.10% 降至 1.13%，这是点估计改善，不是统计验收结论；100 mm 谷区仍高约 14.81%。水模 120 mm 的步长响应仍约 -3.86%，边界错配不能解释这一残差。

下一步应继续验证 Urban 的轨迹/步进移植等效性，但没有依据调散射尺度或尾部系数。优先比较同入口 C12 的角度、位置二阶矩及两者相关，再用 TOPAS 电子/非电子剂量分解区分轨迹误差与局部电子沉积近似。全链还需要锁定相同水入口记录，以隔离 Cu/空气段的角度—能量尾；当前不能将谷区偏高全部归因于水 Urban。

修复位置：`/mnt/sdb/wuwei/MAIGO_review_fix_20260923`。原始 `MAIGO_pristine` 工作树未修改；没有 commit/push。

## 改动与适用范围

这次修复 primary C12 water Urban 与物理计分体素的衔接：用当前体素计算 pre/post safety，按真实过面更新 boundary 状态；内部面按出射方向确定归属，接受过面后只将相应坐标放到共享面上，取消该路径原有的位置 nudge；横向和入口反向逃逸正常记账，内部导航失败报告 fatal reason 11；不完整的末端深度体素明确拒绝。

启用条件是 research primary water Urban、`voxel_scorer_clamps_transport: true`、体素计分、均匀水模。CT、异质插入、分层、次级粒子与 production specialization 不在此修复范围内。全局开关默认值没有改变。没有调整散射系数、tail、stopping scale 或剂量归一化，保留 Water_75eV 和 GPU FP32 dose/atomics。

这些边界来自 TOPAS v4.2.3 `TsBox::ConstructVoxelStructure()` 的物理 replica，并非只为减小剂量差而添加。该修改对齐了已确认不一致的几何上下文，但不能据此宣称整个 GPU/Geant4 输运完全等效。

## 水模比较：0.05 mm 步长上限

同一批 20,000 条 250 MeV/u C12 入口记录、0.5 mm 均匀 beamlet，两引擎各三个独立种子，无 Cu；0.1 × 100 × 0.25 mm 体素。按源粒子数和真实体素质量换算，未拟合归一化。

- 总沉积能量 GPU/TOPAS 差：**+0.003%**。
- 积分深度剂量 R80 差：**-0.002141 mm**。这是 0.25 mm 分箱数据的插值指标，不代表微米级物理精度。
- 冻结剂量验收：**FAIL**，退出码 1；具体条件见 `water050_dose_gate.json`。

| 深度中心 mm | 修复前 FWHM 差 | 修复后 FWHM 差 | 修复后峰区差 | 修复后谷区差 |
|---:|---:|---:|---:|---:|
| 19.875 | -0.623% | -0.357% | +1.964% | -100.000% |
| 39.875 | -0.806% | -0.711% | +1.591% | +6.829% |
| 59.875 | +0.167% | +0.636% | +0.427% | -100.000% |
| 79.875 | -0.760% | +1.694% | -0.230% | -20.092% |
| 99.875 | +1.464% | +0.168% | -0.670% | +24.754% |
| 119.875 | +1.587% | +2.046% | -2.454% | +5.640% |

![水模比较](water050_comparison.png)

## 两引擎相同步长变化

TOPAS 0.025 mm 对照已补到 20,000 histories × 3 seeds；不能继续使用此前“TOPAS 只有 2,000 histories”的判断。以下是深度 119.875 mm 的 FWHM 在步长上限 0.05 → 0.025 mm 时的变化：

| 引擎/版本 | FWHM 变化 |
|---|---:|
| TOPAS | -0.189% |
| GPU 修复前 | -3.857% |
| GPU 物理体素修复后 | -3.859% |

0.025 mm 的 GPU/TOPAS 剂量门槛为 **FAIL**。完整各深度均值、三个种子的数值、run SE 和步长对照保存在 `comparison.json`；三种子 SE 本身仍有不确定性。

先前固定能量的 `sqrt(t/X0) * (a+b*ln(t/X0))` 核心近似不能替代此实测步长响应，也不能单独证明散射公式需要调整。

![步长响应](step_response.png)

## Cu → 空气 → 水全链

GPU 250,000 histories × 3 seeds，对照现有 TOPAS 10,000,000 histories × 1 seed。两引擎源分布配置匹配，但没有共用同一批水入口记录。TOPAS 缺少独立重复，**只能作探索性比较，不能宣告统计验收通过**。

- 总水中沉积能量差：**-0.109%**。
- 积分深度剂量 R80 差：**+0.010266 mm**。

| 深度中心 mm | 修复前 FWHM 差 | 修复后 FWHM 差 | 修复后峰区差 | 修复后谷区差 |
|---:|---:|---:|---:|---:|
| 19.875 | +0.436% | +0.220% | -1.435% | +3.883% |
| 39.875 | -4.117% | -1.538% | -0.416% | +4.144% |
| 59.875 | +1.799% | -0.013% | -1.237% | +5.022% |
| 79.875 | +1.226% | +1.430% | -1.038% | +5.212% |
| 99.875 | -2.312% | -1.828% | -0.686% | +14.806% |
| 119.875 | -2.724% | -1.754% | -1.249% | +5.719% |

![全链比较](fullchain_comparison.png)

## 验证和证据

- `carbon_mc`、`urban_localize` 构建通过；localize 包括所有 999 个横向面的双向归属、过面、反向、角点、安全距离和逃逸测试。
- 最终 11 个 GPU 运行的 quality 均 accepted，Urban fatal/cap/guard/subulp 均为零；完整摘要 `final_run_quality.json`。
- 导航运行中部分初始边缘记录先被上游 absorbing geometry 吸收，因此不能把全部 200 条记录声称为完成了水中逃逸。进入水的斜入射记录用于检查横向逃逸；低能测试覆盖终止和射程边界。
- 最大绝对能量账本相对残差：1.62e-05。
- 首次 final localize 因未创建 CSV 输出目录失败，修正测试脚本后重跑通过；保留失败日志，未将其混入最终通过结果。
- 150.1 mm 水模配 0.25 mm 深度体素的无效配置已做实际运行检查，按预期以非零退出码明确拒绝（`partial_grid_rejection.json`）。
- `boundary_incremental.patch` 是本轮相对前一阶段源码快照的增量；`tracked_changes.patch` 包含工作树相对 HEAD 的全部改动，含原有 draft，不能全部归为本轮修改。

复现示例：

```bash
cd /mnt/sdb/wuwei/MAIGO_review_fix_20260923
export LD_LIBRARY_PATH=/home/wuwei/sycl_workspace/llvm/build/install/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
build/oneapi-nvidia-minibeam/carbon_mc --config config/boundary_final_water_s1.yaml
```

上述命令会使用同名输出目录；复验时应复制配置并改成新文件名，保留已有证据。`run_final.py` 会拒绝覆盖已存在的输出。

# MAIGO 修复后 TOPAS / GPU 比较

已修复明确的实现与验收错误，并完成水模和 Cu→水全链对照。**目前仍有剂量指标未通过，不能宣布 Urban 全面验收通过。**

代码、配置和原始运行证据位于远程 `wuwei@100.125.0.11:/mnt/sdb/wuwei/MAIGO_review_fix_20260923`。这是独立 worktree；原仓库 `/mnt/sdb/wuwei/MAIGO_pristine` 的正在进行中的 coding 工作没有被覆盖，也没有 commit/push。基于 `d51e599e8bf5c566a8e6e96580e7c000e1745e41` 加当时未提交草稿；补丁包含这些继承修改及本次修复。

## 直接看结果

| 对照 | 水中总沉积能量差 GPU/TOPAS−1 | 积分深度剂量 R80 差 | 20–120 mm 横向 FWHM 差 |
|---|---:|---:|---:|
| 0.5 mm 束，无 Cu；双方各 3×20,000 histories | +0.002385% | −0.002230 mm | −0.81%～+1.59% |
| 原始 Cu→水全链；GPU 3×250,000，TOPAS 1×10,000,000 | −0.108644% | +0.006887 mm | −4.12%～+1.80% |

所有剂量按源粒子数归一化，没有拟合归一化、stopping/MCS scale 或尾部参数。R80 用 0.25 mm 深度网格插值，以上小差值不代表达到微米级绝对精度。全链 TOPAS 仅一个独立种子，无法估计其 run SE，所以全链结果只用于诊断。

![Cu→水全链比较](topas_gpu_fullchain_comparison.png)

![同入口水模比较](topas_gpu_water_comparison.png)

全链：C12 250 MeV/u，原始发射度、1.2% 能散、0.5 mm 狭缝、3.6 mm pitch、60 mm Cu、原始空气间隙；仅 EM，Water_75eV，水中 Urban 最大步长 0.05 mm。源分布配置一致，随机粒子独立抽样。水模：双方读取相同的 20,000 条显式入口记录，无 Cu、无初始角发散或能散，水深 150 mm。

TOPAS 参考为 `/mnt/sda/wuwei/minibeam_single_center_em_only_water75ev_e250_10m/topas_7378/`；日志确认恰好 10,000,000 histories，Water_75eV、production cut 0.05 mm。没有使用名称相近但实际 cut 为 0.5 mm 的参考。

## 未通过的部分

水模正式 dose gate 返回 **FAIL，exit 1**。各深度横向积分和 FWHM 通过；峰区在 20、40、100、120 mm 的 `|相对差|+2SE` 超过 2% 门槛。80–120 mm 谷区/PVDR 未通过；浅层存在零谷剂量，PVDR 为 INCONCLUSIVE。完整逐项状态见 [water_dose_gate.json](water_dose_gate.json)。

全链峰区均值差 −0.39%～−2.10%，谷区 +3.69%～+14.23%，PVDR −3.93%～−13.96%。这些是点估计，缺少 TOPAS 独立参考 SE；不把单参考结果标作 PASS。

| 水中深度中心 mm | 峰区差 | 谷区差 | PVDR 差 | FWHM 差 |
|---:|---:|---:|---:|---:|
| 19.875 | −1.18% | +3.84% | −4.61% | +0.44% |
| 39.875 | −0.39% | +3.69% | −3.93% | −4.12% |
| 59.875 | −1.71% | +5.55% | −6.29% | +1.80% |
| 79.875 | −1.54% | +5.80% | −6.72% | +1.23% |
| 99.875 | −2.10% | +14.23% | −13.96% | −2.31% |
| 119.875 | −0.53% | +4.00% | −4.14% | −2.72% |

数值表：[水模 CSV](comparison.csv)、[全链 CSV](fullchain_comparison.csv)；含各次结果与 GPU SE 的 [全链 JSON](fullchain_comparison.json)。

## 修复内容

1. **真实 Geant4 oracle 上下文**：原草稿虽然已绑定 `ionIoni`，但在 `BeamOn` 结束后才查询表，丢失实际 C12 有效电荷状态。3000 MeV、Water_75eV/cut 0.05 mm 的 restricted range 错到约 5055.7 mm；真实活动 step 上约 140.44 mm。两者与实际约 127 mm 的总射程不能混用。新提取器在真实 C12 第一条 step 内记录 range、restricted dEdx、inverse energy，再结束该探针轨迹；覆盖三种材料/cut 上下文共 120,003 条记录。生产读取器拒绝旧状态标记，改用 `VALID_ACTIVE_C12_STEP_CONTEXT`。同时将 10 个现有 primary/Cu Urban 实验配置指向新表；secondary-only replay 未纳入本次输运修复。
2. **一次 Urban 采样对应一次实际输运**：缓存采样结果，能损使用接受的 true path，位置使用 geom path；下一步使用随机能损后的实际能量。去掉宏步预采样后再次采样的执行方式。失败在能损/计分之前报告；range-limited 步按 Geant4 顺序消耗剩余能量后终止，不再对这部分重复做涨落。相空间平面能量使用该步实际能损插值。
3. **稳定数学与上下文检查**：沿用并验证原草稿已加入的稳定 inverse 分支，使用修正后的活动上下文表完成长双精度参考验证。没有把继承的数学修复归为新写实现。
4. **验收脚本**：修复稀疏 survivor ID 的重复索引、把能量列误当权重、全体积失败未汇总到退出码、零谷区/零 GPU 比值和未包围半高交点等问题；检查独立 GPU seed。
5. **恢复完整性校验**：CMake 原先强制关闭文件 SHA 校验；现在 OFF 默认且本次构建实际启用校验。GPU dose/atomics 保持 FP32，FP64 OFF。

## 已执行验证与限制

- `carbon_mc` / `urban_localize` 构建成功；`urban_localize: ALL PASS`，`git diff --check` 通过。R2 inverse 最坏相对误差约 7.57e−8（门槛 2e−6）。
- 独立 771 条活动 C12 step 网格验证：range 插值误差 ≤1.24e−7，dEdx ≤7.01e−8，inverse ≤1.96e−7。最终提取器重新编译并运行，四个输出与用于模拟的数据 SHA256 完全一致。
- 回归测试覆盖 survivor/权重/能量聚合；仅全体积多沉积 10% 的夹具正确 FAIL/exit 1（该聚合测试模拟文件加载，但使用真实指标与总状态逻辑）。实际 CLI 对缺少 TOPAS 独立种子、重复 GPU 种子、错误 header hash 都返回 INCONCLUSIVE/exit 2。
- 水模、全链、低能终止和步长检查均正常退出，Urban fatal/cap/guard/subulp 为 0。低能测试为 200 条 0.1001–4800 MeV C12，包含停止与穿出水模，能量账本残差约 1.32e−7；全链残差约 1.60e−5。quality 的 accepted 表示账本/运行检查通过，不等于剂量精度通过。
- 步长 0.05→0.025 mm 的最初 2,000 histories 检查噪声大，因此补跑 GPU 各 3×20,000。GPU R80 变化 −0.0061 mm，FWHM 点估计变化 −0.03%～−3.86%，100/120 mm 仍有可见敏感性。不能据此声称横向完全收敛；TOPAS 0.025 mm 只有 2,000 histories 的快速检查，尚无同统计量、多种子收敛证据。见 [step_highstat.json](step_highstat.json)。
- 全链 GPU 每 250,000 histories 实际约 15.4 s（kernel 约 12.2 s）；复用的 TOPAS 10M 参考约 5199 s、128 CPU threads。样本量和硬件不同，这不是同条件速度基准。

## 下一步方向

优先用同一批 Cu 出口/水入口 phase-space 做配对 replay，分开定位 Cu 产生的角度尾与水中传播；对尾区角度、位移、能量联合统计及真实权重作比较。核对 GPU 局部 δ 电子能损沉积与 TOPAS 显式电子传播对峰谷的影响，这是待验证的原因，不能先认定为唯一原因。随后补齐 0.025 mm 的 TOPAS 独立种子与步长收敛统计。暂不继续调 stopping/MCS scale、尾部参数或归一化，也不提升 Urban 为默认模型。

## 复现与交付

在远程修复 worktree 中，已编译程序可直接运行：

```bash
cd /mnt/sdb/wuwei/MAIGO_review_fix_20260923
export LD_LIBRARY_PATH=/home/wuwei/sycl_workspace/llvm/build/install/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
build/oneapi-nvidia-minibeam/carbon_mc --config config/review_fullchain_s1.yaml
```

完整证据目录为 `evidence/review_fix_20260923/`。`maigo_compare_results.py` / `maigo_analyze_followup.py` 重算图表；`maigo_fix_tests.py` 执行回归测试。`tracked_changes.patch` 是相对基准 HEAD 的全部 tracked 修改，含继承草稿；应用到仍在修改的原仓库前须逐文件合并。`review_fix_source_bundle.tar.gz` 包含修改源码、配置、新表、提取器、脚本与关键结果，不包含外部物理数据包或编译器。`artifact_manifest.json` 固定源码、程序、输入与输出 SHA256。

# 独立电子响应的三个包

当前独立 CT minibeam 电子响应按 projectile birth 与 electron/photon propagation 拆分。文件内容、SHA、抽样方法和物理模型不变。

| 包目录（`data/ct_electron/` 下） | 内容 | 磁盘大小 |
|---|---|---:|
| `proton_independent_v1` | Proton 电离电子产额、出生能量—方向联合分布 | 328,367,136 bytes（0.328 GB） |
| `carbon_independent_v1` | C12 专属出生谱及提取来源记录 | 990,170,470 bytes（0.990 GB） |
| `shared_independent_v1` | Schneider joint、charged、photon 传播数据 | 3,809,155,986 bytes（3.809 GB） |

大小按十进制 GB，包含清单中数据及来源记录，不含各包自身的 `manifest.json`。三个包唯一文件合计约 5.128 GB。Carbon birth 二进制本身为 990,004,008 bytes。

```text
data/ct_electron/
  proton_independent_v1/
    birth_bank.bin
    manifest.json
  carbon_independent_v1/
    birth_bank.bin
    extraction_provenance.json
    source_identity_check.json
    manifest.json
  shared_independent_v1/
    schneider_joint.bin
    charged/index.txt + node*/operator.bin
    photon/index.txt  + node*/operator.bin
    manifest.json
```

Proton 运行只需要 proton + shared；carbon 独立试验只需要 carbon + shared。两个专属包分别记录共享包 manifest SHA，因此以后更换共享包时必须明确更新依赖和重新验证。包内索引使用相对路径，无 `out/` 运行时依赖；carbon provenance 中保留的原提取绝对路径仅用于溯源。

旧 `proton_independent_v1/{schneider_joint.bin,charged,photon}` 现在是指向 shared 的兼容链接，不占第二份数据空间，历史冻结配置仍可运行。打包分发应按 manifest 的成员清单收集文件，避免跟随兼容链接重复复制 shared。

## 配置和检查

默认 `config/proton_ct_minibeam.yaml` 已引用 proton birth 和 shared 的三个传播组件。Carbon opt-in 配置为 `config/carbon_soft_minibeam_independent_trial.yaml`，使用 `inputs/carbon_soft_tissue/ct.bin` 均匀软组织试验体模（独立的场景输入，不是第四个电子响应包）。其密度源于 RT07575 section7 中位数；不是完整患者 CT。需要包含 C12 独立出生包入口的当前源代码构建；本次已编译的候选位于 `out/carbon_independent_trial_20261004/build/carbon_mc`，正式 proton 二进制未替换。

```bash
python3 scripts/check_electron_packages.py
```

检查三个包的全部 171 个文件、manifest SHA、粒子 Z/A、共享依赖、charged/photon 索引引用及大小。所有运行时组件继续使用 C++ 加载器已有的 SHA 和覆盖范围检查；manifest 用于安装完整性与依赖管理，不替代物理审计。

规范记录在 `docs/electron_packages_manifest.json`；`docs/proton_independent_runtime_manifest.json` 同步记录 proton 当前实际依赖的文件路径。

拆分后验证采用相同二进制、配置和种子，proton / carbon 各连续两批 1,000 primary，与拆分前软组织结果比较。检查 quality、primary/EM、电子/光子工作、能量闭合、零队列溢出、零旧包回退，并检查第二批缓存复用。证据位于 `out/electron_package_split_20261004/`。

验证已通过：171 个文件全部匹配；两种粒子的整数审计均与拆分前一致，四批 quality 通过，第二批均命中缓存。剂量相对 L1 最大值 proton=2.906e-11、carbon=1.434e-10，差异处于 FP32 原子累加量级。最大能量相对残差分别为 5.926e-8 / 1.056e-6；无电子覆盖缺失、无光子缺口、无队列溢出或 legacy 回退。详见 `verification.json` 和 `regression.json`。

## 适用范围

这三个包仅指本次独立电子响应，未把核反应、重离子阻止本领、铜准直器及 broadbeam 的其他数据包合并进来。既有 carbon endpoint、proton 历史响应包保留用于现有入口和对照。

Carbon 包仍是 ≤250 MeV/u 单能 research trial，63 个材料密度节点；完整 RT07575 的部分高密度材料缺失，尚未完成患者/TOPAS验收。拆包不扩大材料或能量覆盖，也不改变核碎片电子损失仍局部处理的现有范围。见 [Carbon 软组织初测](carbon_soft_independent_20261004.md)。

本次减少的是重复存储和依赖混淆，不是输运优化。一个 proton 或 carbon 进程仍须加载其专属出生包和完整 shared 包；不同 GPU 进程不会因为磁盘共用包而自动共用显存。目前同一进程切换 projectile 的 FP32 retained bank 仍以整个 birth + shared 组合为缓存键，不能声称跨粒子切换时 shared 只上传一次。

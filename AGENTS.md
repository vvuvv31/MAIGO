- 当前仓库的主要目标是实现基于TOPAS提取数据复刻出来A_Data-Driven_Fragmentation_Model_for_Carbon_Therapy_GPU-Accelerated_Monte-Carlo_Dose_Recalculation文章中的GPU加速
- gpu需要在沙盒外才能看到，始终允许在沙盒外运行GPU蒙卡code，gpu版本为sm_75，GPU型号为 RTX 2080Ti
- sycl相关的工具也需要在沙盒外调用，始终允许
- Never submit or run GPU jobs on a remote host or cluster.
- 不要尝试使用1D dose scorer，使用3D dose scorer然后对横向求和来代替1D scorer
- 不要考虑FP32的问题，FP64相比较FP32只能提高不到1%的精度
- 当粒子数太多的时候需要拆分成多个任务，例如一次只跑1/10粒子数，然后再合并结果，防止溢出，检测到次级粒子overflow就需要拆分重跑

- ct不要走四分类包，只走 Schneider 分区。

## Git branch workflow

- 当前开发、文档和新提交的目标分支为 `master`，远端为 `origin/master`。
- 原 `fred` 开发线迁移为 `master`；不要再向 `origin/fred` 提交或推送。
- `legacy` 保留迁移前的 `master`（`8716e7c975e5255a747a46286f3b2581d645bf18`），
  仅供历史查询，不作为新功能或当前物理数据的默认分支。
- docs/archive 中的旧分支名称、历史 URL 和原始记录保留原样，不是当前执行指令。
- 仅在用户明确要求时 commit / push；默认分支约定不等于自动推送授权。
- 提交时区分已完成改动与未验收候选，不把无关工作树修改、大数据包或 scratch 自动加入提交。

## Schneider CT minimum validated physics-data stack

- Schneider CT production/research runs must use at least the currently validated
  `schneider_physics_bundle_v2_1` stack. Never downgrade, alias, or silently fall
  back to an older package, rate table, schema, projectile registry, or water/
  four-class data path.
- The current minimum accepted files are:
  - primary rate: `data/schneider/schneider_inelastic_rates_v2_1.bin`
    (`SCHNRATE` v3), SHA256
    `086ef97dbf323dc2681d5f6c5257e78c87446050f6fb628f3b0f8b24211a8370`;
  - primary CINEL03 package: `data/schneider/cinel03_c12_targets_v2_1.bin`
    (`CINPKG04` v4), SHA256
    `a690fb06ae97946bbc167501380fbcfa465b51e61cbedfaba7d3d373655a7ea2`;
  - secondary rate: `data/schneider/secondary_inelastic_rates_v2_1.bin`
    (`SCHN2RAT` v3), SHA256
    `6aa679ee162c7a47b82da715ea1333b21b4056b6480596d8d6243a96edb67258`;
  - secondary CINEL03 package:
    `data/schneider/cinel03_secondary_targets_v2_1_14p.bin`
    (`CINPKG04` v4, 14-projectile registry), SHA256
    `a0dc4259b856f0b0e7665672a64589ea4cd64ca304cae75fca16d6de50fc8006`;
  - Schneider stopping table: `data/schneider/schneider_stopping_v1.bin`
    (`SCHNSTOP` v1), SHA256
    `9786dba071f61e660fcc5940a844e8109c4e480ccb603d7929d2fcfaae152c2f`;
  - strict-dose section-0 delta-tail table:
    `data/schneider/schneider_section0_c12_delta_tail_v1.csv`, SHA256
    `ff6140f13dcfb4c6f739d6efe56941da6af4749d311e29182601f310a1184aa5`.
- `data/schneider/schneider_physics_bundle_v2_1.json` and
  `data/schneider/v2_1_data_manifest.json` are authoritative. Before a Schneider
  CT run, execute `python3 tools/verify_schneider_v2_1_data.py`; any missing file,
  SHA/size mismatch, lower binary version, missing 14-projectile coverage, or v1
  artifact placed at a v2.1 path is a hard failure.
- A newer data package may replace this minimum only when it has explicit
  TOPAS/Geant4 provenance, passes the same or stronger manifest verification,
  host/device lookup tests, 50k closure gates, one-shard A/B Gamma gate, and a
  zero-overflow full validation. Until those gates pass, continue using the exact
  v2.1 stack above; never use a lower version as a compatibility fallback.

- topas任务需要在本地使用sbatch运行，数据放置在本地/mnt/sda/wuwei目录下，
- topas的extension放在/home/wuwei/topas目录下，如果需要重新编译，source code和build都在/home/wuwei/topas目录下
- 所有任务最多一共使用192线程，内存占用160G
- 提交多个任务的时候按照计算量分配线程/内存数量，控制任务差不多时间完成，防止低能快速跑完了，高能还需要跑很久
- 例如100MeV/u分配10个线程的话，200MeV/u分配20线程，300MeV/u分配30线程。按照剩余（192-已使用）的数量动态按比例分配
- 任务在提交的时候会出现短暂的InvalidAccount，不需要当作异常，只需要等1-3分钟后再检查即可

```
#!/bin/bash
#SBATCH --job-name=xxx       # 任务名称
#SBATCH --partition=compute        # 默认计算分区 (无需修改)
#SBATCH --nodes=1                  # 申请 1 台主机
#SBATCH --cpus-per-task=50         # 申请分配的 CPU 核心数 (根据需要设定)
#SBATCH --mem=50G                  # 申请分配的内存 (根据需要设定)
#SBATCH --output=/mnt/sda/%u/job_%j.log   # 运行日志保存到 15TB 数据盘 (%u 代表当前用户, %j 代表任务ID)
#SBATCH --error=/mnt/sda/%u/job_%j.err    # 错误日志

command
```
## File editing rules

When modifying source files:

- Prefer the native file-editing mechanism over manually constructing unified diffs.
- Do not repeatedly retry the same `git apply` strategy after a malformed patch.
- If `git apply` reports `corrupt patch`, retry at most once.
- After the second failure, switch editing strategy:
  - use a direct file edit,
  - or use a small Python/scripted replacement,
  - or regenerate the target section from the current file contents.
- Before constructing a patch, re-read the relevant section of the target file.
- For large edits, make smaller independent edits instead of one large handwritten patch.
- After editing, inspect `git diff --check` and `git diff` before proceeding.
- Never spend more than two attempts fixing patch syntax.

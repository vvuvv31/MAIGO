- 当前仓库的主要目标是实现基于TOPAS提取数据复刻出来A_Data-Driven_Fragmentation_Model_for_Carbon_Therapy_GPU-Accelerated_Monte-Carlo_Dose_Recalculation文章中的GPU加速
- gpu需要在沙盒外才能看到，始终允许在沙盒外运行GPU蒙卡code，gpu版本为sm_75，GPU型号为 RTX 2080Ti
- sycl相关的工具也需要在沙盒外调用，始终允许
- Never submit or run GPU jobs on a remote host or cluster.
- 不要尝试使用1D dose scorer，使用3D dose scorer然后对横向求和来代替1D scorer
- 不要考虑FP32的问题，FP64相比较FP32只能提高不到1%的精度
- 当粒子数太多的时候需要拆分成多个任务，例如一次只跑1/10粒子数，然后再合并结果，防止溢出，检测到次级粒子overflow就需要拆分重跑

- ct不要走四分类包，只走 Schneider 分区。

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
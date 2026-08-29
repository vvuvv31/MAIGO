- 当前仓库的主要目标是实现基于TOPAS提取数据复刻出来A_Data-Driven_Fragmentation_Model_for_Carbon_Therapy_GPU-Accelerated_Monte-Carlo_Dose_Recalculation文章中的GPU加速
- gpu需要在沙盒外才能看到，始终允许在沙盒外运行GPU蒙卡code，gpu版本为sm_75，GPU型号为 RTX 2080Ti
- GPU jobs must always run locally in WSL. Never submit or run GPU jobs on a remote host or cluster.
- 不要尝试使用1D dose scorer，使用3D dose scorer然后对横向求和来代替1D scorer

- topas任务需要在wuwei@127.0.0.1上运行，数据放置在wuwei@127.0.0.1:/mnt/sda/wuwei目录下，
- topas的extension放在/home/wuwei/topas目录下，如果需要重新编译，source code和build都在/home/wuwei/topas目录下
- 所有任务最多一共使用192线程，内存占用128G
- 使用sbatch提交任务
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

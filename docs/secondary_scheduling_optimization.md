# 次级输运调度优化（2026-09-15）

当前两矩 δ 聚合生产物理下，RT07575同一分片3,240,963原发；保持spots、种子、全部已支持次级EM、Schneider v2.1核数据、MCS和精确CT边界。独立核弹性仍关闭。本地RTX2080Ti/sm_75，生产基线commit `3ab0e4e`，二进制SHA256 `db4e7345e05bdd941e38076a8e2c51ed6ff5187d9cb31293808eaf97b02b6316`。

## 测量与结论

硬件分析见 `benchmark/ctbenchmark20260915/result/hardware_profile/`：次级约占输运kernel时间65%，代表性启动active threads/warp为14.92、occupancy约25%。这些是定位依据，不是候选的计数器复测。

- 显式nd_range固定32/64/128/256线程块：各两次，穿插三次基线，整体慢约1.1–1.8%；不采用。
- 保持原range启动，调整完整物理迭代间的续跑间隔：各两次，穿插三次基线；16步最佳。

| 续跑间隔 | Elapsed中位数 s | 次级kernel中位数 s | 吞吐变化 |
|---|---:|---:|---:|
| 原64步基线 | 33.86455 | 19.14470 | 0% |
| 16步 | 32.37058 | 17.61086 | +4.62% |
| 32步 | 32.70305 | 17.97185 | +3.55% |
| 可调候选64步 | 33.87651 | 19.17069 | -0.035% |
| 128步 | 36.95529 | 22.18195 | -8.36% |

16步次级耗时减少约8.01%，整体吞吐提高4.62%。程序Elapsed与完整进程wall分别记录：16步wall中位数34.04558s，基线35.52779s。

改变的是调度间隔，不是物理步长：完整输运迭代后保存粒子及RNG状态，稳定压紧存活索引，再续跑。首代续跑由10轮增至36轮，第二代由8轮增至28轮；压紧本身约从0.008s增至0.026s。更频繁重组存活线程与提速相符，但尚未重测候选active lanes，不能声称已通过计数器量化分歧下降。

全部22次扫描运行质量通过、零overflow。16步相对基线最大同体素剂量差为峰值的0.00029841%。这是相同网格GPU自对照，不代替患者RTSTRUCT BODY TOPAS Gamma或b1–b4验收。

连续测试GPU温度约82–84°C；线程块实验基线有升温漂移。续跑扫描使用正反顺序和穿插基线，64步候选与原程序仅差-0.035%，16步重复32.340807/32.400360s，支持收益超过本组波动。没有改变驱动、GPU时钟或风扇设置。

## 可复用实现

`CARBON_SECONDARY_SEGMENT_STEPS`为CMake参数，允许16/32/64/128，默认仍64；仅当`secondary_step_chunking: true`时使用。无实验环境变量或新物理分支。日志输出实际步数，非64步质量报告标记`secondary_step_chunking_candidate`，不继承此前64步精度验收。

固定16步隔离构建：

```bash
cmake -S . -B scratch/secondary_fixed16/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/home/wuwei/sycl_workspace/llvm/build/install/bin/icpx \
  -DCARBON_ENABLE_SYCL=ON -DCARBON_SYCL_TARGETS=nvptx64-nvidia-cuda \
  -DCARBON_CUDA_ARCH=sm_75 -DCARBON_SECONDARY_SEGMENT_STEPS=16 \
  -DBUILD_TESTING=OFF
cmake --build scratch/secondary_fixed16/build --target carbon_mc -j 8
```

脚本、逐次3D剂量、质量报告、补丁及完整数字分别在：

- `benchmark/ctbenchmark20260915/result/secondary_launch_optimization/`
- `benchmark/ctbenchmark20260915/result/secondary_segment_optimization/`
- `benchmark/ctbenchmark20260915/result/secondary_fixed16_validation/`：最终CMake固定16步构建的独立验证。

生产配置和`build/oneapi-nvidia-release/carbon_mc`保持原样；此次未commit/push。推广前应验证另外两例CT及水模，收益不可直接外推到其他病例或核弹性开启配置。后续更大的提速仍应关注按剩余射程分组和降低寄存器活跃范围。

## 最终固定16步构建验证

最终编译期参数实现另做四次配对运行（基线、16、16、基线），全部质量通过、零overflow。两次16步实际Elapsed为31.535520/31.782762s，基线32.603963/33.693114s。

按重复中位数：Elapsed 33.148539 → 31.659141s，吞吐 **+4.70%**；次级kernel 18.717986 → 17.240403s。最大同体素剂量差为基线峰值的0.00029841%。两次16步约102k histories/s，不能直接与冷GPU单次峰值混比。

固定16步二进制SHA256 `e7f46b15c9406b954bf265820b9998976905d1e61e28248eff9dc6d271c9366b`。数据包验证通过；运行质量报告确认使用16步并带candidate标记。默认64步CPU构建成功，现有`carbon_tests`提前退出：缺少`data/packages/c12_H1_95MeVu_events.bin`旧FRED库，未完成该测试；没有补入旧物理数据或修改生产数据路径。GPU验证与这一历史测试数据缺失分别记录。

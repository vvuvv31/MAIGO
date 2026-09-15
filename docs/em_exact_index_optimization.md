# 精确 EM 查表索引性能验证（2026-09-15）

本地 RTX2080Ti/sm_75，16步次级调度候选作为基线。启用已有 CARBON_EM_EXACT_INDEX：按浮点指数桶缩小二分查找范围，仍使用原数据节点、区间选择与插值，不新增物理近似。索引运行时由固定统一EM包生成，数据包未改变。

## 正确性

现有 unified_em_index GPU 测试覆盖全部记录的节点和相邻浮点值，共82,741,572次查询，逐位比较失败数0。该测试比较节点查询的factor及分段函数值；不是所有物理量的穷举证明。完整输运另做以下剂量与质量对照。

## RT07575 重复配对

3,240,963原发；顺序 baseline16 → indexed16 → indexed16 → baseline16。

| 指标 | baseline16 中位数 | indexed16 中位数 |
|---|---:|---:|
| elapsed_s | 31.713334 | 30.744806 |
| wall_s | 33.393184 | 32.372527 |
| primary_s | 10.032505 | 9.613910 |
| secondary_s | 17.011508 | 16.398833 |

Elapsed口径吞吐提升 3.15%；最大体素差/基线峰值 0.00004618%。基线自身重复最大差 0.00004973%。全部质量通过、零overflow。

计数器未重新采样；结果支持该开关有性能收益，不能据此定量声称寄存器或long_scoreboard已下降。前后基线存在温度/时钟漂移，使用穿插和重复降低影响，不承诺精确百分比。

## 其他病例：单次配对

| 病例 | 原发数 | baseline16 s | indexed16 s | 吞吐变化 | 最大差/峰值 |
|---|---:|---:|---:|---:|---:|
| RT06423 | 3214630 | 29.52543 | 28.93986 | +2.02% | 0.00004547% |
| 20022516 | 3221317 | 37.63469 | 35.77204 | +5.21% | 0.00004661% |
| water | 50000 | 4.40345 | 4.48757 | -1.87% | 0.00013807% |

单次配对的小幅变化可能包含运行波动；所有运行质量通过、零overflow，患者BODY TOPAS Gamma未重算。不要把本轮与前一轮百分比相加当作直接测得的组合收益。

## 复现

```bash
cmake --preset oneapi-nvidia-indexed -DCMAKE_CXX_COMPILER=/home/wuwei/sycl_workspace/llvm/build/install/bin/icpx
cmake --build --preset oneapi-nvidia-indexed --parallel 8
```

已验证程序为 `scratch/lookup_20260915/build/carbon_mc`。基线为 `scratch/secondary_20260915_latest/build16/carbon_mc`。脚本、结果、日志及剂量均保存在 `scratch/lookup_20260915/`。

生产默认 `CARBON_EM_EXACT_INDEX=ON`（2026-09-15，紧凑对齐续跑验收后）。关闭用预设 `oneapi-nvidia-index-off`。

紧凑续跑必须在装载后把 `tables` 重新绑到当前核的 `unified_device`（捕获对象指针不能跨 launch 使用）；`SecondaryResumeState` 按 16 字节对齐（272 字节）。264 字节未对齐且先绑 `tables` 再整体覆盖时，次级续跑会 CUDA misaligned/illegal address。

## 对齐续跑 + 装载后绑定（2026-09-15 晚）

同一 sm_75 标志，RT07575 hardware_profile 3,240,963 原发；顺序 base16 → indexed16 → indexed16 → base16。`state_bytes=272`，`exact_index` 与二进制一致。GPU `unified_em_index` 82,741,572 查询失败数 0。水 50k INDEX ON 通过。

| 指标 | base16 中位数 | indexed16 中位数 |
|---|---:|---:|
| elapsed_s | 30.548580 | 29.575662 |
| primary_s | 9.407599 | 9.657995 |
| secondary_s | 16.789649 | 15.477556 |

Elapsed 口径约 +3.19%；次级核约 +7.81%。审计 `0 1036320939 1429988519 0 0 6897079514135703 528848467657861 0`。最大体素差/峰值 0.00005684%；基线自身重复 0.00003552%。质量通过、零 overflow。日志与剂量在 `scratch/kernel_deadpath_20260915/align_*`。

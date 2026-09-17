# 最新代码的次级调度优化（2026-09-15）

本地 RTX2080Ti，sm_75，固定当前清理后的物理实现与数据。独立核弹性关闭，完整次级统一 EM、原生涨落、精确 CT 边界保持不变。只调整完整迭代之间的暂停/稳定压紧间隔，保存并恢复粒子与 RNG 状态。

RT07575 每次 3,240,963 原发，按 64→候选→候选→64 顺序比较。以下为各组两次运行的中位数，跨组不直接比较绝对耗时。

| 间隔 | 基线 Elapsed s | 候选 Elapsed s | 整体吞吐提升 | 次级时间下降 | 最大差/峰值 |
|---|---:|---:|---:|---:|---:|
| 16 | 32.61258 | 31.24376 | 4.38% | 7.95% | 0.00031262% |
| 8 | 33.14646 | 32.00378 | 3.57% | 6.70% | 0.00031262% |

推荐保留16步作为性能候选；8步没有超过16步的配对收益。全部8次质量通过且无overflow。现有采样次数有限，未记录本轮完整时钟/温度曲线；并非新的硬件计数器测量。

## 补充病例

每个病例64/16各一次，用于跨病例检查；速度数字不是重复统计。

| 病例 | 原发数 | 64步 s | 16步 s | 吞吐提升 | 最大差/峰值 |
|---|---:|---:|---:|---:|---:|
| RT06423 | 3214630 | 30.22637 | 29.24513 | 3.36% | 0.00023492% |
| 20022516 | 3221317 | 37.64243 | 37.35611 | 0.77% | 0.00021188% |
| water | 50000 | 4.60848 | 4.56715 | 0.90% | 0.00026463% |

全部补充运行要求质量通过、零overflow与有限剂量值。剂量差是GPU自对照，不能替代完整RTSTRUCT BODY TOPAS Gamma或b1–b4回归。水模与其他病例若收益不同，不以RT07575数字外推。

## 使用与范围

CMake参数 `CARBON_SECONDARY_SEGMENT_STEPS` 支持8/16/32/64/128。2026-09-15 起生产默认是 16；64 步用预设 `oneapi-nvidia-secondary64`。必须启用配置中的 `secondary_step_chunking` 才生效；非16步质量报告标记 candidate。精确 EM 查表索引默认开启，关闭用 `oneapi-nvidia-index-off`。

```bash
cmake --preset oneapi-nvidia-secondary16 -DCMAKE_CXX_COMPILER=/home/wuwei/sycl_workspace/llvm/build/install/bin/icpx
cmake --build --preset oneapi-nvidia-secondary16 --parallel 8
```

已验证的隔离程序：`scratch/secondary_20260915_latest/build16/carbon_mc`。基线：`scratch/remove_analytic_model/build/carbon_mc`。未覆盖现有生产程序，未修改数据包。

复现脚本及逐次日志/质量/剂量在 `scratch/secondary_20260915_latest/`：`compare.py`、`compare8.py`、`crosscase.py`，结果为同目录JSON。

下一步是降低次级活跃变量/寄存器需求并定位长延迟查表；本轮只证明调度频率的收益，不能宣称已经解决寄存器或访存瓶颈。

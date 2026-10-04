# 凝聚软弹性散射：空间分布筛选

本轮没有修改正式输运，没有新增可接受的吞吐收益。正式 FP32 可执行文件
`build/carbon_mc` SHA256 仍为
`63255a237a44dd4d88656ba5b8196ea1539baf7d391ddfd8f5297e0b9bd0980a`。
已有 matched resident 吞吐约100k primary/s，电子约9.75s/批。
本报告的 KS 是分布累积概率差，**不是剂量误差或 gamma 通过率**。

## 夹具与范围

- 节点12/32/57，电子能量1/10/100/400keV，平均软弹性碰撞数
  μ=0.1/0.5/1/2/4/8/16，共84条件。
- 每条件每臂131072个独立哈希种子历史，共11,010,048个/臂。
- 均匀材料、固定能量、0.05mm路径；截角为90%分位角与0.1rad的较小者。
  这里只测软弹性散射，没有硬碰撞、能损、光子、CT界面或剂量验收。
- 全部 GPU 浮点计算和测试计数 atomic 使用FP32；正式构建
  `CARBON_DOSE_FP32=ON`、`CARBON_DOSE_FP64=OFF`。
- 参考为指数自由程和逐次截断 Rutherford+Mott 散射，保留实际碰撞位置。
  先选能量行再进行拒绝采样；各行分别归一化后按软部分概率混合。
- 角度一/二阶 Legendre 矩、横向位移方差、位置方向协方差与
  compound-Poisson 理论比较（7SE加1e-4相对理论值的分量筛选）。
  另检查角度/半径分布、径向分位数、方向范数及弦长不超过路径长度。
  矩检验通过并不证明分布或剂量符合1%要求。

## 结果

| 原型 | 最大径向 KS | 最大角度 q KS | 结论 |
|---|---:|---:|---|
| 两矩 Beta 角度＋单随机铰链 | 0.209190 | 0.043724 | 否决单段替换 |
| 同角度＋两固定铰链/Gaussian bridge | 0.066620 | 0.043724 | 否决单段替换 |
| 上述方法连续4子段 | 0.009918 | 0.006996 | 仅收敛诊断，未验收 |
| 精确半段角度＋GS位移公式 | 0.027992 | 0.006042 | 少碰撞时空间分布不符 |
| GS位移＋恰好一次碰撞的精确位置 | 0.016151 | 0.006042 | 改善但仍未验收 |
| GS位移＋N≤4拼接真实半段轨迹 | 0.007172 | 0.005882 | 通过分量筛选，未作CT验收 |

最初五组均通过上述矩筛选；无无效样本、方向范数或弦长越界。
单铰链最差条件node32/100keV/μ16的径向中位数−27.87%，q99+24.31%，
q99.9+28.83%。两铰链最差条件node12/100keV/μ4的中位数+11.38%，
q99−11.05%，q99.9−17.50%。因此仅拟合矩不能保护电子横向分布。
四子段最差条件node32/400keV/μ16的中位数−0.53%，q99.9−1.87%；
这些有限统计分位差也不能当成CT剂量结论。

GS位移使用**两次精确半段散射角度**，隔离空间公式本身。
未经单事件修正时μ1的径向中位数约−10%；精确单事件位置改善此问题，
但μ2最大KS仍0.016151，最差条件的q90差−1.91%。
μ8的最大KS已为0.005585，表明后续应分别处理少次与多次碰撞。
两种GS试验各有6个μ16条件因Q1>0.02，直接使用现有逐次软散射；
这部分不能算GS覆盖率或性能收益。
首个GS构建未正确分派这6个条件，产生无效结果；已保留并登记，
修正后用原条件重跑，没有删样本或放宽适用范围。

后续新增的`gs-exact-low`分支在总碰撞数N≤4时，将第二段真实轨迹旋转到
第一段的出射方向再拼接，保留空间与角度的相关性；N>4仍使用GS空间公式。
同样84条件、每臂11,010,048历史，矩/有效性/弦长检查全部通过，最大径向
KS降至0.007172；最差KS条件node57/10keV/μ1的径向中位数差仍−4.21%，
不能把KS当成各分位数或剂量的相对误差。该分支仍逐次生成两段散射角度，
尚无吞吐改善结论。`gs_exact_low/coverage.json`检查N>4概率与Poisson律，
稀疏计数使用精确二项检验，84条件family alpha0.01无失败；不能在期望
计数仅0.01时依靠Gaussian Z值作判断。源码/二进制/原始样本指纹已保存。

## 实现和来源

研究入口为 `include/carbon/electron_condensed_elastic.hpp`，没有被正式
transport include。`scripts/check_electron_condensed_gpu.cpp`提供
`bridge`、`gs-spatial`、`gs-exact-one`三种夹具分派和1–64子段组合。
`scripts/analyze_electron_condensed.py`检查保存的逐历史输出。

GS位移公式来自官方Geant4 v11.3.2
[G4GoudsmitSaundersonMscModel.cc](https://raw.githubusercontent.com/Geant4/geant4/v11.3.2/source/processes/electromagnetic/standard/src/G4GoudsmitSaundersonMscModel.cc)
的`SampleMSC` accurate分支（约986–1116行）。源SHA256为
`8f881bbd66e83332aacad815047ff16b41313957a0bfbdb0490932f38b245104`。
此处是固定能量的小Q1空间公式移植，用截断软Mott法则自己的G2/G1，
不是完整GS模型；没有采用原生range/safety抑制横向位移的优化。
原生角度表源码和下载指纹在证据目录的`native_source/manifest.json`。

This product includes software developed by Members of the Geant4
Collaboration (http://cern.ch/geant4).
Geant4派生公式遵循[Geant4许可证](../LICENSES/Geant4.txt)。

## 复现与后续

`out/electron_condensed_20261004/`保存`prepare_cases.py`、输入cases、
各候选目录中的源码/可执行文件/原始输出/报告，以及`summary.json`指纹。
材料索引SHA256固定为
`ad86b286de05aff663667793c241826e3f46472c992d4687be02a50fc9fe2e14`。
夹具编译器为本机SYCL LLVM安装中的icpx，使用
`-std=c++20 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -DCARBON_DOSE_FP32=1 -Iinclude`。
夹具参数依次为材料index、其SHA、cases.txt、131072、输出前缀、子段数、方法。
分析命令：`python3 scripts/analyze_electron_condensed.py <候选目录>`。

本轮尚未计时：GS角度仍逐次生成，目的是隔离物理误差；不能据此声称加速。
后续优先研究低碰撞数的精确联合空间分布，以及真实compound-Poisson/GS
角度的逆CDF。只有分布先通过独立筛选，才进行能损/硬事件耦合、材料和计分
边界保护、沿弯曲轨迹沉积、光子延续与CT峰谷剂量验证。
1–2s电子阶段目标、B1/B2、B3/C/D和其余A1/A2项目均未完成。

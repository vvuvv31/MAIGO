# 第二阶段：每步一次δ合并能损抽样

隔离候选，基于 a7c382d 的均值补偿实验；原发及全部18种已支持带电离子启用。正式源码、配置、二进制未修改，未commit/push。

## 模型

- 不使用逐次δ碰撞抽样、δ时钟或δ碰撞距离限制。
- 对原生proposal rate乘接受后的谱积分，提取 M1=lambda*E[epsilon*accept]、M2=lambda*E[epsilon²*accept]；包含原spin/form-factor/magnetic veto。
- 每步长度h，μ=h*M1，v=h*M2。v是复合Poisson总能损方差，不能使用h*lambda*(E[epsilon²]−E[epsilon]²)。
- Gamma形状k=μ²/v、尺度theta=v/μ，每步抽取一次总δ能损；内部使用Marsaglia–Tsang rejection，通常需要多个uniform，并非一次RNG调用。
- 保留原restricted均值/涨落、核反应、MCS、几何、1%合并平均能损步长上限、64步续跑。
- 在同一步扣动能并局部沉积；按剩余动能截断。截断会改变理论矩，未单独计量其频率。
- Gamma只匹配前两矩，不精确复现零碰撞概率、单次转移上限和高阶谱形；这是一项近似实验。
- 原EM数据未修改。派生EMDMOMT2表206,977,416字节，SHA/原包绑定见moments_table_manifest.json。
- 两矩共31648个节点的32/64点积分复核最大相对误差1.02e-7；实际C++抽样器7组shape，每组200万次，均值/方差统计测试通过（sampler_test.log）。这不等于剂量精度验收。

## RT07575 1/20 shard：6,481,909原发

沿用均值/δOFF的两个spot权重子任务和seed，粒子数精确一致。RT07575配置不启用独立核弹性；b1–b4保留原全离子弹性。
完整耗时为两个成功进程wall之和，不含锁等待，包含加载和输出；合并另计。患者BODY Gamma尚未计算。

|方案|完整wall s|程序Elapsed s|程序histories/s|primary s|secondary s|
|---|---:|---:|---:|---:|---:|
|生产|189.821|188.032|34472.4|133.009|50.968|
|δ均值|63.978|60.560|107033.6|17.521|34.951|
|δ均值+方差|65.469|62.066|104435.7|18.712|34.960|

全部零overflow、运行质量通过，合并剂量为RT07575_delta_moments_merged.mhd/raw。运行质量通过不等于生产精度验收。

## b1–b4：每例200,000原发，共10例

固定TOPAS参考，没有改动TOPAS参数。全部零overflow，EM查表无错误。
[IDD总览](benchmarks/plots/b1_b4_idd_overview.png) · [全部图](benchmarks/plots/README.md)

|Case|均值版峰误差 %|均值+方差峰误差 %|生产峰误差 %|均值+方差R80偏移 mm|积分比|MARE %|
|---|---:|---:|---:|---:|---:|---:|
|b1_100|+6.097|+1.831|+0.107|+0.007|0.99647|0.927|
|b1_200|+15.897|+2.369|-0.050|+0.069|0.99413|1.357|
|b1_300|+18.954|+0.831|-0.228|+0.235|0.99341|1.058|
|b1_400|+40.667|-1.571|-1.493|+0.241|0.99618|0.763|
|b2_150|+1.271|+0.731|-0.077|+0.092|0.99429|1.193|
|b2_250|+0.106|-0.520|-0.584|+0.196|0.99351|0.949|
|b2_350|-0.258|-0.713|-0.457|+0.201|0.99446|0.826|
|b3_layers|-0.510|-1.012|-0.763|+0.084|0.99258|0.720|
|b4_soft_lung|+0.112|-0.528|-0.640|+0.152|0.99608|0.797|
|b4_soft_bone|-1.351|-1.647|-1.088|+0.165|0.99174|1.003|

峰误差是GPU最大IDD / TOPAS最大IDD − 1；R80按各自峰高定义，记录GPU−TOPAS；MARE取TOPAS>10%峰值层。IDD均由3D横向求和。
各图文件以b1/b2/b3/b4开头，深度为TOPAS峰深度×1.2；原IDD相对误差±5%，另存不限y轴版本，避免掩盖超出范围的误差。
b1–b3包括sigma core/halo及误差，全部case包括9个深度的linear/log横向profile。

## 结果判断

本次RT07575完整wall为65.47秒，低于100秒；程序吞吐104.4k/s，比生产约3.03倍，比均值版下降约2.4%。这是单次配对配置测量，未测重复运行波动。
100/200/300 MeV/u的零能散峰高误差降至+1.83/+2.37/+0.83%，支持补回δ能损方差能显著改善峰区的判断；Gamma谱形近似与较长EM步的影响尚未分离。
该峰高指标不能代替逐层剂量误差或患者Gamma。200 MeV/u对数横向图仍显示GPU与TOPAS的低剂量halo偏差，生产版也存在类似趋势，不能据此声称所有横向尾部一致。
候选待用户审图，未接入默认生产，患者BODY Gamma尚未计算。

## 复现

build_moments_table.py → prepare.py → cmake（Release、SYCL nvptx64、sm_75）→ test_sampler.cpp → prepare_benchmarks.py / run_split_shard.py → benchmarks/run_gpu.py → benchmarks/compare.py → summarize.py。
候选源补丁和哈希见candidate.patch、source_manifest.json；原核数据保持v2.1并在CT运行前校验。

## 提交范围与复现依赖

源码快照通过candidate_from_a7c382d.patch完整记录，restore_candidate.py从提交a7c382d恢复到隔离scratch目录；无需旧scratch源树。原prepare.py保留实验来源链。
需另外准备原physics-data与派生EMDMOMT2表（由delta_moments_review/build_moments_table.py生成），本次不提交二进制数据、可执行文件、原始剂量或患者输入。
基准准备脚本还依赖本地benchmark20260914/production_continuation的TOPAS参考、体模及配置；这些外部输入没有随本次提交打包。报告保留实际已完成测量，不能把仅有Git仓库视为已包含全部重跑数据。

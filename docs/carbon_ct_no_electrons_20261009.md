# RT07575 head carbon broadbeam：关闭电子包的吞吐和TOPAS剂量对照

## 本轮结果

关闭当前选择性电子/光子输运包后，吞吐从75508.6/s提高到105423.9/s，约+39.62%，暖批wall减少28.38%。与历史TOPAS的local1%/1mm通过率从99.874934%变为99.873794%，基本保持。本轮只建立实验配置，没有修改八个发布预设或运行时代码。

## 配置与物理范围

原`config/carbon_ct_broadbeam.yaml`：RT07575头部CT、C12 TPS broadbeam source，853个正权重spot、相同beam model/placement/weights，0.5mm ion cap、涨落、continuous26 ion EM、核反应和全部带电碎片。患者坐标网格0.5×0.5×2mm，GPU packed坐标0.5×2×0.5mm，原CCTG不重采样。

原开启模式是当前broadbeam的**选择性**electron tracking：density threshold0.1g/cm³、halo5mm、low-density/tissue survival0.5、group32。结果针对这一模式与关闭模式的对照。

关闭模式：`broadbeam_electron_tracking:false`；删除birth/joint/charged/photon四个包入口；retain bank和birth order关闭，电子survival置1。保留ion EM continuous26和delta moments。δ电子能量按既有关闭路径沿离子计分位置局部沉积；没有独立电子及其EM光子输运。核中性粒子始终为未追踪能量汇，没有改为局部剂量。没有缺包后静默补偿或新增诊断构建模式。

现有carbon dispatcher由CT共享电子stage14转到既有stage6，也去掉出生、group、入队、包初始化等工作；因而加速不能全部归于最后一个electron kernel。

实验YAML：`out/carbon_ct_no_electrons_20261009/no_electrons/config/carbon_ct_broadbeam.yaml`。同一个carbon-only release binary，SHA256 `26726a83f86eacdb90c185eb9497381736d15a2b79688c3a355657afd04339aa`，RTX2080Ti/sm75，CUDA_DISABLE_PTX_JIT=1。

## 性能

ABBA，每进程4×1M primaries，连续复用GPU context/packages。16M总执行量，每臂8M执行量、6M暖批计时；同模式两个进程重复同seed，这些重复不作为额外独立剂量统计量。

| 模式 | 暖批primary/s | 秒/百万primary |
|---|---:|---:|
| 开启当前电子包 | 75508.6 | 13.243517 |
| 关闭电子包 | 105423.9 | 9.485515 |

吞吐比1.396183，两个配对比1.374018/1.418066。GPU未锁频，温度/频率有漂移；这里只报告本机宽范围消融的观测值。

## 独立剂量统计与参考核对

另外分别执行16×1M连续批次，两臂**各16M独立histories**，同入射source allocation及每spot seed；与性能重复进程分开。whole-campaign largest remainder分配，所有batch的source CSV逐字节匹配。32个高统计batch的quality accepted，energy/coverage/completion/zero-overflow通过；16个配对的核事件、弹性计数、live ion terminal counts和transport ledger匹配。

历史TOPAS：`/mnt/sdb/wuwei/MAIGO/benchmark/ctbenchmark20260915/result/RT07575`，5个独立全计划replica共129,638,170原初粒子；原物理量为DoseToMedium Sum（文件虽名OSMK_Dtotal，并非生物加权剂量）。CT和beam model逐字节匹配，正权重spot位置、能量及patient placement/扫描磁铁参数匹配，TOPAS源中额外64条为零权重。GPU source是历史整数分片，normalized weight L1差0.003190%。这项微小固定差别同时作用于两个GPU臂。

使用历史已验证packed_xneg轴交换/翻转恢复患者坐标，没有dose-based配准。将各batch真实histories的Gy累加，再只乘一次129,638,170/16,000,000=8.102385625，匹配TOPAS入射粒子数。没有拟合剂量比例，没有再乘batch数量。表与图采用历史参考统计量等效Gy，不声称是新4Gy计划；原TOPAS说明中的绝对临床剂量倍率不参与本轮比例/gamma。

两个模式使用相同BODY且TOPAS≥10%全体积peak的351,014体素。有限球形搜索格点、trilinear插值、完整evaluation volume内搜索，禁止外推。

| 与TOPAS的判据 | 开启 | 关闭 | 关闭−开启 |
|---|---:|---:|---:|
| local1% / 1mm，search0.1mm | 99.874934% | 99.873794% | -0.001140 pp |
| local3% / 0.2mm，search0.02mm | 99.392332% | 99.448740% | +0.056408 pp |
| global3% / 1mm，search0.1mm | 100.000000% | 100.000000% | +0.000000 pp |
| local3%同体素剂量差 | 89.959090% | 90.394115% | +0.435025 pp |

## 剂量改变在哪里

以下比较**关闭−开启GPU**，区域和阈值仍固定于TOPAS；signed为区域dose sum差异，RMS/P95为各体素相对开启模式的局部剂量变化。

| 区域 | 体素数 | signed dose sum变化 | RMS局部变化 | P95绝对局部变化 |
|---|---:|---:|---:|---:|
| BODY且TOPAS≥10%peak | 351014 | -0.013530% | 0.333749% | 0.340768% |
| BODY且TOPAS≥50%peak | 166156 | -0.012034% | 0.243578% | 0.129092% |
| 相邻不同材料的界面体素 | 171775 | -0.022207% | 0.406105% | 0.648162% |
| ρ<0.3g/cm³ | 869 | -0.018127% | 2.843008% | 5.795350% |

两GPU模式在共同区域的相互gamma：local1%/1mm 99.873509%；local3%/0.2mm 99.896300%。少量低密度体素的改变明显高于全区域平均，不能据整体gamma相近断言每处剂量完全一致。

相对TOPAS，BODY≥10%peak区域的dose-weighted signed bias：开启-0.561380%、关闭-0.574834%；同体素RMS误差分别1.924817%/1.899181%。

16个独立GPU batch估计的区域内单体素平均剂量SE RMS约1.718%/1.703%，尚未包含TOPAS参考自身噪声。paired treatment delta RMS相对TOPAS约0.336%，其paired SE RMS约0.242%。因此0.0564pp的gamma小幅增加不证明关闭模型物理更准确；这份头部宽束消融未显示明显整体gamma恶化。TOPAS为历史配置，绝对结果也包含其他现存模型差异，不能外推到lung/minibeam或更细Z网格。

原始日志/逐批dose/quality/energy/source保留在`out/carbon_ct_no_electrons_20261009/`。分析脚本、JSON报告和图位于同目录与`comparison/`；`dose_comparison.png`和`isocenter_profiles.png`为本轮独立高统计结果。使用既有16线程CPU gamma evaluator，无运行时代码改动。

后续补充：全网格（包含BODY外）≥10%peak的local1%/1mm，开启99.901614%、关闭96.225276%，关闭不通过>99%。本页上面的99.874%是BODY内结果。完整质子/碳双域必要性验收见[后续报告](rt07575_broadbeam_electron_gate_20261009.md)。

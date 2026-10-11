# RT07575 proton CT minibeam 空气步长优化（2026-10-10）

仅将原head验证配置 `minibeam_air_slit_max_step_mm` 从0.05提高到0.2 mm。保留铜0.05 mm、患者0.1 mm步长，primary/secondary及Cu主粒子/碎片涨落、Urban散射、核反应、strict26电子/光子输运与全部487个spot的原能量/发射度/位置关联。使用当前生产静态库构建offline数字源驱动；runtime源码与八默认配置均未改动。

## 800万粒子/组初筛

四个独立2M full-plan campaigns/组，ABBA+BABA。每campaign全局最大余数分配一次，再分为4批；连续批复用context，首批仅从计时中排除。每campaign剂量归一到原4Gy入射数，先求和4批，再平均4campaign。无剂量拟合。

吞吐 48,331→60,906/s，提升26.02%；四配对比值[1.2443026378485367, 1.2682142410601762, 1.266083836638332, 1.2619545349499233]。
上游暖态总时间49.452→23.937s，患者primary26.839→27.216s，electron37.045→37.139s。候选剩余主要成本是电子与患者primary。

患者BODY∪GTV且TOPAS≥10%max，共185559voxels；原历史AllChargedUrbanPhysics加权DIJ；local3%/0.2mm、搜索0.02mm。对照83.700602%，候选83.425218%，变化−0.275384pp。GPU单体素SE RMS约5.99%参考剂量；候选/对照RMS差7.891%，paired SE RMS7.933%，所以不能将局部差直接当模型偏差，也不能以本轮相对gamma宣称精度已验收。患者评价区域dose sum变化+0.005303%。32批全部accepted/energy/coverage/completion/zero-overflow通过，配对source CSV逐字节一致。

该0.05 mm空气限制是RT07575验证配置的显式设置；默认lung预设没有同样的限制，不能把26%提速直接外推到默认lung工况。空气步长改变是步进近似实验，不是不变物理重构。

## 高统计量确认（已完成）

已安排每组94,434,572histories，4个独立23,608,643-history campaigns/组，相同配对顺序与固定患者gamma评价规则；新种子与初筛独立。每批5.9M输出dose/质量/能量账目，原始结果保存在 `out/proton_ct_minibeam_air_20261010/high_statistics/`。确认结果写入该目录report.json/complete.json；未改变默认配置。

本机RTX2080Ti未锁频，历史TOPAS体素方差不可用。初筛不是绝对99%验收，且未验证其他患者、任意光学参数或LET。

高统计量确认已完成：每组94,434,572histories；暖态性能比1.2842988881568778，相对TOPAS local3%/0.2mm gamma变化0.014011715950189796个百分点，相对下降≤1pp检查通过。原始report.json/complete.json保留。部分期间本地CPU进行了A6000构建，长测吞吐存在CPU重叠条件；此前独立初筛的26.02%收益亦应一并参考。该结论仍只适用于原RT07575头部空气步长近似，不是默认lung或绝对TOPAS99%验收。

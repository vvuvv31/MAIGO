# 当前下一步

TOPAS 祖先归属 3D dose scorer、全带电离子级联 scorer 和两个 10 万粒子正式基准均已完成。新 TOPAS 作业只在 `v@192.168.31.5:~/gpu` 运行，最多使用 56 线程；Windows 原生 Arc B580 负责 SYCL kernel。

正式结果的祖先类别闭合最大误差为 `6.395e-14 MeV/primary/bin`，独立 TOPAS total 交叉检查为 `7.882e-7 MeV/primary/bin`，`unresolved=0`，未使用全局 scale。TOPAS 中性来源占全深度总剂量 `0.711%`，在 90 mm 后尾部占 `5.779%`。

同位素/反应代数 QA 已完成。它证明旧 primary reaction package 来自 TOPAS 4.2.p3/Geant4 11.3.2，而 ancestor/cascade reference 来自 TOPAS 4.1.p1/Geant4 11.1.3；两套末态混用是 He、B、Li 剩余偏差的主因。已从 cascade reference 提取 37,661 个 primary C-12 联合末态并编译为统一的 201-bin package。

统一参考后的 B580 结果：charged-origin total 全深度/90 mm 后为 `-0.11%/+2.93%`；He `-0.20%/+2.58%`、proton `+1.41%/+2.84%`、B `+3.88%/+4.06%`、Li `+3.83%/+3.72%`。所有带电类别尾部最大偏差为 Be `+6.67%`，逐 bin 闭合为 `9.80e-11 MeV/primary/bin`，未使用全局 scale。

GPU 3D scorer 的第一阶段框架已完成：CPU/SYCL 支持可选 `60 x 60 x 800` double-atomic total voxel tally、稀疏 CSV 输出和逐 z 的 x/y→IDD 闭合检查。Arc B580 的 100-history 全级联 smoke 产生 800 个 IDD bin 和 800 个中心轴非零体素，最大闭合误差为 `0 MeV/primary/bin`；完整 oneAPI 测试通过。当前粒子输运仍是一维，因此这些非零体素全部位于 `(ix,iy)=(30,30)`，不能把它当成已完成横向剂量验证。

下一步明确为：把粒子和反应包状态扩展到 x/y/z 与三维方向，沿真实路径跨越体素边界，并加入带电粒子的多重库仑散射。随后增加 3D 分类别闭合并与现有 TOPAS `60 x 60 x 800` 祖先归属剂量逐 voxel/切片比较。带电 3D scorer 稳定后再实现 neutron/gamma 来源输运。新的 TOPAS 作业仍只允许在 `v@192.168.31.5:~/gpu` 上运行，最多 56 线程，禁止使用 WSL 运行 TOPAS。

---

## 历史实施记录（以下项目已完成）

下一步应进入“次级碎片输运”阶段。当前射程、峰宽和主碳离子衰减已经较好，但布拉格峰后的尾部低约 92%，主要缺失正是核反应产生的碎片。
建议按以下顺序推进：
在 WSL/TOPAS 中增加按粒子种类计分的 IDD：
主碳离子
次级碳离子
B、Be、Li、He
质子
其他粒子
先明确每类碎片对峰区和尾部的贡献。

建立 TOPAS 碎片基准数据：
先跑 10 万粒子开发基准。
确认计分正确后，再跑 100 万粒子正式基准。
保存原始 CSV、归一化 CSV、运行参数和随机种子。

在 GPU 端实现最小可用碎片模型：
碳离子发生核反应时停止主粒子。
根据 TOPAS 数据采样碎片种类、能量和数量。
第一版只做沿束流方向的 1D 输运。
使用预分配的 GPU 次级粒子队列，避免 kernel 内动态分配。

分阶段验证：
碎片总产生率。
各粒子种类的 IDD。
布拉格峰后尾部积分。
总能量守恒。
最终比较完整 TOPAS 与 B580 GPU IDD。

每个阶段单独 Git 提交，例如：
topas: add species-resolved fragment scorers
physics: add fragment yield model
sycl: add secondary particle queues
validation: compare fragment-resolved idd

按碎片种类分解的 TOPAS IDD、事件级反应包以及 B580 次级粒子生成队列均已完成。现在缺少的是队列中带电离子的实际输运与分粒种剂量评分。

## 2026-07-14 进展

上述 TOPAS 数据准备已经完成到可运行阶段：

- 已生成 10 万粒子的按粒种 IDD 开发基准；
- 已增加自定义截面计分器，直接从当前 TOPAS/Geant4 物理列表导出 1--400 MeV/u 的 C-12+H、C-12+O 和水中非弹性截面；
- 已增加反应顶点 n-tuple，按事件保存反应前 C-12 能量及全部直接次级粒子的 A/Z、能量和方向；
- 已用 100 粒子 smoke 作业验证扩展编译、截面闭合和反应包标准化。

CPU/SYCL 能量相关截面接入也已完成：配置改用 `nuclear_cross_section_file`，两种后端在每一步按当前 MeV/u 插值宏观截面。10,000-history 的 B580 与 serial 曲线 NRMSE 为 `1.38e-5`，R80 差 `-3.9e-5 mm`，2%/2 mm gamma 为 `100%`。

10 万粒子 `fragment-development` 正式反应包已经完成：100,000 个初级粒子中有 37,657 次主 C-12 非弹性反应，共记录 330,659 个直接次级粒子，平均多重性为 8.781。两个低能反应没有直接可见次级粒子，仍以 `secondary_count=0` 的完整反应头保留。压缩反应表和次级粒子表约 10.1 MiB，header 记录数、CSV 行数、offset/count 闭合和 SHA-256 均已验证。

现在最直接的下一项工作是：实现 B580 上的固定容量次级粒子队列、原子计数器和溢出检测，并按 `reaction_id` 整包采样碎片，而不是独立抽取各粒种。带电离子统一按 A/Z 进入队列，不能遗漏 deuteron、triton、He3 等显著能量分量；gamma、neutron 暂时计入未输运能量账本，并在结果中单独报告。

主机端数据边界已经完成：gzip/CSV 可预编译为 5.9 MB、201 个 1 MeV/u 分箱的版本化二进制表，C++ `ReactionPackageTable` 会独立检查格式、分箱、offset/count 和物理值。GCC 12.2 与 Windows oneAPI IntelLLVM 2025.3.3 的真实正式数据加载测试均已通过。随后这些定长数组已接入 SYCL USM 和 fixed-capacity queue。

B580 fixed-capacity queue 也已完成首轮验证。10,000 个初级粒子产生 3,816 次核反应，全部采样事件包；33,260 个直接次级粒子中有 21,357 个带电离子进入队列，溢出为 0，未支持带电能量为 0。中子/光子能量 553,209.49 MeV 单独记账。generation-only 阶段的 primary-only IDD 逐 bin 完全不变。

上述 A/Z 输运现已完成第一版。100,000-history B580 基准输运 216,133 个带电次级粒子，碎片沉积与逃逸能量闭合，队列溢出为 0，总能量误差为 `3.11e-8`。相对完整 TOPAS，总积分差 `+0.48%`、峰值差 `-0.062%`、R80 差 `+0.104 mm`、FWHM 差 `+2.36%`、NRMSE `1.08%`、2%/2 mm gamma `97.71%`；尾积分差为 `+10.055%`，尚未通过 `<10%` 门槛。

随后完成的 10 万粒子细分 scorer 将 `other` 分为 electron/positron、gamma、neutron、deuteron、triton 和 unclassified，并把 helium 分为 alpha、He-3 和其他 Z=2 离子。旧 broad species 列与原基准逐 bin 完全一致，详细闭合最大误差为 `1.24e-10 MeV/primary/bin`。

全深度 `other` 中 electron/positron 占 `78.61%`，deuteron 占 `12.59%`，triton 占 `5.15%`，unclassified 占 `3.61%`，gamma 与 neutron 的直接轨迹沉积合计仅约 `0.004%`。90 mm 后 `other` 中 deuteron、triton、electron/positron 分别占 `51.55%`、`20.92%`、`24.58%`。扣除 TOPAS electron/positron 后，GPU `other` 全深度只高 `0.50%`，尾部高 `4.10%`。

因此下一步已修正为“统一剂量归属语义”，而不是立即调参或实现 neutron/gamma 输运。应先做 TOPAS 祖先归属 scorer，再依据对齐后的差异决定后续级联模型。

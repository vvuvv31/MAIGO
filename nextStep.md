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

最直接的下一项工作是：先修改 TOPAS 配置，输出按碎片种类分解的 IDD 基准。没有这组数据，GPU 碎片模型的产额和能谱无法可靠标定。之后再实现 B580 上的次级粒子队列。
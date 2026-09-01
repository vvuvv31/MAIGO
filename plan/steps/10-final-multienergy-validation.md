# 10：四能量、多 Seed 最终验收

## 目标

冻结package、代码和scorer contract，对G1/G2完成最终独立统计验证。

## 运行矩阵

- Energy：100、200、300、400 MeV/u。
- Cascade：G1、G2。
- Seeds：至少5个独立100k TOPAS和5个独立100k GPU。
- Scorers：unfiltered total、charged-origin 7 species、other charged、neutral-origin、unclassified。
- Grid：200×200×800，0.4×0.4×0.5 mm；IDD为X/Y求和。

## 报告步骤

1. 保存配置、seed、TOPAS job ID、package/code SHA256、日志和scorer header。
2. 报告均值、95% CI、积分、峰值/峰位、NRMSE、prepeak和tail。
3. 输出2×4 species、total多能量、G1/G2增量、core/halo、coverage/ledger图表。
4. 失败回到最早失败层：rate→coverage→birth→cascade→neutral→transport，不做末端经验拟合。

## 最终验收

- [ ] Total：积分/峰值≤2%，峰位≤0.5 mm，NRMSE≤1%。
- [ ] 七species：积分≤2%，峰位≤1 mm，NRMSE≤2%。
- [ ] core≤2%，halo≤5%。
- [ ] coverage、closure、energy residual全部通过。
- [ ] 确定生产默认G1或G2并记录未采用模式原因。
- [ ] README完成度更新为11/11，全部证据路径可读。

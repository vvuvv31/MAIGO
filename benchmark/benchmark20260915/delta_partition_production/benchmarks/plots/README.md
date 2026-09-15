# Benchmark plots

文件名均以 case ID 开头。IDD 深度范围为 TOPAS 最大 IDD 所在深度 ×1.2，受体模范围限制。三维 scorer 横向求和，没有使用 1D scorer。

b1–b3：idd、idd_relative_error、sigma_core_halo、sigma_relative_error、profiles_linear、profiles_log。b4：idd、idd_relative_error、profiles_linear、profiles_log。各图提供 PNG/PDF。

Relative error = 100 × (GPU−TOPAS)/TOPAS；IDD 另画以 TOPAS 峰值归一的差值，两种 IDD 误差图纵轴均限制为 −5% 至 +5%，原始数据不裁剪。sigma 仅在双方拟合均有效时计算误差，缺失处不表示零误差。log profile 的零值不显示。横向 profile 对 y 和精确 2 mm 深度窗积分，单位 MeV/primary/x-bin；九个深度由参考峰位的固定比例选取，实际深度见图和 manifest。

| Case | GPU straggling scale | IDD | Profiles |
|---|---:|---|---|
| b1_100 | 1.0 | [b1_100_idd](b1_100_idd.png) | [linear](b1_100_profiles_linear.png) / [log](b1_100_profiles_log.png) |
| b1_200 | 1.0 | [b1_200_idd](b1_200_idd.png) | [linear](b1_200_profiles_linear.png) / [log](b1_200_profiles_log.png) |
| b1_300 | 1.0 | [b1_300_idd](b1_300_idd.png) | [linear](b1_300_profiles_linear.png) / [log](b1_300_profiles_log.png) |
| b1_400 | 1.0 | [b1_400_idd](b1_400_idd.png) | [linear](b1_400_profiles_linear.png) / [log](b1_400_profiles_log.png) |
| b2_150 | 1.0 | [b2_150_idd](b2_150_idd.png) | [linear](b2_150_profiles_linear.png) / [log](b2_150_profiles_log.png) |
| b2_250 | 1.0 | [b2_250_idd](b2_250_idd.png) | [linear](b2_250_profiles_linear.png) / [log](b2_250_profiles_log.png) |
| b2_350 | 1.0 | [b2_350_idd](b2_350_idd.png) | [linear](b2_350_profiles_linear.png) / [log](b2_350_profiles_log.png) |
| b3_layers | 1.0 | [b3_layers_idd](b3_layers_idd.png) | [linear](b3_layers_profiles_linear.png) / [log](b3_layers_profiles_log.png) |
| b4_soft_lung | 1.0 | [b4_soft_lung_idd](b4_soft_lung_idd.png) | [linear](b4_soft_lung_profiles_linear.png) / [log](b4_soft_lung_profiles_log.png) |
| b4_soft_bone | 1.0 | [b4_soft_bone_idd](b4_soft_bone_idd.png) | [linear](b4_soft_bone_profiles_linear.png) / [log](b4_soft_bone_profiles_log.png) |

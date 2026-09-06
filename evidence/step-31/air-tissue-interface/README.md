# 双向空气/组织界面基线诊断（2026-09-06）

状态：REFERENCE_INCONSISTENT；纵向候选界面正确性及患者Gamma改善均未证明。

175MeV/u，1%能散，40mm HU=-1000空气 + 5mm HU=100软组织，交换顺序。
两个TOPAS seed各20k，两组GPU基线各20k。核反应关闭，3D DoseToMedium。
CCTG按Schneider公式构造，section0/8、rho=0.011316064745/1.078799724579。
不使用患者源密度缩放；没有打开均匀介质专用flag来绕过混合网格保护。
旧失败参考未复用。生成的新Z命名容器正确完成材料初始化。

界面相邻2mm观察值：
- 空气→组织：上游空气GPU +2.104%，下游组织 −0.133%。
- 组织→空气：上游组织 −0.169%，下游空气 +1.268%。
这些只描述基线局部差异，不是候选修复结果。

空气→组织末端(+2到+5mm，最后一个体素层主导)参考异常：
两replica相差85.27%，总积分不能作验收。无日志中的stuck/kill/exception证据，
故不宣称已查清异常根因。额外真空容器padding对照2363/2364仍有末端异常，
没有据此宣称geometry修复成功。保留全部原始文件，不后验删除末端bin报PASS。

全部TOPAS作业2359–2364 COMPLETED 0:0；峰值96CPU/40GB。
两GPU基线accepted=true；2个诊断单测通过；diff check clean。
参考退出0并不等于数据物理可信，本例由replica一致性检查拒绝。
数据位于/mnt/sda/wuwei/air_tissue_interface_20260906/
及/mnt/sda/wuwei/air_tissue_interface_padded_20260906/。
manifest冻结准备输入；analysis.json记录输出SHA及局限。

下一步以有足够下游材料缓冲、远离出口的界面ROI为判据，检查buffer厚度收敛；
再测试跨材料候选。患者配对口径见plan2/interface_patient_focus.md。
不继续围绕521次低能查询扩表；不改物理/scale，不跑不合规的患者候选。
本轮无新患者Gamma、未commit/push。

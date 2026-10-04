# 独立电子边界输运：公式组件提取记录

最新状态见 [独立电子运行路径](electron_charged_runtime_20261003.md)：主程序和 proton CT minibeam 默认研究配置已接入完整 birth/joint/charged/photon，RT07575 已验证旧电子库零读取。以下保留此前公式组件验证记录；不代表正式 CT 剂量验收。

## 弹性散射：公式加原生小表

`include/carbon/electron_elastic.hpp`实现原生GS的screened Rutherford提议分布、Mott能量结点混合及拒绝抽样。没有拟合剂量、压缩随机轨迹或丢弃大角散射尾。

原生Mott角修正在31个能量结点上使用分段线性插值及末段三次多项式。提取实际材料的修正值，即可还原整个函数；runtime无须保存随机角度样本。末段与前一段存在微小不连续，导出时分别保留两侧，不能直接把一个公共边界查询值当作两侧相同的结点。

- 每节点9,464B，82节点共 **776,048B（约758KiB）**。这是弹性角度组件的总数据，不是整个电子处理包大小。
- 31×4,096个角度点/材料的函数检查最大误差1.11e−15。
- 每节点100,000个10eV–0.65MeV测试，共 **8,200,000次**，以相同随机数流分别调用原生GS和新CPU实现：随机数状态及消耗完全一致。
- GPU重复全部8.2M条原生参考，最大cos(theta)误差 **4.44e−16**，随机数消耗一致，FP32计分闭合。观察到的最大拒绝抽样次数为5；新实现达到1000次上限会报invalid，不会强行接受被拒绝的角度。
- 证据：`elastic82/report.json`、`elastic_gpu.json`。索引SHA `4b703ea9a9520af265daaa74522983431f7ffb4ae7a919b08557d9bce95b4055`。
- 原生导出二进制：`electron_elastic_native_v2/topas`，SHA `c03d0dec30525a7eda21553244c56e109a4e896d7c8f5c112dd5bcdf8247c666`。构建记录保留所用portable header及SHA。

## 能损涨落：实际电子UrbanFluc

原生opt4的PenIoni与MollerBhabha均选择`UrbanFluc`，与已有proton用的universal分支不同。`electron_urban_fluctuation.hpp`保留两类激发、离化尾和相应高计数组件；不会给电子使用heavy-particle Gaussian/Gamma分支。

每种材料只需电子质量、I、E0、E1、E2、F1、F2共7个double。82节点常数合计 **4,592B**；停止本领、射程、production cut及硬碰撞率是另外的数据，不能用这4.6KB代指整个输运模型。

- 82节点×8个电子能量×5个平均损失条件，共3,280条件，每条件每实现50,000次抽样，即原生/新CPU各164,000,000次。
- 原生与新CPU使用各自随机流，比较分布而非逐值相等。均值和二阶矩最大标准化差2.40，最大双样本KS为0.00942，全部通过预先设定的多条件保守门槛。不同材料条件复用随机流，因此不能将所有条件当作独立样本合并推断置信度。
- 对原始涨落超过入射KE的尾概率另做3,280组exact Fisher检验；结果见`fluctuation_tail_audit.json`。这些原始涨落保留原值，未夹断或重抽。
- GPU与已验证CPU公式以相同随机数重复 **3,358,720次**，随机数消耗一致；损失差相对max(loss,10eV)的最大值3.59e−13，FP32计分闭合。
- 证据：`fluctuation82/report.json`、`fluctuation_gpu.json`。索引SHA `bea737e81d90ee5b9df62903fdb9c7d4cd35e17f43c2ebcf36e119b7106daf3f`。
- 原生二进制：`electron_fluctuation_native/topas`，SHA `63ec6e0a4af16fd35b2168a49ce4383b298461dd19012da4ba46c45846ed3a42`。

这只是原始涨落律，**还未实现完整AlongStep规则**。原生过程还会根据真实步长/射程计算平均损失、处理到达射程和最低能量阈值、停止粒子并闭合能量账。现有原生日志的最低electron KE为100eV、PIXE关闭；不能用任意截断或缺覆盖局部沉积代替这些规则，也不能把原始loss>E直接当作物理沉积。

## 硬碰撞和cascade分支

`ElectronCollisionDump`通过实际electron过程的`StartTracking`、GPIL、`PostStepDoIt`提取eIoni/eBrem条件终态，保留production cut、积分法null collision、parent及每个真实子粒子。子粒子只在离线条件提取后释放；这个工具不称为完整analog历史或只读剂量参考。

- `electron_collision_grid82/report.json`：82节点、257能量锚点的有原生支持通道，每通道512次接受采样。
- **8,606,208次碰撞，8,443,279个子粒子**，553,588次null collision单独记录。
- 逐次能量闭合最大残差5.55e−17MeV，无正能量分支被省略。
- 原始训练记录2,045,938,440B，尚未编译成runtime能量加权分支CDF；这不是最终交付包体积。
- 原生二进制：`electron_collision_native/topas`，SHA `3dfaee049645e169785e8e6e656d902d7ce86ee25a9d5499aeefa63dee366f1b`。

源码复核发现，原生`PostStepDoIt`会在null-collision检验前把负的插值lambda按零处理。先前参数探针中的direct-model正截面与这个effective runtime rate在阈值附近不是逐点相同。当前条件提取调用真实过程；后续runtime率表须选择并明确验证其插值规则，不能借助“全能量锚点都有数据”宣称细区间覆盖已经闭合。

## 完全移除旧库仍需完成

1. 编译独立charged bank：停止本领/射程、实际cut、弹性公式、Urban材料参数、硬碰撞全部正能量分支；做能量/过程细区间覆盖检查。
2. 实现实际几何下的带电步进，charged步长不超过0.25mm。先确定自由飞行和真实界面距离，再计算该段能损及相互作用；不裁切一个已经抽完角度/能损的整步。
3. 以已有12个原生界面场景验证正常/掠入射、继续态、回穿，再补CT棱角及多界面验证。保持能量权重W和物理KE分离，分支缺口明确失败。
4. 主程序解除旧birth/continuation映射与旧bank加载，实际运行无旧库的RT07575并检查峰谷、局部剂量、能量和零溢出。上述组件测试不代替这一验收。

公式依据Geant4 11.3.2的`G4GoudsmitSaundersonTable.cc`、`G4GSMottCorrection.cc`、`G4GoudsmitSaundersonMscModel.cc`、`G4UniversalFluctuation.cc`、`G4UrbanFluctuation.cc`及`G4VEnergyLossProcess.cc`。版本源码保存在`geant4_source/`，许可证见`LICENSES/Geant4.txt`。

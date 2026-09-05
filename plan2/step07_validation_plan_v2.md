# Step07 修正版：同材料、出生条件匹配的密度验证

状态：三个密度各 1-history 输出体积/记录 pilot 已完成；正式条件化比较尚未运行。
原 step07_validation_plan.md 保留为已作废设计，
不能拿其跨材料结果证明 section-0 缩放失败。本文件不是生产准入证据。

## 1. 材料和密度（运行前强制检查）

执行：

    python3 tools/audit_schneider_response_scope.py --hu -1000 -975 -951

三点必须全部属于 material section 0 [-1000,-950)：
- -1000：0.0113160652 g/cm3
- -975：0.0393234522 g/cm3
- -951：0.0662101570 g/cm3

这些数仅作回归预期，实际运行需用当前 pinned Schneider 文件重算，并与
ntuple 实测材料密度比较。必须包含 DensityCorrection，禁止用 builtin 的
全 1 correction 替代文件。HU=-550/-100 不得加入本组。

## 2. 记录格式和能量语义

只使用 CarbonElectronDepositNtupleV3，34 列/232 bytes，schema_version=3。
必须 parent_step_binding_verified=true，missing parent=0。

parent_ke/dir 为产生电子的准确父 step 的 pre 状态；额外保存 post 状态。
不使用 v2 末态缓存，不使用最近轨迹段查找。不能把 pre 状态称为 Geant4
模型内部瞬时采样状态，需同时报告产生步 pre/post KE 差。

仍有 parent_conditioning_runtime_eligible=false：它是实验记录，不是已验证核。

## 3. 固定出生群，避免几何/能谱混杂

先做 200 MeV/u；同一 beam、物理列表、cut、网格和源定义。
比较必须同时按父 pre-step KE、电子根 birth KE、出生位置选择样本。
仅匹配名义 beam energy 不够：高密度几何中主粒子慢化不同，production cut
按长度设置也会改变显式电子产生阈值和出生谱。

将比较拆为：
1. 总 stopping 中局部/显式电子出生能量分配；cut 依赖单列。
2. 在相同电子出生能量及父状态条件下的空间响应。
3. 按相同出生位置、距边界条件选择的沉积/逃逸响应。

不得将条件 1 的变化直接解释为条件 2 的位移缩放失效。
不得用“延长 slab 后仍有新出射电子”证明固定出生群尾部不收敛。

## 4. 指标和门禁

- 每片及逐事件出生/沉积/逃逸闭合，沿用 1e-3 硬上限，不放宽。
- 所有 joint bins 同一边界，概率形状 L1 = sum(abs(P0-P1))，范围 [0,2]。
- 沉积总量比独立报告；不与形状 L1 混用。
- 固定出生群的样本数、能谱、位置分布必须报告。
- 按 history/block 估计误差，不把 steps 或 roots 当独立历史。
- 旧 L1 阈值与“16 倍”结论不复用。条件化 bin、统计目标和停止条件
  尚需在新 campaign 启动前进一步冻结；因此当前不可宣称实验准入完成。
- 数据稀疏、尾部仍受几何影响或出生群无法匹配，标 INCONCLUSIVE，不制表。

## 5. 资源和运行顺序

先每种新密度 1 history 的输出体积 pilot，明确数据/内存估计后再分片增加。
不得直接假定 HU=-1000 的每片体积上限适用于所有密度。
TOPAS 仅本地 Slurm，总 CPU<=192、内存<=160 GB；GPU 不参与本步。
生成器的 --submit 仍关闭。独立 execute_electron_response_campaign.py 已实现
共享资源/磁盘轮询/完成监控，pilot 已通过；正式实验仍须先冻结出生群、
统计停止条件、完整外部输入及实际 cut 元数据。不能因 pilot 完成就扩大 campaign。
不修改 GPU 物理，不替换现有数据包，不运行患者 Gamma/full20。

## 6. Pilot 结果（只证明记录和输出规模）

Job 2350/2351/2352，对应 HU -1000/-975/-951，各 1 history、2CPU/4GiB。
实测 density 为 0.0113161/0.0393235/0.0662102 g/cm3，与公式相对差 <3.1e-6。
三组产生步绑定、3D dose 闭合和加强后的逐事件/逐家族闭合全部通过。
原 raw/report 保留；最终复核为各目录 `analysis_closure_verified.json`。
证据：`evidence/step-31/entrance-mask-candidate/section0-density-pilot-v3.json`。
单 history 不是独立统计重复，禁止比较其分位数后宣称 1/rho 模型通过/失败。

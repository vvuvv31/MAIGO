# 所有 GPU 预设共用单进程批次与输入缓存（2026-10-02）

原生批次参数原本已是通用入口；本次补齐其余路径的实际数据复用。GPU 主入口现在始终创建一个进程级 `SyclTransportContext`，普通 manifest、普通源、TPS、TOPAS spot 和逐 spot 运行都可复用已验证的只读数据。每批仍有独立的输运状态、剂量、随机种子、审计和质量报告。

```bash
./scripts/run.sh proton_ct_plan_fe \
  --histories-per-batch 2000000 --batch-count 10 \
  --batch-output out/proton_broadbeam_10x2m
```

将预设替换为 `carbon_ct_plan_fe_joint`、`proton_water_broadbeam`、`carbon_water_broadbeam`、`proton_water_minibeam`、`carbon_water_minibeam`、`proton_ct_minibeam`，并指定新的输出目录即可。参数是每批整份源计划的总 incident primary 数；原来的 spot 权重分配、总批次种子规划和输出布局不变。`--plan-manifest` 继续用其自己的每分区粒子数，不与原生批次参数叠加。

## 复用内容

- 所有 GPU 入口共用已有 Unified EM、Schneider/CINEL03、Urban、弹性、停止本领等只读缓存。
- CT grid、density/material 设备数组不再依赖 `material_electron_retain_bank`。普通 CT broadbeam 同样可以命中缓存。既有电子 bank 的兼容条件和开关保留。
- CT 缓存支持 CCTG 与 DICOM series；坐标原点策略、Schneider 映射、切片文件增删/修改均参与缓存键。换几何时释放旧设备数组和关联电子 bank；切到 water 时释放不用的 CT 数据。
- Carbon CT 联合响应 CSV、ordered-path 数据进入只读 host 缓存；水模体 proton/carbon 电子响应表也复用解析结果。
- 中子事件、macroscopic rates 和反冲停止本领包接入同一个 session cache；缓存命中时不重复启动预取；中子 recoil Urban 与主输运共用已加载的包。
- 空气电离 rate CSV 也复用解析结果。

缓存键包含路径、size/mtime、适用的 SHA pins、粒子种类，以及包引用的伴随文件。首次加载仍经过原有 SHA、格式、覆盖范围和物理适用性检查；失败结果不进入缓存。元数据读取和文件 stat 仍会发生。

这不是所有初始化都消失：部分设备上传、path bounds/索引构建、工作区与 scorer 分配、轻量验证、输出阶段 CT 读取仍按批执行。缓存不会扩大单批安全粒子数或队列容量，也不改变 group size、物理近似或剂量计分精度。固定 FP32 ON、FP64 OFF。

## 验证

证据目录 `out/shared_batches_20261002/`。本次测试在本地 RTX 2080 Ti 上使用每批1000个 primary；小批验证用于功能、缓存命中、状态隔离及剂量回归，不用于认证完整计划吞吐提升。

- `tests/input_dependencies.cpp`：伴随 payload 修改、嵌套 manifest、缺文件/路径越界拒绝、DICOM 切片增删/修改、origin/mapping 变化、CCTG 文件隔离。
- 实际 carbon 联合响应 host 测试：340,747个样本、8,289,928个路径向量复用同一对象；错误 SHA 被拒绝。此项不输运粒子，不绕过质量门禁。
- GPU 对照与 manifest 验证结果见下方完成记录。

Carbon CT 联合响应仍带有既有 `unvalidated_electron_joint_response` 失败。批次循环必须在首批失败后停止；不得为测量第二批缓存收益忽略该门禁。其 host 缓存测试不代表患者剂量模型验证。

## GPU 完成记录

| 预设 | 每版完成批数 | 新版第二批缓存 | 质量结果 |
|---|---:|---|---|
| proton_water_broadbeam | 2 | 物理包 | 两批接受 |
| carbon_water_broadbeam | 2 | 物理包 | 两批接受 |
| proton_ct_plan_fe | 2 | 物理包 + CT | 两批接受 |
| carbon_ct_plan_fe_joint | 1 | 首批 joint response 加载，随后停止 | 原有联合响应未验证，停止后续批次 |
| proton_water_minibeam | 2 | 水响应 + neutron package | 两批接受 |
| carbon_water_minibeam | 2 | 水响应 | 两批接受 |
| proton_ct_minibeam | 2 | CT + material electron bank | 两批接受 |

两版各完成13批GPU输运；每批1000个primary，七个预设共26批。整数输运/EM审计完全一致，最大剂量相对L1=1.14e-07，最大体素差/参考峰值=2.491e-07，最大相对能量残差=5.271e-06，queue overflow均为0。所有剂量有限且非负。

本轮不宣称输运内核更快或给出稳定吞吐提升百分比。`verification.json`保留cold/warm小批计时；它们受到首次磁盘缓存与编译时主机负载影响，不能用首批差异作提速归因。

已构建主程序SHA256：`a29f19e23795c77d709b4c4593f16b516d14f6503faefcc4d4335bc2646a8cb0`。

原有native batch分配/种子/冲突参数测试也通过。

普通water manifest的两分区对照通过：旧入口两次Unified EM miss，新入口首批miss、第二批hit；整数审计一致，剂量相对L1≤2.12e−9。这个测试直接覆盖了原先未创建context的入口。

混合manifest按CT→同一CT→water→另一CT文件运行，得到CT缓存miss→hit→释放→miss；两版四批整数审计一致，剂量相对L1≤5.28e−9。包含上述额外入口测试，共完成38批GPU对照输运。DICOM的增删/修改失效做了host键测试，GPU剂量对照使用现有CCTG输入。

测试命令与原始证据：`verify.py`、`verification.json`、`manifest_verify.py`、`plain_manifest/checked.json`、`mixed_verify.py`、`mixed_manifest/checked.json`、`input_dependencies.log`、`joint_cache.log`、`summary.json`。构建最终完成日志为`build_complete.log`；之前的链接为纳入最后依赖检查而中断，不是剂量或性能结果。

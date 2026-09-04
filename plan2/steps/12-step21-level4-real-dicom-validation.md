# Step 12: Step 21 Level 4 真实 DICOM 临床体模验证与最终全流程验收

## 1. 目标与背景

Level 4 是全套 Schneider CT 输运的最高阶临床实战验证：直接加载真实病人 DICOM CT 与射野计划，进行真实 GPU 蒙卡剂量重算，并与 TOPAS 高统计量参考模拟进行全三维 DoseToMedium 对比。
此前 `evidence/step-21/` 为空，存在虚假 DONE。

本步骤必须：
完成真实 DICOM 病人数据的三阶段对比模拟（初级纯吸收、C12 元素末态、全次级级联），完整核算全部物理与工程指标，生成真实证据，最终将 Step 21 与整个 Plan 2 推进至闭环。

## 2. 涉及文件与输入数据

- 生产主程序：`carbon_mc`
- 输入数据：冻结的病人 DICOM CT 数据与照射计划（包含 source/plan/DICOM 哈希验证）
- 参考基准：TOPAS 4.2.p3 / Geant4 11.3.2 官方模拟结果
- 证据输出：`evidence/step-21/`

## 3. 详细技术规范与验证阶段

必须分三阶段递进式独立验证并记录：

### 阶段 A: 初级碳核吸收模式 (Primary-Only CT)
- 关闭次级反应回放，仅输运初级 C12。
- 验证材料相关 Stopping Power、MCS 及初级衰减截面在复杂解剖结构中的计算精度。
- 门禁：初级射程差 < 0.5 mm，高剂量区差异 < 1.5%。

### 阶段 B: C12 元素目标末态回放 (C12 Elemental Final-State)
- 开启初级非弹性碰撞并抽样 13 元素目标核，生成第一代次级碎片。
- 门禁：初级存活曲线差异 < 1.5%，第一代主要碎片产额与能谱相符。

### 阶段 C: 全次级级联输运 (Full-Secondary Production CT)
- 开启全部 13 种带电次级粒子的独立目标核抽样、次级偏反应率与次级 CINEL03 回放。
- 对比最终 3D DoseToMedium 剂量网格。
- 门禁指标：
  - 全剂量积分相对差异 < 2.0%
  - Bragg 峰位置 / 射程差异 < 1.0 mm
  - 3D Gamma (2%/2mm) 通过率 > 95.0%
  - 初级存活率相对差异 < 2.0%
  - 主要碎片积分相对差异 < 2.0%
  - `unsupported_lookup_count == 0`
  - `queue_overflow_count == 0`

### 3.4 最终全量证据归档
将全套评测报告、Gamma 分析切片、IDD 对比曲线、各物种分布以及全部输入文件的 SHA-256 写入 `evidence/step-21/`。

## 4. 最终全流程交付核验

在将 Step 21 与 Plan 2 标记为完成前，必须核对以下所有条件：
1. 生产输运由 `transport_sycl()` 驱动，无独立 runner 作弊代码。
2. 每一个验证用例均独立通过门禁，无被平均值掩盖的失败。
3. `tests/test_schneider_dicom_reference.cpp` 中无任何硬编码伪造变量。
4. `evidence/step-21/` 包含完整的真实计算结果与出处哈希。
5. 所有数据文件的 companion metadata 均完整无缺。
6. 全套测试（`ctest`）在最新重新构建后 100% 通过，无崩溃。
7. 未支持核素与次级队列溢出计数全流程严格为 0。
8. 水物理回归基线零漂移。

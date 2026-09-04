# Step 11: Step 21 Level 3 非均匀合成几何真实计算与 Gamma 门禁

## 1. 目标与背景

Step 21 的 Level 3 阶段旨在验证非均匀、多材料交界面、斜向入射等复杂几何下的剂量计算准确度。此前该阶段存在硬编码赋值和空证据问题。

本步骤必须：
使用生产程序对 Level 3 复杂非均匀体模（轴向对称与斜向复杂阶梯）进行真实 GPU 模拟，严格与 TOPAS 参考数据比对，计算 3D 剂量、射程、存活率与 3D Gamma 2%/2mm 通过率。

## 2. 涉及文件与工具

- 执行程序：生产主程序 `carbon_mc`
- 验证脚本：`tools/verify_step21_level3.py` (或相应真实评估脚本)
- 输入配置：`config/step21_level3_*.yaml`
- 证据输出目录：`evidence/step-21/level3/`

## 3. 详细技术规范

### 3.1 计算量度与 3D Gamma 分析定义
必须严格固定并记录 Gamma 分析的参数：
- **剂量归一化**：以 TOPAS 最大剂量（Global max dose）归一化。
- **剂量截断阈值**：10%（低于最大剂量 10% 的低剂量体素不参与 Gamma 统计）。
- **空间搜索半径**：3 个体素或 6 mm。
- **插值方式**：三线性三维插值（Trilinear interpolation）。
- **参与统计体素数**：完整记录参与评测的总有效体素数。

### 3.2 评估指标 (硬门禁)
对于每个 Level 3 几何用例：
1. 全剂量积分相对差异：`total_dose_diff < 2.0%`
2. Bragg 峰或射程差异：`range_diff < 1.0 mm`
3. 3D Gamma (2%/2mm) 通过率：`gamma_pass_rate > 95.0%`
4. 初级碳核存活率相对差异：`primary_survival_diff < 2.0%`
5. 目标核反应混合比（Target interaction mix）：与 TOPAS 统计相符（$\rho$\chi^2$\rho$ 检验  > 0.05$\rho$）
6. 主要碎片积分相对差异：`species_diff < 2.0%`
7. `unsupported_lookup_count == 0`
8. `queue_overflow_count == 0`

### 3.3 真实计算与证据记录
严禁任何硬编码伪造！必须从 TOPAS 输出的 `.mhd` / `.raw` 和生产程序输出的 3D dose 网格真实逐点计算。将包含各向剖面、深度剂量曲线及 Gamma 分布的计算报告真实写入 `evidence/step-21/level3/`。

## 4. 验收标准

1. 轴向对齐用例（Axis-aligned）与斜向用例（Oblique）各自独立计算通过所有门禁。
2. 3D Gamma 2%/2mm 通过率实测 > 95%。
3. 生成真实、可重现的证据文件。

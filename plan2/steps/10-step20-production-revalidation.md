# Step 10: Step 20 生产级重新验证与全用例门禁闭环

## 1. 目标与背景

此前 Step 20 的验收被撤销，原因是在 `staircase_200mevu` 等用例中主要碎片相对偏差高达 4.811%（门禁要求 < 2%），且被验证器的平均值逻辑所掩盖。

本步骤必须：
使用已经接入 Schneider 完整物理的生产程序（`carbon_mc` / 生产主干），在本机 RTX 2080 Ti GPU 上完整重跑 Step 20 全套验证用例。执行任务分片防止溢出，确保**每一个用例单独满足原门禁要求**，产出真实证据，将 Step 20 真正推进至 DONE。

## 2. 涉及文件与工具

- 执行程序：生产主程序 `carbon_mc`（禁止使用带有作弊逻辑的独立 runner）
- 验证脚本：`tools/verify_step20_validation.py`
- 验证数据输入：`data/schneider/`、`benchmark/schneider/step20/cases.yaml`
- 证据输出目录：`evidence/step-20/`

## 3. 详细技术规范与执行规范

### 3.1 运行规范
1. **硬件与环境**：必须在本机 NVIDIA GeForce RTX 2080 Ti（`sm_75`）上沙盒外直接执行，记录 GPU 型号与驱动版本。禁止使用远程主机。
2. **任务分片 (Sharding)**：
   - 单次模拟历史数控制在安全阈值（如每次 1/10 粒子数），防止次级粒子堆栈发生任何 overflow。
   - 只有当分片运行的 `queue_overflow_count == 0` 且 `unsupported_count == 0` 时，该分片方可并入最终统计。
   - 若检测到 overflow，自动丢弃分片并降低历史数重跑。
3. **计分规则**：
   - 必须使用 3D dose scorer 网格，横向求和得到 1D 深度剂量（IDD）。严禁使用 1D dose scorer。

### 3.2 验收指标要求 (原门禁阈值)
每一个用例（包括平坦体模、阶梯体模各能量点）：
- 整体剂量积分相对差异 < 2.0%
- 主要带电碎片（p, He, Li, Be, B, C）相对积分差异 < 2.0%
- 射程差异 < 1.0 mm
- `unsupported_lookup_count == 0`
- `queue_overflow_count == 0`
- 套件平均差异 `suite_mean_diff < 0.02`

### 3.3 门禁核验与证据生成
1. 默认执行只读核验：
   ```bash
   python3 tools/verify_step20_validation.py
   ```
   确认控制台输出所有用例全部显示 `[PASS]`，且整体结果为 `ALL PASS`。
2. 仅在全部通过后，显式生成证据：
   ```bash
   python3 tools/verify_step20_validation.py --generate-evidence
   ```
   生成包含每个用例明细的 `evidence/step-20/verification.json`。

## 4. 验收标准

1. 每一个 Step 20 测试用例的实际误差均 < 2%，包括此前失败的 `staircase_200mevu`。
2. 验证器输出 0 失败，整体 PASS。
3. 真实生成的 `evidence/step-20/` 证据文件提交，哈希记录在案。
4. `plan2/README.md` 中将 Step 20 状态正式更新为 `DONE`。

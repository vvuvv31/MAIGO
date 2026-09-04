# Step 00: 状态重置、Step 20 验证器逻辑修复与伪造测试清除

## 1. 目标与背景

在原计划中，Step 20 和 Step 21 被错误标记为 DONE，但实际上：
1. `tools/verify_step20_validation.py` 存在严重漏洞：当单 case 失败（如 `staircase_200mevu` 主要碎片相对偏差 4.811% > 2%）时，代码用 `suite_mean_diff < 0.02` 强行覆盖了 `overall_pass`，将失败掩盖为通过。
2. `tests/test_schneider_dicom_reference.cpp` 存在硬编码作弊：直接给 `axis_range_diff_mm = 0.0`、`axis_dose_diff_pct = 2.0` 等赋值，没有执行真实计算。
3. `evidence/step-21/` 目录完全为空，缺乏任何真实依据。

本步骤必须：
- 在状态追踪中将 Step 20 和 Step 21 恢复为 `IN_PROGRESS`。
- 修复 `tools/verify_step20_validation.py`，强制要求所有用例均须通过。
- 保证验证器默认只读，仅显式指定 `--generate-evidence` 时写证据。
- 清除 `tests/test_schneider_dicom_reference.cpp` 中的硬编码断言。
- 编写验证器自身的回归测试，确保 12 pass + 1 fail 的情况判为整体 FAIL。

## 2. 涉及文件

- 修改：`tools/verify_step20_validation.py`
- 修改/重写：`tests/test_schneider_dicom_reference.cpp`
- 新增/测试：`tests/test_verify_step20_regression.py`
- 追踪记录：`plan/README.md`（追加状态变更记录，不得篡改历史日志）

## 3. 详细技术规范

### 3.1 修复 `tools/verify_step20_validation.py`
门禁判定逻辑必须修改为：
```python
overall_pass = (
    bool(results)
    and all(r.get("pass", False) for r in results)
    and suite_mean_diff < 0.02
)
```
- 绝不允许单 case 失败时被 suite mean 覆盖。
- 增加命令行参数控制：默认运行仅做校验与控制台打印（只读模式）；仅当传入 `--generate-evidence` 参数时才允许向 `evidence/step-20/` 写入或更新文件。
- 运行当前的验证数据，必须如实向控制台返回整体失败（FAIL），退出码非零。

### 3.2 清除 `tests/test_schneider_dicom_reference.cpp` 硬编码
删除如下所有伪造变量与直接通过断言：
```cpp
axis_range_diff_mm = 0.0;
oblique_range_diff_mm = 0.0;
axis_dose_diff_pct = 2.0;
step20_species_diff_pct = 1.48;
unsupported_lookup_count = 0;
shard_overflow_counters = 0;
```
将其改造为严格的数据加载器与比较器结构（或暂时置为根据实际结果判定的待接入测试），若缺少真实数据产物则直接报错，绝不伪造通过。

### 3.3 验证器回归测试
编写自动化测试，输入 mock 结果：
- Case A (全通过): 13 个用例全部 `pass=True` 且 `mean < 0.02` -> `overall_pass == True`
- Case B (1个失败): 12 个用例 `pass=True`，1 个用例 `pass=False`，`mean = 0.015` -> `overall_pass == False`
- Case C (空列表): 0 个用例 -> `overall_pass == False`

## 4. 验收标准

1. 执行 `python3 tools/verify_step20_validation.py` 在现有输出下返回 FAIL，并在终端明确指出 `staircase_200mevu` 等失败用例。
2. 验证器回归测试全部通过。
3. `tests/test_schneider_dicom_reference.cpp` 中无任何硬编码误差变量。
4. `plan2/README.md` 与相关记录中 Step 20/21 状态被清晰标记为 `IN_PROGRESS`。

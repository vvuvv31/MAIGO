# Step 07: SecondaryRateTable 二进制解析加固、边界防护与畸变数据测试

## 1. 目标与背景

`SecondaryRateTable` 是次级核反应抽样的核心数据结构。此前其解析函数 `from_binary()` 缺乏充分的防御性编程，对畸形文件、大小截断、元数据脱钩或索引越界缺乏有效防护。

本步骤必须：
在 `include/carbon/secondary_rate_table.hpp` 和 `src/secondary_rate_table.cpp` 中实现极其严格的格式校验、乘法溢出防护、元数据 SHA 绑定及全量边界防护，并编写专门的畸变数据测试。

## 2. 涉及文件

- 修改：`include/carbon/secondary_rate_table.hpp`
- 修改：`src/secondary_rate_table.cpp`
- 单元测试：`tests/test_secondary_rate_table_hardening.cpp`

## 3. 详细技术规范

### 3.1 `from_binary()` 严格 Header 与规格检查
解析开始时必须按严格顺序逐一检验：
- Magic 标识符：必须严格等于 8 字节 `"SCHN2RAT"`
- 格式版本号：`version == 1`
- 射弹数量：`num_projectiles == 13`
- 截面材料段数：`num_sections == 25`
- 规范目标元素数：`num_targets == 13`
- 能量网格点数：`num_energies == 860`
- 能量范围与步长：`energy_min == 0.5f`，`energy_max == 430.0f`，`energy_step == 0.5f`
- 头部中所有浮点数值必须为有限值（`std::isfinite() == true`）。

### 3.2 尺寸与溢出安全防护
- 计算所需数据大小时，对所有尺寸乘法执行溢出检查：
  `total_floats = num_projectiles * num_sections * num_targets * num_energies`
  必须确保无 `size_t` 乘法溢出。
- 校验文件精确尺寸：`expected_file_size == header_size + total_floats * sizeof(float)`。若文件截断（truncated）或存在尾随字节（trailing bytes），直接抛出异常拒绝加载。

### 3.3 物理数据有效性与一致性检验
- 检验射弹列表合法性：13 个核素 Z/A 合法且唯一。
- 规范目标元素数组必须与标准顺序（H, C, N, O, Na, Mg, P, S, Cl, Ar, K, Ca, Fe）完全一致。
- 遍历所有偏反应率与总反应率：
  - 必须全部有限且非负：`rate >= 0.0f && std::isfinite(rate)`。
  - 对于每一对 `(projectile, section, energy)`，偏反应率之和必须与总反应率一致：
    `std::abs(sum(partial) - total) < 1e-4f * (total + 1e-6f)`。

### 3.4 伴随元数据绑定
- `metadata_path` 绝不能被忽略。
- 读取伴随 `.json` 元数据，计算二进制文件的 SHA-256 并与元数据中的声明校验和比对，不符则拒绝加载。
- 检查元数据中包含完整的 provenance 字段。

### 3.5 访问器边界检查
所有公共索引访问接口（如 `at(proj, sec, target, energy)`）必须校验输入索引范围，越界时抛出 `std::out_of_range` 异常，绝不允许产生未定义行为。

## 4. 验收标准

1. 编写恶意/畸变输入测试套件，全面覆盖：
   - 错误 magic、错误版本号、错误维度。
   - 包含 NaN、Inf 或负反应率。
   - 重复的射弹或错误的目标元素顺序。
   - 偏反应率与总反应率不匹配。
   - 截断文件、带有多余字节的文件。
   - 元数据 hash 不匹配。
   所有畸变用例均成功触发异常，程序安全退出。
2. 正常合法数据加载无报错，访问器在越界索引时准确抛出异常。

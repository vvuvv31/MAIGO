# Step 01: 基线构建排查与 carbon_tests SEGFAULT 崩溃定位与修复

## 1. 目标与背景

用户与基线检查已明确指出：
在当前 HEAD 重新配置 CMake（cmake reconfigure）并构建后，运行测试套件时出现：
```text
carbon_tests -> SEGFAULT
1/3 tests failed
```
在进一步修改任何生产物理代码之前，必须首先定位此崩溃的根本原因，消除内存非法访问或未初始化指针问题，确保基础单元测试套件健康。

## 2. 涉及文件

- 排查与修改：`tests/carbon_tests.cpp` 或引发崩溃的底层实现头文件（如 `include/carbon/*`、`src/*`）
- 构建配置：`CMakeLists.txt`

## 3. 详细排查与修复流程

1. **复现崩溃**：
   - 在构建目录执行全新构建或单目标构建：
     ```bash
     cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
     cmake --build build -j 16 --target carbon_tests
     ```
   - 使用 GDB 或 `valgrind` / `ASan` 捕获崩溃现场堆栈：
     ```bash
     gdb -batch -ex "run" -ex "bt full" ./build/bin/carbon_tests
     ```
2. **根因定位与修复**：
   - 检查崩溃点是否为：
     - 测试夹具初始化时的空指针解引用或悬挂引用。
     - 静态单例或全局查找表（如 SchneiderHuTable、CrossSectionTable）初始化时序问题。
     - 数组越界（如核素表、截面网格、体素材料索引）。
     - 栈溢出或大结构体按值拷贝。
   - 采用外科手术式精准修复，保持原有测试用例意图不变。
3. **回归验证**：
   - 重新编译并在本地运行 `carbon_tests`，确认崩溃消除，所有已有单测通过。
   - 确认 `ctest -N` 能够完整发现所有测试目标，且基础测试无挂死、无崩溃。

## 4. 验收标准

1. `./build/bin/carbon_tests` 运行无 SEGFAULT，全部已有测试用例通过。
2. 记录崩溃根因分析报告及修复的 commit SHA。

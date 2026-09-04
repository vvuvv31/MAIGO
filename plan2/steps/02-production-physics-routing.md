# Step 02: 顶层物理路由架构 (MaterialPhysicsMode) 与启动期 Fail-Closed 校验

## 1. 目标与背景

生产输运必须存在明确、互斥、fail-closed 的两条路径：
1. **Water transport**：未加载 CT 或明确配置为水体模时使用。继续使用当前冻结的水输运、水停止功率、水 MCS、水 H/O 核反应率和水 CINEL02 事件回放。不得加载或访问 Schneider section LUT。不得改变任何水物理参数。水回归测试结果必须绝对保持不变。
2. **CT transport**：一旦启用 CT grid，必须直接进入 Schneider material transport。体素 material ID 必须严格解释为 Schneider section ID 0–24。严禁在生产 CT 路径中使用四分类（air/lung/soft/bone），严禁回退到水 H/O 反应率，严禁将 C、N、Ca 等目标核 alias 到 O。

当 CT 启用时，若 Schneider 数据缺失、SHA 不匹配、目标核或射弹覆盖不全，必须在启动阶段立即终止，绝不允许运行到 GPU kernel 后静默忽略。

## 2. 涉及文件

- 修改：`include/carbon/transport_config.hpp`
- 修改：`src/config.cpp`
- 修改：`include/carbon/transport.hpp`
- 单元测试：`tests/carbon_tests.cpp`

## 3. 详细技术规范

### 3.1 定义显式物理模式枚举
在 `include/carbon/transport_config.hpp` 中定义：
```cpp
enum class MaterialPhysicsMode : uint8_t {
    Water = 0,
    SchneiderCt = 1
};
```

### 3.2 模式判定规则
在主机端配置解析阶段一次性确定模式，禁止在运行时通过多个散乱 bool 隐式推断：
```cpp
MaterialPhysicsMode determine_physics_mode(const TransportConfig& cfg, bool has_ct_grid) {
    if (!has_ct_grid) {
        return MaterialPhysicsMode::Water;
    }
    // 启用了 CT grid，必须是 SchneiderCt
    return MaterialPhysicsMode::SchneiderCt;
}
```

### 3.3 统一配置键与区分
只建立一套规范 key，明确区分水与 Schneider 相关配置：
- `water_cinel_package_file`
- `water_reaction_rate_file`
- `ct_schneider_c12_cinel03_file`
- `ct_schneider_secondary_cinel03_file`
- `ct_schneider_primary_rate_file`
- `ct_schneider_secondary_rate_file`
- `ct_schneider_stopping_power_file`
- `ct_schneider_radiation_length_file`

### 3.4 启动阶段 Fail-Closed 检查
当 `mode == MaterialPhysicsMode::SchneiderCt` 时，在提交 GPU kernel 前执行全套预检：
1. 文件存在性检查。
2. 二进制格式 Header 与 Schema 校验。
3. 伴随元数据存在性检查。
4. 二进制文件实际 SHA-256 与 metadata 绑定检查。
5. Schneider 源文件 SHA-256 绑定检查。
6. 13 个射弹核素（Z=1..6 及对应同位素）覆盖检查。
7. 13 个规范目标元素（H, C, N, O, Na, Mg, P, S, Cl, Ar, K, Ca, Fe）覆盖检查。
8. 能量网格边界与步长一致性检查。
9. Primary 与 Secondary 截面与 CINEL03 包互相一致性检查。
任何一项校验失败，立即抛出 `std::runtime_error` 或特定异常，终止程序，严禁回退到水物理。

## 4. 验收标准

1. 单元测试覆盖：
   - 无 CT 输入 -> 模式为 `Water`。
   - 有 CT 输入 -> 模式为 `SchneiderCt`。
   - 有 CT 输入但缺少 Schneider 数据 -> 在启动时抛出致命异常。
   - 生产 CT 路径绝不触发四分类转换函数。
2. 水物理回归测试：
   - 在固定随机数种子下，水体模的初级射程、能量沉积、碎片能谱与既有冻结基线逐 bit 一致。

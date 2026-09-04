# Step 06: 严格能量分类记账、消除伪造局部沉积与次级溢出熔断

## 1. 目标与背景

在粒子输运与核反应中，能量守恒是物理自洽性的根本底线。
此前存在的危险操作包括：
`local_deposit += fragment_kinetic_sum;`
把原本应输运或未被追踪的次级碎片动能强行作为碰撞点的局部能量沉积，人为制造总能量“守恒”的假象。

本步骤必须：
1. 严禁任何伪造局部沉积的代码。
2. 建立清晰的 8 分类能量账本。
3. 对次级粒子队列溢出（queue overflow）执行严格计数、动能统计与质量门禁熔断。

## 2. 涉及文件

- 修改：`include/carbon/transport_config.hpp`
- 修改：`src/transport_sycl.cpp`
- 修改：`src/detail/sycl_score_device.inc`
- 单元测试：`tests/carbon_tests.cpp`

## 3. 详细技术规范

### 3.1 消除伪造局部沉积
全面排查并删除所有无物理依据的动能强制局部累加代码，例如：
```cpp
// 严格禁止：
// local_deposit += fragment_kinetic_sum;
```
局部能量沉积（process local deposit）仅能记录核反应中由于核激发、极短程反冲核等由 Geant4/TOPAS 明确定义的 `GetLocalEnergyDeposit()`，绝不能包含已被实例化生成的次级碎片动能。

### 3.2 8 分类能量账本架构
在设备端与主机端统一维护能量流向统计：
1. `E_continuous_ionizing`: 带电粒子沿轨迹电离损失连续沉积
2. `E_nuclear_local`: 核反应过程本身的局部沉积 (process local deposit)
3. `E_transported_secondaries`: 成功压入队列并实际输运的次级带电碎片总初动能
4. `E_escaped_charged`: 飞出体模/边界的带电粒子携带动能
5. `E_neutral`: 中子、光子等中性产物携带动能
6. `E_cutoff_kill`: 低于传输能量阈值被终止粒子的残余动能
7. `E_unsupported`: 因未受支持核素而未进入输运的动能
8. `E_queue_overflow`: 因次级队列满而被丢弃的粒子动能

### 3.3 次级队列溢出严格处理 (Overflow Guard)
当次级队列无法容纳新粒子时：
1. 增加原子计数器：`queue_overflow_count++`。
2. 将被丢弃粒子的动能累加至：`E_queue_overflow += kinetic_energy`。
3. 在批次结束时，若 `queue_overflow_count > 0`：
   - 输出严重警告并使该 validation shard 标记为无效。
   - 生产任务触发异常，禁止将含有溢出的数据并入最终结果。
   - 自动指导减小单片粒子历史数（sharding）重跑。

## 4. 验收标准

1. 代码中不再包含任何强制把碎片动能记为局部沉积的语句。
2. 能量闭环校验：
   `E_initial = E_continuous + E_nuclear_local + E_escaped + E_neutral + E_cutoff + E_unsupported + E_overflow + Q_reaction`
   各项分类明确，账目清楚。
3. 队列溢出测试：人工构造超小队列，确认 overflow 被准确捕获、动能准确统计并正确导致质量门禁失败。

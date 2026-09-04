# Step 04: 生产端 CT 初级核反应：体素面步进、偏反应率抽样与 CINEL03 回放

## 1. 目标与背景

在生产 CT 输运中，初级 C12 核反应必须严格依据体素实际介质的元素组分与密度进行偏反应率计算，且光深必须跨体素连续累积，步长必须受体素面截断。

本步骤必须：
在生产级 `src/transport_sycl.cpp` / `src/detail/sycl_inelastic_device.inc` 中实现符合物理规范的初级核反应逻辑，彻底取代旧四分类和硬编码回退。

## 2. 涉及文件

- 修改：`src/transport_sycl.cpp`
- 修改：`src/detail/sycl_inelastic_device.inc`
- 修改/引用：`include/carbon/inelastic.hpp`

## 3. 详细技术规范

### 3.1 体素参数获取与偏反应率计算
在体素 `(ix, iy, iz)` 中：
```cpp
uint8_t section = voxel.section_id; // 0..24
float rho = voxel.density;          // g/cm^3
float E = projectile_energy_per_nucleon; // MeV/u

float total_rate = 0.0f;
float partial_rates[13];
for (int t = 0; t < 13; ++t) {
    // 密度只能乘一次！
    partial_rates[t] = rho * SchneiderPrimaryMassPartialRate(section, t, E);
    total_rate += partial_rates[t];
}
```

### 3.2 步长控制与光深连续累加
步长必须由连续能损步长、体素面距离和核光深耗尽距离三者共同限制：
```cpp
step = min({
    condensed_history_step,
    distance_to_voxel_face,
    remaining_optical_depth / total_rate
});
```
- 粒子向前推进 `step` 后，消耗光深 `delta_tau = total_rate * step`，更新 `remaining_optical_depth -= delta_tau`。
- **跨越体素面时**：当粒子由于到达体素面而终止本步推进时，更新体素索引，保留剩余的 `remaining_optical_depth`，进入新体素后用新 `section`、新 `rho` 和当前 `E` 重新计算 `total_rate`，**严禁重新随机采样新的光深**！

### 3.3 碰撞目标核抽样与 CINEL03 回放
当 `remaining_optical_depth <= 0` 时，触发核非弹性碰撞：
1. 依概率分布 `P(target = t) = partial_rates[t] / total_rate` 进行离散抽样（categorical sampling），获得目标元素序号及对应的原子序数 `target_Z`。
2. 查询 CINEL03 事件库：
   ```cpp
   event = cinel03_find_event_device(c12_package, projectile_Z=6, projectile_A=12, target_Z, E);
   ```
3. **Fail-Closed 异常处理**：
   - 若未检索到有效事件，生产模式下必须触发诊断计数器 `event_lookup_miss++`，并终止该粒子/标记批次无效；
   - 严禁静默回退为弹性继续输运；
   - 严禁借用其他 target 的事件；
   - 严禁回退到水 CINEL02 事件包；
   - 严禁将 C、N、Ca 等目标核 alias 到 O。
4. 将回放产生的有效带电次级产物压入次级粒子堆栈。

## 4. 验收标准

1. 编译通过并在核反应步进中验证体素面截断生效。
2. 单元测试验证：
   - 改变体素密度时，反应截面线性缩放，且密度绝未被二次乘方。
   - 跨越体素面时光深连续，无额外光学突变。
   - 抽样目标核分布与 13 目标偏反应率精确相符。
   - 缺失事件时明确上报 miss 并不产生错误物理产物。

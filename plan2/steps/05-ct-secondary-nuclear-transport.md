# Step 05: 生产端 CT 次级核反应：彻底解耦初级目标核与全关联次级输运

## 1. 目标与背景

当前代码（尤其是 `tools/run_step20_gpu.cpp` 等旧 runner）存在致命物理逻辑错误：
在次级粒子发生核反应时，直接复用了初级碰撞采样的 `target_z`。这在物理上完全错误——次级粒子在输运数毫米甚至厘米后，在新的位置与介质发生碰撞，必须根据次级粒子自身的能量与介质在该处的元素偏反应率独立重新抽样目标核！

本步骤必须：
在生产主干 `src/transport_sycl.cpp` 中实现真正的次级核反应级联，为每一个次级带电粒子独立采样目标核，回放次级 CINEL03 事件，并维持 Be6 的冻结语义。

## 2. 涉及文件

- 修改：`src/transport_sycl.cpp`
- 修改：`src/detail/sycl_inelastic_device.inc`
- 修改：`src/detail/sycl_cinel02_device.inc` (或对应 cinel03 include)
- 清理纠正：`tools/run_step20_gpu.cpp`

## 3. 详细技术规范

### 3.1 独立次级目标核抽样
对于次级队列中提取出的带电粒子（projectile: `pz, pa`）：
1. 计算次级粒子当前比动能 `E = kinetic_energy / pa`。
2. 在当前体素 `section` 中，读取该次级核素对应的 13 目标偏反应率：
   ```cpp
   float sec_partial_rates[13];
   float sec_total_rate = 0.0f;
   for (int t = 0; t < 13; ++t) {
       sec_partial_rates[t] = rho * SecondaryPartialRate(pz, pa, section, t, E);
       sec_total_rate += sec_partial_rates[t];
   }
   ```
3. 发生核反应时，**独立随机抽样**次级碰撞的目标核：
   ```cpp
   uint8_t sec_target_Z = sample_target_categorical(sec_partial_rates, sec_total_rate, rng);
   ```
   **严禁从初级相互作用上下文传递任何 target_Z**。彻底删除 `tools/run_step20_gpu.cpp` 中的历史复用代码。

### 3.2 次级 CINEL03 事件检索与回放
1. 使用次级自身的 `(pz, pa, sec_target_Z, E)` 查询次级 CINEL03 事件包。
2. 回放关联事件中的全部产物，正确转换到世界坐标系。
3. 可输运的带电产物进入正式 cascade 队列，更新 generation（代数）。

### 3.3 Be6 冻结语义保护 (TopasCompatKill)
- 遇到 `Be6`（即 Z=4, A=6）时，必须执行冻结的 `TopasCompatKill`：直接终止该粒子并在局部记账（或按照与基线一致的方式处理）。
- 不得将 Be6 视为常规缺失包的错误。
- 保持水基线中的 Be6 行为不变。

### 3.4 异常拦截与未支持核素处理
- 若次级核素不受支持（如 Z>6 且未在次级截面表中声明），记录未支持核素计数 `unsupported_projectile++`，并计入能量账本，直接终止，绝不映射到 proton 或其他任意同位素。
- 若检索不到 CINEL03 事件，严格报告 miss 并终止，使质量门禁失败。

## 4. 验收标准

1. 静态代码检查确认 `transport_sycl.cpp` 中次级反应循环无任何来自外部/初级的 `target_z` 引用。
2. 单元测试验证：针对同一初级事件生成的多个次级碎片，其后续核反应目标核在统计上独立且符合偏反应率权重。
3. Be6 触发 `TopasCompatKill`，测试通过。

# Step 03: 生产级显存缓冲配置与 Schneider 数据显存上传

## 1. 目标与背景

当前 Schneider target selection、CINEL03 回放与 secondary rates 仅存在于独立 runner（如 `tools/run_step19_gpu.cpp` 等），真正的 `src/transport_sycl.cpp` 并没有把这套数据结构配置进主输运 kernel。

本步骤必须：
在生产路径主函数 `transport_sycl()` 中，当模式为 `SchneiderCt` 时，将全部 Schneider 数据表与 CINEL03 二进制包分配 SYCL device buffer 并上传，将设备端结构指针/句柄传递给 GPU 输运 kernel。

## 2. 涉及文件

- 修改：`src/transport_sycl.cpp`
- 修改：`src/io.cpp`
- 修改：`include/carbon/transport.hpp`
- 涉及/包含：`src/detail/sycl_transport_context_impl.inc` 等设备上下文结构

## 3. 详细技术规范

### 3.1 主机端加载全部 Schneider 数据集
在 `transport_sycl()` 主机端准备阶段：
若 `mode == MaterialPhysicsMode::SchneiderCt`：
1. 加载初级核反应偏反应率表：`schneider_inelastic_rates_v1.bin`
2. 加载 C12 目标元素 CINEL03 包：`cinel03_c12_targets.bin`
3. 加载次级粒子核反应偏反应率张量：`secondary_inelastic_rates_v1.bin`
4. 加载次级全物种 CINEL03 包：`cinel03_secondary_targets.bin`
5. 加载 25-section Schneider 停止功率表：`schneider_stopping_v1.bin`
6. 加载 25-section Schneider 辐射长度/MCS 查找表。

### 3.2 SYCL 显存分配与上传
使用 SYCL device buffer 或 USM device memory：
- 创建对应的 device-accessible 内存块。
- 安全拷贝主机内存数据至设备显存。
- 组装 `SchneiderDeviceContext` 或将其注入生产 `TransportKernelContext`。
- 确保在整个模拟生命周期内显存不被提前释放。

### 3.3 模式标志传递
将 `MaterialPhysicsMode` 作为常量参数传入 kernel lambda，使 GPU 端代码可以根据该模式走编译期或轻量分支路由，完全杜绝运行时不确定性。

## 4. 验收标准

1. 编译通过，无 SYCL 内存生命周期或类型转换警告。
2. 编写测试确认当 `SchneiderCt` 开启时，显存分配成功且数据内容与主机端二进制校验和完全匹配。
3. 当 `Water` 模式时，Schneider 相关显存缓冲不进行任何分配，显存占用保持最小。

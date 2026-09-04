# Step 08: CINEL03 限制加固、元数据出处链补齐与动态路径推导

## 1. 目标与背景

1. **CINEL03 保护不足**：
   - 产物数量转为 uint32 offset 时缺乏上限保护；
   - 交互、事件、节点数超过 uint32 范围时缺乏设备端上传前阻断；
   - offset + count 计算可能发生整数溢出；
   - 元数据未与包自身 SHA-256 紧密绑定。
2. **元数据路径硬编码**：
   - `tools/run_step13_gpu.cpp` 将 stopping metadata 路径硬编码为 `data/schneider/schneider_stopping_v1.metadata.json`，若用户配置了其他路径则产生脱钩甚至 hash 不匹配。
3. **出处链不完整**：
   - Step 20/21 数据元数据存在 placeholder、unknown、空数组或缺失字段。

本步骤必须：
全面加固 CINEL03 上限检查与溢出防护；消除所有硬编码元数据路径；补齐全部必填 provenance 字段。

## 2. 涉及文件

- 修改：CINEL03 编解码与加载逻辑（`include/carbon/*cinel03*`、`src/*cinel03*` 或相关 include）
- 修改：`tools/run_step13_gpu.cpp`
- 修复并验证元数据：`data/schneider/*.metadata.json`、`data/packages/*.metadata.json`

## 3. 详细技术规范

### 3.1 CINEL03 边界与溢出防御
在加载与准备显存上传阶段：
1. 校验 `product_count`：若单个事件产物数超出预设上限（例如 64），抛出异常。
2. 校验总事件数、节点数、交互数：必须严格处于 uint32 范围内，严禁发生向 uint32 强转导致的截断。
3. 对 `offset + count` 采用防溢出形式校验：
   ```cpp
   if (offset > UINT32_MAX - count || offset + count > total_products) {
       throw std::runtime_error("CINEL03 product range overflow or out of bounds");
   }
   ```
4. 计算包文件 SHA-256 并与伴随元数据声明的比对，不一致直接拒绝上传。

### 3.2 动态推导元数据路径
在 `tools/run_step13_gpu.cpp` 及其他工具中，严禁硬编码伴随元数据路径。改为：
```cpp
std::string metadata_file = cfg.ct_schneider_stopping_power_file + ".metadata.json";
if (!std::filesystem::exists(metadata_file)) {
    // 尝试去除后缀替换为 .metadata.json 或通过配置显式指定
    metadata_file = replace_extension(cfg.ct_schneider_stopping_power_file, ".metadata.json");
}
```
确保 hash 的是与当前实际 binary 文件真正配对的 sidecar。

### 3.3 补齐完整 Provenance 元数据字段
检查并补齐所有 Step 20/21 依赖的数据元数据，必须全部包含：
- `topas_version` (如 "4.2.p3")
- `geant4_version` (如 "11.3.2")
- `physics_list` (如 "g4ion-inclxx")
- `schneider_source_file` 及其实际 SHA-256
- `extractor_commit_sha`
- `compiler_commit_sha`
- `raw_campaign_manifest_hash`
- `energy_grid` (min, max, step)
- `projectiles` 完整核素列表
- `target_elements` 完整 13 元素列表
- `units` (MeV, mm, g/cm^3, etc.)
- `generation_timestamp`
- `validation_report_hash`
绝对禁止出现 "placeholder"、"unknown"、空数组或手工随意填写的假哈希。

## 4. 验收标准

1. 单元测试验证：CINEL03 偏移溢出、超大计数与损坏索引被准确捕获并抛出异常。
2. 校验脚本检查所有数据目录下的 `.metadata.json`，100% 字段完整，且计算出的 binary SHA-256 与文件完全一致。

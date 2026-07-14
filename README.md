# carbon-oneapi-mc

面向 Intel oneAPI/SYCL 的碳离子水中一维 condensed-history 蒙特卡洛剂量引擎。当前阶段聚焦 200 MeV/u C-12、均匀水模体和 pristine Bragg curve，并以 TOPAS/Geant4 为参考。

> 研究用途：当前阻止本领模型和输运结果尚未完成 TOPAS 验证，不能用于临床或治疗计划。

## 当前能力

- C++20 一维 CPU CSDA 输运；
- MeV/u 与 C-12 总动能的显式转换；
- CSV 阻止本领表及线性插值；
- 自适应步长和深度 bin 边界限制；
- 每初级粒子的能量沉积与剂量 CSV；
- 可选 Intel oneAPI/SYCL 后端；
- CPU/SYCL 共用配置和物理数据；
- 无第三方 C++ 测试依赖的基础测试。

## WSL 构建

在 VS Code 的 WSL 窗口中打开本目录。CPU 调试构建：

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug
./build/cpu-debug/carbon_mc --config config/beam_200MeVu.yaml --device serial
```

加载 Intel oneAPI 环境后构建 SYCL 版本：

```bash
source /opt/intel/oneapi/setvars.sh
cmake --preset oneapi-release
cmake --build --preset oneapi-release
ctest --preset oneapi-release
./build/oneapi-release/carbon_mc --config config/beam_200MeVu.yaml --device gpu
```

如果 oneAPI 不在 `/opt/intel/oneapi`，请在运行上述命令前加载实际安装位置的 `setvars.sh`；项目本身不硬编码 oneAPI 路径。

## 输出

默认写入 `out/cpu_depth_dose.csv`，列为：

```text
depth_mm,energy_deposition_MeV_per_primary,dose_Gy_per_primary,relative_dose
```

剂量按配置中的 scorer 横截面积、水密度和深度 bin 质量计算。绝对剂量比较时，TOPAS 必须使用完全相同的 scorer 体素体积。

## 物理数据状态

`data/stopping_power_water.csv` 是用于软件联调的透明、可再生 Bethe-Bloch + 有效电荷近似表，不是 ICRU 或 Geant4 参考数据。第一个物理校准任务是从可信数据源生成正式表，并用 TOPAS 的 R80 对它进行验证；禁止通过逐能量手工调参替代该步骤。


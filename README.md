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

## 已验证环境与当前结果

本仓库已在以下本机 WSL 环境实际验证：

- Debian WSL2；
- GCC 12.2 CPU 构建；
- Intel oneAPI DPC++ 2026.1；
- Intel OpenCL CPU device（Core i5-14600K）；
- TOPAS 4.2.p3 / Geant4 11.3.p2。

当前 WSL 的 `sycl-ls` 尚未列出 Level Zero GPU，因此 SYCL kernel 已通过 `icpx` 编译，但只在 SYCL CPU 设备上实际执行。10,000-history TOPAS 开发基准与 CSDA 的比较为：R80 差 `+0.075 mm`、FWHM 相对差 `-29.9%`、峰值差 `+69.4%`、尾积分差 `-89.9%`。射程已经接近，峰宽、峰高和碎裂尾部仍是后续物理模块的工作，不应把这些差异解释为最终准确度。

另有 10,000-history 的 TOPAS 电磁物理隔离基准。使用一次性全局校准 `straggling_scale=1.2` 后，Level 2 结果为：R80 差 `+0.105 mm`、FWHM 相对差 `+2.29%`、峰值差 `+0.12%`、2%/2 mm gamma `97.28%`。后续 100–400 MeV/u 验证必须固定此参数。

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

VS Code 可直接使用仓库中的 CMake Presets。推荐在 Remote WSL 窗口中选择 `cpu-debug` 或 `oneapi-release` preset；`.vscode` 中已包含扩展建议和 preset 设置。

## 输出

默认写入 `out/cpu_depth_dose.csv`，列为：

```text
depth_mm,energy_deposition_MeV_per_primary,dose_Gy_per_primary,relative_dose
```

剂量按配置中的 scorer 横截面积、水密度和深度 bin 质量计算。绝对剂量比较时，TOPAS 必须使用完全相同的 scorer 体素体积。

## 物理数据状态

`data/stopping_power_water.csv` 是用于软件联调的透明、可再生 Bethe-Bloch + 有效电荷近似表，不是 ICRU 或 Geant4 参考数据。第一个物理校准任务是从可信数据源生成正式表，并用 TOPAS 的 R80 对它进行验证；禁止通过逐能量手工调参替代该步骤。

TOPAS 开发参考曲线、版本元数据和当前 CSDA 指标位于 `validation/results/`。原始 TOPAS scorer 文件和完整运行日志位于忽略目录 `validation/topas/output/`。

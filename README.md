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

运行架构已明确分开：WSL 仅运行 TOPAS/Geant4 并生成参考数据；Windows 原生 oneAPI 通过 Level Zero 驱动 Intel Arc B580 执行 SYCL kernel。10,000-history TOPAS 开发基准与 CSDA 的比较为：R80 差 `+0.075 mm`、FWHM 相对差 `-29.9%`、峰值差 `+69.4%`、尾积分差 `-89.9%`。射程已经接近，峰宽、峰高和碎裂尾部仍是后续物理模块的工作，不应把这些差异解释为最终准确度。

另有 10,000-history 的 TOPAS 电磁物理隔离基准。使用一次性全局校准 `straggling_scale=1.2` 后，Level 2 结果为：R80 差 `+0.105 mm`、FWHM 相对差 `+2.29%`、峰值差 `+0.12%`、2%/2 mm gamma `97.28%`。后续 100–400 MeV/u 验证必须固定此参数。

Level 3 使用从 TOPAS primary-C12 生存代理拟合的有效宏观衰减系数 `0.0050613 mm^-1`。相对 primary-C12 scorer，峰值差 `+4.94%`、FWHM 差 `-1.76%`、R80 差 `+0.101 mm`、2%/2 mm gamma `98.29%`。这仍是过渡模型：反应后的剩余能量只记入 `untracked_nuclear_energy`，尚未生成碎片；完整 TOPAS 的尾积分因此仍低约 `92.4%`。

项目现已通过 TOPAS 自定义计分器直接查询同一 Geant4 物理列表中的 C-12 非弹性截面，得到 1--400 MeV/u 的 H、O 微观截面和水中宏观截面表。200 MeV/u 时水中宏观截面为 `0.00474216 mm^-1`，平均自由程为 `210.874 mm`。下一版 GPU 输运应读取该能量相关表，替换上述拟合常数。反应末态使用事件级 n-tuple；相同 `reaction_id` 的碎片保持多重性、能量和方向相关性并作为整体采样。

## WSL 构建

WSL 主要用于 TOPAS；也可在 VS Code 的 WSL 窗口中进行 CPU 调试构建：

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug
./build/cpu-debug/carbon_mc --config config/beam_200MeVu.yaml --device serial
```

如需在 WSL 中构建 SYCL CPU 版本：

```bash
source /opt/intel/oneapi/setvars.sh
cmake --preset oneapi-release
cmake --build --preset oneapi-release
ctest --preset oneapi-release
./build/oneapi-release/carbon_mc --config config/beam_200MeVu.yaml --device gpu
```

如果 oneAPI 不在 `/opt/intel/oneapi`，请在运行上述命令前加载实际安装位置的 `setvars.sh`；项目本身不硬编码 oneAPI 路径。

## Windows Arc B580 构建与运行

在 Windows VS Code 终端中运行：

```bat
scripts\build_windows_oneapi.cmd
scripts\run_windows_b580.cmd
```

脚本会依次初始化 Visual Studio 2026 C++ 工具链与 Intel oneAPI，并使用独立的 `oneapi-windows-release` preset/build 目录。B580 被限制到 `ONEAPI_DEVICE_SELECTOR=level_zero:0`，不会误选 OpenCL CPU。若 oneAPI 安装在其他位置，可预先设置 `ONEAPI_SETVARS`。

100,000-history Windows 原生实测结果：

- serial：`14,287 histories/s`；
- SYCL CPU：`27,265 histories/s`；
- Arc B580 Level Zero：`110,601 histories/s`；
- B580 相对 serial：`7.74x`；
- B580 相对 SYCL CPU：`4.06x`。

为避免大统计量下 float 原子累积误差，SYCL dose scorer 使用 double atomic。B580 与 serial 的 100k 曲线 NRMSE 为 `1.16e-6`，R80 差 `2.1e-5 mm`，2%/2 mm gamma 为 `100%`。

VS Code 可直接使用仓库中的 CMake Presets；`.vscode` 中已包含扩展、preset 设置和 Windows 构建/运行任务。

## 输出

默认写入 `out/cpu_depth_dose.csv`，列为：

```text
depth_mm,energy_deposition_MeV_per_primary,dose_Gy_per_primary,relative_dose
```

剂量按配置中的 scorer 横截面积、水密度和深度 bin 质量计算。绝对剂量比较时，TOPAS 必须使用完全相同的 scorer 体素体积。

## 物理数据状态

`data/stopping_power_water.csv` 是用于软件联调的透明、可再生 Bethe-Bloch + 有效电荷近似表，不是 ICRU 或 Geant4 参考数据。第一个物理校准任务是从可信数据源生成正式表，并用 TOPAS 的 R80 对它进行验证；禁止通过逐能量手工调参替代该步骤。

`data/c12_inelastic_cross_sections_water_geant4_11_3_2.csv` 是由项目内的 TOPAS 扩展在物理初始化后直接调用 `G4HadronicProcessStore` 生成的截面表；配套 JSON 记录 TOPAS/Geant4 版本、原始文件哈希和 200 MeV/u 参考点。它不是从 IDD 衰减曲线拟合得到的数据。

TOPAS 开发参考曲线、版本元数据和当前 CSDA 指标位于 `validation/results/`。原始 TOPAS scorer 文件和完整运行日志位于忽略目录 `validation/topas/output/`。

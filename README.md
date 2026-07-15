# carbon-oneapi-mc

面向 Intel oneAPI/SYCL 的碳离子水中 condensed-history 蒙特卡洛剂量引擎。主 C-12 与带电碎片已支持三维方向、x/y/z voxel 边界、Highland 多重库仑散射和 `60 x 60 x 800` GPU total dose。

> 研究用途：当前阻止本领模型和输运结果尚未完成 TOPAS 验证，不能用于临床或治疗计划。

## 当前能力

- C++20 一维 CPU CSDA 输运；
- MeV/u 与 C-12 总动能的显式转换；
- CSV 阻止本领表及线性插值；
- 自适应步长和深度 bin 边界限制；
- 每初级粒子的能量沉积与剂量 CSV；
- 可选 Intel oneAPI/SYCL 后端；
- CPU/SYCL 共用配置和物理数据；
- 从 TOPAS/Geant4 直接导出的能量相关 C-12 水中非弹性截面；
- TOPAS 事件级反应包的 GPU 联合采样和固定容量带电次级队列；
- 可选 `60 x 60 x 800` double-atomic total voxel tally、稀疏 CSV 和 voxel→IDD 闭合检查；
- 主 C-12 和所有输运带电碎片的逐步 Highland 投影角采样与局部三维方向更新；
- 无第三方 C++ 测试依赖的基础测试。

## 已验证环境与当前结果

当前计算架构已经实际验证：

- 远程 `v@192.168.31.5:~/gpu`：TOPAS 4.1.p1 / Geant4 11.1.3 MT，最多 56 线程；
- Windows：Visual Studio 2026、Intel oneAPI DPC++ 2026.1；
- Windows Level Zero：Intel Arc B580；
- WSL：仅用于历史开发和数据访问辅助，不再运行新的 TOPAS 作业。

新的 TOPAS 脚本只传到远程 `~/gpu` 并在那里运行；Windows 原生 oneAPI 通过 Level Zero 驱动 Arc B580 执行 SYCL kernel。禁止在 WSL 提交新的 TOPAS 计算。

另有 10,000-history 的 TOPAS 电磁物理隔离基准。使用一次性全局校准 `straggling_scale=1.2` 后，Level 2 结果为：R80 差 `+0.105 mm`、FWHM 相对差 `+2.29%`、峰值差 `+0.12%`、2%/2 mm gamma `97.28%`。后续 100–400 MeV/u 验证必须固定此参数。

Level 3 使用从 TOPAS primary-C12 生存代理拟合的有效宏观衰减系数 `0.0050613 mm^-1`。相对 primary-C12 scorer，峰值差 `+4.94%`、FWHM 差 `-1.76%`、R80 差 `+0.101 mm`、2%/2 mm gamma `98.29%`。这仍是过渡模型：反应后的剩余能量只记入 `untracked_nuclear_energy`，尚未生成碎片；完整 TOPAS 的尾积分因此仍低约 `92.4%`。

项目现已通过 TOPAS 自定义计分器直接查询同一 Geant4 物理列表中的 C-12 非弹性截面，得到 1--400 MeV/u 的 H、O 微观截面和水中宏观截面表。200 MeV/u 时水中宏观截面为 `0.00474216 mm^-1`，平均自由程为 `210.874 mm`。CPU 和 SYCL 输运均已按当前能量插值该表并替换上述拟合常数。反应末态使用事件级 n-tuple；相同 `reaction_id` 的碎片保持多重性、能量和方向相关性并作为整体采样。

100,000-history `fragment-development` 正式反应包已固化：37,657 次主 C-12 非弹性反应、330,659 个直接次级粒子，反应率 `37.657%`，平均多重性 `8.781`。两个没有直接可见次级粒子的低能反应以零长度反应包保留。压缩表、运行版本、耗时、输入/输出哈希及完整闭合检查记录在 `validation/results/topas_200MeVu_reaction_packages_development.metadata.json`。

为避免 Windows oneAPI 可执行文件依赖 zlib 或在运行时解析 CSV，`validation/scripts/compile_reaction_package.py` 会把两个 gzip 表编译为版本化的小端定长二进制表。当前 5.9 MB 二进制包含 201 个 1 MeV/u 分箱、37,657 个反应头和 330,659 个精简次级粒子记录，可由 `ReactionPackageTable::from_binary` 严格校验后直接复制到 GPU。GCC 12.2 和 Windows IntelLLVM 2025.3.3 均已通过真实数据加载测试。

Arc B580 的固定容量 secondary-generation queue 已接入 primary kernel。10,000-history 实测的 3,816 次核反应全部按能量分箱采样完整反应包，生成 33,260 个直接次级粒子；其中 21,357 个带电离子整包写入队列，溢出为 0，中子/光子能量单独记账。generation-only 输出与 primary-only 的物理列逐 bin 数值一致；CSV 字节哈希仅因当前固定 LF 与旧 CRLF 参考不同。10k 耗时对 SYCL JIT/磁盘缓存敏感，不用于正式性能比较。

生成阶段可在初始化 oneAPI 环境后运行：

```bat
set ONEAPI_DEVICE_SELECTOR=level_zero:0
build\oneapi-windows-release\carbon_mc.exe --config config\beam_200MeVu_fragment_generation.yaml --device gpu
```

可复现的 B580 版本、输入哈希、队列统计和 primary-only 回归结果记录在 `validation/results/windows_b580_secondary_generation_10k.metadata.json`。

第一版带电碎片输运也已接入。它按采样的 `direction_z` 做正/反向一维 CSDA，并在相同 MeV/u 下用有效电荷平方比从 C-12 表缩放任意 A/Z 离子的停止本领。100,000-history B580 运行输运 216,133 个带电次级粒子和 316,048,099 个碎片步进，队列无溢出，预热后吞吐为 `95,985 histories/s`，总能量误差为 `3.11e-8`。

相对完整 TOPAS，总 IDD 的积分差为 `+0.48%`、峰值差 `-0.062%`、R80 差 `+0.104 mm`、FWHM 差 `+2.36%`、NRMSE `1.08%`、2%/2 mm gamma `97.71%`。峰后尾积分差由未输运碎片时约 `-92%` 改善为 `+10.055%`，非常接近但尚未通过 `<10%` 验收线；禁止为跨线而手工调参。

TOPAS 祖先归属 scorer、带电碎片两代级联、三维方向和多重散射现已完成。100k B580/TOPAS charged-origin 绝对三维比较（无全局 scale）得到：全深度/90 mm 后积分差 `-0.206%/+3.13%`，高剂量 voxel 平均绝对差 `3.24%`、Pearson `0.974`；20/50/86.75 mm 横向 σ 与 TOPAS 的差均小于约 `0.06 mm`。下一任务是 3D 祖先类别 voxel 闭合和中性来源输运。

```bat
set ONEAPI_DEVICE_SELECTOR=level_zero:0
build\oneapi-windows-release\carbon_mc.exe --config config\beam_200MeVu_fragment_transport_100k.yaml --device gpu
```

直接截面版本的 10,000-history 原生 B580 验证得到 3,816 次核反应，serial 得到 3,818 次；曲线 NRMSE 为 `1.38e-5`、R80 差 `-3.9e-5 mm`、1%/1 mm 与 2%/2 mm gamma 均为 `100%`。这两次事件差异来自 SYCL float 与 serial double 的采样边界，不影响当前剂量曲线一致性。

## WSL CPU 调试构建

WSL 只保留 CPU/SYCL 调试和数据访问用途，不运行新的 TOPAS 作业。可在 VS Code 的 WSL 窗口中构建：

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

启用 `enable_voxel_scoring: true` 后，还会写出只包含非零体素的稀疏 CSV：

```text
ix,iy,iz,x_mm,y_mm,z_mm,energy_deposition_MeV_per_primary,dose_Gy_per_primary
```

Arc B580 的 MCS 100-history smoke 和 100k 正式三维基准可直接运行：

```bat
build\oneapi-windows-release\carbon_mc.exe --config config\beam_200MeVu_mcs_voxel_smoke.yaml
build\oneapi-windows-release\carbon_mc.exe --config config\beam_200MeVu_mcs_voxel_100k.yaml
```

正式比较命令：

```bat
python validation\scripts\compare_3d_dose.py ^
  --gpu-voxel validation\results\windows_b580_mcs_3d_100k_voxels.csv ^
  --gpu-idd validation\results\windows_b580_mcs_3d_100k_idd.csv ^
  --topas-npz validation\results\topas_200MeVu_ancestor_dose_3d_development.npz ^
  --output-metrics validation\results\windows_b580_mcs_3d_100k_vs_topas.metrics.json ^
  --output-plot validation\results\windows_b580_mcs_3d_100k_vs_topas.png
```

比较口径是 GPU transported charged total 对 TOPAS 前八个 charged-origin 祖先类别之和，单位均为绝对 `MeV/primary/voxel`，禁止全局 scale。100k 运行 voxel→IDD 最大闭合误差为 `1.71e-10 MeV/primary/bin`。

`enable_multiple_scattering: true` 使用水的辐射长度 `36.08 g/cm²` 和 Lynch–Dahl/Highland 投影 RMS 角宽；每步采样两个独立高斯横向斜率并旋转到当前母粒子局部基。它没有拟合散射 scale；当前在步末施加方向 kick，尚未采样步内相关横向位移，因此仍需用步长收敛、亚毫米网格和多能量束宽验证其适用性。

三维开发数据使用 binary v2：每个 reaction/cascade 产物保存相对入射母粒子的局部
`direction_x/y/z`。加载器继续接受 v1，并把缺失的横向方向置为 NaN，避免误当成
零偏转。`beam_200MeVu_voxel_smoke.yaml` 使用独立 `_3d.bin`；正式一维基准仍使用
原 v1 文件。v2 的 Arc B580 100-history smoke 能量平衡误差为 `3.43e-8`，800 个
voxel z 层逐 bin 还原 IDD 的最大误差为 `0 MeV/primary/bin`。

## 物理数据状态

`data/stopping_power_water.csv` 是用于软件联调的透明、可再生 Bethe-Bloch + 有效电荷近似表，不是 ICRU 或 Geant4 参考数据。第一个物理校准任务是从可信数据源生成正式表，并用 TOPAS 的 R80 对它进行验证；禁止通过逐能量手工调参替代该步骤。

`data/c12_inelastic_cross_sections_water_geant4_11_3_2.csv` 是由项目内的 TOPAS 扩展在物理初始化后直接调用 `G4HadronicProcessStore` 生成的截面表；配套 JSON 记录 TOPAS/Geant4 版本、原始文件哈希和 200 MeV/u 参考点。它不是从 IDD 衰减曲线拟合得到的数据。

输运配置通过 `nuclear_cross_section_file` 指向该表；启用 `enable_primary_attenuation` 后，serial 与 SYCL kernel 都在每一步按 C-12 当前 MeV/u 线性插值水中宏观截面。

TOPAS 开发参考曲线、版本元数据和当前 CSDA 指标位于 `validation/results/`。原始 TOPAS scorer 文件和完整运行日志位于忽略目录 `validation/topas/output/`。

## Charged-origin 3D voxel 分类

设置 `enable_charged_origin_voxel_scoring: true` 后，GPU 在 total voxel 之外维护 8 个互斥 double-atomic 数组：

`primary_c12, secondary_carbon, boron, beryllium, lithium, helium, proton, other_charged`

稀疏输出由 `charged_origin_voxel_output_file` 指定。每行同时包含 total 和 8 个
`MeV/primary/voxel` 分类值；写文件前会检查逐 voxel 分类和、逐分类 z-plane 与
IDD 的闭合，内部门槛为 `1e-9 MeV/primary`。正式 100k Arc B580 结果可再次验证：

```bat
python validation\scripts\validate_charged_origin_voxels.py ^
  --gpu-category-voxel validation\results\windows_b580_mcs_3d_100k_charged_origin_voxels.csv ^
  --gpu-species-idd validation\results\windows_b580_mcs_3d_100k_species.csv ^
  --output-metrics validation\results\windows_b580_mcs_3d_100k_charged_origin_closure.metrics.json
```

该次 60×60×800 运行有 1,104,020 个非零 voxel；最大逐 voxel 闭合误差
`6.70e-11 MeV/primary`，最大逐分类 z-plane 闭合误差
`1.00e-10 MeV/primary/bin`，无 secondary/cascade queue overflow。原始 97 MB
分类 voxel CSV 被 Git 忽略，可由正式 YAML 确定性重建；指标和元数据进入版本控制。

## Neutron/gamma TOPAS 接口（smoke）

`CarbonNeutralNtuple` 记录 neutron/gamma 相互作用的 incident/continuation 能量、
局部沉积、水中宏观总截面，以及相关直接产物。远程 100-history smoke 后本地标准化：

```bat
python validation\scripts\prepare_topas_neutral.py ^
  --header validation\topas\output\neutral_smoke_interactions.header ^
  --phsp validation\topas\output\neutral_smoke_interactions.phsp ^
  --interactions-output validation\results\topas_200MeVu_neutral_smoke_interactions.csv.gz ^
  --products-output validation\results\topas_200MeVu_neutral_smoke_products.csv.gz ^
  --metadata validation\results\topas_200MeVu_neutral_smoke.metadata.json ^
  --case smoke
```

当前 smoke 包含 4,039 次相互作用和 1,961 个产物（2494 neutron / 1545 gamma）；
过程覆盖 `hadElastic`、`neutronInelastic`、`nCapture`、`phot`、`compt`、`conv`、
`Rayl`；全部 interaction 截面为正。

主机端可编译为 GPU 就绪二进制表：

```bat
python validation\scripts\compile_neutral_package.py ^
  --metadata validation\results\topas_200MeVu_neutral_smoke.metadata.json ^
  --interactions validation\results\topas_200MeVu_neutral_smoke_interactions.csv.gz ^
  --products validation\results\topas_200MeVu_neutral_smoke_products.csv.gz ^
  --output validation\results\topas_200MeVu_neutral_smoke.bin ^
  --output-metadata validation\results\topas_200MeVu_neutral_smoke.compiled.json
```

`NeutralPackageTable::from_binary` 加载 gamma/neutron 两个 projectile、按绝对 MeV
分箱的宏观总截面、带 continuation 的 interaction 以及局部系产物方向。smoke 包约
210 KB，C++ 加载测试已通过。

设置 `enable_neutral_transport: true` 后，SYCL 后端将 neutron/gamma 写入独立队列，
用总截面采样自由程，并采样末态 local deposit / continuation / 带电产物。中性后代
剂量进入 neutron/gamma origin IDD 与 `neutral_origin_voxel_output_file`；带电 lineage
标签避免污染 charged-origin 闭合。smoke 配置：

```bat
carbon_mc.exe --config config\beam_200MeVu_neutral_smoke.yaml --device cpu
```

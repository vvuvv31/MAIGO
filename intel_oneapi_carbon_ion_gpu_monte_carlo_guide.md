# 基于 Intel oneAPI 的碳离子水中 GPU 蒙特卡洛模拟：从零到 TOPAS Bragg 曲线匹配

## 1. 项目目标

本文档给出一条从零开始、基于 Intel oneAPI DPC++/SYCL 实现碳离子在水中输运的 GPU 蒙特卡洛开发路线。

最终目标是：

1. 使用 TOPAS 建立可信的 C-12 水中 pristine Bragg curve 基准；
2. 使用 C++ 编写一维 CPU 输运原型；
3. 将输运内核迁移到 Intel oneAPI/SYCL；
4. 逐步加入：
   - 连续能量损失；
   - 能量涨落；
   - 多重库仑散射；
   - 初级碳离子核反应衰减；
   - 次级碎片输运；
5. 在 Intel GPU 上获得与 TOPAS 相匹配的 Bragg 曲线；
6. 完成准确度、稳定性和性能验证；
7. 形成可用于论文发表的实验结果。

第一阶段建议仅研究：

- 粒子：C-12；
- 材料：均匀水；
- 束流：单能 pencil beam；
- 能量：先从 200 MeV/u 开始；
- 几何：一维深度输运；
- 输出：depth-dose curve；
- 参考：TOPAS/Geant4。

---

## 2. 研究路线总览

建议将开发过程分为以下阶段。

| 阶段 | 物理与工程内容 | 主要目标 |
|---|---|---|
| 0 | TOPAS 基准模拟 | 获得可信的参考 Bragg 曲线 |
| 1 | CPU 一维 CSDA | 匹配射程位置 |
| 2 | SYCL GPU 移植 | 在 Intel GPU 上跑通 |
| 3 | 能量涨落 | 匹配峰宽和远端下降 |
| 4 | 多重散射 | 支持有限横向 scorer 和 3D 扩展 |
| 5 | 初级核反应衰减 | 匹配峰高和碳离子存活率 |
| 6 | 次级碎裂输运 | 匹配 Bragg 峰后尾部 |
| 7 | 性能优化 | 提升 histories/s |
| 8 | 多能量验证 | 证明模型可泛化 |
| 9 | 论文实验 | 完成准确度、性能、消融研究 |

必须按照以下顺序调试：

1. 射程；
2. 峰宽；
3. 峰高；
4. 峰后碎裂尾部。

不要一开始同时调整所有物理模块。

### 2.1 当前项目状态（2026-07-14）

本仓库已经不再处于“从零搭框架”阶段。当前实际架构是：

```text
远程 Linux：v@192.168.31.5:~/gpu（最多 56 线程）
  └─ TOPAS 4.1.p1 / Geant4 11.1.3 MT
     ├─ 生成总 IDD、分粒种 IDD
     ├─ 生成 60x60x800 祖先归属 3D dose/IDD
     ├─ 直接导出 C-12+H/O 非弹性截面
     └─ 生成事件级反应末态 n-tuple

WSL Debian
  └─ 仅保留历史开发环境和 TOPAS 数据访问辅助，不再提交新的 TOPAS 作业

Windows 原生
  └─ Intel oneAPI/SYCL + Level Zero + Intel Arc B580
     ├─ serial / SYCL CPU / SYCL GPU 共用物理表
     └─ 执行和验证输运 kernel
```

已经完成：

- 200 MeV/u C-12 水中 TOPAS 总 IDD、电磁隔离 IDD 和 10 万粒子分粒种 IDD；
- CPU 一维 CSDA、自适应步长、Bohr straggling 和 Philox counter-based RNG；
- Windows 原生 Arc B580 Level Zero kernel；
- TOPAS 自定义 `CarbonCrossSectionNtuple`，直接从 `G4HadronicProcessStore` 导出 1--400 MeV/u 的 C-12+H、C-12+O 及水中宏观非弹性截面；
- CPU/SYCL 按当前 MeV/u 插值能量相关截面并采样初级核反应；
- TOPAS 自定义 `CarbonReactionNtuple`，按事件记录反应前 C-12 能量和全部直接次级粒子；
- 反应包标准化脚本和 OriginCount 独立 QA scorer；
- 10 万粒子正式事件级反应包：37,657 次主 C-12 非弹性反应和 330,659 个直接次级粒子；
- B580 事件包联合采样、fixed-capacity `atomic64` 带电次级生成队列和溢出/能量记账。
- TOPAS `CarbonDoseOrigin` 祖先归属 3D scorer、100-history smoke 和远程 56 线程 100000-history 正式基准；
- 祖先归属 TOPAS 与当前 B580 分粒种 IDD 的无 scale 比较。
- TOPAS `CarbonCascadeNtuple` 全带电离子后续反应记录、MT-safe interaction sequence、10 万粒子正式级联包；
- Arc B580 最多两代的 breadth-first 带电碎片级联输运；带电核产物按自身 Z/A 重新分类，电子沉积仍归当前带电母粒子。
- CPU/SYCL 可选 `60 x 60 x 800` total voxel tally、稀疏剂量 CSV 和逐 z 的 voxel→IDD 闭合检查；B580 100-history smoke 已通过。

关键验证结果：

- 200 MeV/u 水中宏观非弹性截面：`0.00474216 mm^-1`；
- 对应平均自由程：`210.874 mm`；
- 10,000-history 直接截面版本中，B580/serial 核反应数为 `3816/3818`；
- B580 对 serial：NRMSE `1.38e-5`、R80 差 `-3.9e-5 mm`、2%/2 mm gamma `100%`；
- 第一版带电次级输运已把完整 TOPAS 尾积分差从约 `-92%` 改善到 `+10.055%`；
- TOPAS `other` 细分显示其中 `78.61%` 是电子/正电子直接轨迹，gamma+neutron 直接沉积积分仅 `0.00407 MeV·mm/primary`；
- 扣除计分归属不同的 TOPAS 电子项后，GPU `other` 全深度差 `+0.50%`、90 mm 后尾部差 `+4.10%`。
- 祖先归属正式基准逐 bin 类别闭合最大误差 `6.395e-14 MeV/primary/bin`；与独立 TOPAS total 的最大差 `7.882e-7 MeV/primary/bin`；
- GPU 对原始 TOPAS total：全深度积分 `+0.402%`，90 mm 后尾部 `+15.452%`；
- 祖先对齐后仍存在明确的带电碎片偏差：He 全深度/尾部 `+36.55%/+50.14%`，proton `-24.86%/-34.15%`。
- 正式级联表包含 34 种 projectile、71,089 次可用 interaction、511,019 个产物和 13,469 个截面采样点；
- 同位素/代际 QA 证明旧 primary package 来自 TOPAS 4.2.p3/Geant4 11.3.2，而 ancestor/cascade reference 来自 TOPAS 4.1.p1/Geant4 11.1.3；混用末态数据是剩余粒种偏差的主因；
- 改用 cascade reference 的 37,661 个 primary C-12 联合末态后，带电类别全深度均在约 `±4%` 内，90 mm 后最大为 Be `+6.67%`；He 为 `-0.20%/+2.58%`，proton 为 `+1.41%/+2.84%`。

GPU 3D voxel scorer 的内存、原子累积、CSV 输出和 IDD 闭合框架已经完成；下一开发目标是真实 x/y/z 与三维方向输运、多重库仑散射和 3D 分类别闭合。当前 charged-origin total 已对齐到全深度 `-0.11%`、尾部 `+2.93%`；raw total 尾部 `-3.02%` 主要反映 TOPAS 尾部 `5.779%` 的中性来源尚未空间输运。禁止用全局 scale；后续在 3D charged scorer 稳定后再加入 neutron/gamma 来源输运。

---

# 第一部分：开发环境

## 3. 推荐软件环境

本项目采用 Windows 与远程 Linux 分工；WSL 不再承担新的 TOPAS 计算：

- Windows：Visual Studio 2026、Intel oneAPI、Level Zero、Arc B580；
- 远程 Linux `v@192.168.31.5:~/gpu`：TOPAS/Geant4 参考模拟，最多 56 线程；
- WSL Debian：仅作历史环境和必要的数据访问辅助；
- VS Code：打开 Windows 工作区；通过 SSH 脚本提交远程 TOPAS 作业。

推荐组件：

- Intel oneAPI Base Toolkit；
- Intel DPC++/C++ Compiler；
- Intel Level Zero Runtime；
- Intel VTune Profiler；
- CMake；
- Git；
- Python 3；
- NumPy；
- Pandas；
- Matplotlib；
- SciPy；
- TOPAS；
- Geant4。

建议项目使用：

- C++20；
- SYCL 2020；
- CMake；
- USM；
- `icpx -fsycl`；
- Intel GPU Level Zero 后端。

---

## 4. 检查 oneAPI 环境

WSL 中加载 oneAPI 环境，用于 SYCL CPU 编译检查：

```bash
source /opt/intel/oneapi/setvars.sh
```

检查编译器：

```bash
icpx --version
```

WSL 中检查 SYCL 设备：

```bash
sycl-ls
```

当前 WSL 只看到 Intel CPU OpenCL 是允许的。Intel GPU kernel 在 Windows 原生 Level Zero/Arc B580 上运行；新的 TOPAS 数据在远程 Linux 主机生成。不要把 WSL 看不到 Level Zero GPU 当成项目阻塞条件。

Windows oneAPI 终端中运行：

```bat
sycl-ls
```

Windows 必须能看到 Arc B580，例如：

```text
[level_zero:gpu][level_zero:0] Intel(R) Arc(TM) B580 Graphics
```

---

## 5. 最小 SYCL 测试程序

创建 `hello_sycl.cpp`：

```cpp
#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    try {
        sycl::queue queue{
            sycl::gpu_selector_v,
            sycl::property::queue::enable_profiling{}
        };

        std::cout << "Device: "
                  << queue.get_device()
                         .get_info<sycl::info::device::name>()
                  << '\n';

        constexpr std::size_t n = 1024;

        float* values = sycl::malloc_shared<float>(n, queue);

        if (values == nullptr) {
            std::cerr << "USM allocation failed\n";
            return 1;
        }

        queue.parallel_for(
            sycl::range<1>{n},
            [=](sycl::id<1> index) {
                values[index] = static_cast<float>(index[0]);
            }
        ).wait();

        std::cout << "values[100] = " << values[100] << '\n';

        sycl::free(values, queue);
    }
    catch (const sycl::exception& error) {
        std::cerr << "SYCL error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
```

编译：

```bash
icpx -O2 -fsycl hello_sycl.cpp -o hello_sycl
```

运行：

```bash
./hello_sycl
```

---

# 第二部分：建立 TOPAS 基准

## 6. 固定第一套基准条件

建议第一套 reference case 固定为：

```text
Particle: C-12
Energy: 200 MeV/u
Material: Water
Phantom length: 40 cm
Beam direction: +z
Beam type: Monoenergetic pencil beam
Depth bin: 0.5 mm
Histories: 100 for syntax/filter smoke test
Histories: 1e4 for total-IDD development
Histories: 1e5 for species/reaction calibration development
Histories: 1e6 for promoted reference
```

必须记录：

- TOPAS 版本；
- Geant4 版本；
- physics list；
- production cuts；
- step limit；
- 随机种子；
- 水密度；
- 模体尺寸；
- 束斑宽度；
- 初始能散；
- 初始角散；
- scorer 尺寸；
- scorer bin 数。

---

## 7. TOPAS 输出内容

至少输出：

```text
depth_mm
dose_Gy_per_primary
energy_deposition_MeV_per_primary
dose_uncertainty
```

建议额外输出：

```text
primary_C12_energy_deposition
secondary_proton_energy_deposition
secondary_alpha_energy_deposition
secondary_heavy_ion_energy_deposition
primary_C12_survival
C12_H_O_inelastic_cross_sections
primary_C12_reaction_headers
correlated_reaction_secondaries
```

这样可以分别验证：

- 电磁能损；
- 初级碳离子衰减；
- 次级碎片剂量；
- Bragg 峰后尾部。

截面、反应末态和 IDD 的职责必须分开：

```text
Geant4/TOPAS 截面查询
  → 决定 GPU 上“何时发生反应”

事件级反应 n-tuple
  → 决定一次反应“共同产生哪些粒子及其联合运动学”

OriginCount / 分粒种 IDD
  → 独立验证产额、深度分布和最终剂量
```

不能仅用分粒种直方图独立抽取每个碎片，否则会破坏同一反应内的多重性、能量和方向相关性。

---

## 8. TOPAS 基准检查

绘制以下图：

1. 原始 depth-dose；
2. 峰值归一化 depth-dose；
3. Bragg 峰局部放大；
4. 峰后尾部对数坐标；
5. 初级碳离子剂量贡献；
6. 次级粒子剂量贡献；
7. 统计不确定度。

提取指标：

- Peak depth；
- R90；
- R80；
- R50；
- R20；
- Distal falloff；
- FWHM；
- Peak-to-entrance ratio；
- Tail integral；
- Primary survival curve。

---

# 第三部分：项目结构

## 9. 推荐目录

```text
carbonGPU/
├── CMakeLists.txt
├── CMakePresets.json
├── README.md
├── nextStep.md
├── config/
│   ├── beam_200MeVu.yaml
│   ├── beam_200MeVu_straggling.yaml
│   ├── beam_200MeVu_attenuation.yaml
│   └── water_phantom.yaml
├── data/
│   ├── stopping_power_water.csv
│   ├── c12_inelastic_cross_sections_water_geant4_11_3_2.csv
│   └── c12_inelastic_cross_sections_water_geant4_11_3_2.metadata.json
├── include/carbon/
│   ├── cross_section.hpp
│   ├── device.hpp
│   ├── io.hpp
│   ├── particle.hpp
│   ├── rng.hpp
│   ├── stopping_power.hpp
│   ├── straggling.hpp
│   ├── transport.hpp
│   └── transport_config.hpp
├── src/
│   ├── config.cpp
│   ├── cross_section.cpp
│   ├── device.cpp
│   ├── io.cpp
│   ├── main.cpp
│   ├── stopping_power.cpp
│   ├── transport_cpu.cpp
│   └── transport_sycl.cpp
├── tests/
│   └── carbon_tests.cpp
├── validation/
│   ├── topas/
│   │   ├── extensions/
│   │   ├── build_extensions.sh
│   │   └── run_topas.sh
│   ├── scripts/
│   │   ├── compare_depth_dose.py
│   │   ├── prepare_topas_cross_sections.py
│   │   ├── prepare_topas_reactions.py
│   │   └── prepare_topas_species.py
│   └── results/
└── scripts/
    ├── build_windows_oneapi.cmd
    └── run_windows_b580.cmd
```

---

# 第四部分：基础数据结构

## 10. 粒子结构

创建 `include/particle.hpp`：

```cpp
#pragma once

struct Particle1D {
    float position_mm;
    float kinetic_energy_MeV;
    int particle_type;
    int alive;
};
```

第一版只做一维输运。

后续三维版本：

```cpp
struct Particle3D {
    sycl::float3 position_mm;
    sycl::float3 direction;
    float kinetic_energy_MeV;
    int particle_type;
    int alive;
};
```

---

## 11. 配置结构

```cpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

struct TransportConfig {
    std::size_t number_of_histories{10'000};
    double initial_energy_MeVu{200.0};
    int mass_number{12};
    double phantom_length_mm{400.0};
    double depth_bin_width_mm{0.5};
    double maximum_step_mm{0.5};
    double maximum_relative_energy_loss{0.005};
    double energy_cutoff_MeV{0.1};
    double water_density_g_per_cm3{1.0};
    bool enable_energy_straggling{false};
    double straggling_scale{1.0};
    bool enable_primary_attenuation{false};
    std::uint64_t random_seed{20'260'714};
    std::filesystem::path stopping_power_file{
        "data/stopping_power_water.csv"};
    std::filesystem::path nuclear_cross_section_file{
        "data/c12_inelastic_cross_sections_water_geant4_11_3_2.csv"};
    std::filesystem::path output_file{"out/cpu_depth_dose.csv"};
    std::string device{"serial"};
};
```

注意：

C-12 总初始动能为：

\[
E_{\mathrm{total}} = 12 E_{\mathrm{MeV/u}}
\]

例如：

\[
200\ \mathrm{MeV/u}
\rightarrow
2400\ \mathrm{MeV}
\]

必须在程序内部明确区分：

- MeV/u；
- 总动能 MeV；
- stopping power 的输入单位；
- stopping power 的输出单位。

---

# 第五部分：Stopping Power

## 12. 连续慢化模型

碳离子平均能量损失：

\[
\frac{dE}{dx} = -S(E)
\]

单步平均能损：

\[
\Delta E_{\mathrm{mean}} = S(E)\Delta x
\]

其中：

- \(E\)：粒子动能；
- \(S(E)\)：线性阻止本领；
- \(\Delta x\)：步长。

建议内部统一使用：

```text
Energy: MeV
Length: mm
Stopping power: MeV/mm
Density: g/cm3
```

---

## 13. 阻止本领数据表

建议数据格式：

```csv
energy_MeVu,stopping_power_MeV_per_mm
1.0,125.1
2.0,99.2
...
400.0,0.8
```

第一版建议使用均匀能量网格。

优点：

- GPU 查表快；
- 不需要二分搜索；
- 可直接计算 index；
- 易于 CPU/GPU 一致性验证。

当前 `data/stopping_power_water.csv` 仍是用于软件联调的透明 Bethe-Bloch + 有效电荷近似表，不应描述为最终 ICRU/临床物理数据。它已用 TOPAS 射程验证，但后续多能量研究仍需固定数据来源和版本。

### 13.1 核反应截面数据表

当前正式表为：

```text
data/c12_inelastic_cross_sections_water_geant4_11_3_2.csv
```

标准列为：

```csv
energy_MeV_per_u,total_kinetic_energy_MeV,c12_h_inelastic_cross_section_barn,c12_o_inelastic_cross_section_barn,hydrogen_macroscopic_cross_section_per_mm,oxygen_macroscopic_cross_section_per_mm,water_macroscopic_cross_section_per_mm,water_mean_free_path_mm
```

这张表不是从 IDD 生存曲线拟合得到的。TOPAS 完成物理初始化后，自定义 scorer 直接调用：

```cpp
G4HadronicProcessStore::GetInelasticCrossSectionPerAtom(...)
G4HadronicProcessStore::GetInelasticCrossSectionPerVolume(...)
```

其中 H/O 微观截面与原子数密度相乘后，必须满足：

\[
\Sigma_{water}(E)
=
\Sigma_H(E)+\Sigma_O(E)
\]

标准化脚本会检查该闭合关系，并记录原始 n-tuple、header、TOPAS 日志的 SHA-256、TOPAS/Geant4 版本和参考能量点。

---

## 14. 线性插值函数

```cpp
inline float interpolate_uniform_table(
    float energy,
    const float* table,
    int table_size,
    float energy_min,
    float inverse_energy_step
) {
    float floating_index =
        (energy - energy_min) * inverse_energy_step;

    int index =
        static_cast<int>(sycl::floor(floating_index));

    index = sycl::max(
        0,
        sycl::min(index, table_size - 2)
    );

    float fraction =
        floating_index - static_cast<float>(index);

    return table[index] +
           fraction * (table[index + 1] - table[index]);
}
```

---

## 15. 自适应步长

建议：

\[
\Delta x =
\min
\left(
\Delta x_{\max},
f_E \frac{E}{S(E)}
\right)
\]

其中：

```text
maximum_step_mm = 0.5 mm
maximum_relative_energy_loss = 0.005 to 0.01
```

函数：

```cpp
inline float choose_step_mm(
    float energy_MeV,
    float stopping_power_MeV_per_mm,
    float maximum_step_mm,
    float maximum_relative_energy_loss
) {
    float energy_limited_step =
        maximum_relative_energy_loss *
        energy_MeV /
        stopping_power_MeV_per_mm;

    return sycl::fmin(
        maximum_step_mm,
        energy_limited_step
    );
}
```

建议在 Bragg 峰附近进一步限制：

```text
minimum remaining energy region:
maximum step = 0.05 to 0.1 mm
```

---

# 第六部分：CPU 一维原型

## 16. CPU 输运伪代码

```cpp
for each primary history:
    energy = initial total kinetic energy
    position = 0

    while particle is alive:
        stopping_power = lookup(energy)
        step = choose_step(energy, stopping_power)
        deposited_energy = stopping_power * step

        deposited_energy =
            min(deposited_energy, energy)

        depth_bin =
            floor(position / depth_bin_width)

        score[depth_bin] += deposited_energy

        energy -= deposited_energy
        position += step

        if energy <= cutoff:
            stop particle

        if position >= phantom length:
            stop particle
```

---

## 17. CPU 版本第一阶段验收

只使用连续能损时，预计：

- Bragg 峰会过尖；
- 峰后没有尾部；
- 峰宽过小；
- 射程应该接近 TOPAS。

第一阶段目标：

```text
|R80_CPU - R80_TOPAS| < 1 mm
```

此时不要调整碎裂模型。

首先确认：

- 能量单位；
- 步长；
- stopping power；
- 水密度；
- scorer bin；
- 初始化能量；
- 每 primary 归一化。

---

# 第七部分：迁移到 SYCL GPU

## 18. 设备选择

```cpp
#include <sycl/sycl.hpp>
#include <iostream>
#include <string>

sycl::queue create_queue(const std::string& device_name) {
    auto async_handler = [](sycl::exception_list exceptions) {
        for (const auto& pointer : exceptions) {
            try {
                std::rethrow_exception(pointer);
            }
            catch (const sycl::exception& error) {
                std::cerr
                    << "Asynchronous SYCL error: "
                    << error.what()
                    << '\n';
            }
        }
    };

    if (device_name == "gpu") {
        return sycl::queue{
            sycl::gpu_selector_v,
            async_handler,
            sycl::property_list{
                sycl::property::queue::enable_profiling{},
                sycl::property::queue::in_order{}
            }
        };
    }

    if (device_name == "cpu") {
        return sycl::queue{
            sycl::cpu_selector_v,
            async_handler,
            sycl::property::queue::enable_profiling{}
        };
    }

    return sycl::queue{
        sycl::default_selector_v,
        async_handler,
        sycl::property::queue::enable_profiling{}
    };
}
```

运行形式：

```bash
./carbon_mc --device cpu
./carbon_mc --device gpu
```

---

## 19. USM 内存策略

### 调试版

使用 shared USM：

```cpp
double* dose = sycl::malloc_shared<double>(
    number_of_bins,
    queue
);
```

### 性能版

使用 device USM。当前剂量数组使用 `double`，因为大统计量下 `float` 原子累积会产生可见舍入误差：

```cpp
double* dose_device = sycl::malloc_device<double>(
    number_of_bins,
    queue
);
```

初始化：

```cpp
queue.memset(
    dose_device,
    0,
    number_of_bins * sizeof(double)
).wait();
```

结果复制：

```cpp
queue.memcpy(
    dose_host.data(),
    dose_device,
    number_of_bins * sizeof(double)
).wait();
```

---

## 20. one-work-item-per-history kernel

```cpp
queue.submit([&](sycl::handler& handler) {
    handler.parallel_for(
        sycl::nd_range<1>{
            sycl::range<1>{global_size},
            sycl::range<1>{local_size}
        },
        [=](sycl::nd_item<1> item) {
            std::size_t history_id =
                item.get_global_linear_id();

            if (history_id >= number_of_histories) {
                return;
            }

            float position_mm = 0.0f;
            float energy_MeV = initial_energy_MeV;
            float inverse_mass_number = 1.0f / 12.0f;

            while (
                energy_MeV > energy_cutoff_MeV &&
                position_mm < phantom_length_mm
            ) {
                float energy_MeVu =
                    energy_MeV * inverse_mass_number;

                float stopping_power =
                    interpolate_uniform_table(
                        energy_MeVu,
                        stopping_power_table,
                        stopping_power_table_size,
                        minimum_table_energy,
                        inverse_table_step
                    );

                float step_mm =
                    choose_step_mm(
                        energy_MeV,
                        stopping_power,
                        maximum_step_mm,
                        maximum_relative_energy_loss
                    );

                float deposited_energy =
                    stopping_power * step_mm;

                deposited_energy =
                    sycl::fmin(
                        deposited_energy,
                        energy_MeV
                    );

                int depth_bin =
                    static_cast<int>(
                        position_mm /
                        depth_bin_width_mm
                    );

                if (
                    depth_bin >= 0 &&
                    depth_bin < number_of_bins
                ) {
                    sycl::atomic_ref<
                        double,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space
                    > atomic_dose(dose_device[depth_bin]);

                    atomic_dose.fetch_add(
                        static_cast<double>(deposited_energy)
                    );
                }

                energy_MeV -= deposited_energy;
                position_mm += step_mm;
            }
        }
    );
}).wait();
```

---

# 第八部分：随机数

## 21. 随机数设计要求

Monte Carlo RNG 必须满足：

1. 不同 history 独立；
2. CPU 和 GPU 结果可复现；
3. work-group size 不影响序列；
4. 线程调度不影响序列；
5. 不在 kernel 内动态分配状态；
6. 支持 uniform、Gaussian 和 exponential。

推荐使用 counter-based RNG。

随机数可以定义为：

\[
r =
RNG(
seed,
history\_id,
interaction\_index,
random\_dimension
)
\]

推荐模型：

- Philox；
- Threefry；
- Random123 风格；
- PCG 仅用于早期验证。

---

## 22. 随机变量用途

每个输运步可能需要：

```text
dimension 0: energy-loss straggling
dimension 1: nuclear interaction
dimension 2: scattering polar angle
dimension 3: scattering azimuth
dimension 4: fragmentation channel
dimension 5: secondary energy
dimension 6: secondary direction
```

建议固定 random dimension 含义，保证可复现。

---

# 第九部分：能量涨落

## 23. 为什么必须加入 straggling

纯 CSDA 会让所有粒子几乎停在同一个位置，因此：

- Bragg 峰太尖；
- 峰宽太小；
- distal falloff 太陡。

加入：

\[
\Delta E =
\overline{\Delta E} + \delta E
\]

第一版可以使用：

\[
\delta E \sim \mathcal{N}(0,\sigma_E^2)
\]

---

## 24. 简化采样

```cpp
float sampled_energy_loss =
    mean_energy_loss +
    sigma_energy_loss *
    gaussian_random;

sampled_energy_loss =
    sycl::fmax(
        0.0f,
        sampled_energy_loss
    );

sampled_energy_loss =
    sycl::fmin(
        sampled_energy_loss,
        particle_energy
    );
```

需要避免：

- 负能损；
- 单步损失超过粒子总能量；
- NaN；
- 极端离群值；
- 不合理方差。

---

## 25. Straggling 验收指标

加入后重点比较：

```text
Peak depth
FWHM
R80
R20
R80 - R20
```

阶段目标：

```text
FWHM relative difference < 5%
|R80_GPU - R80_TOPAS| < 1 mm
```

---

# 第十部分：多重库仑散射

## 26. 第一阶段是否需要实现

如果 scorer 覆盖整个横截面，则一维积分 depth-dose 对横向散射不特别敏感。

可暂时忽略多重散射，直到：

- 使用有限横向 scorer；
- 比较横向束宽；
- 进入三维几何；
- 进入异质模体。

---

## 27. 三维方向更新

方向更新形式：

\[
\vec{u}_{new}
=
R(\theta,\phi)\vec{u}_{old}
\]

其中：

- \(\phi\) 均匀分布于 \([0,2\pi)\)；
- \(\theta\) 根据多重散射模型采样。

验证输出：

- 入射处横向 sigma；
- 中段横向 sigma；
- Bragg 峰附近横向 sigma；
- 径向剂量；
- TOPAS beam profile。

---

# 第十一部分：初级核反应衰减

## 28. 核反应概率

宏观截面：

\[
\Sigma(E) =
n_H \sigma_H(E) +
n_O \sigma_O(E)
\]

单步反应概率：

\[
P_{\mathrm{nuc}}
=
1 -
\exp(
-\Sigma(E)\Delta x
)
\]

采样：

```cpp
if (uniform_random < nuclear_probability) {
    particle.alive = false;
}
```

第一版只终止初级 C-12，不生成次级粒子。

当前实现不再接受单一常数 `nuclear_macroscopic_cross_section_per_mm`。配置使用：

```yaml
enable_primary_attenuation: true
nuclear_cross_section_file: data/c12_inelastic_cross_sections_water_geant4_11_3_2.csv
```

serial 与 SYCL kernel 在每一步完成电磁能损后，用 C-12 当前 MeV/u 线性插值 `water_macroscopic_cross_section_per_mm`。200 MeV/u 的直接值为 `0.00474216 mm^-1`，而早期从 primary survival 拟合的过渡常数是 `0.0050613 mm^-1`，高约 6.7%，因此旧常数不得再用于后续验证。

### 28.1 从 TOPAS 直接生成截面表

首次构建扩展版 TOPAS：

```bash
cd /mnt/d/OneDrive/DoctorDocuments/myproject/carbonGPU
bash validation/topas/build_extensions.sh
```

导出并标准化截面：

```bash
export TOPAS_EXECUTABLE="$PWD/build/opentopas-extension-install/bin/topas"
export TOPAS_G4_DATA_DIR="$HOME/Applications/GEANT4/G4DATA"

./validation/topas/run_topas.sh cross-sections
python3 validation/scripts/prepare_topas_cross_sections.py
```

原始 `.header`、`.phsp` 和 TOPAS 日志保存在忽略目录 `validation/topas/output/`；标准 CSV 和 metadata JSON 位于 `data/` 并进入 Git。

---

## 29. 第一阶段核模型的意义

它可以改善：

- 初级 C-12 存活率；
- Bragg 峰高度；
- 入口到峰值比例；
- 深度方向 primary fluence。

在当前最小模型中会产生以下限制：

- 发生反应后的能量进入 `untracked_nuclear_energy`，能量账目仍闭合，但不会贡献次级粒子剂量；
- 峰后尾部偏低；
- 不能预测分粒种产额和碎裂尾部。

因此它只是碎裂模型前的过渡版本。

---

## 30. Primary survival 验证

输出：

\[
N_C(z) / N_C(0)
\]

与 TOPAS 比较：

- 未发生核反应的初级碳离子数；
- 不同深度的 primary fluence；
- 衰减常数；
- 峰前粒子存活率。

---

# 第十二部分：次级碎裂输运

## 31. 需要处理的主要粒子

第一版带电次级输运可考虑：

- proton；
- alpha；
- Li；
- Be；
- B；
- carbon fragments。

事件级 TOPAS 数据还会保留 neutron、gamma 和反应残核，用于完整能量账目。中子可先采用：

- 能量逃逸；
- 简化局部沉积；
- 参数化非局部贡献。

后续再加入更精细中子输运。

---

## 32. 碎裂过程

发生核反应后：

1. 根据反应前 C-12 能量选择相邻的事件样本区间；
2. 抽取一个完整 `reaction_id`；
3. 一次性读取该反应包内的全部碎片；
4. 保留包内的粒种、多重性、能量和方向相关性；
5. 根据一维或三维模型处理方向；
6. 写入 secondary queue；
7. 输运带电次级粒子；
8. 记录逃逸能量和沉积能量。

必须检查：

\[
E_{\mathrm{initial}}
=
E_{\mathrm{local\ deposit}}
+
E_{\mathrm{secondary}}
+
E_{\mathrm{escape}}
+
E_{\mathrm{residual}}
\]

禁止从 proton、alpha、Li、Be、B 等独立直方图分别采样数量和能量。这种做法虽然可能重现单粒种边缘分布，却会破坏事件内相关性和能量守恒。

### 32.1 生成事件级反应包

100-history smoke 测试：

```bash
./validation/topas/run_topas.sh fragment-smoke

python3 validation/scripts/prepare_topas_reactions.py \
  --case smoke \
  --histories 100 \
  --reactions-output validation/topas/output/fragment_smoke_reactions.csv.gz \
  --secondaries-output validation/topas/output/fragment_smoke_secondaries.csv.gz \
  --metadata validation/topas/output/fragment_smoke_reaction_sampling.metadata.json
```

已验证的 smoke 数据包含 `33/100` 个初级非弹性反应、`271` 个直接次级粒子，平均每个反应包 `8.21` 个次级粒子。正式标定使用：

```bash
./validation/topas/run_topas.sh fragment-development

python3 validation/scripts/prepare_topas_reactions.py \
  --case development \
  --histories 100000 \
  --reactions-output validation/results/topas_200MeVu_reactions_development.csv.gz \
  --secondaries-output validation/results/topas_200MeVu_secondaries_development.csv.gz \
  --metadata validation/results/topas_200MeVu_reaction_packages_development.metadata.json
```

标准化脚本把 TOPAS 全局 `z=-200...200 mm` 转成水深 `0...400 mm`。压缩反应表每个 `reaction_id` 只保存一次入射能量和顶点，并用 `secondary_offset_zero_based + secondary_count` 定位次级粒子表中的连续反应包。脚本还检查 TOPAS header 的 history/entry 数、每个 secondary 的唯一 reaction header、顶点和反应前能量一致性以及方向归一化。

正式开发基准已经完成。TOPAS 4.2.p3 / Geant4 11.3.p2 在 100,000 个初级粒子中记录到 `37,657` 次主 C-12 非弹性反应和 `330,659` 个直接次级粒子，反应率为 `37.657%`，平均多重性为 `8.781`。运行耗时 `814.858 s`。其中两个低能反应没有直接可见次级粒子；它们以 `secondary_count=0` 保留，不能删除，否则会系统性低估反应概率。反应头与次级记录之和为 `368,316`，与 TOPAS header 完全闭合；压缩文件 SHA-256、行数及 offset/count 连续性也已验证。

### 32.2 编译为 GPU 运行时表

Windows oneAPI 程序不在运行时解析 gzip/CSV。使用标准库 Python 脚本先编译为小端定长二进制：

```bash
python3 validation/scripts/compile_reaction_package.py \
  --metadata validation/results/topas_200MeVu_reaction_packages_development.metadata.json \
  --reactions validation/results/topas_200MeVu_reactions_development.csv.gz \
  --secondaries validation/results/topas_200MeVu_secondaries_development.csv.gz \
  --output validation/results/topas_200MeVu_reaction_packages_development.bin \
  --output-metadata validation/results/topas_200MeVu_reaction_packages_development.binary.metadata.json
```

version 1 格式由 64-byte header、201 个 8-byte 能量分箱、37,657 个 16-byte 反应头和 330,659 个 16-byte 次级粒子记录组成，总计 `5,894,728` bytes。次级记录只保留运行时需要的 PDG、Z、A、动能和 z 方向；完整三维方向仍保留在源 gzip 表中。每个 1 MeV/u 分箱至少有 2 个、最多有 358 个完整反应包。

`ReactionPackageTable::from_binary` 独立验证 magic/version、ABI record size、文件长度、能量分箱覆盖、反应到次级 offset/count 闭合、能量和方向范围。真实正式数据测试已经同时通过 GCC 12.2 和 Windows IntelLLVM 2025.3.3。

---

## 33. GPU 队列设计

不要在单个线程中递归生成无限数量粒子。

推荐：

```text
Input particle queue
        ↓
Transport kernel
        ↓
Surviving primary queue
Secondary particle queue
        ↓
Queue compaction
        ↓
Next transport kernel
```

推荐按粒子种类分队列：

```text
C12 queue
proton queue
alpha queue
heavy-fragment queue
```

这样可以减少：

- 分支发散；
- 不同 stopping-power 模型混杂；
- 不同步长策略混杂；
- 不同核过程混杂。

实现顺序应为：

1. 先建立固定容量、可检测溢出的 secondary queue；
2. 将 `reaction_id` 作为联合采样单元；
3. 第一版只输运沿束流方向的带电碎片；
4. 分别统计队列溢出、未输运中子/光子能量和残核能量；
5. 通过 OriginCount、分粒种 IDD、尾积分和总能量同时验收。

当前已完成第 1、2 步以及生成阶段的第 4 步记账。实现使用一次 `atomic64` reservation 为一个反应包的全部带电离子预留连续空间；容量不足时整组拒绝，禁止写入半个相关反应包。所有 `Z>0` 的离子都进入通用 A/Z 队列，包括 proton、deuteron、triton、He3、alpha 和 Z=3--6 碎片；不能只保留 proton/alpha，否则本次 10k 样本会遗漏 `796,679.75 MeV` 的带电次级动能。

Arc B580 10,000-history 实测结果：

- 3,816 次核反应，3,816 个完整反应包；
- 33,260 个直接次级粒子，直接次级动能 5,554,734.6 MeV；
- 21,357 个带电离子进入容量 160,000 的队列，溢出为 0；
- 排队带电能量 5,001,525.1 MeV，中子/光子能量 553,209.49 MeV；
- 未支持带电能量为 0；
- generation-only 配置不输运队列，因此其 primary-only IDD 物理列与未启用生成时逐 bin 数值一致；
- 10k 耗时明显受 SYCL JIT/磁盘缓存状态影响，不应用于正式性能比较。

第 3 步已经完成。第一版对队列中的全部带电离子执行正/反向一维 CSDA：路径方向使用采样的 `direction_z`，停止本领按

\[
S_{A,Z}(E/u)=S_{\mathrm{C12}}(E/u)
\left[\frac{z_{\mathrm{eff}}(Z,\beta)}{z_{\mathrm{eff}}(6,\beta)}\right]^2
\]

缩放，其中 `z_eff` 与 C-12 开发表使用同一 Hubert 型公式。输运分别累计 secondary C、B、Be、Li、He、proton 和 other charged IDD，并将已排队能量从 `untracked_nuclear_energy` 转移到 secondary deposited/escaped 账本。

100,000-history Arc B580 结果：216,133 个带电次级粒子、316,048,099 个碎片步进、零队列溢出、预热后 `95,985 histories/s`、总能量误差 `3.11e-8`。相对完整 TOPAS：总积分 `+0.48%`、峰值 `-0.062%`、R80 `+0.104 mm`、FWHM `+2.36%`、NRMSE `1.08%`、2%/2 mm gamma `97.71%`、尾积分 `+10.055%`。尾部已由此前约 `-92%` 大幅改善，但仍以极小幅度未通过 `<10%` 验收线。

原始分粒种积分显示 secondary C/B/Be/Li/He 偏高约 20--32%、proton 偏低约 31%、other 偏低约 79%，但细分 scorer 证明最后一项主要是计分归属差异。TOPAS 的 `other` 中 electron/positron、deuteron、triton、unclassified 分别占 `78.61%`、`12.59%`、`5.15%`、`3.61%`，gamma+neutron 的直接轨迹沉积积分仅 `0.00407 MeV·mm/primary`。GPU condensed-history 不显式输运 delta electron，而是把电子阻止能量局部计入母离子；扣除 TOPAS electron/positron 后，GPU `other` 全深度只差 `+0.50%`，90 mm 后尾部差 `+4.10%`。

因此当前数据不能直接证明 neutron/gamma 输运是 `other` 差异的主因，也不能把所有重碎片偏高解释为真实产额偏高。应先用祖先归属 scorer 将 TOPAS 的电子沉积回归到母粒子，再评估碎片后续核反应、衰变级联、能量涨落、多重散射和中性粒子输运。经验 scale 仍然禁止。

---

# 第十三部分：剂量评分

## 34. Global atomic 版本

当前准确度版本直接使用 double atomic：

```cpp
sycl::atomic_ref<
    double,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space
> atomic_dose(dose[bin]);

atomic_dose.fetch_add(static_cast<double>(deposited_energy));
```

优点：

- 实现简单；
- 正确性容易验证。

缺点：

- Bragg 峰附近原子冲突严重；
- 大量线程写入相同 bin；
- GPU 利用率下降。

Arc B580 支持所需的 `fp64` 和 `atomic64`。程序启动后会检查这两个 SYCL aspect；不满足时应明确报错，不能静默退回 float。此前 10 万粒子测试已经确认，double atomic 可把 B580 与 serial 的剂量累积误差压到可忽略水平。

---

## 35. Local-memory tally

优化思路：

1. 每个 work-group 分配 local dose；
2. work-group 内局部原子累积；
3. barrier；
4. 将 local dose 合并到 global dose。

限制：

- local memory 容量有限；
- bin 太多时需要 tiling；
- 清零和归约有额外开销。

建议比较：

```text
global atomic
local-memory tally
two-stage reduction
```

这可以成为论文的性能实验之一。

---

# 第十四部分：CMake

## 36. 当前 CMake Presets

项目已提供三个 preset：

```text
cpu-debug                 WSL/Linux GCC CPU
oneapi-release            WSL/Linux oneAPI SYCL CPU
oneapi-windows-release    Windows oneAPI + Arc GPU
```

WSL CPU 调试构建：

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug
```

WSL oneAPI 正确性构建：

```bash
source /opt/intel/oneapi/setvars.sh
cmake --preset oneapi-release
cmake --build --preset oneapi-release
ctest --preset oneapi-release
```

Windows 原生 Arc B580 构建和运行：

```bat
scripts\build_windows_oneapi.cmd
scripts\run_windows_b580.cmd --histories 10000
```

运行脚本固定 `ONEAPI_DEVICE_SELECTOR=level_zero:0`，避免误选 OpenCL CPU。当前 Windows `latest` 编译器实际报告 IntelLLVM 2025.3.3，而 WSL 验证编译器为 IntelLLVM 2026.1.0；论文和 metadata 必须分别记录，不能笼统写成同一版本。

---

# 第十五部分：测试体系

## 37. 单位测试

必须测试：

- MeV/u 到总动能；
- cm 到 mm；
- g/cm3；
- MeV cm2/g 到 MeV/mm；
- depth bin index；
- stopping-power interpolation；
- range integration。

---

## 38. 单粒子确定性测试

关闭随机过程：

```text
histories = 1
fixed step
fixed stopping power
no nuclear interaction
no scattering
```

逐步输出：

```text
step
position
energy_before
stopping_power
step_length
energy_deposit
energy_after
depth_bin
```

CPU 和 GPU 逐步比较。

---

## 39. 能量守恒测试

对于每个 history：

\[
E_0 =
E_{\mathrm{deposit}}
+
E_{\mathrm{remaining}}
+
E_{\mathrm{escape}}
+
E_{\mathrm{secondary}}
\]

报告：

```text
absolute energy balance error
relative energy balance error
maximum history error
mean history error
```

---

## 40. RNG 测试

检查：

- uniform mean；
- uniform variance；
- Gaussian mean；
- Gaussian variance；
- autocorrelation；
- history 间相关性；
- CPU/GPU 重现性；
- 不同 work-group size 的稳定性。

---

## 41. 步长收敛

测试：

```text
1.0 mm
0.5 mm
0.25 mm
0.10 mm
0.05 mm
```

观察：

- R80；
- peak depth；
- FWHM；
- NRMSE；
- runtime。

结果应显示：

- 步长减小时结果趋于稳定；
- 计算时间增加；
- 可确定准确度与性能折中点。

---

## 42. 粒子数收敛

测试：

```text
1e4
1e5
1e6
1e7
```

统计误差应近似满足：

\[
\sigma \propto \frac{1}{\sqrt{N}}
\]

---

# 第十六部分：与 TOPAS 比较

## 43. 曲线预处理原则

GPU 与 TOPAS 必须使用相同：

- 深度 bin；
- scorer 边界；
- 单位；
- 横向积分范围；
- 归一化；
- 束流定义。

论文中建议同时展示：

1. 每初级粒子的绝对剂量或能量沉积；
2. 峰值归一化剂量。

不要只展示归一化曲线。

### 43.1 当前项目的运行与查看方式

在 Windows 原生终端运行 B580：

```bat
scripts\run_windows_b580.cmd --histories 10000
```

结果写入：

```text
out/windows_b580_attenuation_depth_dose.csv
```

新的 TOPAS 作业只在远程主机运行。先将仓库中的 TOPAS 输入、扩展和脚本同步到 `v@192.168.31.5:~/gpu`，再在远端构建扩展：

```bash
ssh v@192.168.31.5
cd ~/gpu
./validation/topas/build_extensions_remote.sh
```

100-history smoke 和 100000-history 正式 3D 基准分别运行：

```bash
cd ~/gpu
./validation/topas/run_ancestor_remote.sh smoke

nohup ./validation/topas/run_ancestor_remote.sh development \
  > validation/topas/output/ancestor_development.remote.log 2>&1 &
```

远程脚本固定使用 56 线程，并调用安装在远端的 TOPAS 4.1.p1 / Geant4 11.1.3 MT。原始 scorer 位于远端 `~/gpu/validation/topas/output/`；正式标准化结果同步回本地 `validation/results/`。

祖先归属 3D scorer 使用 `60 x 60 x 800` 网格，体素为 `5 x 5 x 0.5 mm^3`，覆盖整个 `300 x 300 x 400 mm^3` 水模体。它输出 12 个互斥来源类别，并把电子/正电子沉积归回其带电母粒子；neutron/gamma 及其后代保留为中性来源。标准化命令为：

```bash
python3 validation/scripts/prepare_topas_ancestor_dose.py \
  --case development --histories 100000 --seed 20260714 \
  --execution-host vv \
  --output-npz validation/results/topas_200MeVu_ancestor_dose_3d_development.npz \
  --output-idd validation/results/topas_200MeVu_ancestor_dose_3d_development.idd.csv \
  --metadata validation/results/topas_200MeVu_ancestor_dose_3d_development.metadata.json \
  --plot validation/results/topas_200MeVu_ancestor_dose_3d_development.png

python3 validation/scripts/compare_ancestor_attributed_idd.py \
  validation/results/topas_200MeVu_ancestor_dose_3d_development.idd.csv \
  validation/results/windows_b580_fragment_transport_species_100k.csv \
  --metrics-output validation/results/windows_b580_ancestor_attributed_100k_vs_topas.metrics.json \
  --plot validation/results/windows_b580_ancestor_attributed_100k_vs_topas.png
```

正式基准验收结果：所有祖先类别逐 bin 求和的最大闭合误差为 `6.395e-14 MeV/primary/bin`；与未改动的独立 TOPAS total scorer 最大差为 `7.882e-7 MeV/primary/bin`；`unresolved=0`。这同时证明新 scorer 没有改变原始 TOPAS total IDD，且没有使用全局 scale。

比较时 GPU `other` 只对应 TOPAS `other_charged`；TOPAS 的 `neutron/gamma/neutral_other` 来源单独报告。GPU 对原始 TOPAS total 的全深度积分差为 `+0.402%`，90 mm 后为 `+15.452%`。祖先对齐后的 He 为 `+36.55%/+50.14%`，proton 为 `-24.86%/-34.15%`（全深度/90 mm 后），所以后续工作应转向带电碎片的再反应和衰变级联。

### 43.2 带电碎片后续核反应级联

新的 TOPAS `CarbonCascadeNtuple` 记录所有带电 projectile 的 inelastic interaction、宏观非弹性截面和联合直接末态。MT 模式下唯一键必须使用 `(run, thread, event, interaction_sequence)`；`track_id` 不能唯一表示 interaction，因为同一存活 track 可以再次反应。正式远程作业运行：

```bash
cd ~/gpu
./validation/topas/run_cascade_remote.sh smoke
setsid -f ./validation/topas/run_cascade_remote.sh development \
  > validation/topas/output/cascade-development_nohup.log 2>&1 < /dev/null
```

100000-history 正式作业耗时 `397.414 s`，验证得到 71,091 次 interaction 和 511,034 个产物。编译后 GPU 表保留 34 种 projectile、71,089 次有正非弹性截面的 interaction 和 511,019 个产物；2 条 `Z4A4` 因 TOPAS 返回零截面而在 metadata 中显式跳过。运行 B580：

```bat
scripts\run_windows_b580_cascade.cmd
```

B580 初版正式运行暴露了一个计分语义错误：后续核反应的带电产物继承了原始带电祖先类别。例如 He 产生的 proton 仍被累计在 He 中，这与 `CarbonDoseOrigin` 对核产物按自身 Z/A 分类的规则不一致。修复只改变类别索引，不使用 scale，也不改变反应、阻止能或 total dose 算法。

修正后 100000 histories、最多两代的 B580 运行产生 36,713 次后续 interaction 和 129,401 个后代带电粒子入队，无溢出；能量平衡误差 `2.75e-8`，kernel elapsed `2.590 s`，吞吐量 `38,614 histories/s`。与祖先归属 TOPAS 比较：全深度 raw total `-0.932%`，charged-origin total `-0.223%`；90 mm 后 raw total `+4.197%`。proton 从错误归属时的 `-27.38%/-39.16%` 改善到 `+1.05%/+3.10%`（全深度/90 mm 后），He 从 `+24.79%/+31.00%` 降到 `+15.89%/+18.57%`。GPU 分粒种逐 bin 求和与 total 的最大闭合误差为 `7.70e-11 MeV/primary/bin`。

进一步的同位素/反应代数 QA 恢复了 cascade reference 中 0--4 代反应谱，并发现 primary reaction package 使用 TOPAS 4.2.p3/Geant4 11.3.2，而 ancestor/cascade reference 使用 TOPAS 4.1.p1/Geant4 11.1.3。旧 primary package 相对 cascade reference 的初级 C-12 末态，alpha 能量/反应高约 `15.6%`，He-3 高约 `51.8%`，与 GPU He 偏高一致。

`prepare_primary_reactions_from_cascade.py` 从同一 cascade reference 的 track-1、event-interaction-0 C-12 记录提取 37,661 个联合末态和 323,901 个产物，并编译为覆盖全部 201 个能量 bin 的 `topas_200MeVu_cascade_aligned_primary.bin`。替换初级包后，B580 100000-history 结果为：全深度 raw/charged total `-0.825%/-0.114%`，90 mm 后 `-3.022%/+2.926%`；He `-0.20%/+2.58%`、proton `+1.41%/+2.84%`、B `+3.88%/+4.06%`、Li `+3.83%/+3.72%`（全深度/尾部）。所有带电类别尾部最大偏差为 Be `+6.67%`，逐 bin 闭合为 `9.80e-11 MeV/primary/bin`，未使用全局 scale。

在没有 numpy/matplotlib 的主机上可直接运行标准库版本：

```bat
python validation\scripts\compare_ancestor_attributed_idd_portable.py ^
  validation\results\topas_200MeVu_ancestor_dose_3d_development.idd.csv ^
  validation\results\windows_b580_fragment_cascade_aligned_100k_species.csv ^
  --metrics-output validation\results\windows_b580_fragment_cascade_aligned_100k_vs_topas.metrics.json ^
  --plot validation\results\windows_b580_fragment_cascade_aligned_100k_vs_topas.svg
```

### 43.3 GPU 3D voxel dose scorer 第一阶段

`TransportConfig` 现在支持：

- `enable_voxel_scoring`；
- `voxel_bins_x/voxel_bins_y`；
- `voxel_size_x_mm/voxel_size_y_mm`；
- `voxel_dose_output_file`。

z 方向继续使用 `depth_bin_width_mm` 和 `number_of_bins()`。标准 smoke 网格为 `60 x 60 x 800`，体素 `5 x 5 x 0.5 mm^3`，与 TOPAS 祖先归属 scorer 的几何一致。CPU 和 SYCL 都维护独立 total voxel 数组；SYCL 使用 double atomic，并覆盖主 C-12 步进、带电碎片步进和停止于水中的剩余能量沉积。输出是只写非零体素的 CSV：

```text
ix,iy,iz,x_mm,y_mm,z_mm,energy_deposition_MeV_per_primary,dose_Gy_per_primary
```

Windows Arc B580 smoke：

```bat
build\oneapi-windows-release\carbon_mc.exe --config config\beam_200MeVu_voxel_smoke.yaml
```

YAML 中的 `device: gpu` 已由配置加载器直接解析，不再必须额外传 `--device gpu`。100-history 全级联运行选中 `Intel(R) Arc(TM) B580 Graphics`，能量平衡误差为 `4.12e-8`，输出 800 个 IDD bin 和 800 个非零体素，逐 z 的 x/y 求和与 IDD 最大差为 `0 MeV/primary/bin`。

这一里程碑只验证 3D scorer 的内存布局、所有沉积路径和闭合性。当前粒子状态仍只有深度和 `direction_z`，所以剂量全部落在中心 `(ix,iy)=(30,30)`；它不是物理横向剂量。下一提交必须先扩展 x/y/z 和三维方向、实现体素边界步进与多重库仑散射，再进行 TOPAS 逐 voxel/切片比较。

---

## 44. R80 等指标

远端侧 R80：

\[
D(R_{80}) = 0.8 D_{\max}
\]

采用线性插值求交点。

同理计算：

- R90；
- R80；
- R50；
- R20。

Distal falloff：

\[
DF = R_{80} - R_{20}
\]

---

## 45. NRMSE

\[
NRMSE =
\frac{
\sqrt{
\frac{1}{N}
\sum_i
(D_i^{GPU} - D_i^{TOPAS})^2
}
}{
D_{\max}^{TOPAS}
}
\]

建议分别计算：

- 全曲线；
- entrance region；
- plateau region；
- peak region；
- tail region。

---

## 46. Tail integral error

\[
\epsilon_{\mathrm{tail}}
=
\frac{
\int_{R_{80}}^{z_{\max}} D_{GPU}(z)dz
-
\int_{R_{80}}^{z_{\max}} D_{TOPAS}(z)dz
}{
\int_{R_{80}}^{z_{\max}} D_{TOPAS}(z)dz
}
\]

---

## 47. Gamma analysis

建议报告：

```text
2% / 2 mm
1% / 1 mm
```

同时给出低剂量阈值，例如：

```text
10% maximum dose threshold
1% maximum dose threshold
```

Gamma 不能代替：

- R80；
- FWHM；
- tail integral；
- peak dose error。

---

# 第十七部分：分阶段验收标准

## 48. Level 1：CSDA

```text
|R80_GPU - R80_TOPAS| < 1 mm
```

---

## 49. Level 2：Straggling

```text
FWHM relative difference < 5%
Distal falloff difference < 1 mm
```

---

## 50. Level 3：Primary attenuation

```text
Peak dose difference < 5%
Primary survival curve difference < 5%
```

同时要求：

- 截面来自可追溯的 TOPAS/Geant4 直接查询；
- H/O 分量与水中总宏观截面闭合；
- 禁止使用逐能量单独拟合的衰减常数；
- serial/SYCL 的核反应统计和剂量曲线一致。

---

## 51. Level 4：Fragmentation

```text
Tail integral difference < 10%
```

更高目标：

```text
Tail integral difference < 5%
```

---

## 52. 最终目标

```text
R80 difference < 0.5 to 1.0 mm
Peak dose difference < 3 to 5%
FWHM difference < 5%
Tail integral difference < 5 to 10%
2%/2 mm gamma passing rate > 95%
```

---

# 第十八部分：性能优化

## 53. 主要性能瓶颈

碳离子 GPU Monte Carlo 的常见瓶颈：

- 可变长度 while 循环；
- work-item 执行步数不同；
- Bragg 峰附近步长变小；
- global atomic contention；
- stopping-power table 访问；
- 次级粒子动态生成；
- 队列压缩；
- host-device 数据传输。

---

## 54. 优化顺序

建议严格按以下顺序：

1. 先确保正确；
2. shared USM 改为 device USM；
3. 测试 work-group size；
4. 优化 stopping-power lookup；
5. global atomic 改为 local tally；
6. 缩短长生命周期 kernel；
7. 每若干步做一次 queue compaction；
8. 按粒子类型分类；
9. 按能量分类；
10. 再测试 fast math；
11. 再考虑 float/double 混合精度。

---

## 55. Work-group size 测试

测试：

```text
32
64
128
256
512
```

不要假设某个值一定最佳。

---

## 56. VTune 分析

示例：

```bash
vtune -collect gpu-offload \
  -result-dir vtune_gpu \
  ./build/carbon_mc --device gpu
```

重点分析：

- GPU active time；
- EU active/stalled/idle；
- kernel time；
- atomic bottleneck；
- memory bandwidth；
- host-device transfer；
- SIMD utilization；
- occupancy；
- kernel launch overhead。

---

# 第十九部分：论文实验设计

## 57. 能量矩阵

第一轮：

```text
200 MeV/u
```

最终验证：

```text
100 MeV/u
150 MeV/u
200 MeV/u
250 MeV/u
300 MeV/u
350 MeV/u
400 MeV/u
```

关键要求：

> 不允许每个能量单独调一套参数。

同一物理模型必须直接预测所有能量。

---

## 58. 物理模块消融

依次展示：

```text
Stopping power only
+ Energy straggling
+ Multiple scattering
+ Primary nuclear attenuation
+ Secondary fragmentation
```

这张图能清楚解释每个物理模块控制曲线的哪个区域。

---

## 59. 性能实验

比较：

```text
Serial C++
SYCL CPU
Intel integrated GPU
Intel discrete GPU
TOPAS single thread
TOPAS multi-thread
```

指标：

```text
histories/s
steps/s
total runtime
kernel runtime
memory transfer time
speedup
```

---

## 60. 论文建议题目

### 题目 1

```text
Development and TOPAS Validation of a SYCL-Based
GPU Monte Carlo Dose Engine for Carbon-Ion Beams
```

### 题目 2

```text
A Portable Carbon-Ion Monte Carlo Transport Engine
Using Intel oneAPI and SYCL
```

### 题目 3

```text
GPU-Accelerated Carbon-Ion Depth-Dose Calculation
in Water Using a SYCL Condensed-History Monte Carlo Method
```

---

## 61. 论文结构

### Abstract

1. Background；
2. Purpose；
3. Methods；
4. Results；
5. Conclusion。

### 1. Introduction

- 碳离子治疗；
- Bragg peak；
- fragmentation tail；
- Geant4/TOPAS 计算成本；
- GPU fast Monte Carlo；
- SYCL 跨平台价值；
- 本文贡献。

### 2. Materials and Methods

- oneAPI/SYCL 架构；
- 水模体；
- 连续能损；
- straggling；
- scattering；
- nuclear attenuation；
- fragmentation；
- dose scoring；
- TOPAS benchmark；
- validation metrics；
- performance metrics。

### 3. Results

- 200 MeV/u Bragg curve；
- 峰区放大；
- tail 放大；
- 多能量结果；
- R80 error；
- gamma；
- 消融研究；
- 步长收敛；
- 粒子数收敛；
- GPU 性能。

### 4. Discussion

- 误差来源；
- 核模型限制；
- stopping power 不确定度；
- 单精度误差；
- TOPAS 作为参考的限制；
- 向 3D、CT 和临床 beamline 扩展。

### 5. Conclusion

避免在未完成临床验证前声称：

```text
clinical ready
clinical-grade
treatment planning ready
```

---

# 第二十部分：建议的开发里程碑

## 62. 里程碑 1

```text
完成 TOPAS 200 MeV/u 水中 Bragg curve
```

输出：

- TOPAS 参数文件；
- CSV；
- 绘图；
- R80；
- FWHM；
- statistical uncertainty。

---

## 63. 里程碑 2

```text
完成 CPU 一维 CSDA
```

验收：

```text
R80 difference < 1 mm
```

---

## 64. 里程碑 3

```text
完成 SYCL CPU 和 Intel GPU 版本
```

验收：

- 同一 stopping-power table；
- 同一配置；
- CPU/GPU 射程一致；
- 能量守恒；
- 无 NaN；
- 可重复运行。

---

## 65. 里程碑 4

```text
加入 energy straggling
```

验收：

```text
FWHM difference < 5%
```

---

## 66. 里程碑 5

```text
加入 primary nuclear attenuation
```

验收：

```text
Peak dose difference < 5%
```

当前状态：已完成能量相关截面的 TOPAS 直接提取和 CPU/SYCL/B580 接入；初级反应后能量仍记入 `untracked_nuclear_energy`，等待里程碑 6 的次级输运。

---

## 67. 里程碑 6

```text
加入 secondary fragmentation
```

验收：

```text
Tail integral difference < 10%
```

当前状态：同位素/代际 QA、统一 TOPAS/Geant4 primary package 和 GPU total voxel tally 框架已完成。Arc B580 对 charged-origin total 的全深度/尾部差为 `-0.11%/+2.93%`，所有带电类别尾部偏差均小于 `7%`。下一项是横向坐标/三维方向输运、体素边界步进和多重散射；之后完成 3D category closure，再实现 neutron/gamma 来源输运。不得用全局 scale 掩盖空间或中性来源偏差。

---

## 68. 里程碑 7

```text
完成 100 to 400 MeV/u 多能量验证
```

要求：

- 不逐能量调参；
- 使用统一模型；
- 报告 R80、FWHM、peak、tail 和 gamma。

---

## 69. 里程碑 8

```text
完成 VTune 性能优化和论文图表
```

### 69.1 当前 Git 里程碑

与本 guide 当前状态对应的重要提交包括：

```text
3f152e2  topas: species-resolved fragment scorers
74cc153  data: 100k species-resolved TOPAS baseline
51963db  topas: extract cross sections and reaction final states
a32ff10  physics: use TOPAS energy-dependent reaction cross sections
5dbf1b6  data: add GPU-ready reaction package loader
3bda906  sycl: add correlated secondary generation queue
6c7fd2b  sycl: transport charged reaction fragments
d477929  topas: decompose residual particle dose
51541a4  validation: add particle-dose attribution baseline
```

每次新增正式 TOPAS 数据集、物理模块、SYCL 队列结构或验证结果，均应单独提交，且 metadata 必须记录输入哈希、软件版本、随机种子和运行统计。

---

# 第二十一部分：推荐 AI 提问顺序

## 70. 生成项目框架

```text
请为基于 Intel oneAPI DPC++/SYCL 的碳离子 Monte Carlo
项目生成完整 C++20 工程框架。

要求：
- CMake；
- icpx -fsycl；
- 支持 SYCL CPU 和 Intel GPU；
- 使用 USM；
- 支持设备选择；
- 包含 stopping-power table；
- 包含 CPU/GPU 单元测试；
- 包含 TOPAS 验证目录；
- Linux 环境。
```

---

## 71. 生成一维输运内核

```text
请使用 SYCL 2020 编写 C-12 在水中的一维 CSDA 输运程序。

要求：
- 输入能量单位为 MeV/u；
- 内部转换为总动能；
- one work-item per history；
- USM device memory；
- adaptive step；
- stopping-power table 线性插值；
- sycl::atomic_ref 剂量评分；
- 输出 depth-dose CSV；
- 支持 CPU 和 GPU selector。
```

---

## 72. 生成 Philox RNG

```text
请实现一个 SYCL 兼容的 Philox counter-based RNG。

随机数由：
- global seed；
- history ID；
- interaction index；
- random dimension；
共同决定。

要求：
- CPU/GPU 可复现；
- work-group size 不影响结果；
- 提供 uniform、Gaussian 和 exponential 采样。
```

---

## 73. 生成 straggling 模块

```text
请设计适用于碳离子 condensed-history Monte Carlo 的
energy-loss straggling 模块。

要求：
- 第一版使用 Gaussian 或 Bohr approximation；
- 给出方差表达式；
- 防止负能损；
- 保证单步能量守恒；
- 提供 SYCL device-compatible C++ 代码；
- 说明如何用 FWHM 和 distal falloff 验证。
```

---

## 74. 生成核反应模块

```text
请为 C-12 在水中的输运设计简化核反应模块。

第一阶段：
- H 和 O 宏观截面；
- primary attenuation。

第二阶段：
- proton；
- alpha；
- Li；
- Be；
- B；
- carbon fragments。

要求给出：
- 反应概率；
- 数据表格式；
- secondary queue；
- 能量守恒；
- SYCL 实现方式；
- TOPAS 验证方法。
```

---

## 75. 生成验证脚本

```text
请使用 Python 编写 TOPAS 与 SYCL GPU depth-dose 的比较程序。

输入：
- topas.csv；
- gpu.csv。

输出：
- peak depth；
- R90、R80、R50、R20；
- FWHM；
- distal falloff；
- NRMSE；
- peak dose error；
- tail integral error；
- 1%/1 mm gamma；
- 2%/2 mm gamma；
- 对比图；
- 峰区放大图；
- tail 对数图。
```

---

# 第二十二部分：最终检查表

## 76. 物理正确性

- [x] MeV/u 与总动能区分正确；
- [x] stopping power 单位正确；
- [x] 水密度正确；
- [ ] 步长收敛；
- [x] 当前 primary-only 模型能量账目闭合；
- [x] Philox RNG 由 seed/history/step/dimension 唯一确定；
- [x] Bohr straggling 已完成 200 MeV/u 电磁隔离验证；
- [x] 核反应概率小于等于 1；
- [x] 核截面由 TOPAS/Geant4 直接提取并通过 H/O 闭合检查；
- [ ] 碎片能量守恒；
- [x] 当前 200 MeV/u TOPAS/GPU scorer 深度 bin 一致。

## 77. CPU/GPU 一致性

- [ ] 单粒子逐步一致；
- [x] stopping-power interpolation 一致；
- [x] cross-section interpolation 一致；
- [x] 固定随机数时可复现；
- [ ] 不同 work-group size 统计稳定；
- [x] 当前测试未发现越界访问；
- [x] 无 NaN；
- [x] 无未初始化 USM；
- [x] Windows Arc B580 Level Zero 实际运行通过。

## 78. TOPAS 匹配

- [x] entrance region；
- [x] plateau；
- [x] peak position；
- [x] peak width（Level 2 电磁隔离）；
- [x] primary peak height（Level 3）；
- [x] distal falloff（Level 2 电磁隔离）；
- [ ] fragmentation tail；
- [x] primary attenuation/survival proxy；
- [x] absolute energy-deposition normalization；
- [x] 分粒种 TOPAS IDD 10 万粒子基准；
- [x] 事件级反应 n-tuple smoke；
- [x] 事件级反应 n-tuple 10 万粒子正式基准；
- [x] 祖先归属 3D dose scorer 100-history smoke；
- [x] 祖先归属 3D dose scorer 10 万粒子正式基准；
- [x] 祖先类别逐 bin 闭合与独立 total 不变性验证；
- [x] GPU/TOPAS 祖先归属 IDD 无 scale 比较。
- [x] GPU `60 x 60 x 800` total voxel tally 与 100-history B580 smoke；
- [ ] GPU 真实三维轨迹、多重散射和 3D category closure。

## 79. 性能实验

- [x] CPU serial；
- [x] SYCL CPU；
- [x] Intel Arc B580 GPU；
- [ ] TOPAS single-thread；
- [x] TOPAS multi-thread；
- [x] histories/s；
- [x] total steps；
- [x] kernel elapsed time；
- [ ] transfer time；
- [ ] VTune report。

## 80. 三维方向数据边界（2026-07-14）

reaction package 与 charged-fragment cascade package 已升级为 binary v2。产物记录由
`pdg/Z/A/energy/direction_z` 扩展为 `pdg/Z/A/energy/direction_x/direction_y/direction_z`，
方向坐标定义为相对发生反应时入射母粒子的右手局部正交基。编译器先验证 TOPAS 全局
方向为单位向量，再投影到该局部基；运行时应将局部方向旋转到当前母粒子方向，而不能
把它当成固定实验室坐标。

C++ 加载器同时接受 v1 和 v2。v1 中不存在的 `direction_x/y` 被展开为 NaN，测试会检查
该哨兵；v2 则要求三个方向分量有限且模长平方与 1 的差小于 `2e-3`。旧 v1 二进制和正式
一维基准配置没有覆盖，新生成的 `_3d.bin` 只用于三维开发与回归。

从已保存的 TOPAS 100000-history/smoke gzip 表生成的 v2 数据包含 37,661 个 primary C-12
reaction、323,901 个直接产物，以及 71,089 个可用级联相互作用、511,019 个级联产物。
Windows oneAPI 全量构建和 v1/v2 加载测试通过。Arc B580 上的 100-history 全级联 voxel
smoke 能量平衡误差为 `3.43e-8`，800 个非零中心轴 voxel 对 800-bin IDD 的最大闭合误差
为 `0 MeV/primary/bin`。仍只有中心轴剂量是预期结果，因为运行时粒子状态尚未三维化。

---

# 结论

最稳妥的开发顺序是：

```text
1. TOPAS 基准
2. CPU 一维 CSDA
3. 匹配射程
4. SYCL GPU 移植
5. CPU/GPU 一致性验证
6. 加 energy straggling
7. 匹配峰宽
8. 加 primary nuclear attenuation
9. 匹配峰高
10. 加 secondary fragmentation
11. 匹配峰后尾部
12. 扩展到多个能量
13. 做步长和统计收敛
14. 使用 VTune 优化
15. 整理论文实验
```

本项目已经完成第 10 步的两代带电碎片后续核反应级联、TOPAS 祖先归属 3D scorer、两个 100000-history 正式基准、GPU/TOPAS 对齐比较，以及 GPU total voxel tally 的第一阶段闭合 smoke。紧接着应执行：

```text
1. 将粒子队列和反应包从 depth/direction_z 扩展为 x/y/z 坐标与三维方向
2. 实现沿三维轨迹的体素边界步进，保证 voxel→IDD 和能量闭合
3. 加入带电离子的多重库仑散射与横向展宽，并实现 3D category closure
4. 将 GPU 3D 剂量与 TOPAS `60 x 60 x 800` 祖先归属体素逐 voxel/切片比较
5. 在带电 3D scorer 稳定后加入 neutron/gamma 来源输运
6. 做原子队列可复现性、1000000-history 统计收敛和 100--400 MeV/u 多能量验证
```

第一篇论文的合理边界是：

> 在均匀水模体中，使用基于 SYCL 的 condensed-history GPU Monte Carlo，复现 100–400 MeV/u C-12 的 pristine Bragg curves，并与 TOPAS 在射程、峰宽、峰高、碎裂尾部和计算性能方面进行系统比较。

在完成这一阶段之前，不建议提前加入：

- DICOM；
- CT；
- RBE；
- LET；
- SOBP；
- 临床 beamline；
- GUI；
- 多 GPU；
- 生物剂量。

先完成一维水模体，是最容易建立正确性、可重复性和论文逻辑的路线。

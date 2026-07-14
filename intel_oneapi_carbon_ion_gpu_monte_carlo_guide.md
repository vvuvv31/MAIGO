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

---

# 第一部分：开发环境

## 3. 推荐软件环境

建议优先使用 Linux。

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

加载 oneAPI 环境：

```bash
source /opt/intel/oneapi/setvars.sh
```

检查编译器：

```bash
icpx --version
```

检查 SYCL 设备：

```bash
sycl-ls
```

预期可以看到 Intel GPU，例如：

```text
[level_zero:gpu][level_zero:0] Intel(R) Arc(TM) Graphics
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
Histories: 1e6 for development
Histories: 1e7 or higher for final reference
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
```

这样可以分别验证：

- 电磁能损；
- 初级碳离子衰减；
- 次级碎片剂量；
- Bragg 峰后尾部。

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
carbon-oneapi-mc/
├── CMakeLists.txt
├── README.md
├── config/
│   ├── beam_200MeVu.yaml
│   └── water_phantom.yaml
├── data/
│   ├── stopping_power_water.csv
│   ├── nuclear_cross_section_h.csv
│   ├── nuclear_cross_section_o.csv
│   └── fragmentation_tables/
├── include/
│   ├── particle.hpp
│   ├── transport_config.hpp
│   ├── material.hpp
│   ├── stopping_power.hpp
│   ├── straggling.hpp
│   ├── scattering.hpp
│   ├── nuclear.hpp
│   ├── scoring.hpp
│   ├── rng.hpp
│   └── device.hpp
├── src/
│   ├── main.cpp
│   ├── device.cpp
│   ├── transport_cpu.cpp
│   ├── transport_sycl.cpp
│   ├── io.cpp
│   └── scoring.cpp
├── kernels/
│   ├── transport_kernel.hpp
│   ├── scoring_kernel.hpp
│   └── queue_compaction.hpp
├── tests/
│   ├── test_units.cpp
│   ├── test_interpolation.cpp
│   ├── test_rng.cpp
│   ├── test_energy_conservation.cpp
│   ├── test_cpu_gpu_match.cpp
│   └── test_step_convergence.cpp
├── validation/
│   ├── topas/
│   ├── scripts/
│   └── results/
└── paper/
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

struct TransportConfig {
    std::size_t number_of_histories;
    float initial_energy_MeVu;
    float phantom_length_mm;
    float depth_bin_width_mm;
    float maximum_step_mm;
    float maximum_relative_energy_loss;
    float energy_cutoff_MeV;
    std::uint64_t random_seed;
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
            sycl::property::queue::enable_profiling{}
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
float* dose = sycl::malloc_shared<float>(
    number_of_bins,
    queue
);
```

### 性能版

使用 device USM：

```cpp
float* dose_device = sycl::malloc_device<float>(
    number_of_bins,
    queue
);
```

初始化：

```cpp
queue.memset(
    dose_device,
    0,
    number_of_bins * sizeof(float)
).wait();
```

结果复制：

```cpp
queue.memcpy(
    dose_host.data(),
    dose_device,
    number_of_bins * sizeof(float)
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

            while (
                energy_MeV > energy_cutoff_MeV &&
                position_mm < phantom_length_mm
            ) {
                float stopping_power =
                    interpolate_uniform_table(
                        energy_MeV,
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
                        float,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space
                    > atomic_dose(dose_device[depth_bin]);

                    atomic_dose.fetch_add(
                        deposited_energy
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

---

## 29. 第一阶段核模型的意义

它可以改善：

- 初级 C-12 存活率；
- Bragg 峰高度；
- 入口到峰值比例；
- 深度方向 primary fluence。

但会产生问题：

- 发生反应后的能量消失；
- 峰后尾部偏低；
- 总能量不守恒。

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

第一版可考虑：

- proton；
- alpha；
- Li；
- Be；
- B；
- carbon fragments。

中子可先采用：

- 能量逃逸；
- 简化局部沉积；
- 参数化非局部贡献。

后续再加入更精细中子输运。

---

## 32. 碎裂过程

发生核反应后：

1. 选择碎裂通道；
2. 采样碎片种类；
3. 采样碎片数量；
4. 分配动能；
5. 采样方向；
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

---

# 第十三部分：剂量评分

## 34. Global atomic 版本

第一版直接使用：

```cpp
sycl::atomic_ref<
    float,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space
> atomic_dose(dose[bin]);

atomic_dose.fetch_add(deposited_energy);
```

优点：

- 实现简单；
- 正确性容易验证。

缺点：

- Bragg 峰附近原子冲突严重；
- 大量线程写入相同 bin；
- GPU 利用率下降。

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

## 36. 最小 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.22)

project(carbon_oneapi_mc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(
    carbon_mc
    src/main.cpp
    src/device.cpp
    src/transport_sycl.cpp
    src/scoring.cpp
    src/io.cpp
)

target_include_directories(
    carbon_mc
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(
    carbon_mc
    PRIVATE
    -fsycl
    -O3
)

target_link_options(
    carbon_mc
    PRIVATE
    -fsycl
)
```

构建：

```bash
source /opt/intel/oneapi/setvars.sh

cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j
```

运行：

```bash
./build/carbon_mc --device gpu
```

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

---

## 67. 里程碑 6

```text
加入 secondary fragmentation
```

验收：

```text
Tail integral difference < 10%
```

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

- [ ] MeV/u 与总动能区分正确；
- [ ] stopping power 单位正确；
- [ ] 水密度正确；
- [ ] 步长收敛；
- [ ] 能量守恒；
- [ ] RNG 独立；
- [ ] straggling 方差合理；
- [ ] 核反应概率小于等于 1；
- [ ] 碎片能量守恒；
- [ ] scorer 范围一致。

## 77. CPU/GPU 一致性

- [ ] 单粒子逐步一致；
- [ ] stopping-power interpolation 一致；
- [ ] 固定随机数时可复现；
- [ ] 不同 work-group size 统计稳定；
- [ ] 无越界访问；
- [ ] 无 NaN；
- [ ] 无未初始化 USM。

## 78. TOPAS 匹配

- [ ] entrance region；
- [ ] plateau；
- [ ] peak position；
- [ ] peak width；
- [ ] peak height；
- [ ] distal falloff；
- [ ] fragmentation tail；
- [ ] primary survival；
- [ ] absolute normalization。

## 79. 性能实验

- [ ] CPU serial；
- [ ] SYCL CPU；
- [ ] Intel GPU；
- [ ] TOPAS single-thread；
- [ ] TOPAS multi-thread；
- [ ] histories/s；
- [ ] steps/s；
- [ ] kernel time；
- [ ] transfer time；
- [ ] VTune report。

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

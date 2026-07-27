# 编译与运行

本项目使用 Intel oneAPI DPC++/C++（`icpx`）编译 SYCL 后端。编译时决定可执行文件包含 Intel SPIR-V、NVIDIA CUDA，或同时包含两种目标；运行时通过 `--device` 选择实际后端。

## 1. 准备 oneAPI 环境

Linux：

```bash
cd /home/v/project/MAIGO-cuda

unset SETVARS_COMPLETED
source /opt/intel/oneapi/setvars.sh --force
```

如果 oneAPI 安装在其他位置，请改为对应的 `setvars.sh`。

可用以下命令检查设备：

```bash
sycl-ls
```

## 2. 编译目标

`CARBON_SYCL_TARGETS` 控制编译目标：

| 配置 | `CARBON_SYCL_TARGETS` | 用途 |
|---|---|---|
| Intel only | `spir64` | Intel Arc，通过 Level Zero 或 OpenCL 运行 |
| NVIDIA only | `nvptx64-nvidia-cuda` | NVIDIA GPU，通过 CUDA plugin 运行 |
| Intel + NVIDIA | `spir64,nvptx64-nvidia-cuda` | 同一个可执行文件运行于两种 GPU |

项目已有以下 CMake preset：

```bash
# 仅 Intel SPIR-V
cmake --preset oneapi-intel-release
cmake --build --preset oneapi-intel-release

# 仅 NVIDIA CUDA
cmake --preset oneapi-nvidia-release
cmake --build --preset oneapi-nvidia-release

# Intel SPIR-V + NVIDIA CUDA
cmake --preset oneapi-release
cmake --build --preset oneapi-release
```

`oneapi-release` 是双目标构建。该可执行文件在编译后不需要重新编译，即可在运行时选择 Level Zero 或 CUDA。

### 没有 Ninja 时

preset 默认使用 Ninja。如果系统没有 Ninja，可改用 Unix Makefiles：

```bash
cmake -S . -B build/oneapi-intel-make \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCARBON_ENABLE_SYCL=ON \
  -DCARBON_BUILD_TESTS=ON \
  -DCARBON_SYCL_TARGETS=spir64

cmake --build build/oneapi-intel-make -j"$(nproc)"
```

双目标构建时，把最后一项改为：

```bash
-DCARBON_SYCL_TARGETS=spir64,nvptx64-nvidia-cuda
```

## 3. 在 Intel Arc B580 上运行

当前 Linux B580 驱动与 oneAPI 2026.1 的 Level Zero V2 kernel-with-arguments 提交路径存在兼容问题。需要设置以下回退开关：

```bash
export UR_L0_V2_DISABLE_ZE_LAUNCH_KERNEL_WITH_ARGS=1
```

该开关让 Level Zero V2 adapter 使用旧的稳定 kernel launch 路径。当前 B580 已实际验证基础输运、FP64 dose、voxel scoring、secondary transport、fragment cascade 和 neutral transport 均可运行。

运行示例：

```bash
UR_L0_V2_DISABLE_ZE_LAUNCH_KERNEL_WITH_ARGS=1 \
ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
build/oneapi-intel-make/carbon_mc \
  --config config/beam_200MeVu_voxel_smoke.yaml \
  --device level_zero
```

也可以使用设备别名：

```bash
--device intel
--device arc
```

### 使用 B580 OpenCL 后端

如果不使用 Level Zero，也可以选择 OpenCL：

```bash
ONEAPI_DEVICE_SELECTOR=opencl:gpu \
build/oneapi-intel-make/carbon_mc \
  --config config/beam_200MeVu_voxel_smoke.yaml \
  --device opencl
```

## 4. 在 NVIDIA CUDA 上运行

CUDA 构建需要：

- Intel oneAPI DPC++/C++；
- oneAPI CUDA plugin；
- NVIDIA CUDA Toolkit；
- 可用的 NVIDIA 驱动。

运行示例：

```bash
ONEAPI_DEVICE_SELECTOR=cuda:gpu \
build/oneapi-release/carbon_mc \
  --config config/beam_200MeVu_attenuation.yaml \
  --device cuda
```

也可以使用别名：

```bash
--device nvidia
```

如果需要指定 NVIDIA AOT 架构，可在配置时设置：

```bash
-DCARBON_CUDA_ARCH=sm_75
```

请按实际 GPU 修改 `sm_75`。

## 5. 运行时设备开关

编译了对应目标后，`carbon_mc` 支持：

| 参数 | 后端 |
|---|---|
| `--device level_zero`、`intel`、`arc` | Intel Level Zero |
| `--device cuda`、`nvidia` | NVIDIA CUDA |
| `--device opencl` | OpenCL GPU |
| `--device gpu` | 由 `ONEAPI_DEVICE_SELECTOR` 选择 |
| `--device cpu` | SYCL CPU |
| `--device serial` | 非 SYCL 串行 CPU |

推荐显式指定后端：

```bash
--device level_zero
```

或：

```bash
--device cuda
```

仅编译 `spir64` 的可执行文件不能在 CUDA 后端运行；仅编译 `nvptx64-nvidia-cuda` 的可执行文件也不能在 Level Zero 后端运行。需要单个可执行文件同时支持两者时，必须使用双目标：

```bash
-DCARBON_SYCL_TARGETS=spir64,nvptx64-nvidia-cuda
```

## 6. 测试

```bash
ctest --test-dir build/oneapi-intel-make --output-on-failure
```

运行 Level Zero 相关测试或程序时，保留 B580 兼容变量：

```bash
export UR_L0_V2_DISABLE_ZE_LAUNCH_KERNEL_WITH_ARGS=1
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
```

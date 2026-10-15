cmake_minimum_required(VERSION 3.22)
project(MAIGO_clean LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Clang")
    message(FATAL_ERROR "Use the SYCL icpx compiler")
endif()
set(CARBON_SYCL_TARGETS "nvptx64-nvidia-cuda" CACHE STRING "SYCL GPU backend")
if(NOT CARBON_SYCL_TARGETS STREQUAL "nvptx64-nvidia-cuda" AND
   NOT CARBON_SYCL_TARGETS STREQUAL "spir64_gen")
    message(FATAL_ERROR "Use nvptx64-nvidia-cuda (CUDA AOT) or spir64_gen (Intel GPU AOT)")
endif()
set(CARBON_CUDA_ARCH "sm_75" CACHE STRING "Required CUDA AOT architecture; set for the GPU running this binary")
if(CARBON_SYCL_TARGETS STREQUAL "nvptx64-nvidia-cuda" AND NOT CARBON_CUDA_ARCH MATCHES "^sm_[0-9]+[a-z]?$")
    message(FATAL_ERROR "Set -DCARBON_CUDA_ARCH=sm_75 (RTX 2080 Ti), sm_86 (RTX A6000), or sm_89 (L4), according to your GPU. Empty/PTX-only builds are not supported.")
endif()
set(CARBON_INTEL_GPU_ARCH "" CACHE STRING "Required Intel GPU AOT device, e.g. bmg-g21 for Arc B580")
if(CARBON_SYCL_TARGETS STREQUAL "spir64_gen" AND NOT CARBON_INTEL_GPU_ARCH MATCHES "^[A-Za-z0-9_-]+$")
    message(FATAL_ERROR "Set -DCARBON_INTEL_GPU_ARCH to the destination GPU, e.g. bmg-g21 for Arc B580. SPIR-V/JIT-only builds are not supported.")
endif()
option(CARBON_ENABLE_PROTON_BROADBEAM "Compile proton broadbeam transport" ON)
option(CARBON_ENABLE_PROTON_MINIBEAM "Compile proton minibeam transport" ON)
option(CARBON_ENABLE_CARBON_BROADBEAM "Compile carbon broadbeam transport" ON)
option(CARBON_ENABLE_CARBON_MINIBEAM "Compile carbon minibeam transport" ON)
option(CARBON_ENABLE_GENERIC "Compile generic proton/C12 transport" ON)
if(NOT (CARBON_ENABLE_PROTON_BROADBEAM OR CARBON_ENABLE_PROTON_MINIBEAM OR
        CARBON_ENABLE_CARBON_BROADBEAM OR CARBON_ENABLE_CARBON_MINIBEAM OR CARBON_ENABLE_GENERIC))
    message(FATAL_ERROR "Enable at least one transport pipeline")
endif()
add_library(carbon_core
    src/all_ion_elastic.cpp
    src/batch_run.cpp
    src/cli.cpp
    src/config.cpp
    src/cross_section.cpp
    src/ct_grid.cpp
    src/delta_moments_data.cpp
    src/device.cpp
    src/inelastic_package_v3.cpp
    src/io.cpp
    src/material_nuclear_rates.cpp
    src/plan_run.cpp
    src/run_quality.cpp
    src/run_session.cpp
    src/schneider_rate_table.cpp
    src/schneider_target_sampler.cpp
    src/secondary_rate_table.cpp
    src/source_pose.cpp
    src/stopping_power.cpp
    src/tps_source.cpp
    src/transport_metrics.cpp
    src/transport_sycl.cpp
    src/unified_em_package.cpp
    src/urban_mcs_package.cpp
)
target_include_directories(carbon_core PUBLIC include PRIVATE src)
target_compile_definitions(carbon_core PUBLIC
    CARBON_EM_AUDIT_SHARDS=64 CARBON_SECONDARY_SEGMENT_STEPS=16)
foreach(pipeline PROTON_BROADBEAM PROTON_MINIBEAM CARBON_BROADBEAM CARBON_MINIBEAM GENERIC)
    message(STATUS "Transport pipeline ${pipeline}: ${CARBON_ENABLE_${pipeline}}")
    target_compile_definitions(carbon_core PRIVATE
        CARBON_ENABLE_${pipeline}=$<BOOL:${CARBON_ENABLE_${pipeline}}>)
endforeach()
target_compile_options(carbon_core PRIVATE -Wall -Wextra -Wpedantic -fsycl -fsycl-targets=${CARBON_SYCL_TARGETS})
target_link_options(carbon_core PUBLIC -fsycl -fsycl-targets=${CARBON_SYCL_TARGETS})
if(CARBON_SYCL_TARGETS STREQUAL "nvptx64-nvidia-cuda")
    set(carbon_aot_options "SHELL:-Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=${CARBON_CUDA_ARCH}")
else()
    target_compile_definitions(carbon_core PRIVATE CARBON_INTEL_GPU_AOT=1)
    # Keep native Intel images separate: merged images miscompile CT lookups
    # with the tested oneAPI 2026.1/B580 toolchain. CUDA is unchanged.
    target_compile_options(carbon_core PRIVATE -fsycl-device-code-split=per_kernel)
    target_link_options(carbon_core PUBLIC -fsycl-device-code-split=per_kernel)
    set(carbon_aot_options "SHELL:-Xsycl-target-backend=spir64_gen \"-device ${CARBON_INTEL_GPU_ARCH}\"")
endif()
target_compile_options(carbon_core PRIVATE "${carbon_aot_options}")
target_link_options(carbon_core PUBLIC "${carbon_aot_options}")
add_executable(carbon_mc src/main.cpp)
target_link_libraries(carbon_mc PRIVATE carbon_core)
target_compile_options(carbon_mc PRIVATE $<TARGET_PROPERTY:carbon_core,COMPILE_OPTIONS>)

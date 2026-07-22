cmake_minimum_required(VERSION 3.22)

project(carbon_oneapi_mc VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CARBON_ENABLE_SYCL "Build the oneAPI/SYCL transport backend" OFF)
option(CARBON_BUILD_TESTS "Build unit and integration tests" ON)
option(CARBON_ENABLE_TRANSPORT_PROFILE
       "Enable low-overhead SYCL step-path profiling counters (not for production dose)"
       OFF)
# Global dose USM + device atomics as float instead of double. Much faster on
# consumer NVIDIA GPUs (weak FP64); slightly more statistical noise for huge N.
option(CARBON_DOSE_FP32 "Use FP32 SYCL dose scorers/atomics (faster on CUDA)" OFF)

# Comma-separated SYCL device triple list for -fsycl-targets.
# Empty keeps the compiler default (SPIR-V JIT; fine for Intel Level Zero/OpenCL).
# For NVIDIA CUDA, nvptx64-nvidia-cuda is required — default -fsycl SPIR-V does not
# load kernels on the CUDA plugin ("No kernel named ... was found").
# Dual Intel Arc + NVIDIA: spir64,nvptx64-nvidia-cuda
set(CARBON_SYCL_TARGETS "" CACHE STRING
    "SYCL targets for -fsycl-targets (e.g. spir64,nvptx64-nvidia-cuda)")
# Optional SM arch for NVIDIA AOT (e.g. sm_75 for TITAN RTX). Empty embeds portable PTX.
set(CARBON_CUDA_ARCH "" CACHE STRING
    "Optional NVIDIA GPU arch for AOT (e.g. sm_75); empty = portable PTX")

add_library(carbon_core
    src/config.cpp
    src/cross_section.cpp
    src/ct_grid.cpp
    src/io.cpp
    src/reaction_package.cpp
    src/cascade_package.cpp
    src/neutral_package.cpp
    src/stopping_power.cpp
    src/topas_spots.cpp
    src/tps_source.cpp
    src/transport_cpu.cpp
)

target_include_directories(carbon_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MSVC)
    target_compile_options(carbon_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(carbon_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(CARBON_ENABLE_TRANSPORT_PROFILE)
    target_compile_definitions(carbon_core PUBLIC CARBON_TRANSPORT_PROFILE=1)
    message(STATUS "CARBON_ENABLE_TRANSPORT_PROFILE=ON (SYCL step counters enabled)")
endif()

if(CARBON_DOSE_FP32)
    target_compile_definitions(carbon_core PUBLIC CARBON_DOSE_FP32=1)
    message(STATUS "CARBON_DOSE_FP32=ON (SYCL dose scorers use float atomics)")
else()
    message(STATUS "CARBON_DOSE_FP32=OFF (SYCL dose scorers use double atomics)")
endif()

if(CARBON_ENABLE_SYCL)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
        message(FATAL_ERROR "CARBON_ENABLE_SYCL requires Intel oneAPI icpx (IntelLLVM).")
    endif()

    target_sources(carbon_core PRIVATE src/device.cpp src/transport_sycl.cpp src/transport_profile.cpp)
    target_compile_definitions(carbon_core PUBLIC CARBON_HAS_SYCL=1)
    target_compile_options(carbon_core PRIVATE -fsycl)
    target_link_options(carbon_core PUBLIC -fsycl)

    if(CARBON_SYCL_TARGETS)
        message(STATUS "SYCL targets: ${CARBON_SYCL_TARGETS}")
        target_compile_options(carbon_core PRIVATE
            -fsycl-targets=${CARBON_SYCL_TARGETS})
        target_link_options(carbon_core PUBLIC
            -fsycl-targets=${CARBON_SYCL_TARGETS})
    else()
        message(STATUS "SYCL targets: compiler default (SPIR-V JIT; Intel GPU/CPU)")
    endif()

    if(CARBON_CUDA_ARCH)
        if(NOT CARBON_SYCL_TARGETS MATCHES "nvptx64-nvidia-cuda")
            message(WARNING
                "CARBON_CUDA_ARCH=${CARBON_CUDA_ARCH} is set but CARBON_SYCL_TARGETS "
                "does not include nvptx64-nvidia-cuda; arch flag may be ignored.")
        endif()
        message(STATUS "CUDA GPU arch: ${CARBON_CUDA_ARCH}")
        # SHELL: keep multi-token flag as one argv group for the driver.
        target_compile_options(carbon_core PRIVATE
            "SHELL:-Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=${CARBON_CUDA_ARCH}")
        target_link_options(carbon_core PUBLIC
            "SHELL:-Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=${CARBON_CUDA_ARCH}")
    endif()
else()
    target_sources(carbon_core PRIVATE src/transport_profile.cpp)
endif()

add_executable(carbon_mc src/main.cpp)
target_link_libraries(carbon_mc PRIVATE carbon_core)

if(CARBON_BUILD_TESTS)
    enable_testing()
    add_executable(carbon_tests tests/carbon_tests.cpp)
    target_link_libraries(carbon_tests PRIVATE carbon_core)
    target_compile_definitions(
        carbon_tests PRIVATE CARBON_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_test(NAME carbon_tests COMMAND carbon_tests)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME reorient_ct_grid_tps_90_tests
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/validation/scripts/test_reorient_ct_grid_tps_90.py
        )
        add_test(
            NAME dose_mapping_geometry_tests
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/validation/scripts/test_match_gpu_geometry.py
        )
    endif()
endif()

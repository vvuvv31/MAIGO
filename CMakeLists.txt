cmake_minimum_required(VERSION 3.22)

project(carbon_oneapi_mc VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CARBON_ENABLE_SYCL "Build the oneAPI/SYCL transport backend" OFF)
option(CARBON_ENABLE_TRANSPORT_PROFILE
       "Enable low-overhead SYCL step-path profiling counters"
       OFF)
# FP32 dose atomics: much faster on consumer NVIDIA GPUs (weak FP64).
# FP32 dose atomics default ON; enable FP64 only when CARBON_DOSE_FP64=ON is explicitly specified.
option(CARBON_DOSE_FP32 "Use FP32 SYCL dose scorers/atomics" ON)
option(CARBON_DOSE_FP64 "Use FP64 SYCL dose scorers/atomics" OFF)
option(CARBON_LET_FP64 "Use FP64 SYCL LET scorers/atomics" OFF)

if(CARBON_DOSE_FP64)
    set(CARBON_DOSE_FP32 OFF CACHE BOOL "Use FP32 SYCL dose scorers/atomics" FORCE)
endif()
# Copper minibeam beamline. Default OFF keeps the legacy CT/water/LET kernel.
option(CARBON_ENABLE_MINIBEAM "Build Copper minibeam transport support" OFF)
# Extra physics-validation scorers (fluence, survival, reaction-z, ledger files).
# Default OFF keeps production DoseToMedium + LET_d atomics only.
option(CARBON_VALIDATION_SCORERS
       "Compile validation/debug scorers (species fluence, survival, ledger)"
       OFF)

# Comma-separated SYCL targets for -fsycl-targets.
# Empty → compiler default (SPIR-V JIT for Intel).
# NVIDIA CUDA plugin requires nvptx64-nvidia-cuda.
set(CARBON_SYCL_TARGETS "" CACHE STRING
    "SYCL targets for -fsycl-targets (e.g. spir64,nvptx64-nvidia-cuda)")
set(CARBON_CUDA_ARCH "" CACHE STRING
    "Optional NVIDIA GPU arch for AOT (e.g. sm_75); empty = portable PTX")

add_library(carbon_core
    src/cli.cpp
    src/config.cpp
    src/cross_section.cpp
    src/ct_grid.cpp
    src/ct_dicom.cpp
    src/io.cpp
    src/plan_run.cpp
    src/electron_transport.cpp
    src/energy_loss_fluctuation.cpp
    src/fred_table1.cpp
    src/fred_event_library.cpp
    src/inelastic_package_v2.cpp
    src/inelastic_package_v3.cpp
    src/multiple_scattering.cpp
    src/run_quality.cpp
    src/stopping_power.cpp
    src/schneider_rate_table.cpp
    src/schneider_stopping_table.cpp
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
    message(STATUS "CARBON_ENABLE_TRANSPORT_PROFILE=ON")
endif()

if(CARBON_DOSE_FP32)
    target_compile_definitions(carbon_core PUBLIC CARBON_DOSE_FP32=1)
    message(STATUS "CARBON_DOSE_FP32=ON")
else()
    message(STATUS "CARBON_DOSE_FP32=OFF")
endif()

if(CARBON_LET_FP64)
    target_compile_definitions(carbon_core PUBLIC CARBON_LET_FP64=1)
    message(STATUS "CARBON_LET_FP64=ON")
else()
    message(STATUS "CARBON_LET_FP64=OFF (FP32 LET atomics)")
endif()

if(CARBON_ENABLE_MINIBEAM)
    target_compile_definitions(carbon_core PUBLIC CARBON_ENABLE_MINIBEAM=1)
    message(STATUS "CARBON_ENABLE_MINIBEAM=ON")
else()
    message(STATUS "CARBON_ENABLE_MINIBEAM=OFF")
endif()

if(CARBON_VALIDATION_SCORERS)
    target_compile_definitions(carbon_core PUBLIC CARBON_VALIDATION_SCORERS=1)
    message(STATUS "CARBON_VALIDATION_SCORERS=ON")
else()
    message(STATUS "CARBON_VALIDATION_SCORERS=OFF")
endif()

if(CARBON_ENABLE_SYCL)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Clang")
        message(FATAL_ERROR "CARBON_ENABLE_SYCL requires Intel oneAPI DPC++ / Clang (IntelLLVM or Clang).")
    endif()

    target_include_directories(carbon_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_sources(carbon_core PRIVATE
        src/device.cpp
        src/transport_sycl.cpp
        src/transport_profile.cpp)
    set_property(SOURCE src/transport_sycl.cpp APPEND PROPERTY
        COMPILE_DEFINITIONS CARBON_DEFINE_SYCL_CONTEXT=1)
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
        message(STATUS "SYCL targets: compiler default (SPIR-V JIT)")
    endif()

    if(CARBON_CUDA_ARCH)
        if(NOT CARBON_SYCL_TARGETS MATCHES "nvptx64-nvidia-cuda")
            message(WARNING
                "CARBON_CUDA_ARCH is set but CARBON_SYCL_TARGETS has no nvptx64 target")
        endif()
        message(STATUS "CUDA GPU arch: ${CARBON_CUDA_ARCH}")
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

if(BUILD_TESTING)
    add_executable(carbon_tests tests/carbon_tests.cpp)
    add_executable(inelastic_package_v2_tests tests/inelastic_package_v2_tests.cpp)
    target_link_libraries(inelastic_package_v2_tests PRIVATE carbon_core)
    if(CARBON_ENABLE_SYCL)
        target_compile_options(inelastic_package_v2_tests PRIVATE -fsycl)
        if(CARBON_SYCL_TARGETS)
            target_compile_options(inelastic_package_v2_tests PRIVATE
                -fsycl-targets=${CARBON_SYCL_TARGETS})
        endif()
    endif()
    target_link_libraries(carbon_tests PRIVATE carbon_core)
    if(CARBON_ENABLE_SYCL)
        target_compile_options(carbon_tests PRIVATE -fsycl)
        if(CARBON_SYCL_TARGETS)
            target_compile_options(carbon_tests PRIVATE
                -fsycl-targets=${CARBON_SYCL_TARGETS})
        endif()
    endif()
    target_compile_definitions(carbon_tests PRIVATE
        CARBON_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_executable(run_step13_gpu tools/run_step13_gpu.cpp)
    target_link_libraries(run_step13_gpu PRIVATE carbon_core)
    if(CARBON_ENABLE_SYCL)
        target_compile_options(run_step13_gpu PRIVATE -fsycl)
        if(CARBON_SYCL_TARGETS)
            target_compile_options(run_step13_gpu PRIVATE
                -fsycl-targets=${CARBON_SYCL_TARGETS})
        endif()
    endif()

    add_executable(run_step15_gpu tools/run_step15_gpu.cpp)
    target_link_libraries(run_step15_gpu PRIVATE carbon_core)
    if(CARBON_ENABLE_SYCL)
        target_compile_options(run_step15_gpu PRIVATE -fsycl)
        if(CARBON_SYCL_TARGETS)
            target_compile_options(run_step15_gpu PRIVATE
                -fsycl-targets=${CARBON_SYCL_TARGETS})
        endif()
    endif()
    add_test(NAME carbon_tests COMMAND carbon_tests)
    add_test(NAME inelastic_package_v2_tests COMMAND inelastic_package_v2_tests)
    set_tests_properties(carbon_tests PROPERTIES
        TIMEOUT 300
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

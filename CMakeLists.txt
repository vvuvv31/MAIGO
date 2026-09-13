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
    src/electron_joint_response.cpp
    src/water_electron_response.cpp
    src/energy_loss_fluctuation.cpp
    src/unified_em_package.cpp
    src/fred_table1.cpp
    src/fred_event_library.cpp
    src/inelastic_package_v2.cpp
    src/inelastic_package_v3.cpp
    src/multiple_scattering.cpp
    src/run_quality.cpp
    src/stopping_power.cpp
    src/all_ion_elastic.cpp
    src/schneider_rate_table.cpp
    src/material_nuclear_rates.cpp
    src/schneider_delta_tail.cpp
    src/schneider_stopping_table.cpp
    src/schneider_ion_stopping_table.cpp
    src/schneider_target_sampler.cpp
    src/secondary_rate_table.cpp
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
    if(CARBON_ENABLE_SYCL)
        add_executable(unified_em_gpu tests/unified_em_gpu.cpp)
        target_link_libraries(unified_em_gpu PRIVATE carbon_core)
        target_compile_options(unified_em_gpu PRIVATE $<TARGET_PROPERTY:carbon_core,COMPILE_OPTIONS>)
        # Explicit invocation with a verified real table on the local GPU.
        add_executable(all_ion_elastic_gpu tests/all_ion_elastic_gpu.cpp)
        target_link_libraries(all_ion_elastic_gpu PRIVATE carbon_core)
        target_compile_options(all_ion_elastic_gpu PRIVATE $<TARGET_PROPERTY:carbon_core,COMPILE_OPTIONS>)
        add_executable(schneider_ion_stopping_gpu tests/schneider_ion_stopping_gpu.cpp)
        target_link_libraries(schneider_ion_stopping_gpu PRIVATE carbon_core)
        target_compile_options(schneider_ion_stopping_gpu PRIVATE
            $<TARGET_PROPERTY:carbon_core,COMPILE_OPTIONS>)
        add_executable(material_electron_boundary_gpu_smoke tests/material_electron_boundary_gpu_smoke.cpp)
        target_link_libraries(material_electron_boundary_gpu_smoke PRIVATE carbon_core)
        target_compile_options(material_electron_boundary_gpu_smoke PRIVATE
            $<TARGET_PROPERTY:carbon_core,COMPILE_OPTIONS>)
        # Explicit real-data/local-GPU invocation, not an automatic CTest run.
        add_executable(material_electron_density_gpu_smoke tests/material_electron_density_gpu_smoke.cpp)
        target_link_libraries(material_electron_density_gpu_smoke PRIVATE carbon_core)
        target_compile_options(material_electron_density_gpu_smoke PRIVATE
            $<TARGET_PROPERTY:carbon_core,COMPILE_OPTIONS>)
    endif()
    add_executable(unified_em_production tests/unified_em_production.cpp)
    target_link_libraries(unified_em_production PRIVATE carbon_core)
    add_executable(unified_em_density tests/unified_em_density.cpp)
    target_link_libraries(unified_em_density PRIVATE carbon_core)
    add_executable(unified_em_distribution tests/unified_em_distribution.cpp)
    target_link_libraries(unified_em_distribution PRIVATE carbon_core)
    add_executable(unified_em_delta tests/unified_em_delta.cpp)
    target_link_libraries(unified_em_delta PRIVATE carbon_core)
    add_executable(unified_em_mean tests/unified_em_mean.cpp)
    target_link_libraries(unified_em_mean PRIVATE carbon_core)
    add_executable(unified_em_lookup tests/unified_em_lookup.cpp)
    target_link_libraries(unified_em_lookup PRIVATE carbon_core)
    add_executable(primary_em_mean_candidate_unit tests/primary_em_mean_candidate_unit.cpp)
    target_include_directories(primary_em_mean_candidate_unit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME primary_em_mean_candidate_unit COMMAND primary_em_mean_candidate_unit)
    add_executable(restricted_fluctuation_candidate_unit tests/restricted_fluctuation_candidate_unit.cpp)
    target_include_directories(restricted_fluctuation_candidate_unit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME restricted_fluctuation_candidate_unit COMMAND restricted_fluctuation_candidate_unit)
    add_executable(discrete_delta_candidate_unit tests/discrete_delta_candidate_unit.cpp)
    target_include_directories(discrete_delta_candidate_unit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME discrete_delta_candidate_unit COMMAND discrete_delta_candidate_unit)
    if(UNIX)
        foreach(candidate_target primary_em_mean_candidate_unit restricted_fluctuation_candidate_unit discrete_delta_candidate_unit)
            target_link_libraries(${candidate_target} PRIVATE m)
        endforeach()
    endif()
    add_executable(water_electron_response_unit tests/water_electron_response_unit.cpp)
    target_include_directories(water_electron_response_unit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(water_electron_response_unit PRIVATE carbon_core)
    add_test(NAME water_electron_response_unit COMMAND water_electron_response_unit)
    add_test(NAME water_electron_config_guard COMMAND water_electron_response_unit
        ${CMAKE_CURRENT_SOURCE_DIR}/config/unified_water_production.yaml)
    set_tests_properties(water_electron_config_guard PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    add_executable(carbon_tests tests/carbon_tests.cpp)
    add_executable(inelastic_package_v2_tests tests/inelastic_package_v2_tests.cpp)
    target_link_libraries(inelastic_package_v2_tests PRIVATE carbon_core)
    target_include_directories(inelastic_package_v2_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(CARBON_ENABLE_SYCL)
        target_compile_options(inelastic_package_v2_tests PRIVATE -fsycl)
        if(CARBON_SYCL_TARGETS)
            target_compile_options(inelastic_package_v2_tests PRIVATE
                -fsycl-targets=${CARBON_SYCL_TARGETS})
        endif()
    endif()
    target_link_libraries(carbon_tests PRIVATE carbon_core)
    add_executable(test_material_nuclear_rates tests/test_material_nuclear_rates.cpp)
    target_link_libraries(test_material_nuclear_rates PRIVATE carbon_core)
    if(CARBON_ENABLE_SYCL)
        target_compile_options(test_material_nuclear_rates PRIVATE -fsycl)
        if(CARBON_SYCL_TARGETS)
            target_compile_options(test_material_nuclear_rates PRIVATE -fsycl-targets=${CARBON_SYCL_TARGETS})
        endif()
    endif()
    add_test(NAME material_nuclear_rates COMMAND test_material_nuclear_rates)
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

    add_executable(run_step19_gpu tools/run_step19_gpu.cpp)
    target_link_libraries(run_step19_gpu PRIVATE carbon_core)
    if(CARBON_ENABLE_SYCL)
        target_compile_options(run_step19_gpu PRIVATE -fsycl)
        if(CARBON_SYCL_TARGETS)
            target_compile_options(run_step19_gpu PRIVATE
                -fsycl-targets=${CARBON_SYCL_TARGETS})
        endif()
    endif()

    add_executable(run_step21_level3_gpu tools/run_step21_level3_gpu.cpp)
    target_link_libraries(run_step21_level3_gpu PRIVATE carbon_core)
    if(CARBON_ENABLE_SYCL)
        target_compile_options(run_step21_level3_gpu PRIVATE -fsycl)
        if(CARBON_SYCL_TARGETS)
            target_compile_options(run_step21_level3_gpu PRIVATE
                -fsycl-targets=${CARBON_SYCL_TARGETS})
        endif()
    endif()

    add_executable(test_schneider_dicom_reference tests/test_schneider_dicom_reference.cpp)
    target_link_libraries(test_schneider_dicom_reference PRIVATE carbon_core)
    if(CARBON_ENABLE_SYCL)
        target_compile_options(test_schneider_dicom_reference PRIVATE -fsycl)
        if(CARBON_SYCL_TARGETS)
            target_compile_options(test_schneider_dicom_reference PRIVATE
                -fsycl-targets=${CARBON_SYCL_TARGETS})
        endif()
    endif()

    add_executable(dump_tps_phase_space tools/dump_tps_phase_space.cpp)
    target_link_libraries(dump_tps_phase_space PRIVATE carbon_core)

    add_executable(test_secondary_rate_table_hardening tests/test_secondary_rate_table_hardening.cpp)
    target_link_libraries(test_secondary_rate_table_hardening PRIVATE carbon_core)

    add_test(NAME carbon_tests COMMAND carbon_tests)
    add_executable(test_schneider_longitudinal tests/test_schneider_longitudinal.cpp)
    target_link_libraries(test_schneider_longitudinal PRIVATE carbon_core)
    target_compile_definitions(test_schneider_longitudinal PRIVATE
        CARBON_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    if(CARBON_ENABLE_SYCL)
        target_compile_options(test_schneider_longitudinal PRIVATE -fsycl)
        if(CARBON_SYCL_TARGETS)
            target_compile_options(test_schneider_longitudinal PRIVATE
                -fsycl-targets=${CARBON_SYCL_TARGETS})
        endif()
    endif()
    add_test(NAME test_schneider_longitudinal COMMAND test_schneider_longitudinal)
    add_executable(test_schneider_ion_stopping tests/test_schneider_ion_stopping.cpp)
    target_link_libraries(test_schneider_ion_stopping PRIVATE carbon_core)
    add_test(NAME test_schneider_ion_stopping COMMAND test_schneider_ion_stopping)
    add_executable(test_nuclear_elastic_scattering tests/test_nuclear_elastic_scattering.cpp)
    target_link_libraries(test_nuclear_elastic_scattering PRIVATE carbon_core)
    add_test(NAME test_nuclear_elastic_scattering COMMAND test_nuclear_elastic_scattering)
    add_test(NAME inelastic_package_v2_tests COMMAND inelastic_package_v2_tests)
    add_test(NAME test_schneider_dicom_reference COMMAND test_schneider_dicom_reference)
    add_test(NAME test_secondary_rate_table_hardening COMMAND test_secondary_rate_table_hardening)
    set_tests_properties(carbon_tests PROPERTIES
        TIMEOUT 300
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(test_schneider_dicom_reference PROPERTIES
        TIMEOUT 300
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(test_secondary_rate_table_hardening PROPERTIES
        TIMEOUT 300
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

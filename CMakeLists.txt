cmake_minimum_required(VERSION 3.22)

project(carbon_oneapi_mc VERSION 0.1.0 LANGUAGES CXX)


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
# Copper minibeam beamline, alongside the Unified EM water/CT kernel.
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
    src/batch_run.cpp
    src/config.cpp
    src/cross_section.cpp
    src/ct_grid.cpp
    src/ct_dicom.cpp
    src/io.cpp
    src/plan_run.cpp
    src/electron_transport.cpp
    src/electron_joint_response.cpp
    src/water_electron_response.cpp
    src/neutron_package.cpp
    src/unified_em_package.cpp
    src/urban_mcs_package.cpp
    src/delta_moments_data.cpp
    src/inelastic_package_v2.cpp
    src/inelastic_package_v3.cpp
    src/run_quality.cpp
    src/stopping_power.cpp
    src/all_ion_elastic.cpp
    src/schneider_rate_table.cpp
    src/material_nuclear_rates.cpp
    src/schneider_delta_tail.cpp
    src/schneider_target_sampler.cpp
    src/secondary_rate_table.cpp
    src/topas_spots.cpp
    src/tps_source.cpp
    src/transport_cpu.cpp
)

target_include_directories(carbon_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Bind prepared electron artifacts to all host compiler/validator algorithms.
# Reconfigure automatically if a dependency changes; no stale physics cache.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "scripts/material_prepared_sources.txt")
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/material_prepared_sources.txt" material_prepared_sources)
find_path(ZSTD_INCLUDE_DIR NAMES zstd.h REQUIRED)
find_library(ZSTD_LIBRARY NAMES zstd REQUIRED)
target_include_directories(carbon_core PUBLIC ${ZSTD_INCLUDE_DIR})
target_link_libraries(carbon_core PUBLIC ${ZSTD_LIBRARY})
set(material_prepared_fingerprint "")
foreach(source IN LISTS material_prepared_sources)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source}")
    file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${source}" source_sha)
    string(APPEND material_prepared_fingerprint "${source}:${source_sha}\n")
endforeach()
string(SHA256 material_prepared_version "${material_prepared_fingerprint}")
target_compile_definitions(carbon_core PRIVATE CARBON_MATERIAL_PREPARED_VERSION="${material_prepared_version}")

# These options are consumed by the electron transport headers as well as
# the SYCL backend. Presence-tested macros must stay undefined when OFF.
option(CARBON_ELECTRON_REPLAY_REUSE "Reuse checked electron genealogy within a packet transaction" OFF)
option(CARBON_ELECTRON_CROSSING_ROWS32 "Use lossless 32-bit electron crossing row IDs when representable" OFF)
option(CARBON_CT_ELECTRON_QUEUE "Build separate CT electron packet execution" OFF)
foreach(feature ELECTRON_REPLAY_REUSE ELECTRON_CROSSING_ROWS32 CT_ELECTRON_QUEUE)
    if(CARBON_${feature})
        target_compile_definitions(carbon_core PRIVATE CARBON_${feature}=1)
    endif()
endforeach()
option(CARBON_FE_JOINT_QUEUE "Build separate selected FE joint-response replays" OFF)
target_compile_definitions(carbon_core PRIVATE
    CARBON_FE_JOINT_QUEUE=$<BOOL:${CARBON_FE_JOINT_QUEUE}>)

option(CARBON_EM_EXACT_INDEX "Narrow EM searches with exact exponent buckets" ON)
target_compile_definitions(carbon_core PUBLIC CARBON_EM_EXACT_INDEX=$<BOOL:${CARBON_EM_EXACT_INDEX}>)
option(CARBON_EM_EMPTY_BUCKET_FAST_PATH
       "Research: skip exact-index key loads for exponent buckets containing no knots"
       OFF)
option(CARBON_EM_EMPTY_BUCKET_AUDIT
       "Diagnostic: count exact-index queries and empty exponent buckets by table kind"
       OFF)
option(CARBON_EM_SEARCH_KEY_AUDIT
       "Diagnostic: sample actual exact-index comparison addresses for warp-sharing analysis"
       OFF)
option(CARBON_EM_INTERVAL_WIDTH_AUDIT
       "Diagnostic: histogram exact-index narrowed interval widths for node/raw0/raw1"
       OFF)
option(CARBON_EM_PAIR_PREPARE
       "Research: materialize both density endpoints before their independent EM prepare queries"
       OFF)
if((CARBON_EM_EMPTY_BUCKET_FAST_PATH OR CARBON_EM_EMPTY_BUCKET_AUDIT OR
    CARBON_EM_INTERVAL_WIDTH_AUDIT) AND
   NOT CARBON_EM_EXACT_INDEX)
    message(FATAL_ERROR
        "CARBON_EM_EMPTY_BUCKET_FAST_PATH/AUDIT require CARBON_EM_EXACT_INDEX")
endif()
target_compile_definitions(carbon_core PUBLIC
    CARBON_EM_EMPTY_BUCKET_FAST_PATH=$<BOOL:${CARBON_EM_EMPTY_BUCKET_FAST_PATH}>
    CARBON_EM_EMPTY_BUCKET_AUDIT=$<BOOL:${CARBON_EM_EMPTY_BUCKET_AUDIT}>
    CARBON_EM_SEARCH_KEY_AUDIT=$<BOOL:${CARBON_EM_SEARCH_KEY_AUDIT}>
    CARBON_EM_INTERVAL_WIDTH_AUDIT=$<BOOL:${CARBON_EM_INTERVAL_WIDTH_AUDIT}>
    CARBON_EM_PAIR_PREPARE=$<BOOL:${CARBON_EM_PAIR_PREPARE}>)
option(CARBON_USE_OPENSSL_SHA256 "Use OpenSSL accelerated file SHA256 when available" ON)
option(CARBON_DISABLE_INTEGRITY_CHECKS
    "Skip file SHA256 content checks and provenance hashing" OFF)
target_compile_definitions(carbon_core PUBLIC
    CARBON_DISABLE_INTEGRITY_CHECKS=$<BOOL:${CARBON_DISABLE_INTEGRITY_CHECKS}>)
if(CARBON_DISABLE_INTEGRITY_CHECKS)
    message(WARNING "CARBON_DISABLE_INTEGRITY_CHECKS=ON: file content hashes are not verified")
endif()
if(CARBON_USE_OPENSSL_SHA256)
    find_package(OpenSSL QUIET COMPONENTS Crypto)
    if(OpenSSL_FOUND)
        target_link_libraries(carbon_core PUBLIC OpenSSL::Crypto)
        target_compile_definitions(carbon_core PUBLIC CARBON_OPENSSL_SHA256=1)
    else()
        message(STATUS "OpenSSL unavailable: using portable SHA256 with full verification")
    endif()
endif()
# Compile-time ablations; no runtime branches in particle steps.
foreach(feature MATERIAL_CACHE STEP_CACHE LOCAL_AUDIT)
    set(em_feature_default ON)
    if(feature STREQUAL "LOCAL_AUDIT")
        set(em_feature_default OFF)
    endif()
    option(CARBON_EM_${feature} "Unified EM performance feature ${feature}" ${em_feature_default})
    target_compile_definitions(carbon_core PUBLIC
        CARBON_EM_${feature}=$<BOOL:${CARBON_EM_${feature}}>)
endforeach()
option(CARBON_EM_SPLIT_SEARCH_KEYS "Separate EM node/segment binary-search key arrays (candidate A)" OFF)
target_compile_definitions(carbon_core PUBLIC
    CARBON_EM_SPLIT_SEARCH_KEYS=$<BOOL:${CARBON_EM_SPLIT_SEARCH_KEYS}>)
set(CARBON_EM_AUDIT_SHARDS "64" CACHE STRING
    "Unified EM diagnostic counter shards per counter (positive integer)")
if(NOT CARBON_EM_AUDIT_SHARDS MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "CARBON_EM_AUDIT_SHARDS must be a positive integer")
endif()
target_compile_definitions(carbon_core PUBLIC
    CARBON_EM_AUDIT_SHARDS=${CARBON_EM_AUDIT_SHARDS})
option(CARBON_SECONDARY_STEP_PROFILE "Per-species secondary step profile (diagnostic only)" OFF)
target_compile_definitions(carbon_core PUBLIC
    CARBON_SECONDARY_STEP_PROFILE=$<BOOL:${CARBON_SECONDARY_STEP_PROFILE}>)
option(CARBON_SECONDARY_CONTEXT_POINTER
       "Research: pass immutable secondary transport parameters through a device context pointer"
       OFF)
target_compile_definitions(carbon_core PUBLIC
    CARBON_SECONDARY_CONTEXT_POINTER=$<BOOL:${CARBON_SECONDARY_CONTEXT_POINTER}>)
option(CARBON_SECONDARY_EXPLICIT_ND_RANGE
       "Research: launch the small-closure secondary kernel through an explicit nd_range"
       OFF)
set(CARBON_SECONDARY_ND_RANGE_SIZE "64" CACHE STRING
    "Work-group size for the explicit secondary nd_range candidate (32/64/128)")
set_property(CACHE CARBON_SECONDARY_ND_RANGE_SIZE PROPERTY STRINGS 32 64 128)
if(NOT CARBON_SECONDARY_ND_RANGE_SIZE MATCHES "^(32|64|128)$")
    message(FATAL_ERROR "CARBON_SECONDARY_ND_RANGE_SIZE must be 32, 64 or 128")
endif()
if(CARBON_SECONDARY_EXPLICIT_ND_RANGE AND NOT CARBON_SECONDARY_CONTEXT_POINTER)
    message(FATAL_ERROR
        "CARBON_SECONDARY_EXPLICIT_ND_RANGE requires CARBON_SECONDARY_CONTEXT_POINTER")
endif()
target_compile_definitions(carbon_core PUBLIC
    CARBON_SECONDARY_EXPLICIT_ND_RANGE=$<BOOL:${CARBON_SECONDARY_EXPLICIT_ND_RANGE}>
    CARBON_SECONDARY_ND_RANGE_SIZE=${CARBON_SECONDARY_ND_RANGE_SIZE})
option(CARBON_SECONDARY_REUSE_PROJECTILE_INDEX
       "Research: reuse the per-launch immutable secondary projectile registry index"
       OFF)
target_compile_definitions(carbon_core PUBLIC
    CARBON_SECONDARY_REUSE_PROJECTILE_INDEX=$<BOOL:${CARBON_SECONDARY_REUSE_PROJECTILE_INDEX}>)
option(CARBON_SECONDARY_PRODUCTION_SPECIALIZE
       "Research: host-dispatch a secondary kernel specialized for the validated CT production path"
       OFF)
if(CARBON_SECONDARY_PRODUCTION_SPECIALIZE AND NOT CARBON_SECONDARY_CONTEXT_POINTER)
    message(FATAL_ERROR
        "CARBON_SECONDARY_PRODUCTION_SPECIALIZE requires CARBON_SECONDARY_CONTEXT_POINTER")
endif()
target_compile_definitions(carbon_core PUBLIC
    CARBON_SECONDARY_PRODUCTION_SPECIALIZE=$<BOOL:${CARBON_SECONDARY_PRODUCTION_SPECIALIZE}>)
option(CARBON_SECONDARY_KNOWN_SPECIES_PROBE
       "Compile-only probe: remove generic-recoil transport from the specialized secondary kernel"
       OFF)
if(CARBON_SECONDARY_KNOWN_SPECIES_PROBE AND NOT CARBON_SECONDARY_PRODUCTION_SPECIALIZE)
    message(FATAL_ERROR
        "CARBON_SECONDARY_KNOWN_SPECIES_PROBE requires CARBON_SECONDARY_PRODUCTION_SPECIALIZE")
endif()
target_compile_definitions(carbon_core PUBLIC
    CARBON_SECONDARY_KNOWN_SPECIES_PROBE=$<BOOL:${CARBON_SECONDARY_KNOWN_SPECIES_PROBE}>)
option(CARBON_SECONDARY_NON_HE4_PROBE
       "Compile-only probe: remove He-4 hazard auditing from the known-species kernel"
       OFF)
if(CARBON_SECONDARY_NON_HE4_PROBE AND NOT CARBON_SECONDARY_KNOWN_SPECIES_PROBE)
    message(FATAL_ERROR
        "CARBON_SECONDARY_NON_HE4_PROBE requires CARBON_SECONDARY_KNOWN_SPECIES_PROBE")
endif()
target_compile_definitions(carbon_core PUBLIC
    CARBON_SECONDARY_NON_HE4_PROBE=$<BOOL:${CARBON_SECONDARY_NON_HE4_PROBE}>)
set(CARBON_SECONDARY_EXACT_SPECIES_PROBE "-1" CACHE STRING
    "Compile-only species probe (-1 disables, 0..17 exact species, -2 Hydrogen p+d class)"
)
if(NOT CARBON_SECONDARY_EXACT_SPECIES_PROBE MATCHES "^(-1|-2|[0-9]|1[0-7])$")
    message(FATAL_ERROR
        "CARBON_SECONDARY_EXACT_SPECIES_PROBE must be -1 (off), -2 (Hydrogen class) or a charged-species index 0..17")
endif()
if(NOT CARBON_SECONDARY_EXACT_SPECIES_PROBE STREQUAL "-1" AND
   NOT CARBON_SECONDARY_KNOWN_SPECIES_PROBE)
    message(FATAL_ERROR
        "CARBON_SECONDARY_EXACT_SPECIES_PROBE requires CARBON_SECONDARY_KNOWN_SPECIES_PROBE")
endif()
target_compile_definitions(carbon_core PUBLIC
    CARBON_SECONDARY_EXACT_SPECIES_PROBE=${CARBON_SECONDARY_EXACT_SPECIES_PROBE})
option(CARBON_SECONDARY_HOT_SPECIES_SPECIALIZE
       "Research: dispatch proton/deuteron continuation slices to exact-species kernels"
       OFF)
if(CARBON_SECONDARY_HOT_SPECIES_SPECIALIZE AND
   NOT CARBON_SECONDARY_PRODUCTION_SPECIALIZE)
    message(FATAL_ERROR
        "CARBON_SECONDARY_HOT_SPECIES_SPECIALIZE requires CARBON_SECONDARY_PRODUCTION_SPECIALIZE")
endif()
if(CARBON_SECONDARY_HOT_SPECIES_SPECIALIZE AND
   (CARBON_SECONDARY_KNOWN_SPECIES_PROBE OR
    NOT CARBON_SECONDARY_EXACT_SPECIES_PROBE STREQUAL "-1"))
    message(FATAL_ERROR
        "CARBON_SECONDARY_HOT_SPECIES_SPECIALIZE cannot be combined with compile-only species probes")
endif()
target_compile_definitions(carbon_core PUBLIC
    CARBON_SECONDARY_HOT_SPECIES_SPECIALIZE=$<BOOL:${CARBON_SECONDARY_HOT_SPECIES_SPECIALIZE}>)
option(CARBON_PRIMARY_PRODUCTION_SPECIALIZE
       "Research: host-dispatch a primary kernel specialized for the validated CT production path"
       OFF)
target_compile_definitions(carbon_core PUBLIC
    CARBON_PRIMARY_PRODUCTION_SPECIALIZE=$<BOOL:${CARBON_PRIMARY_PRODUCTION_SPECIALIZE}>)
option(CARBON_PRIMARY_CONTEXT_POINTER
       "Pass immutable primary-kernel closures via a device context pointer (needed for Intel Arc 2048 B kernel-arg limit)"
       OFF)
target_compile_definitions(carbon_core PUBLIC
    CARBON_PRIMARY_CONTEXT_POINTER=$<BOOL:${CARBON_PRIMARY_CONTEXT_POINTER}>)
option(CARBON_PRIMARY_CONTEXT_MOVE_UNIFIED
       "With CARBON_PRIMARY_CONTEXT_POINTER, also move the unified EM view out of the argument list"
       ON)
target_compile_definitions(carbon_core PUBLIC
    CARBON_PRIMARY_CONTEXT_MOVE_UNIFIED=$<BOOL:${CARBON_PRIMARY_CONTEXT_MOVE_UNIFIED}>)

set(CARBON_SECONDARY_SEGMENT_STEPS "16" CACHE STRING
    "Complete secondary iterations per continuation launch (8/16/32/64/128; production default 16)")
set_property(CACHE CARBON_SECONDARY_SEGMENT_STEPS PROPERTY STRINGS 8 16 32 64 128)
if(NOT CARBON_SECONDARY_SEGMENT_STEPS MATCHES "^(8|16|32|64|128)$")
    message(FATAL_ERROR "CARBON_SECONDARY_SEGMENT_STEPS must be 8, 16, 32, 64 or 128")
endif()
target_compile_definitions(carbon_core PUBLIC
    CARBON_SECONDARY_SEGMENT_STEPS=${CARBON_SECONDARY_SEGMENT_STEPS})

set(CARBON_SECONDARY_HOT_SEGMENT_STEPS "16" CACHE STRING
    "Complete iterations per continuation launch for exact proton/deuteron slices (8/16/32/64/128; production default 16)")
set_property(CACHE CARBON_SECONDARY_HOT_SEGMENT_STEPS PROPERTY STRINGS 8 16 32 64 128)
if(NOT CARBON_SECONDARY_HOT_SEGMENT_STEPS MATCHES "^(8|16|32|64|128)$")
    message(FATAL_ERROR "CARBON_SECONDARY_HOT_SEGMENT_STEPS must be 8, 16, 32, 64 or 128")
endif()
target_compile_definitions(carbon_core PUBLIC
    CARBON_SECONDARY_HOT_SEGMENT_STEPS=${CARBON_SECONDARY_HOT_SEGMENT_STEPS})

if(MSVC)
    target_compile_options(carbon_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(carbon_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(CARBON_ENABLE_TRANSPORT_PROFILE)
    target_compile_definitions(carbon_core PUBLIC CARBON_TRANSPORT_PROFILE=1)
    message(STATUS "CARBON_TRANSPORT_PROFILE=ON")
endif()
option(CARBON_SECONDARY_STEP_PROFILE "Per-species secondary step profile (diagnostic only)" OFF)
target_compile_definitions(carbon_core PUBLIC
    CARBON_SECONDARY_STEP_PROFILE=$<BOOL:${CARBON_SECONDARY_STEP_PROFILE}>)

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
    option(CARBON_ELECTRON_SEGMENT_DIAGNOSTICS "Sample unchanged charged states for condensed-segment feasibility (not for timing)" OFF)
    if(CARBON_ELECTRON_SEGMENT_DIAGNOSTICS)
        target_compile_definitions(carbon_core PRIVATE CARBON_ELECTRON_SEGMENT_DIAGNOSTICS=1)
    endif()
    option(CARBON_ELECTRON_RANGE_DIAGNOSTICS "Observe range-guard counterfactual dose without changing transport (not for timing)" OFF)
    if(CARBON_ELECTRON_RANGE_DIAGNOSTICS)
        target_compile_definitions(carbon_core PRIVATE CARBON_ELECTRON_RANGE_DIAGNOSTICS=1)
    endif()
    option(CARBON_ELECTRON_VOXEL_CACHE "Research exact-physics cross-step electron voxel/material cache" OFF)
    if(CARBON_ELECTRON_VOXEL_CACHE)
        target_compile_definitions(carbon_core PRIVATE CARBON_ELECTRON_VOXEL_CACHE=1)
    endif()
    option(CARBON_ELECTRON_FP64_GEOMETRY
           "Precision ablation: FP64 electron positions and faces, FP32 physics/scoring" OFF)
    if(CARBON_ELECTRON_FP64_GEOMETRY)
        target_compile_definitions(carbon_core PRIVATE CARBON_ELECTRON_FP64_GEOMETRY=1)
    endif()
    option(CARBON_ELECTRON_FP64_LOSS_GATE
           "Precision ablation: FP64 mean-product and Urban small-loss gate, FP32 sampling/scoring" OFF)
    if(CARBON_ELECTRON_FP64_LOSS_GATE)
        target_compile_definitions(carbon_core PRIVATE CARBON_ELECTRON_FP64_LOSS_GATE=1)
    endif()
    option(CARBON_ELECTRON_CERTIFIED_RATES
           "Validate FP32 hard-rate majorants once before transport" OFF)
    if(CARBON_ELECTRON_CERTIFIED_RATES)
        target_compile_definitions(carbon_core PRIVATE CARBON_ELECTRON_CERTIFIED_RATES=1)
    endif()
    option(CARBON_ELECTRON_REUSE_PARAMETERS
           "Reuse same-step FP32 electron material parameters" OFF)
    if(CARBON_ELECTRON_REUSE_PARAMETERS)
        target_compile_definitions(carbon_core PRIVATE CARBON_ELECTRON_REUSE_PARAMETERS=1)
    endif()
    option(CARBON_ELECTRON_STEP_DIAGNOSTICS
           "Diagnostic-only FP32 electron energy/step-limiter histogram (not for timing)" OFF)
    if(CARBON_ELECTRON_STEP_DIAGNOSTICS)
        target_compile_definitions(carbon_core PRIVATE CARBON_ELECTRON_STEP_DIAGNOSTICS=1)
    endif()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    if(CARBON_ELECTRON_VOXEL_CACHE)
        add_custom_target(carbon_electron_voxel_cache_generated_check
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_electron_voxel_cache.py --check
            COMMENT "Checking exact-physics electron voxel-cache wrapper" VERBATIM)
        add_dependencies(carbon_core carbon_electron_voxel_cache_generated_check)
    endif()
    add_custom_target(carbon_electron_fp32_generated_check
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_electron_fp32.py --check
        COMMENT "Checking independent FP32 electron equations against their sources"
        VERBATIM)
    add_dependencies(carbon_core carbon_electron_fp32_generated_check)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Clang")
        message(FATAL_ERROR "CARBON_ENABLE_SYCL requires Intel oneAPI DPC++ / Clang (IntelLLVM or Clang).")
    endif()

    target_include_directories(carbon_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_sources(carbon_core PRIVATE
        src/device.cpp
        src/neutron_transport_sycl.cpp
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
if(CARBON_ENABLE_SYCL)
    target_compile_options(carbon_mc PRIVATE $<TARGET_PROPERTY:carbon_core,COMPILE_OPTIONS>)
endif()

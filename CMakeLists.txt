cmake_minimum_required(VERSION 3.22)

project(carbon_oneapi_mc VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CARBON_ENABLE_SYCL "Build the oneAPI/SYCL transport backend" OFF)
option(CARBON_BUILD_TESTS "Build unit and integration tests" ON)

add_library(carbon_core
    src/config.cpp
    src/cross_section.cpp
    src/io.cpp
    src/stopping_power.cpp
    src/transport_cpu.cpp
)

target_include_directories(carbon_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MSVC)
    target_compile_options(carbon_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(carbon_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(CARBON_ENABLE_SYCL)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
        message(FATAL_ERROR "CARBON_ENABLE_SYCL requires Intel oneAPI icpx (IntelLLVM).")
    endif()

    target_sources(carbon_core PRIVATE src/device.cpp src/transport_sycl.cpp)
    target_compile_definitions(carbon_core PUBLIC CARBON_HAS_SYCL=1)
    target_compile_options(carbon_core PRIVATE -fsycl)
    target_link_options(carbon_core PUBLIC -fsycl)
endif()

add_executable(carbon_mc src/main.cpp)
target_link_libraries(carbon_mc PRIVATE carbon_core)

if(CARBON_BUILD_TESTS)
    enable_testing()
    add_executable(carbon_tests tests/carbon_tests.cpp)
    target_link_libraries(carbon_tests PRIVATE carbon_core)
    add_test(NAME carbon_tests COMMAND carbon_tests)
endif()

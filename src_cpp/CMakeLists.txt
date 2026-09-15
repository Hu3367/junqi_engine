cmake_minimum_required(VERSION 3.16)
project(junqi_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    add_compile_options(/W4 /O2 /permissive-)
else()
    add_compile_options(-Wall -Wextra -O3)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CORE_SOURCES
    src/board.cpp
    src/rules.cpp
    src/zobrist.cpp
    src/eval_apk.cpp
    src/eval_expert.cpp
    src/eval_expert_tables.cpp
    src/expert_qsearch.cpp
    src/apk_engine.cpp
)

# 1. 纯 C++ 核心静态库
add_library(junqi_static STATIC ${CORE_SOURCES})

# 2. Python 扩展模块 (junqi_core)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    # 尝试通过 Python 查找 pybind11 include 路径
    find_package(Python3 COMPONENTS Interpreter Development QUIET)
    if(Python3_FOUND)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        find_package(pybind11 CONFIG QUIET)
    endif()
endif()

if(pybind11_FOUND)
    pybind11_add_module(junqi_core bindings/python_bindings.cpp)
    target_link_libraries(junqi_core PRIVATE junqi_static)
    message(STATUS "junqi_core Python module configured successfully.")
else()
    message(WARNING "pybind11 not found! Python bindings 'junqi_core' target will not be built.")
endif()

# 3. 独立 C++ 基准性能测试与自检测试程序
add_executable(junqi_bench tests/bench_main.cpp)
target_link_libraries(junqi_bench PRIVATE junqi_static)

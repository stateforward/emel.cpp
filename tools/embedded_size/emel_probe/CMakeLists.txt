cmake_minimum_required(VERSION 3.20)

project(emel_qwen3_e2e_probe_project LANGUAGES C CXX)

if(NOT DEFINED EMEL_ROOT_DIR)
  message(FATAL_ERROR "EMEL_ROOT_DIR must point at the emel.cpp repository root")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(EMEL_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory("${EMEL_ROOT_DIR}" emel_root)

add_executable(emel_qwen3_e2e_probe
  main.cpp
)

target_link_libraries(emel_qwen3_e2e_probe
  PRIVATE
    emel_core
    emel
)

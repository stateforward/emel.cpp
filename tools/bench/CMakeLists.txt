cmake_minimum_required(VERSION 3.20)
project(emel_bench_tools C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT EMEL_ROOT)
  get_filename_component(EMEL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
endif()

set(REF_IMPL_REPOSITORY "https://github.com/ggml-org/llama.cpp.git" CACHE STRING "")
set(REF_IMPL_REF "master" CACHE STRING "")

include(FetchContent)
FetchContent_Declare(
  reference_impl
  GIT_REPOSITORY ${REF_IMPL_REPOSITORY}
  GIT_TAG ${REF_IMPL_REF}
)
FetchContent_GetProperties(reference_impl)
if(NOT reference_impl_POPULATED)
  FetchContent_Populate(reference_impl)
endif()

set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${reference_impl_SOURCE_DIR}/ggml ggml)

set(EMEL_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${EMEL_ROOT} emel)

add_executable(bench_runner
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_main.cpp
)

target_link_libraries(bench_runner
  PRIVATE
    emel_core
    emel
    ggml
)

target_include_directories(bench_runner
  PRIVATE
    ${EMEL_ROOT}/src
    ${EMEL_ROOT}/include
    ${reference_impl_SOURCE_DIR}/ggml/include
)

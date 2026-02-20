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

set(EMEL_ENABLE_TESTS ON CACHE BOOL "" FORCE)
add_subdirectory(${EMEL_ROOT} emel)

add_executable(reference_allocator_bench
  ${CMAKE_CURRENT_SOURCE_DIR}/reference_allocator_bench.cpp
)

target_link_libraries(reference_allocator_bench
  PRIVATE
    ggml
)

target_include_directories(reference_allocator_bench
  PRIVATE
    ${reference_impl_SOURCE_DIR}/ggml/include
)

add_custom_target(bench_compare
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_compare.sh
          $<TARGET_FILE:emel_bench_bin>
          $<TARGET_FILE:reference_allocator_bench>
  DEPENDS emel_bench_bin reference_allocator_bench
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

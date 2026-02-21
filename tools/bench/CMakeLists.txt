cmake_minimum_required(VERSION 3.20)
project(emel_bench_tools C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT EMEL_ROOT)
  get_filename_component(EMEL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
endif()

set(REF_IMPL_REPOSITORY "https://github.com/ggml-org/llama.cpp.git" CACHE STRING "")
set(REF_IMPL_REF "" CACHE STRING "")
if(REF_IMPL_REF STREQUAL "")
  set(REF_IMPL_REF_FILE "${CMAKE_CURRENT_SOURCE_DIR}/reference_ref.txt")
  if(EXISTS "${REF_IMPL_REF_FILE}")
    file(READ "${REF_IMPL_REF_FILE}" REF_IMPL_REF_CONTENT)
    string(STRIP "${REF_IMPL_REF_CONTENT}" REF_IMPL_REF_CONTENT)
    set(REF_IMPL_REF "${REF_IMPL_REF_CONTENT}" CACHE STRING "" FORCE)
  else()
    set(REF_IMPL_REF "master" CACHE STRING "" FORCE)
  endif()
endif()

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

set(LLAMA_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(LLAMA_FATAL_WARNINGS OFF CACHE BOOL "" FORCE)
include(${reference_impl_SOURCE_DIR}/ggml/cmake/common.cmake)

function(llama_add_compile_flags)
  if(LLAMA_FATAL_WARNINGS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      list(APPEND C_FLAGS -Werror)
      list(APPEND CXX_FLAGS -Werror)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      add_compile_options(/WX)
    endif()
  endif()

  if(LLAMA_ALL_WARNINGS)
    if(NOT MSVC)
      list(APPEND C_FLAGS -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes
                          -Werror=implicit-int -Werror=implicit-function-declaration)

      list(APPEND CXX_FLAGS -Wmissing-declarations -Wmissing-noreturn)

      list(APPEND WARNING_FLAGS -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function)

      list(APPEND C_FLAGS ${WARNING_FLAGS})
      list(APPEND CXX_FLAGS ${WARNING_FLAGS})

      ggml_get_flags(${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})

      add_compile_options("$<$<COMPILE_LANGUAGE:C>:${C_FLAGS};${GF_C_FLAGS}>"
                          "$<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS};${GF_CXX_FLAGS}>")
    else()
      set(C_FLAGS "" PARENT_SCOPE)
      set(CXX_FLAGS "" PARENT_SCOPE)
    endif()
  endif()

  if(NOT MSVC)
    if(LLAMA_SANITIZE_THREAD)
      message(STATUS "Using -fsanitize=thread")
      add_compile_options(-fsanitize=thread)
      link_libraries(-fsanitize=thread)
    endif()

    if(LLAMA_SANITIZE_ADDRESS)
      message(STATUS "Using -fsanitize=address")
      add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
      link_libraries(-fsanitize=address)
    endif()

    if(LLAMA_SANITIZE_UNDEFINED)
      message(STATUS "Using -fsanitize=undefined")
      add_compile_options(-fsanitize=undefined)
      link_libraries(-fsanitize=undefined)
    endif()
  endif()
endfunction()

set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_NUMBER 0 CACHE STRING "" FORCE)
set(LLAMA_INSTALL_VERSION 0.0.0 CACHE STRING "" FORCE)
add_subdirectory(${reference_impl_SOURCE_DIR}/ggml ggml)
add_subdirectory(${reference_impl_SOURCE_DIR}/src llama_src)

set(EMEL_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${EMEL_ROOT} emel)

add_executable(bench_runner
  ${CMAKE_CURRENT_SOURCE_DIR}/batch_splitter_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/buffer_allocator_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/jinja_parser_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/jinja_renderer_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/memory_coordinator_recurrent_bench.cpp
  ${reference_impl_SOURCE_DIR}/common/jinja/lexer.cpp
  ${reference_impl_SOURCE_DIR}/common/jinja/parser.cpp
  ${reference_impl_SOURCE_DIR}/common/jinja/runtime.cpp
  ${reference_impl_SOURCE_DIR}/common/jinja/value.cpp
  ${reference_impl_SOURCE_DIR}/common/jinja/string.cpp
  ${reference_impl_SOURCE_DIR}/common/jinja/caps.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_cases.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_common.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_main.cpp
)

target_link_libraries(bench_runner
  PRIVATE
    emel_core
    emel
    ggml
    llama
)

target_include_directories(bench_runner
  PRIVATE
    ${EMEL_ROOT}/src
    ${EMEL_ROOT}/include
    ${reference_impl_SOURCE_DIR}/src
    ${reference_impl_SOURCE_DIR}/common
    ${reference_impl_SOURCE_DIR}/ggml/include
    ${reference_impl_SOURCE_DIR}/include
)

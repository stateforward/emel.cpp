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
set(BENCH_REFERENCE_SOURCE "cmake_fetch")
set(BENCH_REFERENCE_REF_VALUE "${REF_IMPL_REF}")
string(REGEX MATCH "^[0-9A-Fa-f]{7,40}$" REF_IMPL_REF_IS_HASH "${REF_IMPL_REF}")
FetchContent_Declare(
  reference_impl
  GIT_REPOSITORY ${REF_IMPL_REPOSITORY}
  GIT_TAG ${REF_IMPL_REF}
)
FetchContent_GetProperties(reference_impl)
if(NOT reference_impl_POPULATED)
  FetchContent_Populate(reference_impl)
endif()
if(REF_IMPL_REF_IS_HASH STREQUAL "")
  execute_process(
    COMMAND git -C "${reference_impl_SOURCE_DIR}" fetch --depth 1 origin "${REF_IMPL_REF}"
    RESULT_VARIABLE REF_IMPL_FETCH_STATUS
    OUTPUT_QUIET
    ERROR_QUIET
  )
  if(REF_IMPL_FETCH_STATUS EQUAL 0)
    execute_process(
      COMMAND git -C "${reference_impl_SOURCE_DIR}" checkout --detach FETCH_HEAD
      RESULT_VARIABLE REF_IMPL_CHECKOUT_STATUS
      OUTPUT_QUIET
      ERROR_QUIET
    )
    if(REF_IMPL_CHECKOUT_STATUS EQUAL 0)
      set(BENCH_REFERENCE_SOURCE "cmake_fetch_latest")
    endif()
  endif()
endif()
execute_process(
  COMMAND git -C "${reference_impl_SOURCE_DIR}" rev-parse HEAD
  OUTPUT_VARIABLE BENCH_REFERENCE_REF_VALUE
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(BENCH_REFERENCE_REF_VALUE STREQUAL "")
  set(BENCH_REFERENCE_REF_VALUE "${REF_IMPL_REF}")
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
if(EXISTS "${reference_impl_SOURCE_DIR}/tools/mtmd/CMakeLists.txt")
  add_subdirectory(${reference_impl_SOURCE_DIR}/tools/mtmd mtmd_tools)
else()
  message(STATUS "Reference mtmd tools unavailable; embedding_reference_bench_runner target disabled")
endif()

set(EMEL_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${EMEL_ROOT} emel)

find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp)

set(BENCH_RUNNER_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/batch/planner_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/generation_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gbnf/rule_parser_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/aarch64_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/flash_attention_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/x86_64_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/logits/sampler_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/logits/validator_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/memory/kv_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/memory/recurrent_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/memory/hybrid_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sm_any_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/text/jinja/parser_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/text/jinja/formatter_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/text/tokenizer/preprocessor/bpe_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/text/tokenizer/preprocessor/spm_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/text/tokenizer/preprocessor/ugm_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/text/tokenizer/preprocessor/wpm_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/text/tokenizer/preprocessor/rwkv_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/text/tokenizer/preprocessor/plamo2_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/text/encoders/bpe_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/text/encoders/spm_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/text/encoders/wpm_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/text/encoders/ugm_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/text/encoders/rwkv_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/text/encoders/plamo2_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/text/encoders/fallback_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/text/tokenizer/tokenizer_bench.cpp
  ${reference_impl_SOURCE_DIR}/common/jinja/lexer.cpp
  ${reference_impl_SOURCE_DIR}/common/jinja/parser.cpp
  ${reference_impl_SOURCE_DIR}/common/jinja/runtime.cpp
  ${reference_impl_SOURCE_DIR}/common/jinja/value.cpp
  ${reference_impl_SOURCE_DIR}/common/jinja/string.cpp
  ${reference_impl_SOURCE_DIR}/common/jinja/caps.cpp
  ${reference_impl_SOURCE_DIR}/common/unicode.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_cases.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_common.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/bench_common.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/logits/bench_common.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_main.cpp
)

add_executable(bench_runner
  ${BENCH_RUNNER_SOURCES}
)

target_link_libraries(bench_runner
  PRIVATE
    emel_core
    emel
    ggml
    llama
)

target_compile_definitions(bench_runner
  PRIVATE
    EMEL_BENCH_REPO_ROOT=\"${EMEL_ROOT}\"
    BENCH_REFERENCE_SOURCE=\"${BENCH_REFERENCE_SOURCE}\"
    BENCH_REFERENCE_REF=\"${BENCH_REFERENCE_REF_VALUE}\"
    EMEL_BENCH_EMEL_ARTIFACT_PATH=\"$<TARGET_FILE:emel>\"
    EMEL_BENCH_LLAMA_ARTIFACT_PATH=\"$<TARGET_FILE:llama>\"
)

target_include_directories(bench_runner
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EMEL_ROOT}/src
    ${EMEL_ROOT}/include
    ${reference_impl_SOURCE_DIR}/common
    ${reference_impl_SOURCE_DIR}/src
    ${reference_impl_SOURCE_DIR}/ggml/include
    ${reference_impl_SOURCE_DIR}/include
    ${NLOHMANN_JSON_INCLUDE_DIR}
)

set(DOCTEST_INCLUDE_DIR ${EMEL_ROOT}/third_party/doctest)
if(NOT EXISTS ${DOCTEST_INCLUDE_DIR}/doctest/doctest.h)
  message(FATAL_ERROR "Missing doctest header at ${DOCTEST_INCLUDE_DIR}/doctest/doctest.h")
endif()

add_executable(bench_runner_tests
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_runner_tests.cpp
)

add_dependencies(bench_runner_tests bench_runner)

target_include_directories(bench_runner_tests
  PRIVATE
    ${DOCTEST_INCLUDE_DIR}
)

target_compile_definitions(bench_runner_tests
  PRIVATE
    BENCH_REPO_ROOT=\"${EMEL_ROOT}\"
    BENCH_RUNNER_BINARY_PATH=\"$<TARGET_FILE:bench_runner>\"
)

enable_testing()
add_test(
  NAME bench_runner_tests
  COMMAND bench_runner_tests
)

add_executable(gbnf_bench_runner
  ${CMAKE_CURRENT_SOURCE_DIR}/gbnf/rule_parser_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gbnf/bench_main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_cases.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_common.hpp
)

target_link_libraries(gbnf_bench_runner
  PRIVATE
    emel_core
    emel
    ggml
    llama
)

target_include_directories(gbnf_bench_runner
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EMEL_ROOT}/src
    ${EMEL_ROOT}/include
    ${reference_impl_SOURCE_DIR}/src
    ${reference_impl_SOURCE_DIR}/common
    ${reference_impl_SOURCE_DIR}/ggml/include
    ${reference_impl_SOURCE_DIR}/include
)

add_executable(embedding_generator_bench_runner
  ${CMAKE_CURRENT_SOURCE_DIR}/embedding_generator_bench.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/embedding_generator_bench_main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_common.hpp
)

target_link_libraries(embedding_generator_bench_runner
  PRIVATE
    emel_core
    emel
)

target_include_directories(embedding_generator_bench_runner
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EMEL_ROOT}
    ${EMEL_ROOT}/src
    ${EMEL_ROOT}/include
)

if(TARGET mtmd)
  add_executable(embedding_reference_bench_runner
    ${CMAKE_CURRENT_SOURCE_DIR}/embedding_reference_bench.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/embedding_reference_bench_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bench_common.hpp
  )

  target_link_libraries(embedding_reference_bench_runner
    PRIVATE
      mtmd
      ggml
      llama
  )

  target_compile_definitions(embedding_reference_bench_runner
    PRIVATE
      EMBEDDING_REFERENCE_SOURCE=\"${BENCH_REFERENCE_SOURCE}\"
      EMBEDDING_REFERENCE_REF=\"${BENCH_REFERENCE_REF_VALUE}\"
      EMBEDDING_REFERENCE_REPOSITORY=\"${REF_IMPL_REPOSITORY}\"
  )

  target_include_directories(embedding_reference_bench_runner
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${reference_impl_SOURCE_DIR}/tools/mtmd
      ${reference_impl_SOURCE_DIR}/src
      ${reference_impl_SOURCE_DIR}/ggml/include
      ${reference_impl_SOURCE_DIR}/include
  )
endif()

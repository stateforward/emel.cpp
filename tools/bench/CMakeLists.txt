cmake_minimum_required(VERSION 3.20)
project(emel_bench_tools C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT EMEL_ROOT)
  get_filename_component(EMEL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
endif()

set(EMEL_BENCH_SUITE_FILTER "" CACHE STRING "Build only one bench_runner suite")
set(REF_IMPL_REPOSITORY "https://github.com/ggml-org/llama.cpp.git" CACHE STRING "")
set(REF_IMPL_REF "" CACHE STRING "")
set(EMEL_BENCH_NEEDS_LLAMA ON)
if(NOT EMEL_BENCH_SUITE_FILTER STREQUAL "")
  set(EMEL_BENCH_NEEDS_LLAMA OFF)
endif()
if(EMEL_BENCH_SUITE_FILTER STREQUAL "generation" OR
   EMEL_BENCH_SUITE_FILTER STREQUAL "gbnf_rule_parser" OR
   EMEL_BENCH_SUITE_FILTER STREQUAL "flash_attention" OR
   EMEL_BENCH_SUITE_FILTER STREQUAL "kernel_x86_64" OR
   EMEL_BENCH_SUITE_FILTER STREQUAL "kernel_aarch64")
  set(EMEL_BENCH_NEEDS_LLAMA ON)
endif()

set(BENCH_REFERENCE_SOURCE "cmake_fetch")
set(BENCH_REFERENCE_REF_VALUE "not_used")

if(EMEL_BENCH_NEEDS_LLAMA)
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
endif()

set(EMEL_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${EMEL_ROOT} emel)

find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp)

if(EMEL_BENCH_NEEDS_LLAMA)
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
endif()

set(BENCH_RUNNER_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_disabled_cases.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_cases.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_common.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/bench_common.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/logits/bench_common.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_main.cpp
)
if(EMEL_BENCH_NEEDS_LLAMA)
  list(APPEND BENCH_RUNNER_SOURCES
    ${reference_impl_SOURCE_DIR}/common/jinja/lexer.cpp
    ${reference_impl_SOURCE_DIR}/common/jinja/parser.cpp
    ${reference_impl_SOURCE_DIR}/common/jinja/runtime.cpp
    ${reference_impl_SOURCE_DIR}/common/jinja/value.cpp
    ${reference_impl_SOURCE_DIR}/common/jinja/string.cpp
    ${reference_impl_SOURCE_DIR}/common/jinja/caps.cpp
    ${reference_impl_SOURCE_DIR}/common/unicode.cpp
  )
endif()
set(BENCH_RUNNER_COMPILE_DEFINITIONS "")

function(add_bench_runner_suite suite_name source_path compile_definition)
  if(EMEL_BENCH_SUITE_FILTER STREQUAL "" OR EMEL_BENCH_SUITE_FILTER STREQUAL suite_name)
    list(APPEND BENCH_RUNNER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${source_path}")
    list(APPEND BENCH_RUNNER_COMPILE_DEFINITIONS "${compile_definition}")
    set(BENCH_RUNNER_SOURCES "${BENCH_RUNNER_SOURCES}" PARENT_SCOPE)
    set(BENCH_RUNNER_COMPILE_DEFINITIONS
        "${BENCH_RUNNER_COMPILE_DEFINITIONS}"
        PARENT_SCOPE)
  endif()
endfunction()

add_bench_runner_suite(batch_planner batch/planner_bench.cpp EMEL_BENCH_ENABLE_BATCH_PLANNER)
add_bench_runner_suite(memory_kv memory/kv_bench.cpp EMEL_BENCH_ENABLE_MEMORY_KV)
add_bench_runner_suite(memory_recurrent memory/recurrent_bench.cpp
                       EMEL_BENCH_ENABLE_MEMORY_RECURRENT)
add_bench_runner_suite(memory_hybrid memory/hybrid_bench.cpp EMEL_BENCH_ENABLE_MEMORY_HYBRID)
add_bench_runner_suite(jinja_parser text/jinja/parser_bench.cpp EMEL_BENCH_ENABLE_JINJA_PARSER)
add_bench_runner_suite(jinja_formatter text/jinja/formatter_bench.cpp
                       EMEL_BENCH_ENABLE_JINJA_FORMATTER)
add_bench_runner_suite(gbnf_rule_parser gbnf/rule_parser_bench.cpp
                       EMEL_BENCH_ENABLE_GBNF_RULE_PARSER)
add_bench_runner_suite(generation generation_bench.cpp EMEL_BENCH_ENABLE_GENERATION)
add_bench_runner_suite(diarization_sortformer diarization/sortformer_bench.cpp
                       EMEL_BENCH_ENABLE_DIARIZATION_SORTFORMER)
add_bench_runner_suite(flash_attention kernel/flash_attention_bench.cpp
                       EMEL_BENCH_ENABLE_FLASH_ATTENTION)
add_bench_runner_suite(logits_validator logits/validator_bench.cpp
                       EMEL_BENCH_ENABLE_LOGITS_VALIDATOR)
add_bench_runner_suite(logits_sampler logits/sampler_bench.cpp EMEL_BENCH_ENABLE_LOGITS_SAMPLER)
add_bench_runner_suite(kernel_x86_64 kernel/x86_64_bench.cpp EMEL_BENCH_ENABLE_KERNEL_X86_64)
add_bench_runner_suite(kernel_aarch64 kernel/aarch64_bench.cpp EMEL_BENCH_ENABLE_KERNEL_AARCH64)
add_bench_runner_suite(sm_any sm_any_bench.cpp EMEL_BENCH_ENABLE_SM_ANY)
add_bench_runner_suite(tokenizer_preprocessor_bpe text/tokenizer/preprocessor/bpe_bench.cpp
                       EMEL_BENCH_ENABLE_TOKENIZER_PREPROCESSOR_BPE)
add_bench_runner_suite(tokenizer_preprocessor_spm text/tokenizer/preprocessor/spm_bench.cpp
                       EMEL_BENCH_ENABLE_TOKENIZER_PREPROCESSOR_SPM)
add_bench_runner_suite(tokenizer_preprocessor_ugm text/tokenizer/preprocessor/ugm_bench.cpp
                       EMEL_BENCH_ENABLE_TOKENIZER_PREPROCESSOR_UGM)
add_bench_runner_suite(tokenizer_preprocessor_wpm text/tokenizer/preprocessor/wpm_bench.cpp
                       EMEL_BENCH_ENABLE_TOKENIZER_PREPROCESSOR_WPM)
add_bench_runner_suite(tokenizer_preprocessor_rwkv text/tokenizer/preprocessor/rwkv_bench.cpp
                       EMEL_BENCH_ENABLE_TOKENIZER_PREPROCESSOR_RWKV)
add_bench_runner_suite(tokenizer_preprocessor_plamo2 text/tokenizer/preprocessor/plamo2_bench.cpp
                       EMEL_BENCH_ENABLE_TOKENIZER_PREPROCESSOR_PLAMO2)
add_bench_runner_suite(encoder_bpe text/encoders/bpe_bench.cpp EMEL_BENCH_ENABLE_ENCODER_BPE)
add_bench_runner_suite(encoder_spm text/encoders/spm_bench.cpp EMEL_BENCH_ENABLE_ENCODER_SPM)
add_bench_runner_suite(encoder_wpm text/encoders/wpm_bench.cpp EMEL_BENCH_ENABLE_ENCODER_WPM)
add_bench_runner_suite(encoder_ugm text/encoders/ugm_bench.cpp EMEL_BENCH_ENABLE_ENCODER_UGM)
add_bench_runner_suite(encoder_rwkv text/encoders/rwkv_bench.cpp EMEL_BENCH_ENABLE_ENCODER_RWKV)
add_bench_runner_suite(encoder_plamo2 text/encoders/plamo2_bench.cpp
                       EMEL_BENCH_ENABLE_ENCODER_PLAMO2)
add_bench_runner_suite(encoder_fallback text/encoders/fallback_bench.cpp
                       EMEL_BENCH_ENABLE_ENCODER_FALLBACK)
add_bench_runner_suite(tokenizer text/tokenizer/tokenizer_bench.cpp EMEL_BENCH_ENABLE_TOKENIZER)

if(NOT EMEL_BENCH_SUITE_FILTER STREQUAL "" AND
   NOT EMEL_BENCH_SUITE_FILTER STREQUAL "whisper_compare" AND
   BENCH_RUNNER_COMPILE_DEFINITIONS STREQUAL "")
  message(FATAL_ERROR "Unknown EMEL_BENCH_SUITE_FILTER='${EMEL_BENCH_SUITE_FILTER}'")
endif()

add_executable(bench_runner
  ${BENCH_RUNNER_SOURCES}
)

target_link_libraries(bench_runner PRIVATE emel_core emel)
if(EMEL_BENCH_NEEDS_LLAMA)
  target_link_libraries(bench_runner PRIVATE ggml llama)
endif()

target_compile_definitions(bench_runner
  PRIVATE
    ${BENCH_RUNNER_COMPILE_DEFINITIONS}
    EMEL_BENCH_REPO_ROOT=\"${EMEL_ROOT}\"
    BENCH_REFERENCE_SOURCE=\"${BENCH_REFERENCE_SOURCE}\"
    BENCH_REFERENCE_REF=\"${BENCH_REFERENCE_REF_VALUE}\"
    EMEL_BENCH_EMEL_ARTIFACT_PATH=\"$<TARGET_FILE:emel>\"
)
if(NOT EMEL_BENCH_SUITE_FILTER STREQUAL "")
  target_compile_definitions(bench_runner
    PRIVATE
      EMEL_BENCH_COMPILED_SUITE=\"${EMEL_BENCH_SUITE_FILTER}\"
  )
endif()

add_executable(whisper_emel_parity_runner
  ${CMAKE_CURRENT_SOURCE_DIR}/whisper_emel_parity_runner.cpp
)

target_link_libraries(whisper_emel_parity_runner
  PRIVATE
    emel_core
    emel
)

target_include_directories(whisper_emel_parity_runner
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EMEL_ROOT}
    ${EMEL_ROOT}/src
    ${EMEL_ROOT}/include
)
if(EMEL_BENCH_NEEDS_LLAMA)
  target_compile_definitions(bench_runner
    PRIVATE
      EMEL_BENCH_LLAMA_ARTIFACT_PATH=\"$<TARGET_FILE:llama>\"
  )
endif()

target_include_directories(bench_runner
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EMEL_ROOT}/src
    ${EMEL_ROOT}/include
    ${NLOHMANN_JSON_INCLUDE_DIR}
)
if(EMEL_BENCH_NEEDS_LLAMA)
  target_include_directories(bench_runner
    PRIVATE
      ${reference_impl_SOURCE_DIR}/common
      ${reference_impl_SOURCE_DIR}/src
      ${reference_impl_SOURCE_DIR}/ggml/include
      ${reference_impl_SOURCE_DIR}/include
  )
endif()

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

add_executable(embedding_compare_tests
  ${CMAKE_CURRENT_SOURCE_DIR}/embedding_compare_tests.cpp
)

add_dependencies(embedding_compare_tests embedding_generator_bench_runner)

target_include_directories(embedding_compare_tests
  PRIVATE
    ${DOCTEST_INCLUDE_DIR}
    ${EMEL_ROOT}/src
    ${EMEL_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(embedding_compare_tests
  PRIVATE
    BENCH_REPO_ROOT=\"${EMEL_ROOT}\"
    EMBEDDING_COMPARE_SCRIPT_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/embedding_compare.py\"
    EMBEDDING_REFERENCE_PYTHON_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/embedding_reference_python.py\"
    EMBEDDING_GENERATOR_BENCH_RUNNER_PATH=\"$<TARGET_FILE:embedding_generator_bench_runner>\"
)

add_test(
  NAME embedding_compare_tests
  COMMAND embedding_compare_tests
)

add_executable(generation_compare_tests
  ${CMAKE_CURRENT_SOURCE_DIR}/generation_compare_tests.cpp
)

add_dependencies(generation_compare_tests bench_runner)

target_include_directories(generation_compare_tests
  PRIVATE
    ${DOCTEST_INCLUDE_DIR}
)

target_compile_definitions(generation_compare_tests
  PRIVATE
    BENCH_REPO_ROOT=\"${EMEL_ROOT}\"
    GENERATION_COMPARE_SCRIPT_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/generation_compare.py\"
)

add_test(
  NAME generation_compare_tests
  COMMAND generation_compare_tests
)

add_executable(diarization_compare_tests
  ${CMAKE_CURRENT_SOURCE_DIR}/diarization_compare_tests.cpp
)

add_dependencies(diarization_compare_tests bench_runner)

target_include_directories(diarization_compare_tests
  PRIVATE
    ${DOCTEST_INCLUDE_DIR}
)

target_compile_definitions(diarization_compare_tests
  PRIVATE
    BENCH_REPO_ROOT=\"${EMEL_ROOT}\"
    DIARIZATION_COMPARE_SCRIPT_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/diarization_compare.py\"
)

add_test(
  NAME diarization_compare_tests
  COMMAND diarization_compare_tests
)

add_executable(whisper_benchmark_tests
  ${CMAKE_CURRENT_SOURCE_DIR}/whisper_benchmark_tests.cpp
)

target_include_directories(whisper_benchmark_tests
  PRIVATE
    ${DOCTEST_INCLUDE_DIR}
)

target_compile_definitions(whisper_benchmark_tests
  PRIVATE
    BENCH_REPO_ROOT=\"${EMEL_ROOT}\"
    WHISPER_COMPARE_SCRIPT_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/whisper_compare.py\"
    WHISPER_BENCHMARK_SCRIPT_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/whisper_benchmark.py\"
    WHISPER_EMEL_PARITY_RUNNER_SOURCE_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/whisper_emel_parity_runner.cpp\"
)

add_test(
  NAME whisper_benchmark_tests
  COMMAND whisper_benchmark_tests
)

add_executable(quality_gates_tests
  ${CMAKE_CURRENT_SOURCE_DIR}/quality_gates_tests.cpp
)

target_include_directories(quality_gates_tests
  PRIVATE
    ${DOCTEST_INCLUDE_DIR}
)

target_compile_definitions(quality_gates_tests
  PRIVATE
    BENCH_REPO_ROOT=\"${EMEL_ROOT}\"
)

add_test(
  NAME quality_gates_tests
  COMMAND quality_gates_tests
)

if(EMEL_BENCH_NEEDS_LLAMA)
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
endif()

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

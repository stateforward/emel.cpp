cmake_minimum_required(VERSION 3.20)

project(emel LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
  add_compile_options(/W4 /WX)
else()
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

option(EMEL_ENABLE_TESTS "Build tests" ON)
option(EMEL_ENABLE_FUZZ "Build fuzz targets" OFF)
option(EMEL_ENABLE_TENSOR_PARSER_TEXT_MACHINES
  "Build tensor/parser/text machine surfaces"
  OFF
)

include(FetchContent)
include(cmake/sml_version.cmake)
FetchContent_Declare(
  boost_sml
  GIT_REPOSITORY ${EMEL_BOOST_SML_GIT_REPOSITORY}
  GIT_TAG ${EMEL_BOOST_SML_GIT_TAG}
)
FetchContent_MakeAvailable(boost_sml)

add_library(emel_core INTERFACE)

target_include_directories(emel_core
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${boost_sml_SOURCE_DIR}/include
)

add_library(emel STATIC
  src/emel/model/data.cpp
)

target_link_libraries(emel
  PUBLIC
    emel_core
)


if(EMEL_ENABLE_TENSOR_PARSER_TEXT_MACHINES)
  add_executable(mock_model_load
    tools/mock_main.cpp
  )

  target_link_libraries(mock_model_load
    PRIVATE
      emel_core
  )

  target_include_directories(mock_model_load
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${boost_sml_SOURCE_DIR}/include
  )
endif()

if(EMEL_ENABLE_TESTS)
  include(CTest)
  enable_testing()

  set(DOCTEST_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/doctest)
  if(NOT EXISTS ${DOCTEST_INCLUDE_DIR}/doctest/doctest.h)
    message(FATAL_ERROR "Missing doctest header at ${DOCTEST_INCLUDE_DIR}/doctest/doctest.h")
  endif()

  set(EMEL_TEST_SOURCES
    tests/doctest_main.cpp
    tests/model/loader/lifecycle_tests.cpp
    tests/model/weight_loader/lifecycle_tests.cpp
    tests/gbnf/lexer_tests.cpp
    tests/gbnf/parser_tests.cpp
    tests/gbnf/sampler_tests.cpp
    tests/generator/lifecycle_tests.cpp
    tests/logits/validator/validator_tests.cpp
    tests/logits/sampler/pipeline_tests.cpp
    tests/sm/callback_tests.cpp
    tests/sm/sm_policy_tests.cpp
    tests/sm/sm_any_tests.cpp
    tests/graph/wrapper_visibility_tests.cpp
    tests/graph/allocator/allocator_tests.cpp
    tests/graph/allocator/allocator_action_branch_tests.cpp
    tests/graph/assembler/assembler_tests.cpp
    tests/graph/assembler/assembler_action_branch_tests.cpp
    tests/graph/graph_tests.cpp
    tests/graph/processor/processor_action_branch_tests.cpp
    tests/kernel/lifecycle_tests.cpp
    # Legacy processor tests target pre-cutover API shape and are disabled until migrated.
    # Keep them out of gate builds during the hard cutover.
    tests/batch/planner/planner_tests.cpp
    tests/batch/planner/planner_actions_tests.cpp
    tests/batch/planner/modes/detail_tests.cpp
    tests/batch/planner/modes/simple_actions_tests.cpp
    tests/batch/planner/modes/equal_actions_tests.cpp
    tests/batch/planner/modes/sequential_actions_tests.cpp
    tests/batch/planner/planner_additional_tests.cpp
    tests/batch/planner/planner_action_branch_tests.cpp
    tests/batch/planner/planner_sm_transition_tests.cpp
    tests/batch/planner/planner_sm_flow_tests.cpp
    tests/token/batcher/lifecycle_tests.cpp
    tests/memory/kv/lifecycle_tests.cpp
    tests/memory/recurrent/lifecycle_tests.cpp
    tests/memory/hybrid/lifecycle_tests.cpp
    tests/tensor/lifecycle_tests.cpp
    tests/tensor/view/lifecycle_tests.cpp
  )

  if(EMEL_ENABLE_TENSOR_PARSER_TEXT_MACHINES)
    list(APPEND EMEL_TEST_SOURCES
      tests/gguf/loader/lifecycle_tests.cpp
      tests/text/jinja/parser_tests.cpp
      tests/text/jinja/lexer_tests.cpp
      tests/text/jinja/formatter_tests.cpp
      tests/text/encoders/encoder_tests.cpp
      tests/text/conditioner/conditioner_tests.cpp
      tests/text/renderer/renderer_tests.cpp
      tests/text/unicode/unicode_tests.cpp
      tests/text/tokenizer/preprocessor_tests.cpp
      tests/text/tokenizer/preprocessor_spm_tests.cpp
      tests/text/tokenizer/preprocessor_wpm_tests.cpp
      tests/text/tokenizer/preprocessor_rwkv_tests.cpp
      tests/text/tokenizer/preprocessor_plamo2_tests.cpp
      tests/text/tokenizer/preprocessor_fallback_tests.cpp
      tests/text/tokenizer/bpe_regex_tests.cpp
      tests/text/tokenizer/bpe_split_tests.cpp
      tests/text/tokenizer/tokenizer_tests.cpp
      tests/text/tokenizer/tokenizer_parity_tests.cpp
      tests/text/tokenizer/tokenizer_action_guard_tests.cpp
    )
  endif()

  add_executable(emel_tests_bin
    ${EMEL_TEST_SOURCES}
  )

  target_link_libraries(emel_tests_bin
    PRIVATE
      emel_core
      emel
  )

  target_include_directories(emel_tests_bin
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${boost_sml_SOURCE_DIR}/include
      ${DOCTEST_INCLUDE_DIR}
  )

  add_test(
    NAME emel_tests
    COMMAND emel_tests_bin
  )

  add_test(
    NAME lint_snapshot
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/lint_snapshot.sh
  )

  add_test(
    NAME generate_docs
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_docs.sh --check
  )

endif()

if(EMEL_ENABLE_FUZZ)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "EMEL fuzz targets require Clang/libFuzzer")
  endif()

  set(EMEL_FUZZ_EXTRA_LINK_OPTIONS)
  if(APPLE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
      message(FATAL_ERROR
        "EMEL fuzz targets on macOS require Homebrew LLVM clang++ "
        "(AppleClang does not ship libFuzzer runtime)")
    endif()

    get_filename_component(EMEL_CXX_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(EMEL_CXX_ROOT "${EMEL_CXX_BIN_DIR}/.." ABSOLUTE)
    set(EMEL_CXX_LIBCXX_DIR "${EMEL_CXX_ROOT}/lib/c++")
    if(EXISTS "${EMEL_CXX_LIBCXX_DIR}/libc++.dylib")
      # Homebrew LLVM's libFuzzer runtime depends on Homebrew libc++ symbols.
      list(APPEND EMEL_FUZZ_EXTRA_LINK_OPTIONS
        "-L${EMEL_CXX_LIBCXX_DIR}"
        "-Wl,-rpath,${EMEL_CXX_LIBCXX_DIR}"
      )

      # Apple ld works reliably with system ar/ranlib for static archives.
      set(CMAKE_AR "/usr/bin/ar" CACHE FILEPATH "" FORCE)
      set(CMAKE_RANLIB "/usr/bin/ranlib" CACHE FILEPATH "" FORCE)
    endif()
  endif()

  function(emel_configure_fuzzer target_name)
    target_link_libraries(${target_name}
      PRIVATE
        emel_core
        emel
    )
    target_compile_options(${target_name}
      PRIVATE
        -fsanitize=fuzzer,address,undefined
        -fno-omit-frame-pointer
    )
    target_link_options(${target_name}
      PRIVATE
        -fsanitize=fuzzer,address,undefined
    )
    if(EMEL_FUZZ_EXTRA_LINK_OPTIONS)
      target_link_options(${target_name}
        PRIVATE
          ${EMEL_FUZZ_EXTRA_LINK_OPTIONS}
      )
    endif()
    target_compile_definitions(${target_name}
      PRIVATE
        FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
    )
  endfunction()

  if(EMEL_ENABLE_TENSOR_PARSER_TEXT_MACHINES)
    add_executable(emel_fuzz_gguf_parser
      tests/fuzz/gguf_parser_fuzz.cpp
    )
    emel_configure_fuzzer(emel_fuzz_gguf_parser)
  endif()

  add_executable(emel_fuzz_gbnf_parser
    tests/fuzz/gbnf_parser_fuzz.cpp
  )
  emel_configure_fuzzer(emel_fuzz_gbnf_parser)
endif()

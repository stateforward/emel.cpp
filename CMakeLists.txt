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

include(FetchContent)
FetchContent_Declare(
  boost_sml
  GIT_REPOSITORY https://github.com/stateforward/sml.cpp
  GIT_TAG 119ca42bc7e47e4683d37f8dd40b08881073e268
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
  src/emel/buffer/allocator/c_api.cpp
)

target_link_libraries(emel
  PUBLIC
    emel_core
)


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

if(EMEL_ENABLE_TESTS)
  include(CTest)
  enable_testing()

  set(DOCTEST_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/doctest)
  if(NOT EXISTS ${DOCTEST_INCLUDE_DIR}/doctest/doctest.h)
    message(FATAL_ERROR "Missing doctest header at ${DOCTEST_INCLUDE_DIR}/doctest/doctest.h")
  endif()

  add_executable(emel_tests_bin
    tests/doctest_main.cpp
    tests/model/gguf_tests.cpp
    tests/model/gguf_parser_tests.cpp
    tests/model/gguf_loader_tests.cpp
    tests/model/loader_tests.cpp
    tests/model/parser_tests.cpp
    tests/gbnf/parser_tests.cpp
    tests/text/jinja/parser_tests.cpp
    tests/text/jinja/lexer_tests.cpp
    tests/text/jinja/formatter_tests.cpp
    tests/model/weight_loader_tests.cpp
    tests/logits/validator/validator_tests.cpp
    tests/logits/sampler/pipeline_tests.cpp
    tests/logits/sampler/token_selector_tests.cpp
    tests/telemetry/exporter_tests.cpp
    tests/telemetry/provider_tests.cpp
    tests/sm/sm_policy_tests.cpp
    tests/sm/sm_any_tests.cpp
    tests/decoder/decoder_tests.cpp
    tests/decoder/decoder_sm_transition_tests.cpp
    tests/decoder/decoder_action_branch_tests.cpp
    tests/graph/processor/processor_tests.cpp
    tests/graph/processor/processor_sm_transition_tests.cpp
    tests/graph/processor/processor_action_branch_tests.cpp
    tests/text/encoders/encoder_tests.cpp
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
    tests/batch/planner/planner_tests.cpp
    tests/batch/planner/planner_actions_tests.cpp
    tests/batch/planner/planner_additional_tests.cpp
    tests/batch/planner/planner_action_branch_tests.cpp
    tests/batch/planner/planner_sm_transition_tests.cpp
    tests/batch/planner/planner_sm_flow_tests.cpp
    tests/token/batcher/sanitizer_tests.cpp
    tests/memory/coordinator_tests.cpp
    tests/memory/coordinator_sm_transition_tests.cpp
    tests/memory/coordinator_sm_transition_more_tests.cpp
    tests/memory/coordinator_sm_testing_policy_tests.cpp
    tests/memory/coordinator_action_branch_tests.cpp
    tests/memory/coordinator_guard_tests.cpp
    tests/memory/coordinator_recurrent_tests.cpp
    tests/memory/coordinator_recurrent_sm_transition_tests.cpp
    tests/memory/coordinator_recurrent_sm_transition_more_tests.cpp
    tests/memory/coordinator_recurrent_sm_testing_policy_tests.cpp
    tests/memory/coordinator_recurrent_action_branch_tests.cpp
    tests/memory/coordinator_kv_tests.cpp
    tests/memory/coordinator_kv_sm_transition_tests.cpp
    tests/memory/coordinator_kv_sm_transition_more_tests.cpp
    tests/memory/coordinator_kv_sm_testing_policy_tests.cpp
    tests/memory/coordinator_kv_action_branch_tests.cpp
    tests/memory/coordinator_hybrid_tests.cpp
    tests/memory/coordinator_hybrid_sm_transition_tests.cpp
    tests/memory/coordinator_hybrid_sm_transition_more_tests.cpp
    tests/memory/coordinator_hybrid_sm_testing_policy_tests.cpp
    tests/memory/coordinator_hybrid_action_branch_tests.cpp
    tests/memory/kv/cache_tests.cpp
    tests/memory/kv/cache_sm_transition_tests.cpp
    tests/memory/kv/cache_sm_testing_policy_tests.cpp
    tests/memory/kv/cache_action_branch_tests.cpp
    tests/memory/kv/cache_guard_tests.cpp
    tests/buffer/allocator_tests.cpp
    tests/buffer/allocator_error_tests.cpp
    tests/buffer/allocator_detail_tests.cpp
    tests/buffer/allocator_sm_transition_tests.cpp
    tests/buffer/allocator_guard_tests.cpp
    tests/buffer/allocator_action_branch_tests.cpp
    tests/buffer/allocator_c_api_tests.cpp
    tests/buffer/planner_tests.cpp
    tests/buffer/planner_error_tests.cpp
    tests/buffer/planner_sm_transition_tests.cpp
    tests/buffer/allocator_actions_tests.cpp
    tests/buffer/allocator_actions_more_tests.cpp
    tests/buffer/allocator_action_line_targets_tests.cpp
    tests/buffer/allocator_action_branch_more_tests.cpp
    tests/buffer/allocator_parity_tests.cpp
    tests/buffer/planner_actions_tests.cpp
    tests/buffer/planner_actions_branch_targets_tests.cpp
    tests/buffer/planner_action_branch_tests.cpp
    tests/buffer/chunk_allocator_tests.cpp
    tests/buffer/chunk_allocator_error_tests.cpp
    tests/buffer/chunk_allocator_sm_transition_tests.cpp
    tests/buffer/chunk_allocator_sm_testing_policy_tests.cpp
    tests/buffer/chunk_allocator_action_branch_tests.cpp
    tests/buffer/chunk_allocator_parity_tests.cpp
    tests/buffer/realloc_analyzer_tests.cpp
    tests/buffer/realloc_analyzer_error_tests.cpp
    tests/buffer/realloc_analyzer_sm_transition_tests.cpp
    tests/buffer/realloc_analyzer_sm_flow_tests.cpp
    tests/buffer/realloc_analyzer_action_branch_tests.cpp
    tests/tensor/allocator_tests.cpp
    tests/tensor/allocator_error_tests.cpp
    tests/tensor/allocator_branch_tests.cpp
    tests/tensor/allocator_action_branch_targets_tests.cpp
    tests/tensor/allocator_sm_transition_tests.cpp
    tests/tensor/allocator_sm_transition_more_tests.cpp
    tests/tensor/allocator_sm_testing_policy_tests.cpp
    tests/tensor/allocator_action_branch_tests.cpp
    tests/tensor/allocator_action_branch_more_tests.cpp
    tests/tensor/lifetime_analyzer_tests.cpp
    tests/tensor/lifetime_analyzer_error_tests.cpp
    tests/tensor/lifetime_analyzer_sm_error_tests.cpp
    tests/tensor/lifetime_analyzer_sm_transition_tests.cpp
    tests/tensor/lifetime_analyzer_action_branch_tests.cpp
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
    target_compile_definitions(${target_name}
      PRIVATE
        FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
    )
  endfunction()

  add_executable(emel_fuzz_gguf_parser
    tests/fuzz/gguf_parser_fuzz.cpp
  )
  emel_configure_fuzzer(emel_fuzz_gguf_parser)

  add_executable(emel_fuzz_gbnf_parser
    tests/fuzz/gbnf_parser_fuzz.cpp
  )
  emel_configure_fuzzer(emel_fuzz_gbnf_parser)
endif()

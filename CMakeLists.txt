cmake_minimum_required(VERSION 3.20)

project(emel LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckCXXCompilerFlag)

if(MSVC)
  add_compile_options(/W4 /WX)
else()
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

option(EMEL_ENABLE_TESTS "Build tests" ON)
option(EMEL_ENABLE_FUZZ "Build fuzz targets" OFF)
option(EMEL_ENABLE_AARCH64_HOST_FEATURES
  "Enable host-tuned AArch64 compile flags for EMEL-owned C++ code"
  ON)

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

set(EMEL_AARCH64_HOST_CXX_FLAG "")
if(EMEL_ENABLE_AARCH64_HOST_FEATURES AND NOT CMAKE_CROSSCOMPILING)
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" EMEL_SYSTEM_PROCESSOR_LOWER)
  if(EMEL_SYSTEM_PROCESSOR_LOWER MATCHES "^(arm64|aarch64)$")
    check_cxx_compiler_flag("-mcpu=native+dotprod+i8mm"
                            EMEL_HAS_AARCH64_NATIVE_DOTPROD_I8MM)
    if(APPLE)
      if(EMEL_HAS_AARCH64_NATIVE_DOTPROD_I8MM)
        set(EMEL_AARCH64_HOST_CXX_FLAG "-mcpu=native+dotprod+i8mm")
      endif()
    else()
      if(EMEL_HAS_AARCH64_NATIVE_DOTPROD_I8MM)
        set(EMEL_AARCH64_HOST_CXX_FLAG "-mcpu=native+dotprod+i8mm")
      endif()
    endif()
  endif()
endif()

if(NOT EMEL_AARCH64_HOST_CXX_FLAG STREQUAL "")
  message(STATUS "EMEL enabling AArch64 host compile flag: ${EMEL_AARCH64_HOST_CXX_FLAG}")
  target_compile_options(emel_core
    INTERFACE
      "$<$<COMPILE_LANGUAGE:CXX>:${EMEL_AARCH64_HOST_CXX_FLAG}>"
  )
endif()

add_library(emel STATIC
  src/emel/model/architecture/detail.cpp
  src/emel/model/detail.cpp
  src/emel/model/data.cpp
  src/emel/text/unicode_data.cpp
  src/emel/model/llama/detail.cpp
  src/emel/model/qwen3/detail.cpp
  src/emel/model/lfm2/detail.cpp
  src/emel/model/gemma4/detail.cpp
  src/emel/model/omniembed/detail.cpp
  src/emel/model/sortformer/detail.cpp
  src/emel/diarization/sortformer/detail.cpp
  src/emel/diarization/sortformer/encoder/feature_extractor/detail.cpp
  src/emel/diarization/sortformer/encoder/detail.cpp
  src/emel/diarization/sortformer/modules/detail.cpp
  src/emel/diarization/sortformer/output/detail.cpp
  src/emel/diarization/sortformer/transformer/detail.cpp
)

target_link_libraries(emel
  PUBLIC
    emel_core
)


if(EMEL_ENABLE_TESTS)
  include(CTest)
  enable_testing()

  set(DOCTEST_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/doctest)
  if(NOT EXISTS ${DOCTEST_INCLUDE_DIR}/doctest/doctest.h)
    message(FATAL_ERROR "Missing doctest header at ${DOCTEST_INCLUDE_DIR}/doctest/doctest.h")
  endif()

  set(EMEL_TEST_SOURCES
    tests/doctest_main.cpp
    tests/model/fixture_manifest_tests.cpp
    tests/model/loader/lifecycle_tests.cpp
    tests/model/weight_loader/lifecycle_tests.cpp
    tests/gbnf/lexer_tests.cpp
    tests/gbnf/parser_tests.cpp
    tests/gbnf/sampler_tests.cpp
    tests/generator/lifecycle_tests.cpp
    tests/generator/action_guard_tests.cpp
    tests/generator/detail_tests.cpp
    tests/generator/initializer/lifecycle_tests.cpp
    tests/generator/prefill/lifecycle_tests.cpp
    tests/diarization/request/lifecycle_tests.cpp
    tests/diarization/sortformer/encoder/lifecycle_tests.cpp
    tests/diarization/sortformer/executor/lifecycle_tests.cpp
    tests/diarization/sortformer/modules/lifecycle_tests.cpp
    tests/diarization/sortformer/output/lifecycle_tests.cpp
    tests/diarization/sortformer/pipeline/lifecycle_tests.cpp
    tests/diarization/sortformer/parity/lifecycle_tests.cpp
    tests/diarization/sortformer/transformer/lifecycle_tests.cpp
    tests/embeddings/text_embedding_lane_tests.cpp
    tests/embeddings/vision_embedding_lane_tests.cpp
    tests/embeddings/audio_embedding_lane_tests.cpp
    tests/embeddings/shared_embedding_session_tests.cpp
    tests/embeddings/generator_action_guard_tests.cpp
    tests/embeddings/te_proof_and_regression_tests.cpp
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
    tests/kernel/aarch64_tests.cpp
    tests/kernel/lifecycle_tests.cpp
    tests/kernel/x86_64_tests.cpp
    # Legacy processor tests target pre-cutover API shape and are disabled until migrated.
    # Keep them out of gate builds during the hard cutover.
    tests/batch/planner/planner_tests.cpp
    tests/batch/planner/planner_actions_tests.cpp
    tests/batch/planner/planner_detail_tests.cpp
    tests/batch/planner/modes/simple_actions_tests.cpp
    tests/batch/planner/modes/equal_actions_tests.cpp
    tests/batch/planner/modes/sequential_actions_tests.cpp
    tests/batch/planner/planner_additional_tests.cpp
    tests/batch/planner/planner_action_branch_tests.cpp
    tests/batch/planner/planner_surface_tests.cpp
    tests/batch/planner/planner_sm_transition_tests.cpp
    tests/batch/planner/planner_sm_flow_tests.cpp
    tests/token/batcher/lifecycle_tests.cpp
    tests/memory/kv/lifecycle_tests.cpp
    tests/memory/recurrent/lifecycle_tests.cpp
    tests/memory/hybrid/lifecycle_tests.cpp
    tests/graph/tensor/lifecycle_tests.cpp
    tests/model/tensor/lifecycle_tests.cpp
    tests/tensor/view/lifecycle_tests.cpp
  )

  list(APPEND EMEL_TEST_SOURCES
    tests/gguf/loader/lifecycle_tests.cpp
    tests/text/jinja/parser_tests.cpp
    tests/text/jinja/lexer_tests.cpp
    tests/text/jinja/formatter_tests.cpp
    tests/text/formatter/formatter_tests.cpp
    tests/text/encoders/common_tests.cpp
    tests/text/encoders/bpe_tests.cpp
    tests/text/encoders/spm_tests.cpp
    tests/text/encoders/wpm_tests.cpp
    tests/text/encoders/ugm_tests.cpp
    tests/text/encoders/rwkv_tests.cpp
    tests/text/encoders/plamo2_tests.cpp
    tests/text/encoders/fallback_tests.cpp
    tests/text/conditioner/text_conditioner_tests.cpp
    tests/text/detokenizer/detokenizer_tests.cpp
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
    tests/text/tokenizer/tokenizer_metadata_tests.cpp
    tests/text/tokenizer/preprocessor_metadata_tests.cpp
    tests/text/tokenizer/tokenizer_tests.cpp
    tests/text/tokenizer/tokenizer_parity_tests.cpp
    tests/text/tokenizer/tokenizer_action_guard_tests.cpp
  )

  set(EMEL_TEST_SHARDS "" CACHE STRING "Comma or semicolon separated doctest shards to compile")
  if(NOT EMEL_TEST_SHARDS STREQUAL "")
    string(REPLACE "," ";" EMEL_TEST_SELECTED_SHARDS "${EMEL_TEST_SHARDS}")
    set(EMEL_TEST_FILTERED_SOURCES tests/doctest_main.cpp)
    foreach(test_source IN LISTS EMEL_TEST_SOURCES)
      if(test_source STREQUAL "tests/doctest_main.cpp")
        continue()
      endif()

      set(keep_test_source FALSE)
      foreach(test_shard IN LISTS EMEL_TEST_SELECTED_SHARDS)
        if((test_shard STREQUAL "model_and_batch" AND
              test_source MATCHES "^tests/(model|gguf|gbnf|batch)/") OR
           (test_shard STREQUAL "generator_and_runtime" AND
              test_source MATCHES "^tests/(generator|embeddings|logits|token)/") OR
           (test_shard STREQUAL "diarization" AND
              test_source MATCHES "^tests/diarization/") OR
           (test_shard STREQUAL "sm" AND
              test_source MATCHES "^tests/sm/") OR
           (test_shard STREQUAL "text" AND
              test_source MATCHES "^tests/text/") OR
           (test_shard STREQUAL "text_encoder_plamo2" AND
              test_source MATCHES "^tests/text/encoders/plamo2_tests\\.cpp$") OR
           (test_shard STREQUAL "text_encoders" AND
              test_source MATCHES "^tests/text/encoders/") OR
           (test_shard STREQUAL "text_tokenizer" AND
              test_source MATCHES "^tests/text/tokenizer/") OR
           (test_shard STREQUAL "text_runtime" AND
              test_source MATCHES "^tests/text/(conditioner|detokenizer|formatter|jinja|renderer|unicode)/") OR
           (test_shard STREQUAL "kernel_and_graph" AND
              test_source MATCHES "^tests/(kernel|graph|memory|tensor)/"))
          set(keep_test_source TRUE)
        endif()
      endforeach()

      if(keep_test_source)
        list(APPEND EMEL_TEST_FILTERED_SOURCES "${test_source}")
      endif()
    endforeach()

    list(LENGTH EMEL_TEST_FILTERED_SOURCES EMEL_TEST_FILTERED_SOURCE_COUNT)
    if(EMEL_TEST_FILTERED_SOURCE_COUNT LESS 2)
      message(FATAL_ERROR "EMEL_TEST_SHARDS='${EMEL_TEST_SHARDS}' selected no test sources")
    endif()
    set(EMEL_TEST_SOURCES ${EMEL_TEST_FILTERED_SOURCES})
    message(STATUS "EMEL test build scoped to shards: ${EMEL_TEST_SELECTED_SHARDS}")
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
      ${CMAKE_CURRENT_SOURCE_DIR}/tools/bench
      ${boost_sml_SOURCE_DIR}/include
      ${DOCTEST_INCLUDE_DIR}
  )

  target_compile_definitions(emel_tests_bin
    PRIVATE
      EMEL_TEST_REPO_ROOT=\"${CMAKE_CURRENT_SOURCE_DIR}\"
  )

  set(EMEL_TEST_EXTRA_ARG "" CACHE STRING "Optional extra doctest argument for test shards")

  function(emel_add_doctest_shard shard_name shard_filter)
    add_test(
      NAME emel_tests_${shard_name}
      COMMAND emel_tests_bin --no-breaks --source-file=${shard_filter} ${EMEL_TEST_EXTRA_ARG}
    )
  endfunction()

  emel_add_doctest_shard(
    model_and_batch
    "*tests/model/*,*tests/gguf/*,*tests/gbnf/*,*tests/batch/*"
  )
  emel_add_doctest_shard(
    generator_and_runtime
    "*tests/generator/*,*tests/embeddings/*,*tests/logits/*,*tests/token/*"
  )
  emel_add_doctest_shard(
    diarization
    "*tests/diarization/*"
  )
  emel_add_doctest_shard(
    sm
    "*tests/sm/*"
  )
  emel_add_doctest_shard(
    text
    "*tests/text/*"
  )
  emel_add_doctest_shard(
    text_encoder_plamo2
    "*tests/text/encoders/plamo2_tests.cpp"
  )
  emel_add_doctest_shard(
    text_encoders
    "*tests/text/encoders/*"
  )
  emel_add_doctest_shard(
    text_tokenizer
    "*tests/text/tokenizer/*"
  )
  emel_add_doctest_shard(
    text_runtime
    "*tests/text/conditioner/*,*tests/text/detokenizer/*,*tests/text/formatter/*,*tests/text/jinja/*,*tests/text/renderer/*,*tests/text/unicode/*"
  )
  emel_add_doctest_shard(
    kernel_and_graph
    "*tests/kernel/*,*tests/graph/*,*tests/memory/*,*tests/tensor/*"
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

    # Current ld64 can mis-handle sanitizer-instrumented libFuzzer objects on macOS
    # and fail with invalid relocation diagnostics such as "invalid r_symbolnum=1".
    list(APPEND EMEL_FUZZ_EXTRA_LINK_OPTIONS
      "-Wl,-ld_classic"
    )
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

  add_executable(emel_fuzz_gguf_parser
    tests/fuzz/gguf_parser_fuzz.cpp
  )
  emel_configure_fuzzer(emel_fuzz_gguf_parser)

  add_executable(emel_fuzz_gbnf_parser
    tests/fuzz/gbnf_parser_fuzz.cpp
  )
  emel_configure_fuzzer(emel_fuzz_gbnf_parser)

  add_executable(emel_fuzz_jinja_parser
    tests/fuzz/jinja_parser_fuzz.cpp
  )
  emel_configure_fuzzer(emel_fuzz_jinja_parser)

  add_executable(emel_fuzz_jinja_formatter
    tests/fuzz/jinja_formatter_fuzz.cpp
  )
  emel_configure_fuzzer(emel_fuzz_jinja_formatter)
endif()

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
if(EMEL_BENCH_SUITE_FILTER STREQUAL "batch_planner" OR
   EMEL_BENCH_SUITE_FILTER STREQUAL "memory_kv" OR
   EMEL_BENCH_SUITE_FILTER STREQUAL "memory_recurrent" OR
   EMEL_BENCH_SUITE_FILTER STREQUAL "memory_hybrid" OR
   EMEL_BENCH_SUITE_FILTER STREQUAL "jinja_parser" OR
   EMEL_BENCH_SUITE_FILTER STREQUAL "jinja_formatter" OR
   EMEL_BENCH_SUITE_FILTER STREQUAL "generation" OR
   EMEL_BENCH_SUITE_FILTER STREQUAL "gbnf_rule_parser" OR
   EMEL_BENCH_SUITE_FILTER STREQUAL "flash_attention" OR
   EMEL_BENCH_SUITE_FILTER STREQUAL "kernel_x86_64" OR
   EMEL_BENCH_SUITE_FILTER STREQUAL "kernel_aarch64" OR
   EMEL_BENCH_SUITE_FILTER STREQUAL "decode_wavefront" OR
   EMEL_BENCH_SUITE_FILTER STREQUAL "parallel_matmul" OR
   EMEL_BENCH_SUITE_FILTER STREQUAL "weight_streaming" OR
   # speech_codec_mimi needs the llama.cpp checkout only for its ggml
   # subtree, which the moshi reference driver links against.
   EMEL_BENCH_SUITE_FILTER STREQUAL "speech_codec_mimi")
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

  function(emel_patch_reference_metadata_logging reference_source_dir)
    set(loader_path "${reference_source_dir}/src/llama-model-loader.cpp")
    if(NOT EXISTS "${loader_path}")
      message(FATAL_ERROR "Missing reference loader source at ${loader_path}")
    endif()
    file(READ "${loader_path}" loader_source)
    set(patched_marker "            std::string value          = type == GGUF_TYPE_ARRAY")
    string(FIND "${loader_source}" "${patched_marker}" patch_pos)
    if(patch_pos EQUAL -1)
      set(original_line "            std::string value          = gguf_kv_to_str(meta.get(), i);\n            const size_t MAX_VALUE_LEN = 40;")
      set(patched_line "            std::string value          = type == GGUF_TYPE_ARRAY\n                ? \"<array>\"\n                : gguf_kv_to_str(meta.get(), i);\n            const size_t MAX_VALUE_LEN = 40;")
      string(FIND "${loader_source}" "${original_line}" original_pos)
      if(original_pos EQUAL -1)
        set(original_line "            std::string value          = gguf_kv_to_str(metadata, i);\n            const size_t MAX_VALUE_LEN = 40;")
        set(patched_line "            std::string value          = type == GGUF_TYPE_ARRAY\n                ? \"<array>\"\n                : gguf_kv_to_str(metadata, i);\n            const size_t MAX_VALUE_LEN = 40;")
        string(FIND "${loader_source}" "${original_line}" original_pos)
        if(original_pos EQUAL -1)
          message(FATAL_ERROR
            "Unable to patch reference loader metadata logging in ${loader_path}")
        endif()
      endif()
      string(REPLACE "${original_line}" "${patched_line}" loader_source "${loader_source}")
      file(WRITE "${loader_path}" "${loader_source}")
    endif()

    set(llama_path "${reference_source_dir}/src/llama.cpp")
    if(NOT EXISTS "${llama_path}")
      message(FATAL_ERROR "Missing reference llama source at ${llama_path}")
    endif()
    file(READ "${llama_path}" llama_source)
    set(original_print "        model.load_stats(ml);\n        model.print_info();")
    set(patched_print "        model.load_stats(ml);\n        // EMEL bench disables reference metadata-only logging here; decode remains unchanged.")
    string(FIND "${llama_source}" "${patched_print}" print_patch_pos)
    if(print_patch_pos EQUAL -1)
      string(FIND "${llama_source}" "${original_print}" print_original_pos)
      if(print_original_pos EQUAL -1)
        message(FATAL_ERROR
          "Unable to patch reference model info logging in ${llama_path}")
      endif()
      string(REPLACE "${original_print}" "${patched_print}" llama_source "${llama_source}")
      file(WRITE "${llama_path}" "${llama_source}")
    endif()
  endfunction()

  emel_patch_reference_metadata_logging("${reference_impl_SOURCE_DIR}")
endif()

# moshi.cpp reference (Moshi/PersonaPlex parity lanes). Source-only populate,
# pinned via moshi_reference_ref.txt; the reference driver targets that consume
# it land with the mimi codec parity milestone and flip this gate through their
# suite filters below.
set(MOSHI_REF_REPOSITORY "https://github.com/Codes4Fun/moshi.cpp.git" CACHE STRING "")
set(MOSHI_REF_REF "" CACHE STRING "")
set(EMEL_BENCH_NEEDS_MOSHI OFF)
if(EMEL_BENCH_SUITE_FILTER STREQUAL "speech_codec_mimi" OR
   EMEL_BENCH_SUITE_FILTER STREQUAL "speech_lm_moshi" OR
   EMEL_BENCH_SUITE_FILTER STREQUAL "speech_dialogue_moshi")
  set(EMEL_BENCH_NEEDS_MOSHI ON)
endif()

set(BENCH_MOSHI_REFERENCE_REF_VALUE "not_used")
if(EMEL_BENCH_NEEDS_MOSHI)
  if(MOSHI_REF_REF STREQUAL "")
    set(MOSHI_REF_REF_FILE "${CMAKE_CURRENT_SOURCE_DIR}/moshi_reference_ref.txt")
    if(NOT EXISTS "${MOSHI_REF_REF_FILE}")
      message(FATAL_ERROR "Missing moshi.cpp reference pin at ${MOSHI_REF_REF_FILE}")
    endif()
    file(READ "${MOSHI_REF_REF_FILE}" MOSHI_REF_REF_CONTENT)
    string(STRIP "${MOSHI_REF_REF_CONTENT}" MOSHI_REF_REF_CONTENT)
    set(MOSHI_REF_REF "${MOSHI_REF_REF_CONTENT}" CACHE STRING "" FORCE)
  endif()

  include(FetchContent)
  FetchContent_Declare(
    moshi_reference_impl
    GIT_REPOSITORY ${MOSHI_REF_REPOSITORY}
    GIT_TAG ${MOSHI_REF_REF}
  )
  FetchContent_GetProperties(moshi_reference_impl)
  if(NOT moshi_reference_impl_POPULATED)
    FetchContent_Populate(moshi_reference_impl)
  endif()
  execute_process(
    COMMAND git -C "${moshi_reference_impl_SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE BENCH_MOSHI_REFERENCE_REF_VALUE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(NOT BENCH_MOSHI_REFERENCE_REF_VALUE STREQUAL "${MOSHI_REF_REF}")
    message(FATAL_ERROR
      "moshi.cpp reference resolved to '${BENCH_MOSHI_REFERENCE_REF_VALUE}', "
      "expected pinned '${MOSHI_REF_REF}'")
  endif()

  # SentencePiece is required by moshi.cpp's public header (reference lane
  # only; the EMEL lane never links it).
  set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
  set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    sentencepiece_impl
    GIT_REPOSITORY https://github.com/google/sentencepiece.git
    GIT_TAG v0.2.0
  )
  FetchContent_MakeAvailable(sentencepiece_impl)
endif()

set(EMEL_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${EMEL_ROOT} emel)

find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp)
if(NOT NLOHMANN_JSON_INCLUDE_DIR AND EMEL_BENCH_NEEDS_LLAMA AND
   EXISTS "${reference_impl_SOURCE_DIR}/vendor/nlohmann/json.hpp")
  set(NLOHMANN_JSON_INCLUDE_DIR "${reference_impl_SOURCE_DIR}/vendor")
endif()

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
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_dependency_manifest.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_runner_contract.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_runner.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_runner_registry.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/bench_common.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/logits/bench_common.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_dependency_manifest.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_runner.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_runner_registry.cpp
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
set(BENCH_RUNNER_SUITE_TARGETS "")

function(configure_bench_runner_common_target target_name)
  target_link_libraries(${target_name} PRIVATE emel_core emel)
  if(EMEL_BENCH_NEEDS_LLAMA)
    target_link_libraries(${target_name} PRIVATE ggml llama)
  endif()

  target_include_directories(${target_name}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${EMEL_ROOT}/src
      ${EMEL_ROOT}/include
  )
  if(NLOHMANN_JSON_INCLUDE_DIR)
    target_include_directories(${target_name}
      PRIVATE
        ${NLOHMANN_JSON_INCLUDE_DIR}
    )
  endif()
  if(EMEL_BENCH_NEEDS_LLAMA)
    target_include_directories(${target_name}
      PRIVATE
        ${reference_impl_SOURCE_DIR}/common
        ${reference_impl_SOURCE_DIR}/src
        ${reference_impl_SOURCE_DIR}/ggml/include
        ${reference_impl_SOURCE_DIR}/include
        ${reference_impl_SOURCE_DIR}/vendor
    )
  endif()
endfunction()

function(configure_bench_runner_artifact_definitions target_name)
  target_compile_definitions(${target_name}
    PRIVATE
      EMEL_BENCH_REPO_ROOT=\"${EMEL_ROOT}\"
      BENCH_REFERENCE_SOURCE=\"${BENCH_REFERENCE_SOURCE}\"
      BENCH_REFERENCE_REF=\"${BENCH_REFERENCE_REF_VALUE}\"
      EMEL_BENCH_EMEL_ARTIFACT_PATH=\"$<TARGET_FILE:emel>\"
  )
  if(NOT EMEL_BENCH_SUITE_FILTER STREQUAL "")
    target_compile_definitions(${target_name}
      PRIVATE
        EMEL_BENCH_COMPILED_SUITE=\"${EMEL_BENCH_SUITE_FILTER}\"
    )
  endif()
  if(EMEL_BENCH_NEEDS_LLAMA)
    target_compile_definitions(${target_name}
      PRIVATE
        EMEL_BENCH_LLAMA_ARTIFACT_PATH=\"$<TARGET_FILE:llama>\"
    )
  endif()
endfunction()

function(add_bench_runner_suite suite_name source_path compile_definition)
  if(EMEL_BENCH_SUITE_FILTER STREQUAL "" OR EMEL_BENCH_SUITE_FILTER STREQUAL suite_name)
    set(target_name "bench_runner_suite_${suite_name}")
    add_library(${target_name} OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/${source_path}")
    configure_bench_runner_common_target(${target_name})
    configure_bench_runner_artifact_definitions(${target_name})
    target_compile_definitions(${target_name} PRIVATE ${compile_definition})
    list(APPEND BENCH_RUNNER_SOURCES "$<TARGET_OBJECTS:${target_name}>")
    list(APPEND BENCH_RUNNER_COMPILE_DEFINITIONS "${compile_definition}")
    list(APPEND BENCH_RUNNER_SUITE_TARGETS "${target_name}")
    set(BENCH_RUNNER_SOURCES "${BENCH_RUNNER_SOURCES}" PARENT_SCOPE)
    set(BENCH_RUNNER_COMPILE_DEFINITIONS
        "${BENCH_RUNNER_COMPILE_DEFINITIONS}"
        PARENT_SCOPE)
    set(BENCH_RUNNER_SUITE_TARGETS "${BENCH_RUNNER_SUITE_TARGETS}" PARENT_SCOPE)
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
add_bench_runner_suite(speech_codec_mimi speech/codec_mimi_bench.cpp
                       EMEL_BENCH_ENABLE_SPEECH_CODEC_MIMI)
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
add_bench_runner_suite(sm_scheduler sm_scheduler_bench.cpp EMEL_BENCH_ENABLE_SM_SCHEDULER)
add_bench_runner_suite(graph_processor graph/processor_bench.cpp
                       EMEL_BENCH_ENABLE_GRAPH_PROCESSOR)
add_bench_runner_suite(decode_wavefront text/generator/decode_wavefront_bench.cpp
                       EMEL_BENCH_ENABLE_DECODE_WAVEFRONT)
add_bench_runner_suite(parallel_matmul text/generator/parallel_matmul_bench.cpp
                       EMEL_BENCH_ENABLE_PARALLEL_MATMUL)
add_bench_runner_suite(weight_streaming model/tensor/window_streaming_bench.cpp
                       EMEL_BENCH_ENABLE_WEIGHT_STREAMING)
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

configure_bench_runner_common_target(bench_runner)
configure_bench_runner_artifact_definitions(bench_runner)
if(NOT BENCH_RUNNER_COMPILE_DEFINITIONS STREQUAL "")
  target_compile_definitions(bench_runner
    PRIVATE
      ${BENCH_RUNNER_COMPILE_DEFINITIONS}
  )
endif()

add_executable(whisper_emel_parity_runner
  ${CMAKE_CURRENT_SOURCE_DIR}/whisper_emel_parity_runner.cpp
)

add_executable(mimi_emel_parity_runner
  ${CMAKE_CURRENT_SOURCE_DIR}/speech/mimi_emel_parity_runner.cpp
)

target_link_libraries(mimi_emel_parity_runner
  PRIVATE
    emel_core
    emel
)

target_include_directories(mimi_emel_parity_runner
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EMEL_ROOT}
    ${EMEL_ROOT}/src
    ${EMEL_ROOT}/include
)

# Reference-lane Mimi driver: moshi.cpp lib sources + ggml + SentencePiece.
# Never linked into bench_runner or the EMEL lane; invoked as a subprocess
# via reference_backends/moshi_cpp_mimi.json (two-lane isolation).
if(EMEL_BENCH_NEEDS_MOSHI AND EMEL_BENCH_NEEDS_LLAMA)
  add_executable(moshi_reference_driver
    ${CMAKE_CURRENT_SOURCE_DIR}/speech/moshi_reference_driver.cpp
    ${moshi_reference_impl_SOURCE_DIR}/src/moshi.cpp
    ${moshi_reference_impl_SOURCE_DIR}/src/json.cpp
    ${moshi_reference_impl_SOURCE_DIR}/src/safetensor.cpp
  )

  set_target_properties(moshi_reference_driver PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
  )

  target_include_directories(moshi_reference_driver
    PRIVATE
      ${moshi_reference_impl_SOURCE_DIR}/include
      ${moshi_reference_impl_SOURCE_DIR}
      ${reference_impl_SOURCE_DIR}/ggml/include
      ${sentencepiece_impl_SOURCE_DIR}/src
      ${sentencepiece_impl_BINARY_DIR}/src
  )

  target_link_libraries(moshi_reference_driver
    PRIVATE
      ggml
      sentencepiece-static
  )
endif()

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
set(DOCTEST_INCLUDE_DIR ${EMEL_ROOT}/third_party/doctest)
if(NOT EXISTS ${DOCTEST_INCLUDE_DIR}/doctest/doctest.h)
  message(FATAL_ERROR "Missing doctest header at ${DOCTEST_INCLUDE_DIR}/doctest/doctest.h")
endif()

add_executable(bench_runner_tests
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_runner_tests.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_disabled_cases.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_dependency_manifest.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bench_runner_registry.cpp
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

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

include(FetchContent)
FetchContent_Declare(
  boost_sml
  GIT_REPOSITORY https://github.com/boost-ext/sml.git
  GIT_TAG v1.1.11
)
FetchContent_MakeAvailable(boost_sml)

add_library(emel_core INTERFACE)

target_include_directories(emel_core
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${boost_sml_SOURCE_DIR}/include
)


add_executable(generate_architecture_puml
  tools/generate_architecture_puml.cpp
)

target_link_libraries(generate_architecture_puml
  PRIVATE
    emel_core
)

target_include_directories(generate_architecture_puml
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${boost_sml_SOURCE_DIR}/include
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
    tests/model/loader_tests.cpp
    tests/model/weight_loader_tests.cpp
    tests/decoder/decoder_tests.cpp
    tests/decoder/compute_executor_tests.cpp
    tests/decoder/ubatch_executor_tests.cpp
    tests/batch/splitter_tests.cpp
    tests/batch/splitter_actions_tests.cpp
    tests/memory/coordinator_tests.cpp
    tests/kv/cache_tests.cpp
    tests/buffer/allocator_tests.cpp
    tests/buffer/planner_tests.cpp
    tests/buffer/allocator_actions_tests.cpp
    tests/buffer/planner_actions_tests.cpp
    tests/buffer/chunk_allocator_tests.cpp
    tests/buffer/realloc_analyzer_tests.cpp
    tests/tensor/allocator_tests.cpp
    tests/tensor/lifetime_analyzer_tests.cpp
  )

  target_link_libraries(emel_tests_bin
    PRIVATE
      emel_core
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
    NAME architecture_puml
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_puml.sh --check
  )
endif()

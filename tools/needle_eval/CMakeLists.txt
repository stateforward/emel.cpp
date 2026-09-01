cmake_minimum_required(VERSION 3.20)

project(emel_needle_eval_project LANGUAGES C CXX)

if(NOT DEFINED EMEL_ROOT_DIR)
  get_filename_component(EMEL_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(EMEL_ENABLE_NEEDLE_EVAL OFF)
if(NOT TARGET emel_core)
  set(EMEL_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  add_subdirectory("${EMEL_ROOT_DIR}" emel_root)
endif()

add_executable(emel_needle_eval
  main.cpp
)

target_link_libraries(emel_needle_eval
  PRIVATE
    emel_core
    emel
)

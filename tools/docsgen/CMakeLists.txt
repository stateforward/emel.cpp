cmake_minimum_required(VERSION 3.20)
project(docsgen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/sml_version.cmake")
FetchContent_Declare(
  boost_sml
  GIT_REPOSITORY ${EMEL_BOOST_SML_GIT_REPOSITORY}
  GIT_TAG ${EMEL_BOOST_SML_GIT_TAG}
)
FetchContent_MakeAvailable(boost_sml)

set(ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

file(GLOB_RECURSE SM_HEADERS CONFIGURE_DEPENDS "${ROOT_DIR}/src/emel/**/sm.hpp")
list(FILTER SM_HEADERS EXCLUDE REGEX ".*/src/emel/sm.hpp$")
list(SORT SM_HEADERS)

set(DOCS_ENABLED_SM_HEADERS "")
foreach(sm_header IN LISTS SM_HEADERS)
  file(STRINGS "${sm_header}" docs_markers REGEX "docs: (enabled|disabled)")
  if("${docs_markers}" MATCHES "docs: disabled")
    continue()
  endif()
  list(APPEND DOCS_ENABLED_SM_HEADERS "${sm_header}")
endforeach()
set(SM_HEADERS "${DOCS_ENABLED_SM_HEADERS}")

set(GEN_HEADER "${CMAKE_CURRENT_BINARY_DIR}/docsgen_machines.hpp")
file(WRITE "${GEN_HEADER}" "#pragma once\n\n")
foreach(sm_header IN LISTS SM_HEADERS)
  file(RELATIVE_PATH rel "${ROOT_DIR}/src" "${sm_header}")
  file(APPEND "${GEN_HEADER}" "#include \"${rel}\"\n")
endforeach()

file(APPEND "${GEN_HEADER}" "\ninline void register_machines(std::vector<machine_spec> & out) {\n")
foreach(sm_header IN LISTS SM_HEADERS)
  file(RELATIVE_PATH rel_emel "${ROOT_DIR}/src/emel" "${sm_header}")
  string(REPLACE "/sm.hpp" "" dir "${rel_emel}")
  string(REPLACE "/" "_" name "${dir}")
  string(REPLACE "/" "::" ns "${dir}")
  file(APPEND "${GEN_HEADER}" "  register_machine<emel::${ns}::sm::model_type>(out, \"${name}\", \"emel/${dir}/sm.hpp\");\n")
endforeach()
file(APPEND "${GEN_HEADER}" "}\n")

add_executable(docsgen docsgen.cpp "${GEN_HEADER}")

target_include_directories(docsgen PRIVATE
  "${ROOT_DIR}/src"
  "${ROOT_DIR}/include"
  "${boost_sml_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_BINARY_DIR}"
)

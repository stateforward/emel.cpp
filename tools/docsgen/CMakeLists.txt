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

set(DOCS_MODEL_SM_HEADERS "")
foreach(sm_header IN LISTS SM_HEADERS)
  file(STRINGS "${sm_header}" docs_markers REGEX "docs: (enabled|disabled)")
  if("${docs_markers}" MATCHES "docs: disabled")
    continue()
  endif()

  file(STRINGS "${sm_header}" model_markers REGEX "^struct model *\\{")
  if(NOT model_markers)
    continue()
  endif()

  list(APPEND DOCS_MODEL_SM_HEADERS "${sm_header}")
endforeach()
set(SM_HEADERS "${DOCS_MODEL_SM_HEADERS}")

set(GEN_HEADER "${CMAKE_CURRENT_BINARY_DIR}/docsgen_machines.hpp")
set(GEN_HEADER_TMP "${CMAKE_CURRENT_BINARY_DIR}/docsgen_machines.hpp.tmp")
set(DOCSGEN_PART_SIZE "1" CACHE STRING "Number of state machines per docsgen registration TU")
file(WRITE "${GEN_HEADER_TMP}" "#pragma once\n\n#include <vector>\n\nstruct machine_spec;\n\n")

set(GEN_SOURCES "")
list(LENGTH SM_HEADERS SM_HEADER_COUNT)
if(SM_HEADER_COUNT GREATER 0)
  math(EXPR DOCSGEN_PART_COUNT
       "(${SM_HEADER_COUNT} + ${DOCSGEN_PART_SIZE} - 1) / ${DOCSGEN_PART_SIZE}")
  math(EXPR DOCSGEN_LAST_PART "${DOCSGEN_PART_COUNT} - 1")
  foreach(part_idx RANGE 0 ${DOCSGEN_LAST_PART})
    set(GEN_PART "${CMAKE_CURRENT_BINARY_DIR}/docsgen_machines_part_${part_idx}.cpp")
    set(GEN_PART_TMP "${GEN_PART}.tmp")
    math(EXPR PART_START "${part_idx} * ${DOCSGEN_PART_SIZE}")
    math(EXPR PART_END "${PART_START} + ${DOCSGEN_PART_SIZE} - 1")
    math(EXPR LAST_HEADER_INDEX "${SM_HEADER_COUNT} - 1")
    if(PART_END GREATER LAST_HEADER_INDEX)
      set(PART_END "${LAST_HEADER_INDEX}")
    endif()

    set(PART_CONTEXT_HEADERS "")
    foreach(header_index RANGE ${PART_START} ${PART_END})
      list(GET SM_HEADERS ${header_index} sm_header)
      file(RELATIVE_PATH rel_emel "${ROOT_DIR}/src/emel" "${sm_header}")
      string(REPLACE "/sm.hpp" "" dir "${rel_emel}")
      string(REPLACE "/" ";" dir_parts "${dir}")
      set(context_prefix "")
      foreach(dir_part IN LISTS dir_parts)
        if(context_prefix STREQUAL "")
          set(context_prefix "${dir_part}")
        else()
          string(APPEND context_prefix "/${dir_part}")
        endif()

        set(context_header "${ROOT_DIR}/src/emel/${context_prefix}/context.hpp")
        if(EXISTS "${context_header}")
          file(RELATIVE_PATH context_rel "${ROOT_DIR}/src" "${context_header}")
          list(APPEND PART_CONTEXT_HEADERS "${context_rel}")
        endif()
      endforeach()
    endforeach()
    list(REMOVE_DUPLICATES PART_CONTEXT_HEADERS)

    file(WRITE "${GEN_PART_TMP}" "#include \"docsgen_machine_emit.hpp\"\n\n")
    set(PART_NEEDS_TENSOR_VIEW_POLICY false)
    foreach(context_header IN LISTS PART_CONTEXT_HEADERS)
      file(APPEND "${GEN_PART_TMP}" "#include \"${context_header}\"\n")
    endforeach()
    if(PART_CONTEXT_HEADERS)
      file(APPEND "${GEN_PART_TMP}" "\n")
    endif()
    foreach(header_index RANGE ${PART_START} ${PART_END})
      list(GET SM_HEADERS ${header_index} sm_header)
      file(RELATIVE_PATH rel_emel "${ROOT_DIR}/src/emel" "${sm_header}")
      string(REPLACE "/sm.hpp" "" dir "${rel_emel}")
      if(dir STREQUAL "tensor/view")
        set(PART_NEEDS_TENSOR_VIEW_POLICY true)
      endif()
      file(RELATIVE_PATH rel "${ROOT_DIR}/src" "${sm_header}")
      file(APPEND "${GEN_PART_TMP}" "#include \"${rel}\"\n")
    endforeach()
    if(PART_NEEDS_TENSOR_VIEW_POLICY)
      file(APPEND "${GEN_PART_TMP}" "#include \"docsgen_tensor_view_policy.hpp\"\n")
    endif()

    file(APPEND "${GEN_PART_TMP}"
         "\nvoid register_machines_part_${part_idx}(std::vector<machine_spec> & out) {\n")
    foreach(header_index RANGE ${PART_START} ${PART_END})
      list(GET SM_HEADERS ${header_index} sm_header)
      file(RELATIVE_PATH rel_emel "${ROOT_DIR}/src/emel" "${sm_header}")
      string(REPLACE "/sm.hpp" "" dir "${rel_emel}")
      string(REPLACE "/" "_" name "${dir}")
      string(REPLACE "/" "::" ns "${dir}")
      if(dir STREQUAL "tensor/view")
        file(APPEND "${GEN_PART_TMP}" "  register_machine<emel::tensor::view::model<docsgen_tensor_view_policy>>(out, \"${name}\", \"emel/${dir}/sm.hpp\");\n")
      else()
        file(APPEND "${GEN_PART_TMP}" "  register_machine<emel::${ns}::model>(out, \"${name}\", \"emel/${dir}/sm.hpp\");\n")
      endif()
    endforeach()
    file(APPEND "${GEN_PART_TMP}" "}\n")
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${GEN_PART_TMP}" "${GEN_PART}"
    )

    list(APPEND GEN_SOURCES "${GEN_PART}")
    file(APPEND "${GEN_HEADER_TMP}"
         "void register_machines_part_${part_idx}(std::vector<machine_spec> & out);\n")
  endforeach()
endif()

file(APPEND "${GEN_HEADER_TMP}" "\ninline void register_machines(std::vector<machine_spec> & out) {\n")
if(SM_HEADER_COUNT GREATER 0)
  foreach(part_idx RANGE 0 ${DOCSGEN_LAST_PART})
    file(APPEND "${GEN_HEADER_TMP}" "  register_machines_part_${part_idx}(out);\n")
  endforeach()
endif()
file(APPEND "${GEN_HEADER_TMP}" "}\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${GEN_HEADER_TMP}" "${GEN_HEADER}"
)

add_executable(docsgen docsgen.cpp "${GEN_HEADER}" ${GEN_SOURCES})

# The docs generator is an introspection utility, not runtime code. Release optimization makes
# generated SML registration translation units dominate quality-gate latency without improving
# the generated output.
target_compile_options(docsgen PRIVATE
  "$<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang,GNU>:-O0;-g0;-fno-sanitize=all>"
  "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/Od>"
)

target_include_directories(docsgen PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${ROOT_DIR}/src"
  "${ROOT_DIR}/include"
  "${boost_sml_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_BINARY_DIR}"
)

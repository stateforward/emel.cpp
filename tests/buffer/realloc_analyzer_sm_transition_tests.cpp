#include <array>
#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/buffer/realloc_analyzer/sm.hpp"

namespace {

TEST_CASE("buffer_realloc_analyzer_sm_analyze_success_and_reset") {
  emel::buffer::realloc_analyzer::sm machine{};
  int32_t needs_realloc = 0;
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::buffer::realloc_analyzer::event::analyze{
    .graph = {},
    .node_allocs = nullptr,
    .node_alloc_count = 0,
    .leaf_allocs = nullptr,
    .leaf_alloc_count = 0,
    .needs_realloc_out = &needs_realloc,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  err = EMEL_OK;
  CHECK(machine.process_event(emel::buffer::realloc_analyzer::event::reset{
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
}

TEST_CASE("buffer_realloc_analyzer_sm_validation_error_path") {
  emel::buffer::realloc_analyzer::sm machine{};
  int32_t err = EMEL_OK;

  machine.process_event(emel::buffer::realloc_analyzer::event::analyze{
    .graph = {},
    .node_allocs = nullptr,
    .node_alloc_count = -1,
    .leaf_allocs = nullptr,
    .leaf_alloc_count = 0,
    .needs_realloc_out = nullptr,
    .error_out = &err,
  });
  CHECK(err != EMEL_OK);
}

}  // namespace

#include <boost/sml.hpp>
#include <doctest/doctest.h>

#include "emel/buffer/realloc_analyzer/sm.hpp"
#include "emel/emel.h"

namespace {

TEST_CASE("realloc_analyzer_sm_success_path") {
  emel::buffer::realloc_analyzer::sm machine{};
  int32_t needs_realloc = 1;
  int32_t err = EMEL_OK;

  emel::buffer::realloc_analyzer::event::analyze request{
    .graph = {},
    .node_allocs = nullptr,
    .node_alloc_count = 0,
    .leaf_allocs = nullptr,
    .leaf_alloc_count = 0,
    .needs_realloc_out = &needs_realloc,
    .error_out = &err,
  };

  CHECK(machine.process_event(request));
  CHECK(err == EMEL_OK);
  CHECK(needs_realloc == 0);
}

TEST_CASE("realloc_analyzer_sm_detects_realloc_need") {
  emel::buffer::realloc_analyzer::sm machine{};
  int32_t needs_realloc = 0;
  int32_t err = EMEL_OK;

  emel::buffer::allocator::event::tensor_desc nodes[1] = {
    emel::buffer::allocator::event::tensor_desc{
      .tensor_id = 0,
      .alloc_size = 16,
    },
  };
  emel::buffer::realloc_analyzer::event::node_alloc node_allocs[1] = {};
  emel::buffer::allocator::event::graph_view graph{
    .nodes = nodes,
    .n_nodes = 1,
    .leafs = nullptr,
    .n_leafs = 0,
  };

  emel::buffer::realloc_analyzer::event::analyze request{
    .graph = graph,
    .node_allocs = node_allocs,
    .node_alloc_count = 0,
    .leaf_allocs = nullptr,
    .leaf_alloc_count = 0,
    .needs_realloc_out = &needs_realloc,
    .error_out = &err,
  };

  CHECK(machine.process_event(request));
  CHECK(err == EMEL_OK);
  CHECK(needs_realloc == 1);
}

TEST_CASE("realloc_analyzer_sm_reports_validation_error") {
  emel::buffer::realloc_analyzer::sm machine{};
  int32_t needs_realloc = 0;
  int32_t err = EMEL_OK;

  emel::buffer::allocator::event::graph_view graph{
    .nodes = nullptr,
    .n_nodes = 1,
    .leafs = nullptr,
    .n_leafs = 0,
  };

  emel::buffer::realloc_analyzer::event::analyze request{
    .graph = graph,
    .node_allocs = nullptr,
    .node_alloc_count = 0,
    .leaf_allocs = nullptr,
    .leaf_alloc_count = 0,
    .needs_realloc_out = &needs_realloc,
    .error_out = &err,
  };

  CHECK_FALSE(machine.process_event(request));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("realloc_analyzer_sm_records_phase_error_on_invalid_request") {
  emel::buffer::realloc_analyzer::action::context ctx{};
  boost::sml::sm<emel::buffer::realloc_analyzer::model, boost::sml::testing> machine{ctx};
  int32_t err = EMEL_OK;

  emel::buffer::realloc_analyzer::event::analyze request{
    .graph = {
      .nodes = nullptr,
      .n_nodes = 1,
      .leafs = nullptr,
      .n_leafs = 0,
    },
    .node_allocs = nullptr,
    .node_alloc_count = 0,
    .leaf_allocs = nullptr,
    .leaf_alloc_count = 0,
    .error_out = &err,
  };

  CHECK(machine.process_event(request));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);
}

}  // namespace

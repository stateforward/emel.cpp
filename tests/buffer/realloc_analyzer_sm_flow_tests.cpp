#include <boost/sml.hpp>
#include <doctest/doctest.h>

#include "emel/buffer/realloc_analyzer/sm.hpp"
#include "emel/emel.h"

namespace {

struct noop_queue {
  using container_type = void;

  template <class Event>
  void push(const Event &) noexcept {}
};

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

TEST_CASE("realloc_analyzer_sm_manual_evaluate_error_path") {
  emel::buffer::realloc_analyzer::action::context ctx{};
  noop_queue queue{};
  emel::buffer::realloc_analyzer::Process process{queue};
  boost::sml::sm<
    emel::buffer::realloc_analyzer::model,
    boost::sml::testing,
    emel::buffer::realloc_analyzer::Process>
    machine{ctx, process};
  int32_t needs_realloc = 0;
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
  CHECK(machine.process_event(emel::buffer::realloc_analyzer::events::validate_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::buffer::realloc_analyzer::events::evaluate_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  }));
  CHECK(machine.process_event(emel::buffer::realloc_analyzer::events::analyze_error{
    .err = EMEL_ERR_BACKEND,
    .error_out = &err,
    .request = &request,
  }));
}

TEST_CASE("realloc_analyzer_sm_manual_publish_error_path") {
  emel::buffer::realloc_analyzer::action::context ctx{};
  noop_queue queue{};
  emel::buffer::realloc_analyzer::Process process{queue};
  boost::sml::sm<
    emel::buffer::realloc_analyzer::model,
    boost::sml::testing,
    emel::buffer::realloc_analyzer::Process>
    machine{ctx, process};
  int32_t needs_realloc = 0;
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
  CHECK(machine.process_event(emel::buffer::realloc_analyzer::events::validate_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::buffer::realloc_analyzer::events::evaluate_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::buffer::realloc_analyzer::events::publish_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  }));
  CHECK(machine.process_event(emel::buffer::realloc_analyzer::events::analyze_error{
    .err = EMEL_ERR_BACKEND,
    .error_out = &err,
    .request = &request,
  }));
}

}  // namespace

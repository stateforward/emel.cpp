#include <array>
#include <cstdint>

#include <doctest/doctest.h>

#include "emel/buffer/allocator/events.hpp"
#include "emel/buffer/realloc_analyzer/actions.hpp"
#include "emel/buffer/realloc_analyzer/guards.hpp"
#include "emel/buffer/realloc_analyzer/sm.hpp"
#include "emel/emel.h"

namespace {

using tensor_desc = emel::buffer::allocator::event::tensor_desc;
using graph_view = emel::buffer::allocator::event::graph_view;
using node_alloc = emel::buffer::realloc_analyzer::event::node_alloc;
using leaf_alloc = emel::buffer::realloc_analyzer::event::leaf_alloc;

struct graph_storage {
  std::array<tensor_desc, 1> nodes = {};
  std::array<tensor_desc, 1> leafs = {};
  int32_t n_nodes = 1;
  int32_t n_leafs = 1;
};

graph_view as_view(const graph_storage & g) {
  return graph_view{
    .nodes = g.nodes.data(),
    .n_nodes = g.n_nodes,
    .leafs = g.leafs.data(),
    .n_leafs = g.n_leafs,
  };
}

graph_storage make_graph() {
  graph_storage g{};
  g.leafs[0] = tensor_desc{
    .tensor_id = 1,
    .alloc_size = 16,
    .src_ids = emel::buffer::allocator::event::make_src_ids(),
    .is_view = false,
    .view_src_id = -1,
    .is_input = true,
    .is_output = false,
    .has_external_data = false,
  };
  g.nodes[0] = tensor_desc{
    .tensor_id = 2,
    .alloc_size = 16,
    .src_ids = [] {
      auto ids = emel::buffer::allocator::event::make_src_ids();
      ids[0] = 1;
      return ids;
    }(),
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = true,
    .has_external_data = false,
  };
  return g;
}

}  // namespace

TEST_CASE("buffer_realloc_analyzer_rejects_invalid_graph") {
  emel::buffer::realloc_analyzer::sm machine{};
  int32_t needs_realloc = 0;
  int32_t error = EMEL_OK;

  graph_view invalid{
    .nodes = nullptr,
    .n_nodes = 1,
    .leafs = nullptr,
    .n_leafs = 0,
  };

  CHECK_FALSE(machine.process_event(emel::buffer::realloc_analyzer::event::analyze{
    .graph = invalid,
    .node_allocs = nullptr,
    .node_alloc_count = 0,
    .leaf_allocs = nullptr,
    .leaf_alloc_count = 0,
    .needs_realloc_out = &needs_realloc,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_realloc_analyzer_rejects_missing_alloc_arrays") {
  emel::buffer::realloc_analyzer::sm machine{};
  int32_t needs_realloc = 0;
  int32_t error = EMEL_OK;
  graph_storage g = make_graph();

  CHECK_FALSE(machine.process_event(emel::buffer::realloc_analyzer::event::analyze{
    .graph = as_view(g),
    .node_allocs = nullptr,
    .node_alloc_count = g.n_nodes,
    .leaf_allocs = nullptr,
    .leaf_alloc_count = g.n_leafs,
    .needs_realloc_out = &needs_realloc,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_realloc_analyzer_ignores_mismatched_allocations") {
  emel::buffer::realloc_analyzer::sm machine{};
  int32_t needs_realloc = 0;
  int32_t error = EMEL_OK;
  graph_storage g = make_graph();

  std::array<node_alloc, 1> node_allocs = {};
  std::array<leaf_alloc, 1> leaf_allocs = {};
  node_allocs[0].dst.tensor_id = 99;
  node_allocs[0].dst.buffer_id = 0;
  node_allocs[0].dst.size_max = 16;
  node_allocs[0].src[0].buffer_id = 0;
  node_allocs[0].src[0].size_max = 16;
  leaf_allocs[0].leaf.tensor_id = 1;
  leaf_allocs[0].leaf.buffer_id = 0;
  leaf_allocs[0].leaf.size_max = 16;

  CHECK(machine.process_event(emel::buffer::realloc_analyzer::event::analyze{
    .graph = as_view(g),
    .node_allocs = node_allocs.data(),
    .node_alloc_count = static_cast<int32_t>(node_allocs.size()),
    .leaf_allocs = leaf_allocs.data(),
    .leaf_alloc_count = static_cast<int32_t>(leaf_allocs.size()),
    .needs_realloc_out = &needs_realloc,
    .error_out = &error,
  }));

  CHECK(error == EMEL_OK);
  CHECK(needs_realloc == 0);
}

TEST_CASE("buffer_realloc_analyzer_graph_needs_realloc_ignores_src_id_without_alloc") {
  graph_storage g = make_graph();
  g.nodes[0].src_ids = emel::buffer::allocator::event::make_src_ids();

  std::array<node_alloc, 1> node_allocs = {};
  std::array<leaf_alloc, 1> leaf_allocs = {};
  node_allocs[0].dst.tensor_id = 2;
  node_allocs[0].dst.buffer_id = 0;
  node_allocs[0].dst.size_max = 16;
  node_allocs[0].src[0].tensor_id = 7;
  leaf_allocs[0].leaf.tensor_id = 1;
  leaf_allocs[0].leaf.buffer_id = 0;
  leaf_allocs[0].leaf.size_max = 16;

  CHECK_FALSE(emel::buffer::realloc_analyzer::action::detail::graph_needs_realloc(
    as_view(g),
    node_allocs.data(),
    static_cast<int32_t>(node_allocs.size()),
    leaf_allocs.data(),
    static_cast<int32_t>(leaf_allocs.size())));
}

TEST_CASE("buffer_realloc_analyzer_graph_needs_realloc_flags_missing_src_tensor") {
  graph_storage g = make_graph();
  g.nodes[0].src_ids = emel::buffer::allocator::event::make_src_ids();
  g.nodes[0].src_ids[0] = 99;

  std::array<node_alloc, 1> node_allocs = {};
  std::array<leaf_alloc, 1> leaf_allocs = {};
  node_allocs[0].dst.tensor_id = 2;
  node_allocs[0].dst.buffer_id = 0;
  node_allocs[0].dst.size_max = 16;
  node_allocs[0].src[0].tensor_id = 99;
  leaf_allocs[0].leaf.tensor_id = 1;
  leaf_allocs[0].leaf.buffer_id = 0;
  leaf_allocs[0].leaf.size_max = 16;

  CHECK(emel::buffer::realloc_analyzer::action::detail::graph_needs_realloc(
    as_view(g),
    node_allocs.data(),
    static_cast<int32_t>(node_allocs.size()),
    leaf_allocs.data(),
    static_cast<int32_t>(leaf_allocs.size())));
}

TEST_CASE("buffer_realloc_analyzer_validate_rejects_negative_alloc_counts") {
  emel::buffer::realloc_analyzer::action::context ctx{};
  graph_storage g = make_graph();
  int32_t err = EMEL_OK;
  emel::buffer::realloc_analyzer::event::analyze request{
    .graph = as_view(g),
    .node_allocs = nullptr,
    .node_alloc_count = -1,
    .leaf_allocs = nullptr,
    .leaf_alloc_count = -1,
  };
  emel::buffer::realloc_analyzer::event::validate validate{
    .graph = request.graph,
    .node_allocs = request.node_allocs,
    .node_alloc_count = request.node_alloc_count,
    .leaf_allocs = request.leaf_allocs,
    .leaf_alloc_count = request.leaf_alloc_count,
    .error_out = &err,
    .request = &request,
  };
  CHECK(emel::buffer::realloc_analyzer::guard::invalid_analyze_request{}(validate, ctx));
  CHECK(err == EMEL_OK);
}

TEST_CASE("buffer_realloc_analyzer_on_analyze_done_updates_output") {
  emel::buffer::realloc_analyzer::action::context ctx{};
  int32_t needs_realloc = 0;
  int32_t error = EMEL_OK;

  emel::buffer::realloc_analyzer::action::on_analyze_done(
    emel::buffer::realloc_analyzer::events::analyze_done{
      .needs_realloc = 1,
      .needs_realloc_out = &needs_realloc,
      .error_out = &error,
    },
    ctx);

  CHECK(needs_realloc == 1);
  CHECK(error == EMEL_OK);
}

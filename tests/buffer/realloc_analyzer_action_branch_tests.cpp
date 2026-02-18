#include <array>
#include <cstdint>

#include <doctest/doctest.h>

#include "emel/buffer/allocator/events.hpp"
#include "emel/buffer/realloc_analyzer/actions.hpp"
#include "emel/buffer/realloc_analyzer/events.hpp"
#include "emel/buffer/realloc_analyzer/guards.hpp"
#include "emel/emel.h"

namespace {

using graph_view = emel::buffer::allocator::event::graph_view;
using tensor_desc = emel::buffer::allocator::event::tensor_desc;
using node_alloc = emel::buffer::realloc_analyzer::event::node_alloc;
using leaf_alloc = emel::buffer::realloc_analyzer::event::leaf_alloc;

struct graph_storage {
  std::array<tensor_desc, 1> nodes = {};
  std::array<tensor_desc, 1> leafs = {};
};

graph_storage make_graph() {
  graph_storage g{};
  g.nodes[0] = tensor_desc{
    .tensor_id = 1,
    .alloc_size = 16,
    .src_ids = {{2, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = true,
    .has_external_data = false,
  };
  g.leafs[0] = tensor_desc{
    .tensor_id = 2,
    .alloc_size = 32,
    .src_ids = {{-1, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = true,
    .is_output = false,
    .has_external_data = false,
  };
  return g;
}

graph_view as_view(const graph_storage & g) {
  return graph_view{
    .nodes = g.nodes.data(),
    .n_nodes = static_cast<int32_t>(g.nodes.size()),
    .leafs = g.leafs.data(),
    .n_leafs = static_cast<int32_t>(g.leafs.size()),
  };
}

node_alloc make_node_alloc(int32_t dst_id, int32_t dst_buffer, int32_t dst_size,
                           int32_t src_id, int32_t src_buffer, int32_t src_size) {
  node_alloc alloc{};
  alloc.dst.tensor_id = dst_id;
  alloc.dst.buffer_id = dst_buffer;
  alloc.dst.size_max = dst_size;
  alloc.src[0].tensor_id = src_id;
  alloc.src[0].buffer_id = src_buffer;
  alloc.src[0].size_max = src_size;
  return alloc;
}

leaf_alloc make_leaf_alloc(int32_t tensor_id, int32_t buffer_id, int32_t size_max) {
  leaf_alloc alloc{};
  alloc.leaf.tensor_id = tensor_id;
  alloc.leaf.buffer_id = buffer_id;
  alloc.leaf.size_max = size_max;
  return alloc;
}

}  // namespace

TEST_CASE("buffer_realloc_analyzer_detail_helpers_cover_error_fallbacks") {
  using emel::buffer::realloc_analyzer::action::detail::align_up;
  using emel::buffer::realloc_analyzer::action::detail::normalize_error;

  CHECK(align_up(0, 16) == 0);
  CHECK(align_up(1, 16) == 16);

  CHECK(normalize_error(EMEL_ERR_INVALID_ARGUMENT, EMEL_OK) == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(normalize_error(EMEL_OK, EMEL_ERR_INVALID_ARGUMENT) == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(normalize_error(EMEL_OK, EMEL_OK) == EMEL_ERR_BACKEND);
}

TEST_CASE("buffer_realloc_analyzer_detail_valid_graph_tensors_rejects_invalid") {
  using emel::buffer::realloc_analyzer::action::detail::valid_graph_tensors;

  graph_view invalid{};
  invalid.n_nodes = 1;
  CHECK_FALSE(valid_graph_tensors(invalid));

  graph_storage g = make_graph();
  g.nodes[0].tensor_id = -1;
  CHECK_FALSE(valid_graph_tensors(as_view(g)));

  graph_storage ok = make_graph();
  CHECK(valid_graph_tensors(as_view(ok)));
}

TEST_CASE("buffer_realloc_analyzer_detail_tensor_snapshot_valid_branches") {
  using emel::buffer::realloc_analyzer::action::detail::tensor_snapshot_valid;
  using emel::buffer::realloc_analyzer::action::detail::align_up;

  tensor_desc tensor{
    .tensor_id = 1,
    .alloc_size = 16,
    .src_ids = {{-1, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = false,
    .has_external_data = false,
  };
  emel::buffer::realloc_analyzer::event::tensor_alloc alloc{};

  alloc.buffer_id = -1;
  alloc.size_max = 0;
  CHECK_FALSE(tensor_snapshot_valid(tensor, alloc));

  alloc.buffer_id = 0;
  alloc.size_max = align_up(tensor.alloc_size, 16) - 1;
  CHECK_FALSE(tensor_snapshot_valid(tensor, alloc));

  alloc.size_max = align_up(tensor.alloc_size, 16);
  CHECK(tensor_snapshot_valid(tensor, alloc));

  tensor.has_external_data = true;
  alloc.buffer_id = -1;
  alloc.size_max = 0;
  CHECK(tensor_snapshot_valid(tensor, alloc));
}

TEST_CASE("buffer_realloc_analyzer_detail_graph_needs_realloc_paths") {
  using emel::buffer::realloc_analyzer::action::detail::graph_needs_realloc;
  using emel::buffer::realloc_analyzer::action::detail::align_up;

  graph_storage g = make_graph();
  graph_view view = as_view(g);
  std::array<node_alloc, 1> nodes = {{
    make_node_alloc(
      1, 0, align_up(g.nodes[0].alloc_size, 16),
      2, 0, align_up(g.leafs[0].alloc_size, 16)),
  }};
  std::array<leaf_alloc, 1> leafs = {{
    make_leaf_alloc(2, 0, align_up(g.leafs[0].alloc_size, 16)),
  }};

  CHECK(graph_needs_realloc(view, nodes.data(), 0, leafs.data(), 1));
  CHECK(graph_needs_realloc(view, nodes.data(), 1, leafs.data(), 0));

  nodes[0].dst.tensor_id = 99;
  CHECK(graph_needs_realloc(view, nodes.data(), 1, leafs.data(), 1));
  nodes[0].dst.tensor_id = 1;

  nodes[0].dst.buffer_id = -1;
  CHECK(graph_needs_realloc(view, nodes.data(), 1, leafs.data(), 1));
  nodes[0].dst.buffer_id = 0;

  g.nodes[0].src_ids[0] = -1;
  view = as_view(g);
  nodes[0].src[0].tensor_id = 5;
  CHECK(graph_needs_realloc(view, nodes.data(), 1, leafs.data(), 1));

  g.nodes[0].src_ids[0] = 2;
  view = as_view(g);
  nodes[0].src[0].tensor_id = 3;
  CHECK(graph_needs_realloc(view, nodes.data(), 1, leafs.data(), 1));

  nodes[0].src[0].tensor_id = 99;
  g.nodes[0].src_ids[0] = 99;
  view = as_view(g);
  CHECK(graph_needs_realloc(view, nodes.data(), 1, leafs.data(), 1));

  g.nodes[0].src_ids[0] = 2;
  view = as_view(g);
  nodes[0].src[0].tensor_id = 2;
  nodes[0].src[0].size_max = 0;
  CHECK(graph_needs_realloc(view, nodes.data(), 1, leafs.data(), 1));

  nodes[0].src[0].size_max = align_up(g.leafs[0].alloc_size, 16);
  CHECK_FALSE(graph_needs_realloc(view, nodes.data(), 1, leafs.data(), 1));
}

TEST_CASE("buffer_realloc_analyzer_actions_cover_validation_and_publish") {
  emel::buffer::realloc_analyzer::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::buffer::realloc_analyzer::event::analyze invalid_request{
    .graph = {},
    .node_allocs = nullptr,
    .node_alloc_count = -1,
    .leaf_allocs = nullptr,
    .leaf_alloc_count = 0,
  };
  emel::buffer::realloc_analyzer::event::validate invalid_validate{
    .graph = invalid_request.graph,
    .node_allocs = invalid_request.node_allocs,
    .node_alloc_count = invalid_request.node_alloc_count,
    .leaf_allocs = invalid_request.leaf_allocs,
    .leaf_alloc_count = invalid_request.leaf_alloc_count,
    .error_out = &err,
    .request = &invalid_request,
  };
  CHECK(emel::buffer::realloc_analyzer::guard::invalid_analyze_request{}(
    invalid_validate, ctx));
  CHECK(err == EMEL_OK);

  graph_storage g = make_graph();
  std::array<node_alloc, 1> nodes = {{
    make_node_alloc(1, 0, 16, 2, 0, 32),
  }};
  std::array<leaf_alloc, 1> leafs = {{
    make_leaf_alloc(2, 0, 32),
  }};
  err = EMEL_OK;
  emel::buffer::realloc_analyzer::event::analyze valid_request{
    .graph = as_view(g),
    .node_allocs = nodes.data(),
    .node_alloc_count = 1,
    .leaf_allocs = leafs.data(),
    .leaf_alloc_count = 1,
  };
  emel::buffer::realloc_analyzer::event::validate valid_validate{
    .graph = valid_request.graph,
    .node_allocs = valid_request.node_allocs,
    .node_alloc_count = valid_request.node_alloc_count,
    .leaf_allocs = valid_request.leaf_allocs,
    .leaf_alloc_count = valid_request.leaf_alloc_count,
    .error_out = &err,
    .request = &valid_request,
  };
  CHECK(emel::buffer::realloc_analyzer::guard::valid_analyze_request{}(
    valid_validate, ctx));
  emel::buffer::realloc_analyzer::action::run_validate(valid_validate, ctx);
  CHECK(err == EMEL_OK);

  ctx.needs_realloc = true;
  int32_t needs_realloc_out = 0;
  err = EMEL_OK;
  emel::buffer::realloc_analyzer::action::run_publish(
    emel::buffer::realloc_analyzer::event::publish{
      .needs_realloc_out = &needs_realloc_out,
      .error_out = &err,
    },
    ctx);
  CHECK(needs_realloc_out == 1);
  CHECK(err == EMEL_OK);
}

TEST_CASE("buffer_realloc_analyzer_guard_rejects_missing_request") {
  emel::buffer::realloc_analyzer::action::context ctx{};
  emel::buffer::realloc_analyzer::event::validate validate{
    .graph = {},
    .node_allocs = nullptr,
    .node_alloc_count = 0,
    .leaf_allocs = nullptr,
    .leaf_alloc_count = 0,
    .error_out = nullptr,
    .request = nullptr,
  };
  CHECK_FALSE(emel::buffer::realloc_analyzer::guard::valid_analyze_request{}(
    validate, ctx));
}

TEST_CASE("buffer_realloc_analyzer_action_on_unexpected_reports_invalid_argument") {
  emel::buffer::realloc_analyzer::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::buffer::realloc_analyzer::events::analyze_error ev{
    .err = EMEL_ERR_BACKEND,
    .error_out = &err,
  };

  emel::buffer::realloc_analyzer::action::on_unexpected(ev, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(ctx.step == 1);
}

TEST_CASE("buffer_realloc_analyzer_action_error_handlers_normalize") {
  emel::buffer::realloc_analyzer::action::context ctx{};
  int32_t err_out = EMEL_OK;

  emel::buffer::realloc_analyzer::action::on_analyze_error(
    emel::buffer::realloc_analyzer::events::analyze_error{
      .err = 0,
      .error_out = &err_out,
      .request = nullptr,
    },
    ctx);
  CHECK(err_out == EMEL_ERR_BACKEND);

  err_out = EMEL_OK;
  emel::buffer::realloc_analyzer::action::on_reset_error(
    emel::buffer::realloc_analyzer::events::reset_error{
      .err = EMEL_ERR_INVALID_ARGUMENT,
      .error_out = &err_out,
      .request = nullptr,
    },
    ctx);
  CHECK(err_out == EMEL_ERR_INVALID_ARGUMENT);
}

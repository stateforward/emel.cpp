#include <array>
#include <cstdint>
#include <limits>

#include <doctest/doctest.h>

#include "emel/buffer/allocator/actions.hpp"
#include "emel/buffer/planner/sm.hpp"
#include "emel/buffer/realloc_analyzer/sm.hpp"
#include "emel/emel.h"

namespace {

using tensor_desc = emel::buffer::allocator::event::tensor_desc;
using graph_view = emel::buffer::allocator::event::graph_view;

struct graph_storage {
  std::array<tensor_desc, 1> nodes = {};
  std::array<tensor_desc, 1> leafs = {};
  int32_t n_nodes = 1;
  int32_t n_leafs = 1;
};

graph_storage make_graph() {
  graph_storage g{};
  g.leafs[0] = tensor_desc{
    .tensor_id = 1,
    .alloc_size = 16,
    .src_ids = {{-1, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = true,
    .is_output = false,
    .has_external_data = false,
  };
  g.nodes[0] = tensor_desc{
    .tensor_id = 2,
    .alloc_size = 16,
    .src_ids = {{1, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = true,
    .has_external_data = false,
  };
  return g;
}

graph_view as_view(const graph_storage & g) {
  return graph_view{
    .nodes = g.nodes.data(),
    .n_nodes = g.n_nodes,
    .leafs = g.leafs.data(),
    .n_leafs = g.n_leafs,
  };
}

}  // namespace

TEST_CASE("buffer_allocator_detail_valid_graph_tensors_rejects_bad_inputs") {
  const graph_storage g = make_graph();
  CHECK(emel::buffer::allocator::action::detail::valid_graph_tensors(as_view(g)));

  graph_view missing_nodes{
    .nodes = nullptr,
    .n_nodes = 1,
    .leafs = g.leafs.data(),
    .n_leafs = g.n_leafs,
  };
  CHECK_FALSE(emel::buffer::allocator::action::detail::valid_graph_tensors(missing_nodes));

  graph_view missing_leafs{
    .nodes = g.nodes.data(),
    .n_nodes = g.n_nodes,
    .leafs = nullptr,
    .n_leafs = 1,
  };
  CHECK_FALSE(emel::buffer::allocator::action::detail::valid_graph_tensors(missing_leafs));

  graph_storage neg_id = g;
  neg_id.nodes[0].tensor_id = -4;
  CHECK_FALSE(emel::buffer::allocator::action::detail::valid_graph_tensors(as_view(neg_id)));

  graph_storage neg_size = g;
  neg_size.nodes[0].alloc_size = -1;
  CHECK_FALSE(emel::buffer::allocator::action::detail::valid_graph_tensors(as_view(neg_size)));
}

TEST_CASE("buffer_allocator_detail_graph_needs_realloc_detects_mismatches") {
  graph_storage g = make_graph();
  const graph_view view = as_view(g);

  emel::buffer::allocator::action::context c{};
  c.buffer_count = 1;
  c.has_reserve_snapshot = true;
  c.last_n_nodes = 1;
  c.last_n_leafs = 1;
  c.node_allocs[0].dst = emel::buffer::allocator::action::tensor_alloc{
    .tensor_id = 2,
    .buffer_id = 0,
    .size_max = 16,
  };
  c.node_allocs[0].src[0] = emel::buffer::allocator::action::tensor_alloc{
    .tensor_id = 1,
    .buffer_id = 0,
    .size_max = 16,
  };
  c.node_allocs[0].src[1] = emel::buffer::allocator::action::tensor_alloc{};
  c.node_allocs[0].src[2] = emel::buffer::allocator::action::tensor_alloc{};
  c.node_allocs[0].src[3] = emel::buffer::allocator::action::tensor_alloc{};
  c.leaf_allocs[0].leaf = emel::buffer::allocator::action::tensor_alloc{
    .tensor_id = 1,
    .buffer_id = 0,
    .size_max = 16,
  };

  CHECK_FALSE(emel::buffer::allocator::action::detail::graph_needs_realloc(view, c));

  emel::buffer::allocator::action::context no_snapshot = c;
  no_snapshot.has_reserve_snapshot = false;
  CHECK(emel::buffer::allocator::action::detail::graph_needs_realloc(view, no_snapshot));

  emel::buffer::allocator::action::context mismatched_counts = c;
  mismatched_counts.last_n_nodes = 2;
  CHECK(emel::buffer::allocator::action::detail::graph_needs_realloc(view, mismatched_counts));

  emel::buffer::allocator::action::context mismatched_dst = c;
  mismatched_dst.node_allocs[0].dst.tensor_id = 99;
  CHECK(emel::buffer::allocator::action::detail::graph_needs_realloc(view, mismatched_dst));

  emel::buffer::allocator::action::context mismatched_src = c;
  mismatched_src.node_allocs[0].src[0].tensor_id = 99;
  CHECK(emel::buffer::allocator::action::detail::graph_needs_realloc(view, mismatched_src));

  graph_storage no_src = g;
  no_src.nodes[0].src_ids = {{-1, -1, -1, -1}};
  emel::buffer::allocator::action::context src_missing = c;
  src_missing.node_allocs[0].src[0].tensor_id = 1;
  CHECK(emel::buffer::allocator::action::detail::graph_needs_realloc(as_view(no_src), src_missing));

  emel::buffer::allocator::action::context size_mismatch = c;
  size_mismatch.node_allocs[0].dst.size_max = 8;
  CHECK(emel::buffer::allocator::action::detail::graph_needs_realloc(view, size_mismatch));

  emel::buffer::allocator::action::context leaf_mismatch = c;
  leaf_mismatch.leaf_allocs[0].leaf.tensor_id = 5;
  CHECK_FALSE(emel::buffer::allocator::action::detail::graph_needs_realloc(view, leaf_mismatch));
}

TEST_CASE("buffer_allocator_detail_run_planner_and_realloc_analyzer") {
  graph_storage g = make_graph();
  const graph_view view = as_view(g);

  emel::buffer::allocator::action::context ctx{};
  std::array<int32_t, emel::buffer::allocator::action::k_max_buffers> sizes = {};
  int32_t err = EMEL_OK;
  CHECK_FALSE(emel::buffer::allocator::action::detail::run_planner(
    nullptr,
    view,
    nullptr,
    nullptr,
    1,
    true,
    sizes.data(),
    1,
    nullptr,
    0,
    nullptr,
    0,
    nullptr,
    ctx,
    err));
  CHECK(err != EMEL_OK);

  emel::buffer::planner::sm planner{};
  err = EMEL_OK;
  CHECK(emel::buffer::allocator::action::detail::run_planner(
    &planner,
    view,
    nullptr,
    nullptr,
    1,
    true,
    sizes.data(),
    1,
    nullptr,
    0,
    nullptr,
    0,
    &emel::buffer::planner::default_strategies::reserve,
    ctx,
    err));
  CHECK(err == EMEL_OK);

  emel::buffer::allocator::action::context c{};
  c.last_n_nodes = 0;
  c.last_n_leafs = 0;
  bool needs_realloc = false;
  err = EMEL_OK;
  emel::buffer::realloc_analyzer::sm analyzer{};
  graph_view empty{
    .nodes = nullptr,
    .n_nodes = 0,
    .leafs = nullptr,
    .n_leafs = 0,
  };
  CHECK(emel::buffer::allocator::action::detail::run_realloc_analyzer(
    &analyzer,
    empty,
    c,
    needs_realloc,
    err));
  CHECK(err == EMEL_OK);
}

TEST_CASE("buffer_allocator_detail_size_helpers_cover_growth") {
  std::array<int32_t, emel::buffer::allocator::action::k_max_buffers> committed = {};
  std::array<int32_t, emel::buffer::allocator::action::k_max_buffers> required = {};

  CHECK_FALSE(emel::buffer::allocator::action::detail::requires_growth(committed, required, 1));

  required[0] = 64;
  CHECK(emel::buffer::allocator::action::detail::requires_growth(committed, required, 1));

  emel::buffer::allocator::action::detail::commit_sizes(committed, required, 1);
  CHECK(committed[0] == 64);
}

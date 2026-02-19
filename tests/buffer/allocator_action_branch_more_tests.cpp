#include <array>
#include <doctest/doctest.h>

#include "emel/buffer/allocator/actions.hpp"
#include "emel/buffer/allocator/events.hpp"
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

graph_view as_view(const graph_storage & g) {
  return graph_view{
    .nodes = g.nodes.data(),
    .n_nodes = g.n_nodes,
    .leafs = g.leafs.data(),
    .n_leafs = g.n_leafs,
  };
}

}  // namespace

TEST_CASE("buffer_allocator_detail_tensor_needs_realloc_skips_view_and_external") {
  emel::buffer::allocator::action::context ctx{};
  emel::buffer::allocator::action::tensor_alloc alloc{};
  alloc.tensor_id = 1;
  alloc.buffer_id = -1;
  alloc.size_max = 0;

  tensor_desc view_tensor{
    .tensor_id = 1,
    .alloc_size = 16,
    .src_ids = emel::buffer::allocator::event::make_src_ids(),
    .is_view = true,
    .view_src_id = 0,
    .is_input = false,
    .is_output = false,
    .has_external_data = false,
  };
  CHECK_FALSE(emel::buffer::allocator::action::detail::tensor_needs_realloc(ctx, view_tensor, alloc));

  tensor_desc external_tensor = view_tensor;
  external_tensor.is_view = false;
  external_tensor.has_external_data = true;
  CHECK_FALSE(emel::buffer::allocator::action::detail::tensor_needs_realloc(ctx, external_tensor, alloc));
}

TEST_CASE("buffer_allocator_capture_snapshot_success") {
  emel::buffer::allocator::action::context c{};
  c.buffer_count = 1;
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
  std::array<int32_t, 1> node_ids = {{0}};
  std::array<int32_t, 1> leaf_ids = {{0}};

  CHECK(emel::buffer::allocator::action::detail::capture_alloc_snapshot(
    c,
    as_view(g),
    node_ids.data(),
    leaf_ids.data()));
}

TEST_CASE("buffer_allocator_graph_needs_realloc_ignores_view_and_external") {
  emel::buffer::allocator::action::context c{};
  c.buffer_count = 1;
  c.has_reserve_snapshot = true;
  c.last_n_nodes = 1;
  c.last_n_leafs = 1;

  graph_storage g{};
  g.leafs[0] = tensor_desc{
    .tensor_id = 1,
    .alloc_size = 0,
    .src_ids = emel::buffer::allocator::event::make_src_ids(),
    .is_view = false,
    .view_src_id = -1,
    .is_input = true,
    .is_output = false,
    .has_external_data = true,
  };
  g.nodes[0] = tensor_desc{
    .tensor_id = 2,
    .alloc_size = 0,
    .src_ids = [] {
      auto ids = emel::buffer::allocator::event::make_src_ids();
      ids[0] = 1;
      return ids;
    }(),
    .is_view = true,
    .view_src_id = 1,
    .is_input = false,
    .is_output = true,
    .has_external_data = false,
  };

  c.node_allocs[0].dst.tensor_id = 2;
  c.node_allocs[0].dst.buffer_id = -1;
  c.node_allocs[0].dst.size_max = 0;
  c.node_allocs[0].src[0].tensor_id = 1;
  c.node_allocs[0].src[0].buffer_id = -1;
  c.node_allocs[0].src[0].size_max = 0;
  c.leaf_allocs[0].leaf.tensor_id = 1;
  c.leaf_allocs[0].leaf.buffer_id = -1;
  c.leaf_allocs[0].leaf.size_max = 0;

  CHECK_FALSE(emel::buffer::allocator::action::detail::graph_needs_realloc(
    as_view(g),
    c));
}

TEST_CASE("buffer_allocator_action_error_handlers_cover_reserve_and_alloc_graph") {
  emel::buffer::allocator::action::context c{};
  int32_t err = EMEL_OK;

  emel::buffer::allocator::action::on_reserve_error(
    emel::buffer::allocator::events::reserve_error{.err = 0, .error_out = &err},
    c);
  CHECK(err == EMEL_ERR_BACKEND);

  err = EMEL_OK;
  emel::buffer::allocator::action::on_alloc_graph_error(
    emel::buffer::allocator::events::alloc_graph_error{.err = 0, .error_out = &err},
    c);
  CHECK(err == EMEL_ERR_BACKEND);
}

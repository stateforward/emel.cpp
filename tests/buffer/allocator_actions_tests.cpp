#include <array>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/buffer/allocator/actions.hpp"
#include "emel/buffer/allocator/sm.hpp"
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
    .alloc_size = 64,
    .src_ids = {{-1, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = true,
    .is_output = false,
    .has_external_data = false,
  };
  g.nodes[0] = tensor_desc{
    .tensor_id = 2,
    .alloc_size = 128,
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

graph_view invalid_view() {
  return graph_view{
    .nodes = nullptr,
    .n_nodes = 1,
    .leafs = nullptr,
    .n_leafs = 0,
  };
}

}  // namespace

TEST_CASE("buffer_allocator_actions_normalize_error_fallbacks") {
  CHECK(
    emel::buffer::allocator::action::detail::normalize_error(EMEL_ERR_INVALID_ARGUMENT, EMEL_OK) ==
    EMEL_ERR_INVALID_ARGUMENT);
  CHECK(
    emel::buffer::allocator::action::detail::normalize_error(0, EMEL_ERR_INVALID_ARGUMENT) ==
    EMEL_ERR_INVALID_ARGUMENT);
  CHECK(emel::buffer::allocator::action::detail::normalize_error(0, 0) == EMEL_ERR_BACKEND);
}

TEST_CASE("buffer_allocator_actions_error_handlers_set_backend_fallback") {
  emel::buffer::allocator::action::context ctx{};

  emel::buffer::allocator::action::on_initialize_error(
    emel::buffer::allocator::events::initialize_error{.err = 0}, ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);

  ctx.pending_error = EMEL_OK;
  emel::buffer::allocator::action::on_reserve_n_size_error(
    emel::buffer::allocator::events::reserve_n_size_error{.err = 0}, ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);

  ctx.pending_error = EMEL_OK;
  emel::buffer::allocator::action::on_release_error(
    emel::buffer::allocator::events::release_error{.err = 0}, ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);
}

TEST_CASE("buffer_allocator_actions_validate_graph_and_planner_presence") {
  const graph_storage g = make_graph();
  std::array<int32_t, 1> sizes = {{0}};

  {
    emel::buffer::allocator::action::context ctx{};
    ctx.buffer_count = 1;
    emel::buffer::allocator::action::begin_reserve_n_size(
      emel::buffer::allocator::event::reserve_n_size{
        .graph = invalid_view(),
        .node_buffer_ids = nullptr,
        .leaf_buffer_ids = nullptr,
        .sizes_out = sizes.data(),
        .sizes_out_count = static_cast<int32_t>(sizes.size()),
        .buffer_planner_sm = nullptr,
      },
      ctx);
    CHECK(ctx.pending_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    emel::buffer::allocator::action::context ctx{};
    ctx.buffer_count = 1;
    emel::buffer::allocator::action::begin_reserve_n_size(
      emel::buffer::allocator::event::reserve_n_size{
        .graph = as_view(g),
        .node_buffer_ids = nullptr,
        .leaf_buffer_ids = nullptr,
        .sizes_out = sizes.data(),
        .sizes_out_count = static_cast<int32_t>(sizes.size()),
        .buffer_planner_sm = nullptr,
      },
      ctx);
    CHECK(ctx.pending_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    emel::buffer::allocator::action::context ctx{};
    ctx.buffer_count = 1;
    emel::buffer::allocator::action::begin_reserve_n(
      emel::buffer::allocator::event::reserve_n{
        .graph = invalid_view(),
        .node_buffer_ids = nullptr,
        .leaf_buffer_ids = nullptr,
        .buffer_planner_sm = nullptr,
      },
      ctx);
    CHECK(ctx.pending_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    emel::buffer::allocator::action::context ctx{};
    ctx.buffer_count = 1;
    emel::buffer::allocator::action::begin_reserve_n(
      emel::buffer::allocator::event::reserve_n{
        .graph = as_view(g),
        .node_buffer_ids = nullptr,
        .leaf_buffer_ids = nullptr,
        .buffer_planner_sm = nullptr,
      },
      ctx);
    CHECK(ctx.pending_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    emel::buffer::allocator::action::context ctx{};
    ctx.buffer_count = 1;
    emel::buffer::allocator::action::begin_reserve(
      emel::buffer::allocator::event::reserve{
        .graph = invalid_view(),
        .buffer_planner_sm = nullptr,
      },
      ctx);
    CHECK(ctx.pending_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    emel::buffer::allocator::action::context ctx{};
    ctx.buffer_count = 1;
    emel::buffer::allocator::action::begin_reserve(
      emel::buffer::allocator::event::reserve{
        .graph = as_view(g),
        .buffer_planner_sm = nullptr,
      },
      ctx);
    CHECK(ctx.pending_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    emel::buffer::allocator::action::context ctx{};
    ctx.buffer_count = 1;
    emel::buffer::allocator::action::begin_alloc_graph(
      emel::buffer::allocator::event::alloc_graph{
        .graph = invalid_view(),
        .buffer_planner_sm = nullptr,
      },
      ctx);
    CHECK(ctx.pending_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    emel::buffer::allocator::action::context ctx{};
    ctx.buffer_count = 1;
    emel::buffer::allocator::action::begin_alloc_graph(
      emel::buffer::allocator::event::alloc_graph{
        .graph = as_view(g),
        .buffer_planner_sm = nullptr,
      },
      ctx);
    CHECK(ctx.pending_error == EMEL_ERR_INVALID_ARGUMENT);
  }
}

TEST_CASE("buffer_allocator_wrapper_rejects_reserve_calls_when_uninitialized") {
  emel::buffer::allocator::sm machine{};
  const graph_storage g = make_graph();

  CHECK_FALSE(machine.process_event(emel::buffer::allocator::event::reserve_n{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
  }));
  CHECK_FALSE(machine.process_event(emel::buffer::allocator::event::reserve{
    .graph = as_view(g),
  }));
  CHECK_FALSE(machine.process_event(emel::buffer::allocator::event::alloc_graph{
    .graph = as_view(g),
  }));
}

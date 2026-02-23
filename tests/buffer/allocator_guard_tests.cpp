#include <array>

#include <doctest/doctest.h>

#include "emel/buffer/allocator/actions.hpp"
#include "emel/buffer/allocator/events.hpp"
#include "emel/buffer/allocator/guards.hpp"

namespace {

using tensor_desc = emel::buffer::allocator::event::tensor_desc;
using graph_view = emel::buffer::allocator::event::graph_view;

struct graph_storage {
  std::array<tensor_desc, 1> nodes = {};
  std::array<tensor_desc, 1> leafs = {};
};

graph_storage make_graph() {
  graph_storage g{};
  g.leafs[0] = tensor_desc{
    .tensor_id = 1,
    .alloc_size = 64,
    .src_ids = emel::buffer::allocator::event::make_src_ids(),
    .is_input = true,
  };
  g.nodes[0] = tensor_desc{
    .tensor_id = 2,
    .alloc_size = 128,
    .src_ids = [] {
      auto ids = emel::buffer::allocator::event::make_src_ids();
      ids[0] = 1;
      return ids;
    }(),
    .is_output = true,
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

graph_view invalid_view() {
  return graph_view{
    .nodes = nullptr,
    .n_nodes = 1,
    .leafs = nullptr,
    .n_leafs = 0,
  };
}

void prepare_snapshot_context(
    emel::buffer::allocator::action::context & ctx,
    const graph_view graph,
    const int32_t * node_ids,
    const int32_t * leaf_ids) {
  ctx.buffer_count = 1;
  ctx.has_required_sizes = true;
  emel::buffer::allocator::action::detail::capture_buffer_map(ctx, graph, node_ids, leaf_ids);
  CHECK(
      emel::buffer::allocator::action::detail::capture_alloc_snapshot(ctx, graph, node_ids, leaf_ids));
  ctx.committed_sizes[0] = ctx.last_required_sizes[0];
}

}  // namespace

TEST_CASE("buffer_allocator_cached_guard_branches_cover_failure_paths") {
  const graph_storage g = make_graph();
  const graph_view graph = as_view(g);
  const std::array<int32_t, 1> ids = {{0}};
  std::array<int32_t, 1> sizes = {{0}};

  {
    emel::buffer::allocator::action::context ctx{};
    ctx.buffer_count = 1;
    ctx.has_required_sizes = true;
    emel::buffer::allocator::action::detail::capture_buffer_map(ctx, graph, nullptr, nullptr);
    CHECK_FALSE(emel::buffer::allocator::guard::can_reserve_n_size_cached{}(
        emel::buffer::allocator::event::reserve_n_size{
          .graph = graph,
          .sizes_out = sizes.data(),
          .sizes_out_count = static_cast<int32_t>(sizes.size()),
        },
        ctx));
  }

  {
    emel::buffer::allocator::action::context ctx{};
    CHECK_FALSE(emel::buffer::allocator::guard::can_reserve_n_cached{}(
        emel::buffer::allocator::event::reserve_n{
          .graph = invalid_view(),
        },
        ctx));
  }

  {
    emel::buffer::allocator::action::context ctx{};
    ctx.buffer_count = 1;
    ctx.has_required_sizes = true;
    CHECK_FALSE(emel::buffer::allocator::guard::can_reserve_n_cached{}(
        emel::buffer::allocator::event::reserve_n{
          .graph = graph,
          .node_buffer_ids = ids.data(),
          .leaf_buffer_ids = ids.data(),
        },
        ctx));
  }

  {
    emel::buffer::allocator::action::context ctx{};
    ctx.buffer_count = 1;
    ctx.has_required_sizes = true;
    emel::buffer::allocator::action::detail::capture_buffer_map(ctx, graph, ids.data(), ids.data());
    CHECK_FALSE(emel::buffer::allocator::guard::can_reserve_n_cached{}(
        emel::buffer::allocator::event::reserve_n{
          .graph = graph,
          .node_buffer_ids = ids.data(),
          .leaf_buffer_ids = ids.data(),
        },
        ctx));
  }

  {
    emel::buffer::allocator::action::context ctx{};
    prepare_snapshot_context(ctx, graph, ids.data(), ids.data());
    ctx.last_required_sizes[0] = 64;
    ctx.committed_sizes[0] = 64;
    ctx.committed_chunk_counts[0] = 0;
    CHECK_FALSE(emel::buffer::allocator::guard::can_reserve_n_cached{}(
        emel::buffer::allocator::event::reserve_n{
          .graph = graph,
          .node_buffer_ids = ids.data(),
          .leaf_buffer_ids = ids.data(),
        },
        ctx));
  }

  {
    emel::buffer::allocator::action::context ctx{};
    CHECK_FALSE(emel::buffer::allocator::guard::can_reserve_cached{}(
        emel::buffer::allocator::event::reserve{
          .graph = invalid_view(),
        },
        ctx));
  }

  {
    emel::buffer::allocator::action::context ctx{};
    ctx.buffer_count = 1;
    ctx.has_required_sizes = true;
    CHECK_FALSE(emel::buffer::allocator::guard::can_reserve_cached{}(
        emel::buffer::allocator::event::reserve{
          .graph = graph,
        },
        ctx));
  }

  {
    emel::buffer::allocator::action::context ctx{};
    ctx.buffer_count = 1;
    ctx.has_required_sizes = true;
    emel::buffer::allocator::action::detail::capture_buffer_map(ctx, graph, nullptr, nullptr);
    CHECK_FALSE(emel::buffer::allocator::guard::can_reserve_cached{}(
        emel::buffer::allocator::event::reserve{
          .graph = graph,
        },
        ctx));
  }

  {
    emel::buffer::allocator::action::context ctx{};
    prepare_snapshot_context(ctx, graph, nullptr, nullptr);
    ctx.last_required_sizes[0] = 64;
    ctx.committed_sizes[0] = 64;
    ctx.committed_chunk_counts[0] = 0;
    CHECK_FALSE(emel::buffer::allocator::guard::can_reserve_cached{}(
        emel::buffer::allocator::event::reserve{
          .graph = graph,
        },
        ctx));
  }

  {
    emel::buffer::allocator::action::context ctx{};
    CHECK_FALSE(emel::buffer::allocator::guard::can_alloc_graph_cached{}(
        emel::buffer::allocator::event::alloc_graph{
          .graph = invalid_view(),
        },
        ctx));
  }

  {
    emel::buffer::allocator::action::context ctx{};
    prepare_snapshot_context(ctx, graph, nullptr, nullptr);
    ctx.last_required_sizes[0] = 64;
    ctx.committed_sizes[0] = 64;
    ctx.committed_chunk_counts[0] = 0;
    CHECK_FALSE(emel::buffer::allocator::guard::can_alloc_graph_cached{}(
        emel::buffer::allocator::event::alloc_graph{
          .graph = graph,
        },
        ctx));
  }
}

#include <array>
#include <cstdint>
#include <limits>
#include <doctest/doctest.h>

#include "emel/buffer/allocator/actions.hpp"
#include "emel/buffer/allocator/guards.hpp"
#include "emel/buffer/chunk_allocator/sm.hpp"
#include "emel/buffer/allocator/sm.hpp"
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
    .alloc_size = 64,
    .src_ids = emel::buffer::allocator::event::make_src_ids(),
    .is_view = false,
    .view_src_id = -1,
    .is_input = true,
    .is_output = false,
    .has_external_data = false,
  };
  g.nodes[0] = tensor_desc{
    .tensor_id = 2,
    .alloc_size = 128,
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
  int32_t error_out = EMEL_OK;

  emel::buffer::allocator::action::on_initialize_error(
    emel::buffer::allocator::events::initialize_error{.err = 0, .error_out = &error_out}, ctx);
  CHECK(error_out == EMEL_ERR_BACKEND);

  emel::buffer::allocator::action::on_reserve_n_size_error(
    emel::buffer::allocator::events::reserve_n_size_error{.err = 0, .error_out = &error_out}, ctx);
  CHECK(error_out == EMEL_ERR_BACKEND);

  emel::buffer::allocator::action::on_release_error(
    emel::buffer::allocator::events::release_error{.err = 0, .error_out = &error_out}, ctx);
  CHECK(error_out == EMEL_ERR_BACKEND);
}

TEST_CASE("buffer_allocator_actions_validate_graph_and_planner_presence") {
  const graph_storage g = make_graph();
  std::array<int32_t, 1> sizes = {{0}};

  emel::buffer::allocator::action::context ctx{};
  ctx.buffer_count = 1;

  CHECK_FALSE(emel::buffer::allocator::guard::can_reserve_n_size{}(
    emel::buffer::allocator::event::reserve_n_size{
      .graph = invalid_view(),
      .node_buffer_ids = nullptr,
      .leaf_buffer_ids = nullptr,
      .sizes_out = sizes.data(),
      .sizes_out_count = static_cast<int32_t>(sizes.size()),
    },
    ctx));
  CHECK(emel::buffer::allocator::guard::can_reserve_n_size{}(
    emel::buffer::allocator::event::reserve_n_size{
      .graph = as_view(g),
      .node_buffer_ids = nullptr,
      .leaf_buffer_ids = nullptr,
      .sizes_out = sizes.data(),
      .sizes_out_count = static_cast<int32_t>(sizes.size()),
    },
    ctx));

  CHECK_FALSE(emel::buffer::allocator::guard::can_reserve_n{}(
    emel::buffer::allocator::event::reserve_n{
      .graph = invalid_view(),
      .node_buffer_ids = nullptr,
      .leaf_buffer_ids = nullptr,
    },
    ctx));
  CHECK(emel::buffer::allocator::guard::can_reserve_n{}(
    emel::buffer::allocator::event::reserve_n{
      .graph = as_view(g),
      .node_buffer_ids = nullptr,
      .leaf_buffer_ids = nullptr,
    },
    ctx));

  CHECK_FALSE(emel::buffer::allocator::guard::can_reserve{}(
    emel::buffer::allocator::event::reserve{
      .graph = invalid_view(),
    },
    ctx));
  CHECK(emel::buffer::allocator::guard::can_reserve{}(
    emel::buffer::allocator::event::reserve{
      .graph = as_view(g),
    },
    ctx));

  CHECK_FALSE(emel::buffer::allocator::guard::can_alloc_graph{}(
    emel::buffer::allocator::event::alloc_graph{
      .graph = invalid_view(),
    },
    ctx));
  CHECK(emel::buffer::allocator::guard::can_alloc_graph{}(
    emel::buffer::allocator::event::alloc_graph{
      .graph = as_view(g),
    },
    ctx));
}

TEST_CASE("buffer_allocator_actions_detail_alignment_and_tensor_alloc") {
  using emel::buffer::allocator::action::detail::align_up;
  using emel::buffer::allocator::action::detail::build_tensor_alloc;

  emel::buffer::allocator::action::context ctx{};
  emel::buffer::allocator::action::tensor_alloc alloc{};
  tensor_desc tensor{
    .tensor_id = 9,
    .alloc_size = 33,
    .src_ids = emel::buffer::allocator::event::make_src_ids(),
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = false,
    .has_external_data = true,
  };

  CHECK(build_tensor_alloc(alloc, ctx, tensor, 0, 1));
  CHECK(alloc.buffer_id == -1);
  CHECK(alloc.size_max == 0);

  tensor.has_external_data = false;
  tensor.is_view = true;
  CHECK(build_tensor_alloc(alloc, ctx, tensor, 0, 1));
  CHECK(alloc.buffer_id == -1);

  tensor.is_view = false;
  CHECK_FALSE(build_tensor_alloc(alloc, ctx, tensor, -1, 1));
  CHECK_FALSE(build_tensor_alloc(alloc, ctx, tensor, 2, 1));
  CHECK(build_tensor_alloc(alloc, ctx, tensor, 0, 1));
  CHECK(alloc.buffer_id == 0);
  CHECK(alloc.size_max == align_up(33, 16));

  CHECK(align_up(-4, 16) == 0);
  CHECK(align_up(std::numeric_limits<int32_t>::max(), 16) ==
        std::numeric_limits<int32_t>::max());
}

TEST_CASE("buffer_allocator_actions_capture_snapshot_and_realloc_checks") {
  graph_storage g = make_graph();
  const graph_view view = as_view(g);
  std::array<int32_t, 1> node_ids = {{0}};
  std::array<int32_t, 1> leaf_ids = {{0}};

  {
    emel::buffer::allocator::action::context c{};
    c.buffer_count = 1;
    CHECK_FALSE(emel::buffer::allocator::action::detail::capture_alloc_snapshot(
      c, invalid_view(), node_ids.data(), leaf_ids.data()));
  }

  {
    emel::buffer::allocator::action::context c{};
    c.buffer_count = 1;
    emel::buffer::allocator::action::detail::capture_buffer_map(
      c, view, node_ids.data(), leaf_ids.data());
    CHECK(emel::buffer::allocator::action::detail::capture_alloc_snapshot(
      c, view, node_ids.data(), leaf_ids.data()));
    CHECK(c.has_reserve_snapshot);
    CHECK_FALSE(emel::buffer::allocator::action::detail::graph_needs_realloc(view, c));

    graph_storage bigger = g;
    bigger.nodes[0].alloc_size += 64;
    CHECK(emel::buffer::allocator::action::detail::graph_needs_realloc(as_view(bigger), c));
  }

  {
    emel::buffer::allocator::action::context c{};
    c.buffer_count = 1;
    std::array<int32_t, 1> bad_ids = {{2}};
    CHECK_FALSE(emel::buffer::allocator::action::detail::capture_alloc_snapshot(
      c, view, bad_ids.data(), leaf_ids.data()));
  }

  {
    emel::buffer::allocator::action::context c{};
    c.buffer_count = 1;
    graph_storage invalid = make_graph();
    invalid.nodes[0].src_ids[0] = 999;
    CHECK_FALSE(emel::buffer::allocator::action::detail::capture_alloc_snapshot(
      c, as_view(invalid), node_ids.data(), leaf_ids.data()));
  }
}

TEST_CASE("buffer_allocator_actions_apply_required_sizes_and_release_bindings") {
  emel::buffer::allocator::action::context c{};
  c.buffer_count = 1;
  c.committed_chunk_counts.fill(0);
  c.committed_chunk_ids.fill(-1);
  c.committed_chunk_offsets.fill(0);
  c.committed_chunk_sizes.fill(0);
  c.committed_sizes.fill(0);

  emel::buffer::chunk_allocator::sm chunk_allocator{};
  int32_t err = EMEL_OK;
  CHECK(chunk_allocator.process_event(emel::buffer::chunk_allocator::event::configure{
    .alignment = 16,
    .max_chunk_size = 256,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  std::array<int32_t, emel::buffer::allocator::action::k_max_buffers> required = {};
  required[0] = 0;
  CHECK(emel::buffer::allocator::action::detail::apply_required_sizes_to_chunks(
    c, required, nullptr, nullptr, &chunk_allocator, err));
  CHECK(err == EMEL_OK);
  CHECK(c.committed_chunk_counts[0] == 0);

  required[0] = 64;
  CHECK(emel::buffer::allocator::action::detail::apply_required_sizes_to_chunks(
    c, required, nullptr, nullptr, &chunk_allocator, err));
  CHECK(err == EMEL_OK);
  CHECK(c.committed_chunk_counts[0] == 1);
  CHECK(c.committed_chunk_ids[0] >= 0);
  const uint64_t first_size = c.committed_chunk_sizes[0];

  required[0] = 32;
  CHECK(emel::buffer::allocator::action::detail::apply_required_sizes_to_chunks(
    c, required, nullptr, nullptr, &chunk_allocator, err));
  CHECK(err == EMEL_OK);
  CHECK(c.committed_chunk_sizes[0] == first_size);

  CHECK(emel::buffer::allocator::action::detail::release_all_chunk_bindings(
    c, &chunk_allocator, err));
  CHECK(err == EMEL_OK);
  CHECK(c.committed_chunk_counts[0] == 0);
  CHECK(c.committed_chunk_ids[0] == -1);
  CHECK(c.committed_chunk_sizes[0] == 0);
}

TEST_CASE("buffer_allocator_actions_apply_required_sizes_uses_chunk_plan") {
  emel::buffer::allocator::action::context c{};
  c.buffer_count = 1;
  c.buffer_alignments[0] = 16;
  c.buffer_max_sizes[0] = 32;
  c.committed_chunk_counts.fill(0);
  c.committed_chunk_ids.fill(-1);
  c.committed_chunk_offsets.fill(0);
  c.committed_chunk_sizes.fill(0);
  c.committed_sizes.fill(0);

  emel::buffer::chunk_allocator::sm chunk_allocator{};
  int32_t err = EMEL_OK;
  CHECK(chunk_allocator.process_event(emel::buffer::chunk_allocator::event::configure{
    .alignment = 16,
    .max_chunk_size = 32,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  std::array<int32_t, emel::buffer::allocator::action::k_max_buffers> required = {};
  required[0] = 96;
  std::array<int32_t, emel::buffer::allocator::action::k_max_buffers> chunk_counts = {};
  std::array<int32_t, emel::buffer::allocator::action::k_max_chunk_bindings> chunk_sizes = {};
  chunk_counts[0] = 3;
  chunk_sizes[0] = 32;
  chunk_sizes[1] = 32;
  chunk_sizes[2] = 32;

  CHECK(emel::buffer::allocator::action::detail::apply_required_sizes_to_chunks(
    c, required, chunk_counts.data(), chunk_sizes.data(), &chunk_allocator, err));
  CHECK(err == EMEL_OK);
  CHECK(c.committed_chunk_counts[0] == 3);
  CHECK(c.committed_chunk_sizes[0] >= 32);
  CHECK(c.committed_chunk_sizes[1] >= 32);
  CHECK(c.committed_chunk_sizes[2] >= 32);
}

TEST_CASE("buffer_allocator_actions_begin_initialize_and_release_errors") {
  emel::buffer::allocator::action::context c{};
  emel::buffer::chunk_allocator::sm chunk_allocator{};
  int32_t err = EMEL_OK;

  CHECK_FALSE(emel::buffer::allocator::guard::valid_initialize{}(
    emel::buffer::allocator::event::initialize{
      .buffer_count = 0,
      .chunk_allocator_sm = nullptr,
    }));

  emel::buffer::allocator::action::begin_release(
    emel::buffer::allocator::event::release{
      .chunk_allocator_sm = nullptr,
      .error_out = &err,
    },
    c,
    chunk_allocator);
  CHECK(err == EMEL_OK);
}

TEST_CASE("buffer_allocator_actions_begin_alloc_graph_rejects_invalid_graph") {
  emel::buffer::allocator::action::context c{};
  c.buffer_count = 1;
  CHECK_FALSE(emel::buffer::allocator::guard::can_alloc_graph{}(
    emel::buffer::allocator::event::alloc_graph{
      .graph = invalid_view(),
    },
    c));
}

TEST_CASE("buffer_allocator_wrapper_rejects_reserve_calls_when_uninitialized") {
  emel::buffer::allocator::sm machine{};
  const graph_storage g = make_graph();

  CHECK(machine.process_event(emel::buffer::allocator::event::reserve_n{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
  }));
  CHECK(machine.process_event(emel::buffer::allocator::event::reserve{
    .graph = as_view(g),
  }));
  CHECK(machine.process_event(emel::buffer::allocator::event::alloc_graph{
    .graph = as_view(g),
  }));
}

TEST_CASE("buffer_allocator_actions_chunk_allocator_helper_paths") {
  emel::buffer::allocator::action::context ctx{};
  int32_t err = EMEL_OK;

  std::array<int32_t, emel::buffer::allocator::action::k_max_buffers> required = {};
  required[0] = 64;
  emel::buffer::chunk_allocator::sm chunk_machine{};
  CHECK(emel::buffer::allocator::action::detail::configure_chunk_allocator(&chunk_machine, err));
  CHECK(err == EMEL_OK);
  ctx.buffer_count = 1;
  CHECK(
      emel::buffer::allocator::action::detail::apply_required_sizes_to_chunks(
          ctx, required, nullptr, nullptr, &chunk_machine, err));
  CHECK(err == EMEL_OK);
  CHECK(ctx.committed_sizes[0] == 64);
  CHECK(ctx.committed_chunk_counts[0] == 1);
  CHECK(ctx.committed_chunk_ids[0] >= 0);
  CHECK(ctx.committed_chunk_sizes[0] >= 64);

  ctx.committed_chunk_counts[0] = 1;
  ctx.committed_chunk_ids[0] = 999;
  ctx.committed_chunk_offsets[0] = 0;
  ctx.committed_chunk_sizes[0] = 64;
  CHECK_FALSE(
      emel::buffer::allocator::action::detail::release_all_chunk_bindings(
          ctx, &chunk_machine, err));
  CHECK(err != EMEL_OK);
}

TEST_CASE("buffer_allocator_actions_require_chunk_allocator_in_initialize_reserve_and_release") {
  const graph_storage g = make_graph();
  emel::buffer::allocator::action::context ctx{};
  emel::buffer::planner::sm planner{};
  emel::buffer::chunk_allocator::sm chunk_machine{};
  emel::buffer::realloc_analyzer::sm analyzer{};
  int32_t error_out = EMEL_OK;

  emel::buffer::allocator::action::begin_initialize(
      emel::buffer::allocator::event::initialize{
        .buffer_count = 1,
        .chunk_allocator_sm = nullptr,
        .error_out = &error_out,
      },
      ctx,
      chunk_machine);
  CHECK(error_out == EMEL_OK);

  ctx = {};
  ctx.buffer_count = 1;
  error_out = EMEL_OK;
  CHECK_FALSE(emel::buffer::allocator::guard::can_reserve{}(
    emel::buffer::allocator::event::reserve{
      .graph = invalid_view(),
    },
    ctx));

  ctx = {};
  ctx.buffer_count = 1;
  error_out = EMEL_OK;
  emel::buffer::allocator::action::begin_release(
      emel::buffer::allocator::event::release{
        .chunk_allocator_sm = nullptr,
        .error_out = &error_out,
      },
      ctx,
      chunk_machine);
  CHECK(error_out == EMEL_OK);

  ctx = {};
  ctx.buffer_count = 1;
  error_out = EMEL_OK;
  emel::buffer::allocator::action::begin_reserve(
      emel::buffer::allocator::event::reserve{
        .graph = as_view(g),
        .buffer_planner_sm = &planner,
        .chunk_allocator_sm = &chunk_machine,
        .error_out = &error_out,
      },
      ctx,
      planner,
      chunk_machine);
  CHECK(error_out != EMEL_ERR_INVALID_ARGUMENT);

  ctx = {};
  ctx.buffer_count = 1;
  error_out = EMEL_OK;
  CHECK_FALSE(emel::buffer::allocator::guard::can_alloc_graph{}(
    emel::buffer::allocator::event::alloc_graph{
      .graph = invalid_view(),
    },
    ctx));

  ctx = {};
  ctx.buffer_count = 1;
  error_out = EMEL_OK;
  emel::buffer::allocator::action::begin_alloc_graph(
      emel::buffer::allocator::event::alloc_graph{
        .graph = as_view(g),
        .buffer_planner_sm = &planner,
        .chunk_allocator_sm = &chunk_machine,
        .buffer_realloc_analyzer_sm = &analyzer,
        .error_out = &error_out,
      },
      ctx,
      planner,
      chunk_machine,
      analyzer);
  CHECK(error_out != EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_allocator_actions_helper_edges_for_branch_coverage") {
  namespace detail = emel::buffer::allocator::action::detail;

  CHECK(detail::align_up(0, 16) == 0);
  CHECK(
      detail::align_up(std::numeric_limits<int32_t>::max(), 16) ==
      std::numeric_limits<int32_t>::max());

  const graph_storage g = make_graph();
  bool is_node = true;
  int32_t index = -1;
  CHECK(detail::find_tensor(as_view(g), g.leafs[0].tensor_id, is_node, index) != nullptr);
  CHECK_FALSE(is_node);
  CHECK(index == 0);
  CHECK(detail::find_tensor(as_view(g), 9999, is_node, index) == nullptr);

  emel::buffer::allocator::action::context c{};
  emel::buffer::allocator::action::tensor_alloc alloc{};
  tensor_desc external_tensor{
    .tensor_id = 7,
    .alloc_size = 4096,
    .src_ids = emel::buffer::allocator::event::make_src_ids(),
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = false,
    .has_external_data = true,
  };
  CHECK(detail::build_tensor_alloc(alloc, c, external_tensor, 0, 1));
  CHECK(alloc.buffer_id == -1);
  CHECK(alloc.size_max == 0);

  tensor_desc alloc_tensor = external_tensor;
  alloc_tensor.has_external_data = false;
  alloc_tensor.alloc_size = 64;
  CHECK_FALSE(detail::build_tensor_alloc(alloc, c, alloc_tensor, -1, 1));
  CHECK(detail::tensor_needs_realloc(c, alloc_tensor, emel::buffer::allocator::action::tensor_alloc{
                                                     .tensor_id = alloc_tensor.tensor_id,
                                                     .buffer_id = -1,
                                                     .size_max = 0,
                                                 }));
  CHECK_FALSE(detail::tensor_needs_realloc(c, external_tensor, alloc));

  c.buffer_count = 1;
  CHECK_FALSE(detail::capture_alloc_snapshot(c, invalid_view(), nullptr, nullptr));

  const auto view = as_view(g);
  CHECK(detail::capture_alloc_snapshot(c, view, nullptr, nullptr));
  c.has_reserve_snapshot = true;
  c.last_n_nodes = view.n_nodes + 1;
  c.last_n_leafs = view.n_leafs;
  CHECK(detail::graph_needs_realloc(view, c));
  c.last_n_nodes = view.n_nodes;
  CHECK_FALSE(detail::graph_needs_realloc(view, c));

  std::array<int32_t, emel::buffer::allocator::action::k_max_buffers> required = {};
  required[0] = 0;
  CHECK(detail::chunk_bindings_cover_required(c, required, 1));
  required[0] = 16;
  c.committed_chunk_counts[0] = 0;
  c.committed_chunk_ids[0] = -1;
  CHECK_FALSE(detail::chunk_bindings_cover_required(c, required, 1));

  emel::buffer::chunk_allocator::sm chunk_machine{};
  int32_t err = EMEL_OK;
  required[0] = 0;
  CHECK(detail::apply_required_sizes_to_chunks(c, required, nullptr, nullptr, &chunk_machine, err));

  c = {};
  c.buffer_count = 1;
  required[0] = 16;
  CHECK(detail::apply_required_sizes_to_chunks(c, required, nullptr, nullptr, &chunk_machine, err));
  CHECK(detail::apply_required_sizes_to_chunks(c, required, nullptr, nullptr, &chunk_machine, err));
}

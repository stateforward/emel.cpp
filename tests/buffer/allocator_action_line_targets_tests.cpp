#include <array>
#include <cstdint>

#include <doctest/doctest.h>

#include "emel/buffer/allocator/actions.hpp"
#include "emel/buffer/chunk_allocator/actions.hpp"
#include "emel/buffer/chunk_allocator/sm.hpp"
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

graph_view as_view(const graph_storage & g) {
  return graph_view{
    .nodes = g.nodes.data(),
    .n_nodes = g.n_nodes,
    .leafs = g.leafs.data(),
    .n_leafs = g.n_leafs,
  };
}

graph_storage make_graph(bool with_src) {
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
    .src_ids = {{with_src ? 1 : -1, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = true,
    .has_external_data = false,
  };
  return g;
}

}  // namespace

TEST_CASE("buffer_allocator_capture_snapshot_rejects_invalid_src_buffer") {
  emel::buffer::allocator::action::context c{};
  c.buffer_count = 1;
  const graph_storage g = make_graph(true);
  std::array<int32_t, 1> node_ids = {{0}};
  std::array<int32_t, 1> leaf_ids = {{2}};

  CHECK_FALSE(emel::buffer::allocator::action::detail::capture_alloc_snapshot(
    c, as_view(g), node_ids.data(), leaf_ids.data()));
}

TEST_CASE("buffer_allocator_capture_snapshot_rejects_invalid_leaf_buffer") {
  emel::buffer::allocator::action::context c{};
  c.buffer_count = 1;
  const graph_storage g = make_graph(false);
  std::array<int32_t, 1> node_ids = {{0}};
  std::array<int32_t, 1> leaf_ids = {{2}};

  CHECK_FALSE(emel::buffer::allocator::action::detail::capture_alloc_snapshot(
    c, as_view(g), node_ids.data(), leaf_ids.data()));
}

TEST_CASE("buffer_allocator_graph_needs_realloc_flags_src_size_mismatch") {
  emel::buffer::allocator::action::context c{};
  c.buffer_count = 1;
  c.has_reserve_snapshot = true;
  c.last_n_nodes = 1;
  c.last_n_leafs = 1;
  c.node_allocs[0].dst.tensor_id = 2;
  c.node_allocs[0].dst.buffer_id = 0;
  c.node_allocs[0].dst.size_max = 16;
  c.node_allocs[0].src[0].tensor_id = 1;
  c.node_allocs[0].src[0].buffer_id = 0;
  c.node_allocs[0].src[0].size_max = 8;
  c.leaf_allocs[0].leaf.tensor_id = 1;
  c.leaf_allocs[0].leaf.buffer_id = 0;
  c.leaf_allocs[0].leaf.size_max = 16;

  const graph_storage g = make_graph(true);
  CHECK(emel::buffer::allocator::action::detail::graph_needs_realloc(as_view(g), c));
}

TEST_CASE("buffer_allocator_run_realloc_analyzer_rejects_invalid_graph") {
  emel::buffer::allocator::action::context c{};
  emel::buffer::realloc_analyzer::sm analyzer{};
  bool needs_realloc = false;
  int32_t err = EMEL_OK;

  graph_view invalid{
    .nodes = nullptr,
    .n_nodes = 1,
    .leafs = nullptr,
    .n_leafs = 0,
  };

  CHECK_FALSE(emel::buffer::allocator::action::detail::run_realloc_analyzer(
    &analyzer,
    invalid,
    c,
    needs_realloc,
    err));
  CHECK(err != EMEL_OK);
}

TEST_CASE("buffer_allocator_apply_required_sizes_propagates_allocate_failure") {
  emel::buffer::allocator::action::context c{};
  c.buffer_count = 1;
  c.buffer_alignments[0] = 16;
  c.buffer_max_sizes[0] = 32;
  std::array<int32_t, emel::buffer::allocator::action::k_max_buffers> required = {};
  required[0] = 128;

  emel::buffer::chunk_allocator::sm chunk_allocator{};
  int32_t err = EMEL_OK;
  CHECK(chunk_allocator.process_event(emel::buffer::chunk_allocator::event::configure{
    .alignment = 16,
    .max_chunk_size = 32,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  std::array<int32_t, emel::buffer::allocator::action::k_max_buffers> chunk_counts = {};
  std::array<int32_t, emel::buffer::allocator::action::k_max_chunk_bindings> chunk_sizes = {};
  chunk_counts[0] = 4;
  chunk_sizes[0] = 32;
  chunk_sizes[1] = 32;
  chunk_sizes[2] = 32;
  chunk_sizes[3] = 32;

  CHECK(emel::buffer::allocator::action::detail::apply_required_sizes_to_chunks(
    c, required, chunk_counts.data(), chunk_sizes.data(), &chunk_allocator, err));
  CHECK(err == EMEL_OK);
}

TEST_CASE("buffer_allocator_apply_required_sizes_propagates_release_failure") {
  emel::buffer::allocator::action::context c{};
  c.buffer_count = 1;
  c.committed_chunk_counts[0] = 1;
  c.committed_chunk_ids[0] = 99;
  c.committed_chunk_offsets[0] = 0;
  c.committed_chunk_sizes[0] = 16;
  c.committed_sizes[0] = 16;

  emel::buffer::chunk_allocator::sm chunk_allocator{};
  int32_t err = EMEL_OK;
  CHECK(chunk_allocator.process_event(emel::buffer::chunk_allocator::event::configure{
    .alignment = 16,
    .max_chunk_size = 64,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  std::array<int32_t, emel::buffer::allocator::action::k_max_buffers> required = {};
  required[0] = 32;
  CHECK_FALSE(emel::buffer::allocator::action::detail::apply_required_sizes_to_chunks(
    c, required, nullptr, nullptr, &chunk_allocator, err));
  CHECK(err != EMEL_OK);
}

TEST_CASE("buffer_allocator_actions_on_reserve_and_alloc_done_set_error_out") {
  emel::buffer::allocator::action::context c{};
  int32_t err = EMEL_OK;

  emel::buffer::allocator::action::on_reserve_n_size_done(
    emel::buffer::allocator::events::reserve_n_size_done{.error_out = &err},
    c);
  CHECK(err == EMEL_OK);

  emel::buffer::allocator::action::on_reserve_done(
    emel::buffer::allocator::events::reserve_done{.error_out = &err},
    c);
  CHECK(err == EMEL_OK);

  emel::buffer::allocator::action::on_alloc_graph_done(
    emel::buffer::allocator::events::alloc_graph_done{.error_out = &err},
    c);
  CHECK(err == EMEL_OK);
}

TEST_CASE("buffer_allocator_begin_alloc_graph_reports_planner_failure") {
  emel::buffer::allocator::action::context c{};
  c.buffer_count = 1;
  c.last_n_nodes = 0;
  c.last_n_leafs = 0;
  int32_t err = EMEL_OK;
  emel::buffer::realloc_analyzer::sm analyzer{};
  emel::buffer::planner::sm planner{};
  emel::buffer::chunk_allocator::sm chunk_allocator{};
  const graph_storage g = make_graph(true);

  emel::buffer::allocator::action::begin_alloc_graph(
    emel::buffer::allocator::event::alloc_graph{
      .graph = as_view(g),
      .buffer_planner_sm = nullptr,
      .chunk_allocator_sm = nullptr,
      .buffer_realloc_analyzer_sm = &analyzer,
      .strategy = nullptr,
      .error_out = &err,
    },
    c,
    planner,
    chunk_allocator,
    analyzer);
  CHECK(err == EMEL_OK);
}

TEST_CASE("buffer_allocator_begin_alloc_graph_reports_multibuffer_growth") {
  emel::buffer::allocator::action::context c{};
  c.buffer_count = 2;
  c.last_n_nodes = 1;
  c.last_n_leafs = 1;
  c.node_allocs[0].dst.tensor_id = 2;
  c.node_allocs[0].src[0].tensor_id = 1;
  c.leaf_allocs[0].leaf.tensor_id = 1;
  int32_t err = EMEL_OK;

  emel::buffer::planner::sm planner{};
  emel::buffer::realloc_analyzer::sm analyzer{};
  emel::buffer::chunk_allocator::sm chunk_allocator{};

  emel::buffer::allocator::action::begin_alloc_graph(
    emel::buffer::allocator::event::alloc_graph{
      .graph = as_view(make_graph(true)),
      .buffer_planner_sm = &planner,
      .chunk_allocator_sm = &chunk_allocator,
      .buffer_realloc_analyzer_sm = &analyzer,
      .strategy = &emel::buffer::planner::default_strategies::alloc_graph,
      .error_out = &err,
    },
    c,
    planner,
    chunk_allocator,
    analyzer);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("buffer_allocator_begin_alloc_graph_reports_apply_required_failure") {
  emel::buffer::allocator::action::context c{};
  c.buffer_count = 1;
  c.last_n_nodes = 1;
  c.last_n_leafs = 1;
  c.node_allocs[0].dst.tensor_id = 2;
  c.node_allocs[0].src[0].tensor_id = 1;
  c.leaf_allocs[0].leaf.tensor_id = 1;
  int32_t err = EMEL_OK;

  emel::buffer::planner::sm planner{};
  emel::buffer::realloc_analyzer::sm analyzer{};
  emel::buffer::chunk_allocator::sm chunk_allocator{};

  emel::buffer::allocator::action::begin_alloc_graph(
    emel::buffer::allocator::event::alloc_graph{
      .graph = as_view(make_graph(true)),
      .buffer_planner_sm = &planner,
      .chunk_allocator_sm = nullptr,
      .buffer_realloc_analyzer_sm = &analyzer,
      .strategy = &emel::buffer::planner::default_strategies::alloc_graph,
      .error_out = &err,
    },
    c,
    planner,
    chunk_allocator,
    analyzer);
  CHECK(err == EMEL_OK);
}

TEST_CASE("buffer_allocator_capture_snapshot_rejects_oversized_graph") {
  emel::buffer::allocator::action::context c{};
  c.buffer_count = 1;
  std::array<tensor_desc, emel::buffer::allocator::action::k_max_graph_tensors + 1> nodes = {};
  for (size_t i = 0; i < nodes.size(); ++i) {
    nodes[i] = tensor_desc{
      .tensor_id = static_cast<int32_t>(i),
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_input = false,
      .is_output = true,
      .has_external_data = false,
    };
  }

  graph_view graph{
    .nodes = nodes.data(),
    .n_nodes = emel::buffer::allocator::action::k_max_graph_tensors + 1,
    .leafs = nullptr,
    .n_leafs = 0,
  };

  CHECK_FALSE(emel::buffer::allocator::action::detail::capture_alloc_snapshot(
    c, graph, nullptr, nullptr));
}

TEST_CASE("buffer_allocator_apply_required_sizes_uses_default_allocator_state") {
  emel::buffer::allocator::action::context c{};
  c.buffer_count = 1;
  std::array<int32_t, emel::buffer::allocator::action::k_max_buffers> required = {};
  required[0] = 32;

  emel::buffer::chunk_allocator::sm chunk_allocator{};
  int32_t err = EMEL_OK;
  CHECK_FALSE(chunk_allocator.process_event(emel::buffer::chunk_allocator::event::configure{
    .alignment = 0,
    .max_chunk_size = 0,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  CHECK(emel::buffer::allocator::action::detail::apply_required_sizes_to_chunks(
    c, required, nullptr, nullptr, &chunk_allocator, err));
  CHECK(err == EMEL_OK);
}

TEST_CASE("buffer_allocator_release_all_chunk_bindings_handles_empty") {
  emel::buffer::allocator::action::context c{};
  c.buffer_count = 1;

  emel::buffer::chunk_allocator::sm chunk_allocator{};
  int32_t err = EMEL_OK;
  CHECK(chunk_allocator.process_event(emel::buffer::chunk_allocator::event::configure{
    .alignment = 16,
    .max_chunk_size = 64,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  CHECK(emel::buffer::allocator::action::detail::release_all_chunk_bindings(
    c, &chunk_allocator, err));
  CHECK(err == EMEL_OK);
}

TEST_CASE("buffer_allocator_begin_reserve_n_size_reports_snapshot_failure") {
  emel::buffer::allocator::action::context c{};
  c.buffer_count = 1;
  const graph_storage g = make_graph(true);
  std::array<int32_t, 1> node_ids = {{2}};
  std::array<int32_t, 1> leaf_ids = {{2}};
  std::array<int32_t, 1> sizes = {};
  emel::buffer::planner::sm planner{};
  int32_t err = EMEL_OK;

  emel::buffer::allocator::action::begin_reserve_n_size(
    emel::buffer::allocator::event::reserve_n_size{
      .graph = as_view(g),
      .node_buffer_ids = node_ids.data(),
      .leaf_buffer_ids = leaf_ids.data(),
      .sizes_out = sizes.data(),
      .sizes_out_count = static_cast<int32_t>(sizes.size()),
      .buffer_planner_sm = &planner,
      .strategy = &emel::buffer::planner::default_strategies::reserve_n_size,
      .error_out = &err,
    },
    c,
    planner);

  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_allocator_begin_reserve_n_reports_snapshot_failure") {
  emel::buffer::allocator::action::context c{};
  c.buffer_count = 1;
  const graph_storage g = make_graph(true);
  std::array<int32_t, 1> node_ids = {{2}};
  std::array<int32_t, 1> leaf_ids = {{2}};
  emel::buffer::planner::sm planner{};
  emel::buffer::chunk_allocator::sm chunk_allocator{};
  int32_t err = EMEL_OK;

  emel::buffer::allocator::action::begin_reserve_n(
    emel::buffer::allocator::event::reserve_n{
      .graph = as_view(g),
      .node_buffer_ids = node_ids.data(),
      .leaf_buffer_ids = leaf_ids.data(),
      .buffer_planner_sm = &planner,
      .chunk_allocator_sm = &chunk_allocator,
      .strategy = &emel::buffer::planner::default_strategies::reserve_n,
      .error_out = &err,
    },
    c,
    planner,
    chunk_allocator);

  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_allocator_begin_release_and_on_release_done_set_error_out") {
  emel::buffer::allocator::action::context c{};
  int32_t err = EMEL_OK;

  emel::buffer::chunk_allocator::sm chunk_allocator{};
  CHECK(chunk_allocator.process_event(emel::buffer::chunk_allocator::event::configure{
    .alignment = 16,
    .max_chunk_size = 64,
    .error_out = &err,
  }));

  emel::buffer::allocator::action::begin_release(
    emel::buffer::allocator::event::release{
      .chunk_allocator_sm = &chunk_allocator,
      .error_out = &err,
    },
    c,
    chunk_allocator);
  CHECK(err == EMEL_OK);

  emel::buffer::allocator::action::on_release_done(
    emel::buffer::allocator::events::release_done{.error_out = &err},
    c);
  CHECK(err == EMEL_OK);
}

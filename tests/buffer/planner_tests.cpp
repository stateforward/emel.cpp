#include <array>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/buffer/allocator/events.hpp"
#include "emel/buffer/planner/actions.hpp"
#include "emel/buffer/planner/sm.hpp"
#include "emel/emel.h"

namespace {

using tensor_desc = emel::buffer::allocator::event::tensor_desc;
using graph_view = emel::buffer::allocator::event::graph_view;

struct graph_storage {
  std::array<tensor_desc, 6> nodes = {};
  std::array<tensor_desc, 3> leafs = {};
  int32_t n_nodes = 0;
  int32_t n_leafs = 0;
};

graph_storage make_valid_graph() {
  graph_storage g{};
  g.n_leafs = 1;
  g.n_nodes = 2;
  g.leafs[0] = tensor_desc{
    .tensor_id = 100,
    .alloc_size = 128,
    .src_ids = {{-1, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = true,
    .is_output = false,
    .has_external_data = false,
  };
  g.nodes[0] = tensor_desc{
    .tensor_id = 200,
    .alloc_size = 256,
    .src_ids = {{100, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = false,
    .has_external_data = false,
  };
  g.nodes[1] = tensor_desc{
    .tensor_id = 201,
    .alloc_size = 512,
    .src_ids = {{200, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = true,
    .has_external_data = false,
  };
  return g;
}

graph_storage make_invalid_source_graph() {
  graph_storage g = make_valid_graph();
  g.nodes[0].src_ids[0] = 9999;
  return g;
}

graph_storage make_inplace_reuse_graph() {
  graph_storage g{};
  g.n_leafs = 1;
  g.n_nodes = 2;
  g.leafs[0] = tensor_desc{
    .tensor_id = 300,
    .alloc_size = 512,
    .src_ids = {{-1, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = true,
    .is_output = false,
    .has_external_data = false,
  };
  g.nodes[0] = tensor_desc{
    .tensor_id = 301,
    .alloc_size = 512,
    .src_ids = {{300, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = false,
    .has_external_data = false,
  };
  g.nodes[1] = tensor_desc{
    .tensor_id = 302,
    .alloc_size = 256,
    .src_ids = {{301, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = true,
    .has_external_data = false,
  };
  return g;
}

graph_storage make_prefer_freed_block_graph() {
  graph_storage g{};
  g.n_leafs = 1;
  g.n_nodes = 2;
  g.leafs[0] = tensor_desc{
    .tensor_id = 400,
    .alloc_size = 192,
    .src_ids = {{-1, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = true,
    .is_output = false,
    .has_external_data = false,
  };
  g.nodes[0] = tensor_desc{
    .tensor_id = 401,
    .alloc_size = 32,
    .src_ids = {{400, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = false,
    .has_external_data = false,
  };
  g.nodes[1] = tensor_desc{
    .tensor_id = 402,
    .alloc_size = 96,
    .src_ids = {{401, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = true,
    .has_external_data = false,
  };
  return g;
}

graph_storage make_view_inplace_graph() {
  graph_storage g{};
  g.n_leafs = 2;
  g.n_nodes = 3;
  g.leafs[0] = tensor_desc{
    .tensor_id = 500,
    .alloc_size = 256,
    .src_ids = {{-1, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = true,
    .is_output = false,
    .has_external_data = false,
  };
  g.leafs[1] = tensor_desc{
    .tensor_id = 501,
    .alloc_size = 256,
    .src_ids = {{-1, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = true,
    .is_output = false,
    .has_external_data = false,
  };
  g.nodes[0] = tensor_desc{
    .tensor_id = 510,
    .alloc_size = 0,
    .src_ids = {{500, -1, -1, -1}},
    .is_view = true,
    .view_src_id = 500,
    .is_input = false,
    .is_output = false,
    .has_external_data = false,
  };
  g.nodes[1] = tensor_desc{
    .tensor_id = 511,
    .alloc_size = 256,
    .src_ids = {{510, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = false,
    .has_external_data = false,
  };
  g.nodes[2] = tensor_desc{
    .tensor_id = 512,
    .alloc_size = 256,
    .src_ids = {{511, 501, -1, -1}},
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

bool dispatch_plan_done(
    void *, const emel::buffer::planner::events::plan_done &) noexcept {
  return true;
}

bool dispatch_plan_error(
    void *, const emel::buffer::planner::events::plan_error &) noexcept {
  return true;
}

}  // namespace

TEST_CASE("buffer_planner_starts_idle") {
  emel::buffer::planner::sm planner{};
  int state_count = 0;
  planner.visit_current_states([&](auto) { state_count += 1; });
  CHECK(state_count == 1);
}

TEST_CASE("buffer_planner_plans_sizes_successfully") {
  emel::buffer::planner::sm planner{};
  const graph_storage g = make_valid_graph();
  std::array<int32_t, 1> sizes = {{0}};
  int32_t error_code = -1;

  CHECK(planner.process_event(emel::buffer::planner::event::plan{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .buffer_count = 1,
    .size_only = true,
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
    .error_out = &error_code,
    .owner_sm = &planner,
    .dispatch_done = &dispatch_plan_done,
    .dispatch_error = &dispatch_plan_error,
  }));
  CHECK(error_code == EMEL_OK);
  CHECK(sizes[0] > 0);
  CHECK(planner.total_bytes() > 0);
}

TEST_CASE("buffer_planner_allocates_leaf_inputs") {
  emel::buffer::planner::sm planner{};
  graph_storage g{};
  g.n_leafs = 1;
  g.n_nodes = 0;
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
  std::array<int32_t, 1> sizes = {{0}};
  int32_t error_code = -1;

  CHECK(planner.process_event(emel::buffer::planner::event::plan{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .buffer_count = 1,
    .size_only = false,
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
    .error_out = &error_code,
    .owner_sm = &planner,
    .dispatch_done = &dispatch_plan_done,
    .dispatch_error = &dispatch_plan_error,
  }));
  CHECK(error_code == EMEL_OK);
  CHECK(sizes[0] >= 64);
}

TEST_CASE("buffer_planner_splits_by_max_size") {
  emel::buffer::planner::sm planner{};
  graph_storage g = make_valid_graph();
  std::array<int32_t, 1> sizes = {{0}};
  std::array<int32_t, 1> chunk_counts = {{0}};
  std::array<int32_t, emel::buffer::planner::action::k_max_chunk_plan_entries> chunk_sizes = {};
  std::array<int32_t, 1> leaf_buffer_ids = {{0}};
  std::array<int32_t, 1> max_sizes = {{64}};
  std::array<int32_t, 1> alignments = {{64}};
  int32_t error_code = 0;

  CHECK(planner.process_event(emel::buffer::planner::event::plan{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = leaf_buffer_ids.data(),
    .buffer_count = 1,
    .buffer_alignments = alignments.data(),
    .buffer_max_sizes = max_sizes.data(),
    .size_only = false,
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
    .chunk_sizes_out = chunk_sizes.data(),
    .chunk_sizes_out_count = static_cast<int32_t>(chunk_sizes.size()),
    .chunk_counts_out = chunk_counts.data(),
    .chunk_counts_out_count = static_cast<int32_t>(chunk_counts.size()),
    .error_out = &error_code,
    .owner_sm = &planner,
    .dispatch_done = &dispatch_plan_done,
    .dispatch_error = &dispatch_plan_error,
  }));
  CHECK(error_code == EMEL_OK);
  CHECK(chunk_counts[0] > 1);
  CHECK(chunk_sizes[0] == 64);
}

TEST_CASE("buffer_planner_reports_invalid_arguments") {
  emel::buffer::planner::sm planner{};
  const graph_storage g = make_valid_graph();
  std::array<int32_t, 1> sizes = {{0}};
  int32_t error_code = 0;

  CHECK_FALSE(planner.process_event(emel::buffer::planner::event::plan{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .buffer_count = 0,
    .size_only = true,
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
    .error_out = &error_code,
    .owner_sm = &planner,
    .dispatch_done = &dispatch_plan_done,
    .dispatch_error = &dispatch_plan_error,
  }));
  CHECK(error_code == EMEL_ERR_INVALID_ARGUMENT);

  CHECK_FALSE(planner.process_event(emel::buffer::planner::event::plan{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .buffer_count = 2,
    .size_only = true,
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
    .error_out = &error_code,
    .owner_sm = &planner,
    .dispatch_done = &dispatch_plan_done,
    .dispatch_error = &dispatch_plan_error,
  }));
  CHECK(error_code == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_planner_reports_invalid_sources") {
  emel::buffer::planner::sm planner{};
  const graph_storage g = make_invalid_source_graph();
  std::array<int32_t, 1> sizes = {{0}};
  int32_t error_code = 0;

  CHECK_FALSE(planner.process_event(emel::buffer::planner::event::plan{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .buffer_count = 1,
    .size_only = false,
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
    .error_out = &error_code,
    .owner_sm = &planner,
    .dispatch_done = &dispatch_plan_done,
    .dispatch_error = &dispatch_plan_error,
  }));
  CHECK(error_code == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_planner_reuses_parent_storage_for_inplace_chain") {
  emel::buffer::planner::sm planner{};
  const graph_storage g = make_inplace_reuse_graph();
  std::array<int32_t, 1> sizes = {{0}};
  int32_t error_code = 0;

  CHECK(planner.process_event(emel::buffer::planner::event::plan{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .buffer_count = 1,
    .size_only = false,
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
    .error_out = &error_code,
    .owner_sm = &planner,
    .dispatch_done = &dispatch_plan_done,
    .dispatch_error = &dispatch_plan_error,
  }));
  CHECK(error_code == EMEL_OK);
  CHECK(sizes[0] == 512);
}

TEST_CASE("buffer_planner_prefers_freed_blocks_over_growth") {
  emel::buffer::planner::sm planner{};
  const graph_storage g = make_prefer_freed_block_graph();
  std::array<int32_t, 1> sizes = {{0}};
  int32_t error_code = 0;

  CHECK(planner.process_event(emel::buffer::planner::event::plan{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .buffer_count = 1,
    .size_only = false,
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
    .error_out = &error_code,
    .owner_sm = &planner,
    .dispatch_done = &dispatch_plan_done,
    .dispatch_error = &dispatch_plan_error,
  }));
  CHECK(error_code == EMEL_OK);
  CHECK(sizes[0] == 192);
}

TEST_CASE("buffer_planner_handles_view_parent_inplace_reuse") {
  emel::buffer::planner::sm planner{};
  const graph_storage g = make_view_inplace_graph();
  std::array<int32_t, 1> sizes = {{0}};
  int32_t error_code = 0;

  CHECK(planner.process_event(emel::buffer::planner::event::plan{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .buffer_count = 1,
    .size_only = false,
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
    .error_out = &error_code,
    .owner_sm = &planner,
    .dispatch_done = &dispatch_plan_done,
    .dispatch_error = &dispatch_plan_error,
  }));
  CHECK(error_code == EMEL_OK);
  CHECK(sizes[0] == 512);
}

TEST_CASE("buffer_planner_reports_zero_for_unused_secondary_buffer") {
  emel::buffer::planner::sm planner{};
  const graph_storage g = make_valid_graph();
  const std::array<int32_t, 2> node_ids = {{0, 0}};
  const std::array<int32_t, 1> leaf_ids = {{0}};
  std::array<int32_t, 2> sizes = {{0, 0}};
  int32_t error_code = 0;

  CHECK(planner.process_event(emel::buffer::planner::event::plan{
    .graph = as_view(g),
    .node_buffer_ids = node_ids.data(),
    .leaf_buffer_ids = leaf_ids.data(),
    .buffer_count = 2,
    .size_only = false,
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
    .error_out = &error_code,
    .owner_sm = &planner,
    .dispatch_done = &dispatch_plan_done,
    .dispatch_error = &dispatch_plan_error,
  }));
  CHECK(error_code == EMEL_OK);
  CHECK(sizes[0] > 0);
  CHECK(sizes[1] == 0);
}

TEST_CASE("buffer_planner_testing_sm_can_target_error_and_recovery_paths") {
  emel::buffer::planner::sm planner{};
  const graph_storage g = make_valid_graph();
  std::array<int32_t, 1> sizes = {{0}};
  int32_t error_code = EMEL_OK;

  emel::buffer::planner::strategy failing = emel::buffer::planner::default_strategies::gallocr_parity;
  failing.seed_leafs = +[](emel::buffer::planner::action::context &) noexcept -> int32_t {
    return EMEL_ERR_BACKEND;
  };

  CHECK_FALSE(planner.process_event(emel::buffer::planner::event::plan{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .buffer_count = 1,
    .size_only = false,
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
    .error_out = &error_code,
    .owner_sm = &planner,
    .dispatch_done = &dispatch_plan_done,
    .dispatch_error = &dispatch_plan_error,
    .strategy = &failing,
  }));
  CHECK(error_code == EMEL_ERR_BACKEND);

  error_code = EMEL_OK;
  CHECK(planner.process_event(emel::buffer::planner::event::plan{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .buffer_count = 1,
    .size_only = false,
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
    .error_out = &error_code,
    .owner_sm = &planner,
    .dispatch_done = &dispatch_plan_done,
    .dispatch_error = &dispatch_plan_error,
    .strategy = &emel::buffer::planner::default_strategies::gallocr_parity,
  }));
  CHECK(error_code == EMEL_OK);
}

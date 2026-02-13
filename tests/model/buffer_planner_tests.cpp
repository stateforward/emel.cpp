#include <array>
#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/buffer_allocator/events.hpp"
#include "emel/buffer_planner/guards.hpp"
#include "emel/buffer_planner/sm.hpp"
#include "emel/emel.h"

namespace {

using tensor_desc = emel::buffer_allocator::event::tensor_desc;
using graph_view = emel::buffer_allocator::event::graph_view;

struct graph_storage {
  std::array<tensor_desc, 2> nodes = {};
  std::array<tensor_desc, 1> leafs = {};
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

graph_view as_view(const graph_storage & g) {
  return graph_view{
    .nodes = g.nodes.data(),
    .n_nodes = g.n_nodes,
    .leafs = g.leafs.data(),
    .n_leafs = g.n_leafs,
  };
}

}  // namespace

TEST_CASE("buffer_planner_starts_idle") {
  emel::buffer_planner::sm planner{};
  int state_count = 0;
  planner.visit_current_states([&](auto) { state_count += 1; });
  CHECK(state_count == 1);
}

TEST_CASE("buffer_planner_plans_sizes_successfully") {
  emel::buffer_planner::sm planner{};
  const graph_storage g = make_valid_graph();
  std::array<int32_t, 1> sizes = {{0}};
  int32_t error_code = -1;

  CHECK(planner.process_event(emel::buffer_planner::event::plan{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .buffer_count = 1,
    .size_only = true,
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
    .error_out = &error_code,
  }));
  CHECK(error_code == EMEL_OK);
  CHECK(sizes[0] > 0);
  CHECK(planner.total_bytes() > 0);
}

TEST_CASE("buffer_planner_reports_invalid_arguments") {
  emel::buffer_planner::sm planner{};
  const graph_storage g = make_valid_graph();
  std::array<int32_t, 1> sizes = {{0}};
  int32_t error_code = 0;

  CHECK_FALSE(planner.process_event(emel::buffer_planner::event::plan{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .buffer_count = 0,
    .size_only = true,
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
    .error_out = &error_code,
  }));
  CHECK(error_code == EMEL_ERR_INVALID_ARGUMENT);

  CHECK_FALSE(planner.process_event(emel::buffer_planner::event::plan{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .buffer_count = 2,
    .size_only = true,
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
    .error_out = &error_code,
  }));
  CHECK(error_code == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_planner_reports_invalid_sources") {
  emel::buffer_planner::sm planner{};
  const graph_storage g = make_invalid_source_graph();
  std::array<int32_t, 1> sizes = {{0}};
  int32_t error_code = 0;

  CHECK_FALSE(planner.process_event(emel::buffer_planner::event::plan{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .buffer_count = 1,
    .size_only = false,
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
    .error_out = &error_code,
  }));
  CHECK(error_code == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_planner_reuses_parent_storage_for_inplace_chain") {
  emel::buffer_planner::sm planner{};
  const graph_storage g = make_inplace_reuse_graph();
  std::array<int32_t, 1> sizes = {{0}};
  int32_t error_code = 0;

  CHECK(planner.process_event(emel::buffer_planner::event::plan{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .buffer_count = 1,
    .size_only = false,
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
    .error_out = &error_code,
  }));
  CHECK(error_code == EMEL_OK);
  CHECK(sizes[0] == 512);
}

TEST_CASE("buffer_planner_guards_reflect_pending_error") {
  emel::buffer_planner::action::context ctx{};
  emel::buffer_planner::event::reset_done ev{};

  CHECK(emel::buffer_planner::guard::no_error{}(ev, ctx));
  CHECK_FALSE(emel::buffer_planner::guard::has_error{}(ev, ctx));

  ctx.pending_error = EMEL_ERR_BACKEND;
  CHECK_FALSE(emel::buffer_planner::guard::no_error{}(ev, ctx));
  CHECK(emel::buffer_planner::guard::has_error{}(ev, ctx));
}

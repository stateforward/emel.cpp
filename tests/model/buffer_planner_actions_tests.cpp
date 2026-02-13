#include <array>
#include <cstdint>
#include <doctest/doctest.h>
#include <limits>

#include "emel/buffer_allocator/events.hpp"
#include "emel/buffer_planner/actions.hpp"
#include "emel/buffer_planner/events.hpp"
#include "emel/buffer_planner/sm.hpp"
#include "emel/emel.h"

namespace {

using tensor_desc = emel::buffer_allocator::event::tensor_desc;
using graph_view = emel::buffer_allocator::event::graph_view;

struct graph_storage {
  std::array<tensor_desc, 1> nodes = {};
  std::array<tensor_desc, 1> leafs = {};
  int32_t n_nodes = 1;
  int32_t n_leafs = 1;
};

graph_storage make_graph() {
  graph_storage g{};
  g.leafs[0] = tensor_desc{
    .tensor_id = 10,
    .alloc_size = 64,
    .src_ids = {{-1, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = true,
    .is_output = false,
    .has_external_data = false,
  };
  g.nodes[0] = tensor_desc{
    .tensor_id = 11,
    .alloc_size = 128,
    .src_ids = {{10, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = true,
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

TEST_CASE("buffer_planner_action_math_helpers_handle_edges") {
  CHECK(
    emel::buffer_planner::action::detail::sat_add(std::numeric_limits<int32_t>::max(), 1) ==
    std::numeric_limits<int32_t>::max());
  CHECK(
    emel::buffer_planner::action::detail::sat_add(std::numeric_limits<int32_t>::min(), -1) ==
    std::numeric_limits<int32_t>::min());
  CHECK(emel::buffer_planner::action::detail::align_up(0) == 0);
  CHECK(emel::buffer_planner::action::detail::align_up(-8) == 0);
}

TEST_CASE("buffer_planner_begin_plan_rejects_invalid_event_shapes") {
  const graph_storage g = make_graph();
  int32_t error_code = EMEL_OK;
  emel::buffer_planner::action::context ctx{};

  emel::buffer_planner::action::begin_plan(
    emel::buffer_planner::event::plan{
      .graph = as_view(g),
      .node_buffer_ids = nullptr,
      .leaf_buffer_ids = nullptr,
      .buffer_count = 0,
      .size_only = false,
      .sizes_out = nullptr,
      .sizes_out_count = 0,
      .error_out = &error_code,
    },
    ctx);
  CHECK(ctx.pending_error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(error_code == EMEL_ERR_INVALID_ARGUMENT);

  emel::buffer_planner::action::begin_plan(
    emel::buffer_planner::event::plan{
      .graph =
          graph_view{
            .nodes = nullptr,
            .n_nodes = 1,
            .leafs = nullptr,
            .n_leafs = 0,
          },
      .node_buffer_ids = nullptr,
      .leaf_buffer_ids = nullptr,
      .buffer_count = 1,
      .size_only = false,
      .sizes_out = nullptr,
      .sizes_out_count = 0,
      .error_out = &error_code,
    },
    ctx);
  CHECK(ctx.pending_error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(error_code == EMEL_ERR_INVALID_ARGUMENT);

  emel::buffer_planner::action::begin_plan(
    emel::buffer_planner::event::plan{
      .graph = as_view(g),
      .node_buffer_ids = nullptr,
      .leaf_buffer_ids = nullptr,
      .buffer_count = 1,
      .size_only = false,
      .sizes_out = nullptr,
      .sizes_out_count = -1,
      .error_out = &error_code,
    },
    ctx);
  CHECK(ctx.pending_error == EMEL_OK);
  CHECK(error_code == EMEL_OK);
}

TEST_CASE("buffer_planner_leaf_and_reference_actions_cover_error_paths") {
  const graph_storage base_graph = make_graph();

  {
    emel::buffer_planner::action::context ctx{};
    ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
    ctx.graph = as_view(base_graph);
    ctx.buffer_count = 1;
    emel::buffer_planner::action::on_seed_leafs_done(
      emel::buffer_planner::event::seed_leafs_done{}, ctx);
    CHECK(ctx.planned_leafs == 0);
  }

  {
    graph_storage g = make_graph();
    g.leafs[0].tensor_id = -1;
    emel::buffer_planner::action::context ctx{};
    ctx.graph = as_view(g);
    ctx.buffer_count = 1;
    emel::buffer_planner::action::on_seed_leafs_done(
      emel::buffer_planner::event::seed_leafs_done{}, ctx);
    CHECK(ctx.pending_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    graph_storage g = make_graph();
    std::array<int32_t, 1> leaf_ids = {{2}};
    emel::buffer_planner::action::context ctx{};
    ctx.graph = as_view(g);
    ctx.buffer_count = 1;
    ctx.leaf_buffer_ids = leaf_ids.data();
    emel::buffer_planner::action::on_seed_leafs_done(
      emel::buffer_planner::event::seed_leafs_done{}, ctx);
    CHECK(ctx.pending_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    emel::buffer_planner::action::context ctx{};
    ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
    ctx.graph = as_view(base_graph);
    emel::buffer_planner::action::on_count_references_done(
      emel::buffer_planner::event::count_references_done{}, ctx);
    CHECK(ctx.reference_edges == 0);
  }

  {
    graph_storage g = make_graph();
    g.nodes[0].tensor_id = -1;
    emel::buffer_planner::action::context ctx{};
    ctx.graph = as_view(g);
    emel::buffer_planner::action::on_count_references_done(
      emel::buffer_planner::event::count_references_done{}, ctx);
    CHECK(ctx.pending_error == EMEL_ERR_INVALID_ARGUMENT);
  }
}

TEST_CASE("buffer_planner_node_allocation_actions_cover_error_paths") {
  const graph_storage base_graph = make_graph();

  {
    emel::buffer_planner::action::context ctx{};
    ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
    ctx.graph = as_view(base_graph);
    emel::buffer_planner::action::on_alloc_explicit_inputs_done(
      emel::buffer_planner::event::alloc_explicit_inputs_done{}, ctx);
    CHECK(ctx.planned_nodes == 0);
  }

  {
    graph_storage g = make_graph();
    std::array<int32_t, 1> node_ids = {{2}};
    emel::buffer_planner::action::context ctx{};
    ctx.graph = as_view(g);
    ctx.buffer_count = 1;
    ctx.node_buffer_ids = node_ids.data();
    emel::buffer_planner::action::on_reset_done(emel::buffer_planner::event::reset_done{}, ctx);
    emel::buffer_planner::action::on_alloc_explicit_inputs_done(
      emel::buffer_planner::event::alloc_explicit_inputs_done{}, ctx);
    CHECK(ctx.pending_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    graph_storage g = make_graph();
    std::array<int32_t, 1> node_ids = {{0}};
    emel::buffer_planner::action::context ctx{};
    ctx.graph = as_view(g);
    ctx.buffer_count = 1;
    ctx.node_buffer_ids = node_ids.data();
    emel::buffer_planner::action::on_reset_done(emel::buffer_planner::event::reset_done{}, ctx);
    emel::buffer_planner::action::on_alloc_explicit_inputs_done(
      emel::buffer_planner::event::alloc_explicit_inputs_done{}, ctx);
    CHECK(ctx.pending_error == EMEL_OK);
    CHECK(ctx.planned_nodes == 2);
    CHECK(ctx.bytes_by_buffer[0] > 0);
  }

  {
    emel::buffer_planner::action::context ctx{};
    ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
    ctx.graph = as_view(base_graph);
    emel::buffer_planner::action::on_plan_nodes_done(
      emel::buffer_planner::event::plan_nodes_done{}, ctx);
    CHECK(ctx.planned_nodes == 0);
  }

  {
    graph_storage g = make_graph();
    g.nodes[0].is_input = false;
    std::array<int32_t, 1> node_ids = {{2}};
    emel::buffer_planner::action::context ctx{};
    ctx.graph = as_view(g);
    ctx.buffer_count = 1;
    ctx.node_buffer_ids = node_ids.data();
    emel::buffer_planner::action::on_reset_done(emel::buffer_planner::event::reset_done{}, ctx);
    emel::buffer_planner::action::on_plan_nodes_done(
      emel::buffer_planner::event::plan_nodes_done{}, ctx);
    CHECK(ctx.pending_error == EMEL_ERR_INVALID_ARGUMENT);
  }
}

TEST_CASE("buffer_planner_finalize_and_error_actions_update_outputs") {
  emel::buffer_planner::action::context ctx{};
  int32_t error_code = EMEL_OK;
  ctx.error_out = &error_code;
  ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
  ctx.total_bytes = 123;
  emel::buffer_planner::action::on_finalize_done(emel::buffer_planner::event::finalize_done{}, ctx);
  CHECK(ctx.total_bytes == 123);

  emel::buffer_planner::action::on_plan_error(
    emel::buffer_planner::events::plan_error{.err = 0}, ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);
  CHECK(error_code == EMEL_ERR_BACKEND);
}

TEST_CASE("buffer_planner_wrapper_rejects_plan_while_busy") {
  emel::buffer_planner::sm planner{};
  const graph_storage g = make_graph();
  int32_t error_code = EMEL_OK;
  const emel::buffer_planner::event::plan plan_event{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .buffer_count = 1,
    .size_only = false,
    .sizes_out = nullptr,
    .sizes_out_count = 0,
    .error_out = &error_code,
  };

  auto & base = static_cast<emel::buffer_planner::sm::base_type &>(planner);
  CHECK(base.process_event(plan_event));
  CHECK_FALSE(planner.process_event(plan_event));
}

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
  CHECK(emel::buffer_planner::action::detail::sat_sub_floor_zero(17, 0) == 17);
  CHECK(
    emel::buffer_planner::action::detail::align_up(std::numeric_limits<int32_t>::max()) ==
    std::numeric_limits<int32_t>::max());
}

TEST_CASE("buffer_planner_layout_split_and_merge_blocks") {
  emel::buffer_planner::action::context ctx{};
  ctx.buffer_count = 1;
  emel::buffer_planner::action::detail::reset_layouts(ctx);

  int32_t a0 = -1;
  int32_t a1 = -1;
  CHECK(emel::buffer_planner::action::detail::alloc_bytes_from_layout(ctx, 0, 64, a0));
  CHECK(emel::buffer_planner::action::detail::alloc_bytes_from_layout(ctx, 0, 64, a1));
  CHECK(a0 == 0);
  CHECK(a1 == 64);
  CHECK(ctx.bytes_by_buffer[0] == 128);

  CHECK(emel::buffer_planner::action::detail::free_bytes_to_layout(ctx, 0, a0, 64));
  CHECK(emel::buffer_planner::action::detail::free_bytes_to_layout(ctx, 0, a1, 64));
  CHECK(ctx.buffer_layouts[0].free_block_count == 1);
  CHECK(ctx.buffer_layouts[0].free_blocks[0].offset == 0);
  CHECK(ctx.buffer_layouts[0].free_blocks[0].size == 128);

  int32_t a2 = -1;
  CHECK(emel::buffer_planner::action::detail::alloc_bytes_from_layout(ctx, 0, 96, a2));
  CHECK(a2 == 0);
  CHECK(ctx.bytes_by_buffer[0] == 128);
}

TEST_CASE("buffer_planner_layout_prefers_reuse_before_growth") {
  emel::buffer_planner::action::context ctx{};
  ctx.buffer_count = 1;
  emel::buffer_planner::action::detail::reset_layouts(ctx);

  int32_t a0 = -1;
  int32_t a1 = -1;
  CHECK(emel::buffer_planner::action::detail::alloc_bytes_from_layout(ctx, 0, 128, a0));
  CHECK(emel::buffer_planner::action::detail::alloc_bytes_from_layout(ctx, 0, 32, a1));
  CHECK(a0 == 0);
  CHECK(a1 == 128);
  CHECK(ctx.bytes_by_buffer[0] == 160);

  CHECK(emel::buffer_planner::action::detail::free_bytes_to_layout(ctx, 0, a0, 128));

  int32_t a2 = -1;
  CHECK(emel::buffer_planner::action::detail::alloc_bytes_from_layout(ctx, 0, 64, a2));
  CHECK(a2 == 0);
  CHECK(ctx.bytes_by_buffer[0] == 160);
}

TEST_CASE("buffer_planner_layout_and_strategy_error_edges") {
  {
    emel::buffer_planner::action::context ctx{};
    ctx.buffer_count = 1;
    emel::buffer_planner::action::detail::reset_layouts(ctx);

    int32_t offset = -1;
    CHECK_FALSE(emel::buffer_planner::action::detail::alloc_bytes_from_layout(ctx, 2, 16, offset));
    CHECK_FALSE(
      emel::buffer_planner::action::detail::free_bytes_to_layout(ctx, 3, 0, 16));

    auto & layout = ctx.buffer_layouts[0];
    layout.free_block_count = emel::buffer_planner::action::k_max_free_blocks;
    CHECK_FALSE(emel::buffer_planner::action::detail::insert_free_block(layout, 0, 16));
  }

  {
    const graph_storage g = make_graph();
    int32_t error_code = EMEL_OK;
    emel::buffer_planner::action::context ctx{};
    emel::buffer_planner::strategy bad_strategy{};
    bad_strategy.seed_leafs = emel::buffer_planner::default_strategies::gallocr_parity.seed_leafs;
    bad_strategy.count_references = emel::buffer_planner::default_strategies::gallocr_parity.count_references;
    bad_strategy.alloc_explicit_inputs =
      emel::buffer_planner::default_strategies::gallocr_parity.alloc_explicit_inputs;
    bad_strategy.plan_nodes = emel::buffer_planner::default_strategies::gallocr_parity.plan_nodes;
    bad_strategy.release_expired =
      emel::buffer_planner::default_strategies::gallocr_parity.release_expired;
    bad_strategy.finalize = nullptr;

    emel::buffer_planner::action::begin_plan(
      emel::buffer_planner::event::plan{
        .graph = as_view(g),
        .node_buffer_ids = nullptr,
        .leaf_buffer_ids = nullptr,
        .buffer_count = 1,
        .size_only = false,
        .sizes_out = nullptr,
        .sizes_out_count = 0,
        .error_out = &error_code,
        .strategy = &bad_strategy,
      },
      ctx);
    CHECK(ctx.pending_error == EMEL_ERR_INVALID_ARGUMENT);
    CHECK(error_code == EMEL_ERR_INVALID_ARGUMENT);
  }
}

TEST_CASE("buffer_planner_low_level_layout_edges_and_reset_failures") {
  {
    emel::buffer_planner::action::buffer_layout layout{};
    layout.free_block_count = 2;
    layout.free_blocks[0] = {.offset = 0, .size = 16};
    layout.free_blocks[1] = {.offset = 16, .size = 16};
    emel::buffer_planner::action::detail::remove_free_block(layout, 0);
    CHECK(layout.free_block_count == 1);
    CHECK(layout.free_blocks[0].offset == 16);
  }

  {
    emel::buffer_planner::action::buffer_layout layout{};
    CHECK(emel::buffer_planner::action::detail::insert_free_block(layout, -1, 16));
    CHECK(emel::buffer_planner::action::detail::insert_free_block(layout, 0, 0));
    CHECK(layout.free_block_count == 0);
  }

  {
    emel::buffer_planner::action::buffer_layout layout{};
    CHECK(emel::buffer_planner::action::detail::insert_free_block(layout, 0, 16));
    CHECK(emel::buffer_planner::action::detail::insert_free_block(layout, 32, 16));
    CHECK(emel::buffer_planner::action::detail::insert_free_block(layout, 16, 16));
    CHECK(layout.free_block_count == 1);
    CHECK(layout.free_blocks[0].offset == 0);
    CHECK(layout.free_blocks[0].size == 48);
  }

  {
    emel::buffer_planner::action::context ctx{};
    ctx.buffer_count = 1;
    emel::buffer_planner::action::detail::reset_layouts(ctx);
    CHECK(emel::buffer_planner::action::detail::free_bytes_to_layout(ctx, 0, 0, 32));
    int32_t offset = -1;
    CHECK(emel::buffer_planner::action::detail::alloc_bytes_from_layout(ctx, 0, 32, offset));
    CHECK(offset == 0);
    CHECK(ctx.buffer_layouts[0].free_block_count == 0);
  }

  {
    graph_storage g = make_graph();
    g.nodes[0].tensor_id = g.leafs[0].tensor_id;
    emel::buffer_planner::action::context ctx{};
    ctx.graph = as_view(g);
    ctx.buffer_count = 1;
    emel::buffer_planner::action::on_reset_done(emel::buffer_planner::event::reset_done{}, ctx);
    CHECK(ctx.pending_error == EMEL_ERR_INVALID_ARGUMENT);
  }
}

TEST_CASE("buffer_planner_phase_actions_validate_strategy_callbacks") {
  const graph_storage g = make_graph();
  emel::buffer_planner::strategy base = emel::buffer_planner::default_strategies::gallocr_parity;

  {
    emel::buffer_planner::action::context ctx{};
    ctx.graph = as_view(g);
    ctx.buffer_count = 1;
    ctx.strategy = &base;
    emel::buffer_planner::action::on_reset_done(emel::buffer_planner::event::reset_done{}, ctx);

    emel::buffer_planner::strategy broken = base;
    broken.seed_leafs = nullptr;
    ctx.strategy = &broken;
    emel::buffer_planner::action::on_seed_leafs_done(
      emel::buffer_planner::event::seed_leafs_done{}, ctx);
    CHECK(ctx.pending_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    emel::buffer_planner::action::context ctx{};
    ctx.graph = as_view(g);
    ctx.buffer_count = 1;
    ctx.strategy = &base;
    emel::buffer_planner::action::on_reset_done(emel::buffer_planner::event::reset_done{}, ctx);
    emel::buffer_planner::action::on_seed_leafs_done(
      emel::buffer_planner::event::seed_leafs_done{}, ctx);

    emel::buffer_planner::strategy broken = base;
    broken.count_references = nullptr;
    ctx.strategy = &broken;
    emel::buffer_planner::action::on_count_references_done(
      emel::buffer_planner::event::count_references_done{}, ctx);
    CHECK(ctx.pending_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    emel::buffer_planner::action::context ctx{};
    ctx.graph = as_view(g);
    ctx.buffer_count = 1;
    ctx.strategy = &base;
    emel::buffer_planner::action::on_reset_done(emel::buffer_planner::event::reset_done{}, ctx);
    emel::buffer_planner::action::on_seed_leafs_done(
      emel::buffer_planner::event::seed_leafs_done{}, ctx);
    emel::buffer_planner::action::on_count_references_done(
      emel::buffer_planner::event::count_references_done{}, ctx);

    emel::buffer_planner::strategy broken = base;
    broken.alloc_explicit_inputs = nullptr;
    ctx.strategy = &broken;
    emel::buffer_planner::action::on_alloc_explicit_inputs_done(
      emel::buffer_planner::event::alloc_explicit_inputs_done{}, ctx);
    CHECK(ctx.pending_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    emel::buffer_planner::action::context ctx{};
    ctx.graph = as_view(g);
    ctx.buffer_count = 1;
    ctx.strategy = &base;
    emel::buffer_planner::action::on_reset_done(emel::buffer_planner::event::reset_done{}, ctx);
    emel::buffer_planner::action::on_seed_leafs_done(
      emel::buffer_planner::event::seed_leafs_done{}, ctx);
    emel::buffer_planner::action::on_count_references_done(
      emel::buffer_planner::event::count_references_done{}, ctx);
    emel::buffer_planner::action::on_alloc_explicit_inputs_done(
      emel::buffer_planner::event::alloc_explicit_inputs_done{}, ctx);

    emel::buffer_planner::strategy broken = base;
    broken.plan_nodes = nullptr;
    ctx.strategy = &broken;
    emel::buffer_planner::action::on_plan_nodes_done(
      emel::buffer_planner::event::plan_nodes_done{}, ctx);
    CHECK(ctx.pending_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    emel::buffer_planner::action::context ctx{};
    ctx.graph = as_view(g);
    ctx.buffer_count = 1;
    ctx.strategy = &base;
    emel::buffer_planner::action::on_reset_done(emel::buffer_planner::event::reset_done{}, ctx);
    emel::buffer_planner::action::on_seed_leafs_done(
      emel::buffer_planner::event::seed_leafs_done{}, ctx);
    emel::buffer_planner::action::on_count_references_done(
      emel::buffer_planner::event::count_references_done{}, ctx);
    emel::buffer_planner::action::on_alloc_explicit_inputs_done(
      emel::buffer_planner::event::alloc_explicit_inputs_done{}, ctx);
    emel::buffer_planner::action::on_plan_nodes_done(
      emel::buffer_planner::event::plan_nodes_done{}, ctx);

    emel::buffer_planner::strategy broken = base;
    broken.release_expired = nullptr;
    ctx.strategy = &broken;
    emel::buffer_planner::action::on_release_expired_done(
      emel::buffer_planner::event::release_expired_done{}, ctx);
    CHECK(ctx.pending_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    emel::buffer_planner::action::context ctx{};
    ctx.graph = as_view(g);
    ctx.buffer_count = 1;
    ctx.strategy = &base;
    emel::buffer_planner::action::on_reset_done(emel::buffer_planner::event::reset_done{}, ctx);
    emel::buffer_planner::action::on_seed_leafs_done(
      emel::buffer_planner::event::seed_leafs_done{}, ctx);
    emel::buffer_planner::action::on_count_references_done(
      emel::buffer_planner::event::count_references_done{}, ctx);
    emel::buffer_planner::action::on_alloc_explicit_inputs_done(
      emel::buffer_planner::event::alloc_explicit_inputs_done{}, ctx);
    emel::buffer_planner::action::on_plan_nodes_done(
      emel::buffer_planner::event::plan_nodes_done{}, ctx);
    emel::buffer_planner::action::on_release_expired_done(
      emel::buffer_planner::event::release_expired_done{}, ctx);

    emel::buffer_planner::strategy broken = base;
    broken.finalize = nullptr;
    ctx.strategy = &broken;
    emel::buffer_planner::action::on_finalize_done(
      emel::buffer_planner::event::finalize_done{}, ctx);
    CHECK(ctx.pending_error == EMEL_ERR_INVALID_ARGUMENT);
  }
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

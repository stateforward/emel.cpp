#include <array>
#include <cstdint>
#include <doctest/doctest.h>
#include <limits>

#include "emel/buffer/allocator/events.hpp"
#include "emel/buffer/planner/actions.hpp"
#include "emel/buffer/planner/events.hpp"
#include "emel/buffer/planner/sm.hpp"
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
    .tensor_id = 10,
    .alloc_size = 64,
    .src_ids = emel::buffer::allocator::event::make_src_ids(),
    .is_view = false,
    .view_src_id = -1,
    .is_input = true,
    .is_output = false,
    .has_external_data = false,
  };
  g.nodes[0] = tensor_desc{
    .tensor_id = 11,
    .alloc_size = 128,
    .src_ids = [] {
      auto ids = emel::buffer::allocator::event::make_src_ids();
      ids[0] = 10;
      return ids;
    }(),
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

bool dispatch_plan_done(
    void *, const emel::buffer::planner::events::plan_done &) noexcept {
  return true;
}

bool dispatch_plan_error(
    void *, const emel::buffer::planner::events::plan_error &) noexcept {
  return true;
}

}  // namespace

TEST_CASE("buffer_planner_action_math_helpers_handle_edges") {
  CHECK(
    emel::buffer::planner::action::detail::sat_add(std::numeric_limits<int32_t>::max(), 1) ==
    std::numeric_limits<int32_t>::max());
  CHECK(
    emel::buffer::planner::action::detail::sat_add(std::numeric_limits<int32_t>::min(), -1) ==
    std::numeric_limits<int32_t>::min());
  CHECK(emel::buffer::planner::action::detail::align_up(0, 16) == 0);
  CHECK(emel::buffer::planner::action::detail::align_up(-8, 16) == 0);
  CHECK(emel::buffer::planner::action::detail::sat_sub_floor_zero(17, 0) == 17);
  CHECK(
    emel::buffer::planner::action::detail::align_up(std::numeric_limits<int32_t>::max(), 16) ==
    std::numeric_limits<int32_t>::max());
}

TEST_CASE("buffer_planner_layout_split_and_merge_blocks") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  emel::buffer::planner::action::detail::reset_layouts(ctx);

  int32_t a0 = -1;
  int32_t a1 = -1;
  CHECK(emel::buffer::planner::action::detail::alloc_bytes_from_layout(ctx, 0, 64, a0));
  CHECK(emel::buffer::planner::action::detail::alloc_bytes_from_layout(ctx, 0, 64, a1));
  CHECK(a0 == 0);
  CHECK(a1 == 64);
  CHECK(ctx.bytes_by_buffer[0] == 128);

  CHECK(emel::buffer::planner::action::detail::free_bytes_to_layout(ctx, 0, a0, 64));
  CHECK(emel::buffer::planner::action::detail::free_bytes_to_layout(ctx, 0, a1, 64));
  CHECK(ctx.buffer_layouts[0].free_block_count == 1);
  CHECK(ctx.buffer_layouts[0].free_blocks[0].offset == 0);
  CHECK(ctx.buffer_layouts[0].free_blocks[0].size == 128);

  int32_t a2 = -1;
  CHECK(emel::buffer::planner::action::detail::alloc_bytes_from_layout(ctx, 0, 96, a2));
  CHECK(a2 == 0);
  CHECK(ctx.bytes_by_buffer[0] == 128);
}

TEST_CASE("buffer_planner_layout_prefers_reuse_before_growth") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  emel::buffer::planner::action::detail::reset_layouts(ctx);

  int32_t a0 = -1;
  int32_t a1 = -1;
  CHECK(emel::buffer::planner::action::detail::alloc_bytes_from_layout(ctx, 0, 128, a0));
  CHECK(emel::buffer::planner::action::detail::alloc_bytes_from_layout(ctx, 0, 32, a1));
  CHECK(a0 == 0);
  CHECK(a1 == 128);
  CHECK(ctx.bytes_by_buffer[0] == 160);

  CHECK(emel::buffer::planner::action::detail::free_bytes_to_layout(ctx, 0, a0, 128));

  int32_t a2 = -1;
  CHECK(emel::buffer::planner::action::detail::alloc_bytes_from_layout(ctx, 0, 64, a2));
  CHECK(a2 == 0);
  CHECK(ctx.bytes_by_buffer[0] == 160);
}

TEST_CASE("buffer_planner_layout_and_strategy_error_edges") {
  {
    emel::buffer::planner::action::context ctx{};
    ctx.buffer_count = 1;
    emel::buffer::planner::action::detail::reset_layouts(ctx);

    int32_t offset = -1;
    CHECK_FALSE(emel::buffer::planner::action::detail::alloc_bytes_from_layout(ctx, 2, 16, offset));
    CHECK_FALSE(
      emel::buffer::planner::action::detail::free_bytes_to_layout(ctx, 3, 0, 16));

    auto & layout = ctx.buffer_layouts[0];
    layout.free_block_count = emel::buffer::planner::action::k_max_free_blocks;
    CHECK_FALSE(emel::buffer::planner::action::detail::insert_free_block(layout, 0, 16));
  }

  {
    const graph_storage g = make_graph();
    int32_t error_code = EMEL_OK;
    emel::buffer::planner::action::context ctx{};
    emel::buffer::planner::strategy bad_strategy{};
    bad_strategy.seed_leafs = emel::buffer::planner::default_strategies::gallocr_parity.seed_leafs;
    bad_strategy.count_references = emel::buffer::planner::default_strategies::gallocr_parity.count_references;
    bad_strategy.alloc_explicit_inputs =
      emel::buffer::planner::default_strategies::gallocr_parity.alloc_explicit_inputs;
    bad_strategy.plan_nodes = emel::buffer::planner::default_strategies::gallocr_parity.plan_nodes;
    bad_strategy.release_expired =
      emel::buffer::planner::default_strategies::gallocr_parity.release_expired;
    bad_strategy.finalize = nullptr;

    emel::buffer::planner::action::begin_plan(
      emel::buffer::planner::event::plan{
        .graph = as_view(g),
        .node_buffer_ids = nullptr,
        .leaf_buffer_ids = nullptr,
        .buffer_count = 1,
        .size_only = false,
        .sizes_out = nullptr,
        .sizes_out_count = 0,
        .error_out = &error_code,
        .owner_sm = &ctx,
        .dispatch_done = &dispatch_plan_done,
        .dispatch_error = &dispatch_plan_error,
        .strategy = &bad_strategy,
      },
      ctx);
    CHECK(error_code == EMEL_OK);
    CHECK_FALSE(emel::buffer::planner::action::detail::valid_strategy(&ctx.strategy));
  }
}

TEST_CASE("buffer_planner_low_level_layout_edges_and_reset_failures") {
  {
    emel::buffer::planner::action::buffer_layout layout{};
    layout.free_block_count = 2;
    layout.free_blocks[0] = {.offset = 0, .size = 16};
    layout.free_blocks[1] = {.offset = 16, .size = 16};
    emel::buffer::planner::action::detail::remove_free_block(layout, 0);
    CHECK(layout.free_block_count == 1);
    CHECK(layout.free_blocks[0].offset == 16);
  }

  {
    emel::buffer::planner::action::buffer_layout layout{};
    CHECK(emel::buffer::planner::action::detail::insert_free_block(layout, -1, 16));
    CHECK(emel::buffer::planner::action::detail::insert_free_block(layout, 0, 0));
    CHECK(layout.free_block_count == 0);
  }

  {
    emel::buffer::planner::action::buffer_layout layout{};
    CHECK(emel::buffer::planner::action::detail::insert_free_block(layout, 0, 16));
    CHECK(emel::buffer::planner::action::detail::insert_free_block(layout, 32, 16));
    CHECK(emel::buffer::planner::action::detail::insert_free_block(layout, 16, 16));
    CHECK(layout.free_block_count == 1);
    CHECK(layout.free_blocks[0].offset == 0);
    CHECK(layout.free_blocks[0].size == 48);
  }

  {
    emel::buffer::planner::action::context ctx{};
    ctx.buffer_count = 1;
    emel::buffer::planner::action::detail::reset_layouts(ctx);
    CHECK(emel::buffer::planner::action::detail::free_bytes_to_layout(ctx, 0, 0, 32));
    int32_t offset = -1;
    CHECK(emel::buffer::planner::action::detail::alloc_bytes_from_layout(ctx, 0, 32, offset));
    CHECK(offset == 0);
    CHECK(ctx.buffer_layouts[0].free_block_count == 0);
  }

  {
    graph_storage g = make_graph();
    g.nodes[0].tensor_id = g.leafs[0].tensor_id;
    emel::buffer::planner::action::context ctx{};
    emel::buffer::planner::action::begin_plan(
        emel::buffer::planner::event::plan{
            .graph = as_view(g),
            .node_buffer_ids = nullptr,
            .leaf_buffer_ids = nullptr,
            .buffer_count = 1,
            .size_only = false,
            .sizes_out = nullptr,
            .sizes_out_count = 0,
            .error_out = nullptr,
            .owner_sm = &ctx,
            .dispatch_done = &dispatch_plan_done,
            .dispatch_error = &dispatch_plan_error,
            .strategy = &emel::buffer::planner::default_strategies::gallocr_parity,
        },
        ctx);
    int32_t error_out = EMEL_OK;
    emel::buffer::planner::action::on_reset_done(
        emel::buffer::planner::event::reset_done{.error_out = &error_out}, ctx);
    CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);
  }
}

TEST_CASE("buffer_planner_phase_actions_validate_strategy_callbacks") {
  const graph_storage g = make_graph();
  emel::buffer::planner::strategy base = emel::buffer::planner::default_strategies::gallocr_parity;
  const auto seed_context = [&](emel::buffer::planner::action::context & ctx,
                                const emel::buffer::planner::strategy * strategy) {
    emel::buffer::planner::action::begin_plan(
        emel::buffer::planner::event::plan{
            .graph = as_view(g),
            .node_buffer_ids = nullptr,
            .leaf_buffer_ids = nullptr,
            .buffer_count = 1,
            .size_only = false,
            .sizes_out = nullptr,
            .sizes_out_count = 0,
            .error_out = nullptr,
            .owner_sm = &ctx,
            .dispatch_done = &dispatch_plan_done,
            .dispatch_error = &dispatch_plan_error,
            .strategy = strategy,
        },
        ctx);
  };

  {
    emel::buffer::planner::action::context ctx{};
    seed_context(ctx, &base);
    int32_t reset_err = EMEL_OK;
    emel::buffer::planner::action::on_reset_done(
        emel::buffer::planner::event::reset_done{.error_out = &reset_err}, ctx);
    CHECK(reset_err == EMEL_OK);

    emel::buffer::planner::strategy broken = base;
    broken.seed_leafs = nullptr;
    ctx.strategy = broken;
    CHECK_FALSE(emel::buffer::planner::action::detail::valid_strategy(&ctx.strategy));
  }

  {
    emel::buffer::planner::action::context ctx{};
    seed_context(ctx, &base);
    int32_t reset_err = EMEL_OK;
    emel::buffer::planner::action::on_reset_done(
        emel::buffer::planner::event::reset_done{.error_out = &reset_err}, ctx);
    CHECK(reset_err == EMEL_OK);
    emel::buffer::planner::action::on_seed_leafs_done(
      emel::buffer::planner::event::seed_leafs_done{.error_out = &reset_err}, ctx);

    emel::buffer::planner::strategy broken = base;
    broken.count_references = nullptr;
    ctx.strategy = broken;
    CHECK_FALSE(emel::buffer::planner::action::detail::valid_strategy(&ctx.strategy));
  }

  {
    emel::buffer::planner::action::context ctx{};
    seed_context(ctx, &base);
    int32_t reset_err = EMEL_OK;
    emel::buffer::planner::action::on_reset_done(
        emel::buffer::planner::event::reset_done{.error_out = &reset_err}, ctx);
    CHECK(reset_err == EMEL_OK);
    emel::buffer::planner::action::on_seed_leafs_done(
      emel::buffer::planner::event::seed_leafs_done{.error_out = &reset_err}, ctx);
    emel::buffer::planner::action::on_count_references_done(
      emel::buffer::planner::event::count_references_done{.error_out = &reset_err}, ctx);

    emel::buffer::planner::strategy broken = base;
    broken.alloc_explicit_inputs = nullptr;
    ctx.strategy = broken;
    CHECK_FALSE(emel::buffer::planner::action::detail::valid_strategy(&ctx.strategy));
  }

  {
    emel::buffer::planner::action::context ctx{};
    seed_context(ctx, &base);
    int32_t reset_err = EMEL_OK;
    emel::buffer::planner::action::on_reset_done(
        emel::buffer::planner::event::reset_done{.error_out = &reset_err}, ctx);
    CHECK(reset_err == EMEL_OK);
    emel::buffer::planner::action::on_seed_leafs_done(
      emel::buffer::planner::event::seed_leafs_done{.error_out = &reset_err}, ctx);
    emel::buffer::planner::action::on_count_references_done(
      emel::buffer::planner::event::count_references_done{.error_out = &reset_err}, ctx);
    emel::buffer::planner::action::on_alloc_explicit_inputs_done(
      emel::buffer::planner::event::alloc_explicit_inputs_done{.error_out = &reset_err}, ctx);

    emel::buffer::planner::strategy broken = base;
    broken.plan_nodes = nullptr;
    ctx.strategy = broken;
    CHECK_FALSE(emel::buffer::planner::action::detail::valid_strategy(&ctx.strategy));
  }

  {
    emel::buffer::planner::action::context ctx{};
    seed_context(ctx, &base);
    int32_t reset_err = EMEL_OK;
    emel::buffer::planner::action::on_reset_done(
        emel::buffer::planner::event::reset_done{.error_out = &reset_err}, ctx);
    CHECK(reset_err == EMEL_OK);
    emel::buffer::planner::action::on_seed_leafs_done(
      emel::buffer::planner::event::seed_leafs_done{.error_out = &reset_err}, ctx);
    emel::buffer::planner::action::on_count_references_done(
      emel::buffer::planner::event::count_references_done{.error_out = &reset_err}, ctx);
    emel::buffer::planner::action::on_alloc_explicit_inputs_done(
      emel::buffer::planner::event::alloc_explicit_inputs_done{.error_out = &reset_err}, ctx);
    emel::buffer::planner::action::on_plan_nodes_done(
      emel::buffer::planner::event::plan_nodes_done{.error_out = &reset_err}, ctx);

    emel::buffer::planner::strategy broken = base;
    broken.release_expired = nullptr;
    ctx.strategy = broken;
    CHECK_FALSE(emel::buffer::planner::action::detail::valid_strategy(&ctx.strategy));
  }

  {
    emel::buffer::planner::action::context ctx{};
    seed_context(ctx, &base);
    int32_t reset_err = EMEL_OK;
    emel::buffer::planner::action::on_reset_done(
        emel::buffer::planner::event::reset_done{.error_out = &reset_err}, ctx);
    CHECK(reset_err == EMEL_OK);
    emel::buffer::planner::action::on_seed_leafs_done(
      emel::buffer::planner::event::seed_leafs_done{.error_out = &reset_err}, ctx);
    emel::buffer::planner::action::on_count_references_done(
      emel::buffer::planner::event::count_references_done{.error_out = &reset_err}, ctx);
    emel::buffer::planner::action::on_alloc_explicit_inputs_done(
      emel::buffer::planner::event::alloc_explicit_inputs_done{.error_out = &reset_err}, ctx);
    emel::buffer::planner::action::on_plan_nodes_done(
      emel::buffer::planner::event::plan_nodes_done{.error_out = &reset_err}, ctx);
    emel::buffer::planner::action::on_release_expired_done(
      emel::buffer::planner::event::release_expired_done{.error_out = &reset_err}, ctx);

    emel::buffer::planner::strategy broken = base;
    broken.finalize = nullptr;
    ctx.strategy = broken;
    CHECK_FALSE(emel::buffer::planner::action::detail::valid_strategy(&ctx.strategy));
  }
}

TEST_CASE("buffer_planner_additional_branch_edges_for_coverage_gate") {
  namespace detail = emel::buffer::planner::action::detail;
  using context = emel::buffer::planner::action::context;
  using tensor_record = emel::buffer::planner::action::tensor_record;
  using buffer_layout = emel::buffer::planner::action::buffer_layout;

  {
    buffer_layout layout{};
    layout.free_block_count = 1;
    layout.free_blocks[0] = {.offset = 0, .size = 16};
    detail::remove_free_block(layout, -1);
    CHECK(layout.free_block_count == 1);
  }

  {
    context ctx{};
    tensor_desc bad_tensor{
      .tensor_id = -1,
      .alloc_size = 4,
      .src_ids = emel::buffer::allocator::event::make_src_ids(),
      .is_view = false,
      .view_src_id = -1,
      .is_input = false,
      .is_output = false,
      .has_external_data = false,
    };
    CHECK_FALSE(detail::register_tensor(ctx, bad_tensor));

    tensor_desc good_tensor = bad_tensor;
    good_tensor.tensor_id = 42;
    good_tensor.alloc_size = 16;
    CHECK(detail::register_tensor(ctx, good_tensor));
    CHECK_FALSE(detail::register_tensor(ctx, good_tensor));

    ctx.tensor_count = emel::buffer::planner::action::k_max_tensors;
    good_tensor.tensor_id = 43;
    CHECK_FALSE(detail::register_tensor(ctx, good_tensor));
  }

  {
    context ctx{};
    ctx.buffer_count = 1;
    tensor_record rec{};
    rec.allocatable = true;
    rec.allocated = false;
    rec.alloc_size = 16;
    CHECK_FALSE(detail::allocate_record(ctx, rec, 7));

    rec.allocated = true;
    rec.buffer_id = 7;
    CHECK_FALSE(detail::free_record(ctx, rec));
    CHECK_FALSE(rec.allocated);
  }

  {
    const graph_storage g = make_graph();
    int dummy_owner = 0;
    auto ev = emel::buffer::planner::event::plan{
      .graph = as_view(g),
      .node_buffer_ids = nullptr,
      .leaf_buffer_ids = nullptr,
      .buffer_count = 1,
      .size_only = true,
      .sizes_out = nullptr,
      .sizes_out_count = 0,
      .error_out = nullptr,
      .owner_sm = &dummy_owner,
      .dispatch_done = &dispatch_plan_done,
      .dispatch_error = &dispatch_plan_error,
      .strategy = nullptr,
    };
    CHECK(detail::valid_plan_event(ev));

    ev.graph.n_nodes = -1;
    CHECK_FALSE(detail::valid_plan_event(ev));
    ev.graph.n_nodes = 1;

    std::array<int32_t, 1> sizes = {{0}};
    ev.sizes_out = sizes.data();
    ev.sizes_out_count = 0;
    CHECK_FALSE(detail::valid_plan_event(ev));

    CHECK(detail::valid_strategy(nullptr));
  }

  {
    context ctx{};
    const graph_storage g = make_graph();
    emel::buffer::planner::action::begin_plan(
        emel::buffer::planner::event::plan{
            .graph = as_view(g),
            .node_buffer_ids = nullptr,
            .leaf_buffer_ids = nullptr,
            .buffer_count = 1,
            .size_only = false,
            .sizes_out = nullptr,
            .sizes_out_count = 0,
            .error_out = nullptr,
            .owner_sm = &ctx,
            .dispatch_done = &dispatch_plan_done,
            .dispatch_error = &dispatch_plan_error,
            .strategy = nullptr,
        },
        ctx);
    const int32_t leafs_before = ctx.planned_leafs;
    const int32_t refs_before = ctx.reference_edges;
    CHECK(detail::default_seed_leafs(ctx) == EMEL_ERR_INVALID_ARGUMENT);
    CHECK(detail::default_count_references(ctx) == EMEL_ERR_INVALID_ARGUMENT);
    CHECK(ctx.planned_leafs == leafs_before);
    CHECK(ctx.reference_edges == refs_before);
  }
}

TEST_CASE("buffer_planner_begin_plan_rejects_invalid_event_shapes") {
  const graph_storage g = make_graph();
  namespace detail = emel::buffer::planner::action::detail;

  const auto zero_buffers = emel::buffer::planner::event::plan{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .buffer_count = 0,
    .size_only = false,
    .sizes_out = nullptr,
    .sizes_out_count = 0,
    .error_out = nullptr,
    .owner_sm = const_cast<graph_storage *>(&g),
    .dispatch_done = &dispatch_plan_done,
    .dispatch_error = &dispatch_plan_error,
  };
  CHECK_FALSE(detail::valid_plan_event(zero_buffers));

  const auto missing_nodes = emel::buffer::planner::event::plan{
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
    .error_out = nullptr,
    .owner_sm = const_cast<graph_storage *>(&g),
    .dispatch_done = &dispatch_plan_done,
    .dispatch_error = &dispatch_plan_error,
  };
  CHECK_FALSE(detail::valid_plan_event(missing_nodes));

  const auto negative_sizes = emel::buffer::planner::event::plan{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .buffer_count = 1,
    .size_only = false,
    .sizes_out = nullptr,
    .sizes_out_count = -1,
    .error_out = nullptr,
    .owner_sm = const_cast<graph_storage *>(&g),
    .dispatch_done = &dispatch_plan_done,
    .dispatch_error = &dispatch_plan_error,
  };
  CHECK(detail::valid_plan_event(negative_sizes));
}

TEST_CASE("buffer_planner_leaf_and_reference_actions_cover_error_paths") {
  const graph_storage base_graph = make_graph();

  {
    emel::buffer::planner::action::context ctx{};
    emel::buffer::planner::action::begin_plan(
        emel::buffer::planner::event::plan{
            .graph = as_view(base_graph),
            .node_buffer_ids = nullptr,
            .leaf_buffer_ids = nullptr,
            .buffer_count = 1,
            .size_only = false,
            .sizes_out = nullptr,
            .sizes_out_count = 0,
            .error_out = nullptr,
            .owner_sm = &ctx,
            .dispatch_done = &dispatch_plan_done,
            .dispatch_error = &dispatch_plan_error,
            .strategy = nullptr,
        },
        ctx);
    int32_t err = EMEL_OK;
    emel::buffer::planner::action::on_seed_leafs_done(
      emel::buffer::planner::event::seed_leafs_done{.error_out = &err}, ctx);
    CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
    CHECK(ctx.planned_leafs == 0);
  }

  {
    graph_storage g = make_graph();
    g.leafs[0].tensor_id = -1;
    emel::buffer::planner::action::context ctx{};
    emel::buffer::planner::action::begin_plan(
        emel::buffer::planner::event::plan{
            .graph = as_view(g),
            .node_buffer_ids = nullptr,
            .leaf_buffer_ids = nullptr,
            .buffer_count = 1,
            .size_only = false,
            .sizes_out = nullptr,
            .sizes_out_count = 0,
            .error_out = nullptr,
            .owner_sm = &ctx,
            .dispatch_done = &dispatch_plan_done,
            .dispatch_error = &dispatch_plan_error,
            .strategy = nullptr,
        },
        ctx);
    int32_t err = EMEL_OK;
    emel::buffer::planner::action::on_seed_leafs_done(
      emel::buffer::planner::event::seed_leafs_done{.error_out = &err}, ctx);
    CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    graph_storage g = make_graph();
    std::array<int32_t, 1> leaf_ids = {{2}};
    emel::buffer::planner::action::context ctx{};
    emel::buffer::planner::action::begin_plan(
        emel::buffer::planner::event::plan{
            .graph = as_view(g),
            .node_buffer_ids = nullptr,
            .leaf_buffer_ids = leaf_ids.data(),
            .buffer_count = 1,
            .size_only = false,
            .sizes_out = nullptr,
            .sizes_out_count = 0,
            .error_out = nullptr,
            .owner_sm = &ctx,
            .dispatch_done = &dispatch_plan_done,
            .dispatch_error = &dispatch_plan_error,
            .strategy = nullptr,
        },
        ctx);
    int32_t err = EMEL_OK;
    emel::buffer::planner::action::on_seed_leafs_done(
      emel::buffer::planner::event::seed_leafs_done{.error_out = &err}, ctx);
    CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    emel::buffer::planner::action::context ctx{};
    emel::buffer::planner::action::begin_plan(
        emel::buffer::planner::event::plan{
            .graph = as_view(base_graph),
            .node_buffer_ids = nullptr,
            .leaf_buffer_ids = nullptr,
            .buffer_count = 1,
            .size_only = false,
            .sizes_out = nullptr,
            .sizes_out_count = 0,
            .error_out = nullptr,
            .owner_sm = &ctx,
            .dispatch_done = &dispatch_plan_done,
            .dispatch_error = &dispatch_plan_error,
            .strategy = nullptr,
        },
        ctx);
    int32_t err = EMEL_OK;
    emel::buffer::planner::action::on_count_references_done(
      emel::buffer::planner::event::count_references_done{.error_out = &err}, ctx);
    CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
    CHECK(ctx.reference_edges == 0);
  }

  {
    graph_storage g = make_graph();
    g.nodes[0].tensor_id = -1;
    emel::buffer::planner::action::context ctx{};
    emel::buffer::planner::action::begin_plan(
        emel::buffer::planner::event::plan{
            .graph = as_view(g),
            .node_buffer_ids = nullptr,
            .leaf_buffer_ids = nullptr,
            .buffer_count = 1,
            .size_only = false,
            .sizes_out = nullptr,
            .sizes_out_count = 0,
            .error_out = nullptr,
            .owner_sm = &ctx,
            .dispatch_done = &dispatch_plan_done,
            .dispatch_error = &dispatch_plan_error,
            .strategy = nullptr,
        },
        ctx);
    int32_t err = EMEL_OK;
    emel::buffer::planner::action::on_count_references_done(
      emel::buffer::planner::event::count_references_done{.error_out = &err}, ctx);
    CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  }
}

TEST_CASE("buffer_planner_node_allocation_actions_cover_error_paths") {
  const graph_storage base_graph = make_graph();

  {
    emel::buffer::planner::action::context ctx{};
    emel::buffer::planner::action::begin_plan(
        emel::buffer::planner::event::plan{
            .graph = as_view(base_graph),
            .node_buffer_ids = nullptr,
            .leaf_buffer_ids = nullptr,
            .buffer_count = 1,
            .size_only = false,
            .sizes_out = nullptr,
            .sizes_out_count = 0,
            .error_out = nullptr,
            .owner_sm = &ctx,
            .dispatch_done = &dispatch_plan_done,
            .dispatch_error = &dispatch_plan_error,
            .strategy = nullptr,
        },
        ctx);
    int32_t err = EMEL_OK;
    emel::buffer::planner::action::on_alloc_explicit_inputs_done(
      emel::buffer::planner::event::alloc_explicit_inputs_done{.error_out = &err}, ctx);
    CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
    CHECK(ctx.planned_nodes == 0);
  }

  {
    graph_storage g = make_graph();
    std::array<int32_t, 1> node_ids = {{2}};
    emel::buffer::planner::action::context ctx{};
    emel::buffer::planner::action::begin_plan(
        emel::buffer::planner::event::plan{
            .graph = as_view(g),
            .node_buffer_ids = node_ids.data(),
            .leaf_buffer_ids = nullptr,
            .buffer_count = 1,
            .size_only = false,
            .sizes_out = nullptr,
            .sizes_out_count = 0,
            .error_out = nullptr,
            .owner_sm = &ctx,
            .dispatch_done = &dispatch_plan_done,
            .dispatch_error = &dispatch_plan_error,
            .strategy = nullptr,
        },
        ctx);
    int32_t reset_err = EMEL_OK;
    emel::buffer::planner::action::on_reset_done(
        emel::buffer::planner::event::reset_done{.error_out = &reset_err}, ctx);
    CHECK(reset_err == EMEL_OK);
    int32_t err = EMEL_OK;
    emel::buffer::planner::action::on_alloc_explicit_inputs_done(
      emel::buffer::planner::event::alloc_explicit_inputs_done{.error_out = &err}, ctx);
    CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    graph_storage g = make_graph();
    std::array<int32_t, 1> node_ids = {{0}};
    emel::buffer::planner::action::context ctx{};
    emel::buffer::planner::action::begin_plan(
        emel::buffer::planner::event::plan{
            .graph = as_view(g),
            .node_buffer_ids = node_ids.data(),
            .leaf_buffer_ids = nullptr,
            .buffer_count = 1,
            .size_only = false,
            .sizes_out = nullptr,
            .sizes_out_count = 0,
            .error_out = nullptr,
            .owner_sm = &ctx,
            .dispatch_done = &dispatch_plan_done,
            .dispatch_error = &dispatch_plan_error,
            .strategy = nullptr,
        },
        ctx);
    int32_t reset_err = EMEL_OK;
    emel::buffer::planner::action::on_reset_done(
        emel::buffer::planner::event::reset_done{.error_out = &reset_err}, ctx);
    CHECK(reset_err == EMEL_OK);
    int32_t err = EMEL_OK;
    emel::buffer::planner::action::on_alloc_explicit_inputs_done(
      emel::buffer::planner::event::alloc_explicit_inputs_done{.error_out = &err}, ctx);
    CHECK(err == EMEL_OK);
    CHECK(ctx.planned_nodes == 1);
    CHECK(ctx.bytes_by_buffer[0] > 0);
  }

  {
    emel::buffer::planner::action::context ctx{};
    emel::buffer::planner::action::begin_plan(
        emel::buffer::planner::event::plan{
            .graph = as_view(base_graph),
            .node_buffer_ids = nullptr,
            .leaf_buffer_ids = nullptr,
            .buffer_count = 1,
            .size_only = false,
            .sizes_out = nullptr,
            .sizes_out_count = 0,
            .error_out = nullptr,
            .owner_sm = &ctx,
            .dispatch_done = &dispatch_plan_done,
            .dispatch_error = &dispatch_plan_error,
            .strategy = nullptr,
        },
        ctx);
    int32_t err = EMEL_OK;
    emel::buffer::planner::action::on_plan_nodes_done(
      emel::buffer::planner::event::plan_nodes_done{.error_out = &err}, ctx);
    CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
    CHECK(ctx.planned_nodes == 0);
  }

  {
    graph_storage g = make_graph();
    g.nodes[0].is_input = false;
    std::array<int32_t, 1> node_ids = {{2}};
    emel::buffer::planner::action::context ctx{};
    emel::buffer::planner::action::begin_plan(
        emel::buffer::planner::event::plan{
            .graph = as_view(g),
            .node_buffer_ids = node_ids.data(),
            .leaf_buffer_ids = nullptr,
            .buffer_count = 1,
            .size_only = false,
            .sizes_out = nullptr,
            .sizes_out_count = 0,
            .error_out = nullptr,
            .owner_sm = &ctx,
            .dispatch_done = &dispatch_plan_done,
            .dispatch_error = &dispatch_plan_error,
            .strategy = nullptr,
        },
        ctx);
    int32_t reset_err = EMEL_OK;
    emel::buffer::planner::action::on_reset_done(
        emel::buffer::planner::event::reset_done{.error_out = &reset_err}, ctx);
    CHECK(reset_err == EMEL_OK);
    int32_t err = EMEL_OK;
    emel::buffer::planner::action::on_plan_nodes_done(
      emel::buffer::planner::event::plan_nodes_done{.error_out = &err}, ctx);
    CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  }
}

TEST_CASE("buffer_planner_finalize_and_error_actions_update_outputs") {
  emel::buffer::planner::action::context ctx{};
  int32_t error_code = EMEL_OK;
  ctx.buffer_count = 1;
  ctx.bytes_by_buffer[0] = 123;
  ctx.strategy = emel::buffer::planner::default_strategies::gallocr_parity;
  emel::buffer::planner::action::on_finalize_done(
      emel::buffer::planner::event::finalize_done{.error_out = &error_code}, ctx);
  CHECK(ctx.total_bytes == 123);
  CHECK(error_code == EMEL_OK);

  emel::buffer::planner::action::on_plan_error(
    emel::buffer::planner::events::plan_error{.err = 0, .error_out = &error_code}, ctx);
  CHECK(error_code == EMEL_ERR_BACKEND);
}

TEST_CASE("buffer_planner_split_required_populates_chunk_plan") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.bytes_by_buffer[0] = 96;
  ctx.buffer_alignments[0] = 16;
  ctx.buffer_max_sizes[0] = 32;

  std::array<int32_t, emel::buffer::planner::action::k_max_chunk_plan_entries> chunk_sizes = {};
  std::array<int32_t, emel::buffer::planner::action::k_max_buffers> chunk_counts = {};
  emel::buffer::planner::event::plan plan{};
  plan.chunk_sizes_out = chunk_sizes.data();
  plan.chunk_sizes_out_count = static_cast<int32_t>(chunk_sizes.size());
  plan.chunk_counts_out = chunk_counts.data();
  plan.chunk_counts_out_count = static_cast<int32_t>(chunk_counts.size());

  const int32_t err = emel::buffer::planner::action::detail::run_split_required(ctx, &plan);
  CHECK(err == EMEL_OK);
  CHECK(chunk_counts[0] == 3);
  CHECK(chunk_sizes[0] == 32);
  CHECK(chunk_sizes[1] == 32);
  CHECK(chunk_sizes[2] == 32);
}

TEST_CASE("buffer_planner_allows_sequential_plans") {
  emel::buffer::planner::sm planner{};
  const graph_storage g = make_graph();
  int32_t error_code = EMEL_OK;
  const emel::buffer::planner::event::plan plan_event{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .buffer_count = 1,
    .size_only = false,
    .sizes_out = nullptr,
    .sizes_out_count = 0,
    .error_out = &error_code,
    .owner_sm = &planner,
    .dispatch_done = &dispatch_plan_done,
    .dispatch_error = &dispatch_plan_error,
  };

  auto & base = static_cast<emel::buffer::planner::sm::base_type &>(planner);
  CHECK(base.process_event(plan_event));
  CHECK(planner.process_event(plan_event));
}

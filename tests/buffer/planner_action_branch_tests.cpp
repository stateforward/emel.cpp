#include <array>
#include <doctest/doctest.h>

#include "emel/buffer/allocator/events.hpp"
#include "emel/buffer/planner/actions.hpp"
#include "emel/buffer/planner/events.hpp"
#include "emel/emel.h"

namespace {

struct graph_storage {
  std::array<emel::buffer::allocator::event::tensor_desc, 1> nodes = {};
  std::array<emel::buffer::allocator::event::tensor_desc, 1> leafs = {};
  emel::buffer::allocator::event::graph_view view = {};
};

bool plan_done(void *, const emel::buffer::planner::events::plan_done &) noexcept { return true; }
bool plan_error(void *, const emel::buffer::planner::events::plan_error &) noexcept { return true; }

}  // namespace

TEST_CASE("buffer_planner_detail_insert_free_block_merges_ranges") {
  emel::buffer::planner::action::buffer_layout layout{};
  layout.free_block_count = 2;
  layout.free_blocks[0] = emel::buffer::planner::action::free_block{.offset = 0, .size = 8};
  layout.free_blocks[1] = emel::buffer::planner::action::free_block{.offset = 20, .size = 4};

  CHECK(emel::buffer::planner::action::detail::insert_free_block(layout, 8, 12));
  CHECK(layout.free_block_count == 1);
  CHECK(layout.free_blocks[0].offset == 0);
  CHECK(layout.free_blocks[0].size == 24);

  emel::buffer::planner::action::buffer_layout full{};
  full.free_block_count = emel::buffer::planner::action::k_max_free_blocks;
  CHECK_FALSE(emel::buffer::planner::action::detail::insert_free_block(full, 4, 4));
}

TEST_CASE("buffer_planner_detail_alloc_bytes_from_layout_paths") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;

  ctx.buffer_layouts[0].free_block_count = 1;
  ctx.buffer_layouts[0].free_blocks[0] =
    emel::buffer::planner::action::free_block{.offset = 0, .size = 16};
  int32_t offset = -1;
  CHECK(emel::buffer::planner::action::detail::alloc_bytes_from_layout(ctx, 0, 1, offset));
  CHECK(offset == 0);
  CHECK(ctx.buffer_layouts[0].free_block_count == 0);

  ctx.buffer_layouts[0].free_block_count = 0;
  ctx.buffer_layouts[0].high_watermark = 0;
  ctx.bytes_by_buffer[0] = 0;
  offset = -1;
  CHECK(emel::buffer::planner::action::detail::alloc_bytes_from_layout(ctx, 0, 1, offset));
  CHECK(offset == 0);
  CHECK(ctx.bytes_by_buffer[0] == 16);

  CHECK_FALSE(emel::buffer::planner::action::detail::alloc_bytes_from_layout(ctx, 2, 1, offset));
}

TEST_CASE("buffer_planner_detail_free_bytes_to_layout_validates_buffer_id") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  CHECK_FALSE(emel::buffer::planner::action::detail::free_bytes_to_layout(ctx, 2, 0, 16));
  CHECK(emel::buffer::planner::action::detail::free_bytes_to_layout(ctx, 0, 0, 16));
  CHECK(ctx.buffer_layouts[0].free_block_count > 0);
}

TEST_CASE("buffer_planner_register_tensor_and_graph_validation") {
  emel::buffer::planner::action::context ctx{};
  emel::buffer::allocator::event::tensor_desc tensor{};
  tensor.tensor_id = -1;
  tensor.alloc_size = 4;
  CHECK_FALSE(emel::buffer::planner::action::detail::register_tensor(ctx, tensor));

  tensor.tensor_id = 1;
  CHECK(emel::buffer::planner::action::detail::register_tensor(ctx, tensor));
  CHECK_FALSE(emel::buffer::planner::action::detail::register_tensor(ctx, tensor));

  ctx.tensor_count = emel::buffer::planner::action::k_max_tensors;
  tensor.tensor_id = 2;
  CHECK_FALSE(emel::buffer::planner::action::detail::register_tensor(ctx, tensor));

  emel::buffer::planner::action::context dup_ctx{};
  dup_ctx.leaf_count = 1;
  dup_ctx.node_count = 1;
  dup_ctx.leafs[0].tensor_id = 0;
  dup_ctx.leafs[0].alloc_size = 4;
  dup_ctx.nodes[0].tensor_id = 0;
  dup_ctx.nodes[0].alloc_size = 4;
  CHECK_FALSE(emel::buffer::planner::action::detail::register_graph_tensors(dup_ctx));
}

TEST_CASE("buffer_planner_allocate_and_free_record_branches") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;

  emel::buffer::planner::action::tensor_record rec{};
  rec.allocatable = false;
  CHECK(emel::buffer::planner::action::detail::allocate_record(ctx, rec, 0));

  rec.allocatable = true;
  rec.alloc_size = 16;
  CHECK(emel::buffer::planner::action::detail::allocate_record(ctx, rec, 0));
  CHECK(rec.allocated);
  CHECK(emel::buffer::planner::action::detail::free_record(ctx, rec));
  CHECK_FALSE(rec.allocated);

  rec.allocated = true;
  rec.buffer_id = 2;
  rec.alloc_reserved = 16;
  CHECK_FALSE(emel::buffer::planner::action::detail::free_record(ctx, rec));
  CHECK_FALSE(rec.allocated);
}

TEST_CASE("buffer_planner_valid_plan_event_and_strategy_checks") {
  emel::buffer::planner::event::plan plan{};
  plan.buffer_count = 0;
  CHECK_FALSE(emel::buffer::planner::action::detail::valid_plan_event(plan));

  plan.buffer_count = 1;
  plan.graph.n_nodes = 1;
  plan.graph.nodes = nullptr;
  CHECK_FALSE(emel::buffer::planner::action::detail::valid_plan_event(plan));

  plan.graph.n_nodes = 0;
  plan.graph.n_leafs = 0;
  plan.graph.nodes = nullptr;
  plan.graph.leafs = nullptr;
  plan.sizes_out = nullptr;
  plan.sizes_out_count = 0;
  plan.owner_sm = nullptr;
  CHECK_FALSE(emel::buffer::planner::action::detail::valid_plan_event(plan));

  plan.owner_sm = &plan;
  plan.dispatch_done = &plan_done;
  plan.dispatch_error = &plan_error;
  CHECK(emel::buffer::planner::action::detail::valid_plan_event(plan));

  emel::buffer::planner::strategy bad_strategy{};
  CHECK_FALSE(emel::buffer::planner::action::detail::valid_strategy(&bad_strategy));
  CHECK(emel::buffer::planner::action::detail::valid_strategy(nullptr));
}

TEST_CASE("buffer_planner_begin_plan_sets_error_for_invalid") {
  emel::buffer::planner::action::context ctx{};
  int32_t err = EMEL_OK;
  emel::buffer::planner::event::plan plan{};
  plan.buffer_count = 0;
  plan.error_out = &err;

  emel::buffer::planner::action::begin_plan(plan, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_planner_finalize_copies_sizes") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.bytes_by_buffer[0] = 64;
  ctx.strategy = emel::buffer::planner::default_strategies::gallocr_parity;

  std::array<int32_t, 1> sizes = {{0}};
  emel::buffer::planner::event::plan plan{};
  plan.sizes_out = sizes.data();
  plan.sizes_out_count = 1;

  const int32_t err = emel::buffer::planner::action::detail::run_finalize(ctx, &plan);
  CHECK(err == EMEL_OK);
  CHECK(sizes[0] == 64);
}

TEST_CASE("buffer_planner_default_phase_helpers_report_errors") {
  emel::buffer::planner::action::context ctx{};

  ctx.leaf_count = 1;
  ctx.leafs[0].tensor_id = 1;
  ctx.leafs[0].alloc_size = 4;
  CHECK(emel::buffer::planner::action::detail::default_seed_leafs(ctx) == EMEL_ERR_INVALID_ARGUMENT);

  ctx.node_count = 1;
  ctx.nodes[0].tensor_id = 0;
  ctx.nodes[0].is_view = true;
  ctx.nodes[0].view_src_id = -1;
  ctx.tensor_count = 1;
  ctx.tensors[0].tensor_id = 0;
  CHECK(emel::buffer::planner::action::detail::default_count_references(ctx) == EMEL_ERR_INVALID_ARGUMENT);

  ctx.nodes[0].is_view = false;
  ctx.nodes[0].src_ids[0] = 42;
  CHECK(emel::buffer::planner::action::detail::default_alloc_explicit_inputs(ctx) == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_planner_run_helpers_validate_strategy") {
  emel::buffer::planner::action::context ctx{};
  CHECK(emel::buffer::planner::action::detail::run_seed_leafs(ctx) == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(emel::buffer::planner::action::detail::run_count_references(ctx) == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(emel::buffer::planner::action::detail::run_alloc_explicit_inputs(ctx) == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(emel::buffer::planner::action::detail::run_plan_nodes(ctx) == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(emel::buffer::planner::action::detail::run_release_expired(ctx) == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_planner_default_plan_nodes_reuses_parent_allocation") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.leaf_count = 1;
  ctx.node_count = 1;
  ctx.leafs[0].tensor_id = 0;
  ctx.leafs[0].alloc_size = 64;
  ctx.nodes[0].tensor_id = 1;
  ctx.nodes[0].alloc_size = 32;
  ctx.nodes[0].src_ids = {{0, -1, -1, -1}};
  ctx.leaf_buffer_ids[0] = 0;
  ctx.node_buffer_ids[0] = 0;
  ctx.strategy = emel::buffer::planner::default_strategies::gallocr_parity;

  CHECK(emel::buffer::planner::action::detail::run_reset(ctx) == EMEL_OK);
  CHECK(emel::buffer::planner::action::detail::run_seed_leafs(ctx) == EMEL_OK);
  CHECK(emel::buffer::planner::action::detail::run_count_references(ctx) == EMEL_OK);
  CHECK(emel::buffer::planner::action::detail::run_alloc_explicit_inputs(ctx) == EMEL_OK);
  CHECK(emel::buffer::planner::action::detail::run_plan_nodes(ctx) == EMEL_OK);
}

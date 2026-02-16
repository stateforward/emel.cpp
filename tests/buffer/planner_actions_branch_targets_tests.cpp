#include <array>
#include <cstdint>

#include <doctest/doctest.h>

#include "emel/buffer/allocator/events.hpp"
#include "emel/buffer/planner/actions.hpp"
#include "emel/emel.h"

namespace {

using tensor_desc = emel::buffer::allocator::event::tensor_desc;

tensor_desc make_leaf(const int32_t id) {
  return tensor_desc{
    .tensor_id = id,
    .alloc_size = 16,
    .src_ids = {{-1, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = true,
    .is_output = false,
    .has_external_data = false,
  };
}

tensor_desc make_node(const int32_t id, const int32_t src_id) {
  return tensor_desc{
    .tensor_id = id,
    .alloc_size = 16,
    .src_ids = {{src_id, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = true,
    .has_external_data = false,
  };
}

}  // namespace

TEST_CASE("planner_valid_plan_event_rejects_overflow_and_missing_callbacks") {
  std::array<tensor_desc, 1> nodes = {{make_node(1, -1)}};
  std::array<tensor_desc, 1> leafs = {{make_leaf(2)}};
  std::array<int32_t, 1> sizes = {{0}};

  emel::buffer::planner::event::plan ev{
    .graph = emel::buffer::allocator::event::graph_view{
      .nodes = nodes.data(),
      .n_nodes = emel::buffer::planner::action::k_max_tensors,
      .leafs = leafs.data(),
      .n_leafs = 1,
    },
    .buffer_count = 1,
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
    .owner_sm = reinterpret_cast<void *>(0x1),
    .dispatch_done = reinterpret_cast<bool (*)(void *, const emel::buffer::planner::events::plan_done &)>(0x1),
    .dispatch_error = reinterpret_cast<bool (*)(void *, const emel::buffer::planner::events::plan_error &)>(0x1),
  };

  CHECK_FALSE(emel::buffer::planner::action::detail::valid_plan_event(ev));

  ev.graph.n_nodes = 0;
  ev.graph.n_leafs = 0;
  ev.owner_sm = nullptr;
  CHECK_FALSE(emel::buffer::planner::action::detail::valid_plan_event(ev));
}

TEST_CASE("planner_register_graph_tensors_rejects_invalid_leaf") {
  emel::buffer::planner::action::context ctx{};
  ctx.leaf_count = 1;
  ctx.node_count = 0;
  ctx.leafs[0] = make_leaf(-1);

  CHECK_FALSE(emel::buffer::planner::action::detail::register_graph_tensors(ctx));
}

TEST_CASE("planner_allocate_record_rejects_invalid_buffer") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;

  emel::buffer::planner::action::tensor_record rec{};
  rec.tensor_id = 1;
  rec.alloc_size = 16;
  rec.allocatable = true;
  rec.allocated = false;

  CHECK_FALSE(emel::buffer::planner::action::detail::allocate_record(ctx, rec, 2));
}

TEST_CASE("planner_free_record_rejects_invalid_buffer") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;

  emel::buffer::planner::action::tensor_record rec{};
  rec.tensor_id = 1;
  rec.alloc_size = 16;
  rec.allocatable = true;
  rec.allocated = true;
  rec.buffer_id = 2;

  CHECK_FALSE(emel::buffer::planner::action::detail::free_record(ctx, rec));
  CHECK_FALSE(rec.allocated);
}

TEST_CASE("planner_free_record_handles_full_free_block_list") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.buffer_layouts[0].free_block_count = emel::buffer::planner::action::k_max_free_blocks;

  emel::buffer::planner::action::tensor_record rec{};
  rec.tensor_id = 1;
  rec.alloc_size = 16;
  rec.allocatable = true;
  rec.allocated = true;
  rec.buffer_id = 0;
  rec.alloc_offset = 0;
  rec.alloc_reserved = 16;

  CHECK_FALSE(emel::buffer::planner::action::detail::free_record(ctx, rec));
}

TEST_CASE("planner_default_seed_leafs_propagates_allocate_error") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.leaf_count = 1;
  ctx.leafs[0] = make_leaf(1);
  ctx.leaf_buffer_ids[0] = 2;

  CHECK(emel::buffer::planner::action::detail::register_graph_tensors(ctx));
  CHECK(emel::buffer::planner::action::detail::default_seed_leafs(ctx) == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("planner_default_count_references_rejects_missing_view_source") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.node_count = 1;
  ctx.nodes[0] = make_node(2, -1);
  ctx.nodes[0].is_view = true;
  ctx.nodes[0].view_src_id = 99;

  CHECK(emel::buffer::planner::action::detail::register_graph_tensors(ctx));
  CHECK(emel::buffer::planner::action::detail::default_count_references(ctx) == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("planner_default_alloc_explicit_inputs_rejects_missing_src") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.node_count = 1;
  ctx.nodes[0] = make_node(2, 99);
  ctx.node_buffer_ids[0] = 0;

  CHECK(emel::buffer::planner::action::detail::register_graph_tensors(ctx));
  CHECK(emel::buffer::planner::action::detail::default_alloc_explicit_inputs(ctx) ==
        EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("planner_default_alloc_explicit_inputs_rejects_allocate_failure") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.node_count = 1;
  ctx.leaf_count = 1;
  ctx.leafs[0] = make_leaf(1);
  ctx.nodes[0] = make_node(2, 1);
  ctx.node_buffer_ids[0] = 2;
  ctx.leaf_buffer_ids[0] = 2;

  CHECK(emel::buffer::planner::action::detail::register_graph_tensors(ctx));
  CHECK(emel::buffer::planner::action::detail::default_alloc_explicit_inputs(ctx) ==
        EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("planner_default_plan_nodes_rejects_missing_parent") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.node_count = 1;
  ctx.nodes[0] = make_node(2, 99);
  ctx.node_buffer_ids[0] = 0;

  CHECK(emel::buffer::planner::action::detail::register_graph_tensors(ctx));
  CHECK(emel::buffer::planner::action::detail::default_plan_nodes(ctx) ==
        EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("planner_allocate_record_handles_zero_aligned_size") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  emel::buffer::planner::action::detail::reset_layouts(ctx);

  emel::buffer::planner::action::tensor_record rec{};
  rec.tensor_id = 7;
  rec.alloc_size = -16;
  rec.allocatable = true;
  rec.allocated = false;

  CHECK(emel::buffer::planner::action::detail::allocate_record(ctx, rec, 0));
  CHECK(rec.allocated);
}

TEST_CASE("planner_free_record_noops_when_not_allocatable") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;

  emel::buffer::planner::action::tensor_record rec{};
  rec.tensor_id = 8;
  rec.allocatable = false;
  rec.allocated = true;

  CHECK(emel::buffer::planner::action::detail::free_record(ctx, rec));
  CHECK(rec.allocated);
}

TEST_CASE("planner_default_plan_nodes_skips_view_without_source") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.node_count = 1;
  ctx.leaf_count = 1;
  ctx.leafs[0] = make_leaf(1);
  ctx.leafs[0].is_view = true;
  ctx.leafs[0].view_src_id = -1;
  ctx.nodes[0] = make_node(2, 1);
  ctx.node_buffer_ids[0] = 0;
  ctx.leaf_buffer_ids[0] = 0;

  CHECK(emel::buffer::planner::action::detail::register_graph_tensors(ctx));
  auto * parent = emel::buffer::planner::action::detail::find_record(ctx, 1);
  REQUIRE(parent != nullptr);
  parent->n_children = 2;
  emel::buffer::planner::action::detail::reset_layouts(ctx);
  CHECK(emel::buffer::planner::action::detail::default_plan_nodes(ctx) == EMEL_OK);
}

TEST_CASE("planner_default_plan_nodes_rejects_missing_view_source_in_reuse") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.node_count = 1;
  ctx.leaf_count = 1;
  ctx.leafs[0] = make_leaf(1);
  ctx.leafs[0].is_view = true;
  ctx.leafs[0].view_src_id = 99;
  ctx.nodes[0] = make_node(2, 1);
  ctx.node_buffer_ids[0] = 0;
  ctx.leaf_buffer_ids[0] = 0;

  CHECK(emel::buffer::planner::action::detail::register_graph_tensors(ctx));
  CHECK(emel::buffer::planner::action::detail::default_plan_nodes(ctx) ==
        EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("planner_default_plan_nodes_skips_unallocatable_view_source") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.node_count = 1;
  ctx.leaf_count = 2;
  ctx.leafs[0] = make_leaf(1);
  ctx.leafs[0].has_external_data = true;
  ctx.leafs[1] = make_leaf(2);
  ctx.leafs[1].is_view = true;
  ctx.leafs[1].view_src_id = 1;
  ctx.nodes[0] = make_node(3, 2);
  ctx.node_buffer_ids[0] = 0;
  ctx.leaf_buffer_ids[0] = 0;
  ctx.leaf_buffer_ids[1] = 0;

  CHECK(emel::buffer::planner::action::detail::register_graph_tensors(ctx));
  emel::buffer::planner::action::detail::reset_layouts(ctx);
  CHECK(emel::buffer::planner::action::detail::default_plan_nodes(ctx) == EMEL_OK);
}

TEST_CASE("planner_default_plan_nodes_rejects_missing_parent_in_release") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.node_count = 1;
  ctx.leaf_count = 0;
  ctx.nodes[0] = make_node(4, 99);
  ctx.nodes[0].is_input = true;
  ctx.node_buffer_ids[0] = 0;

  CHECK(emel::buffer::planner::action::detail::register_graph_tensors(ctx));
  CHECK(emel::buffer::planner::action::detail::default_plan_nodes(ctx) ==
        EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("planner_default_plan_nodes_reports_free_record_failure_for_parent") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.node_count = 1;
  ctx.leaf_count = 1;
  ctx.leafs[0] = make_leaf(1);
  ctx.nodes[0] = make_node(5, 1);
  ctx.nodes[0].is_input = true;
  ctx.node_buffer_ids[0] = 0;
  ctx.leaf_buffer_ids[0] = 0;

  CHECK(emel::buffer::planner::action::detail::register_graph_tensors(ctx));
  auto * parent = emel::buffer::planner::action::detail::find_record(ctx, 1);
  REQUIRE(parent != nullptr);
  parent->allocated = true;
  parent->is_output = false;
  parent->buffer_id = 2;
  parent->n_children = 1;
  parent->n_views = 0;

  CHECK(emel::buffer::planner::action::detail::default_plan_nodes(ctx) == EMEL_ERR_BACKEND);
}

TEST_CASE("planner_default_plan_nodes_skips_view_source_count_mismatch") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.node_count = 1;
  ctx.leaf_count = 2;
  ctx.leafs[0] = make_leaf(1);
  ctx.leafs[1] = make_leaf(2);
  ctx.leafs[1].is_view = true;
  ctx.leafs[1].view_src_id = 1;
  ctx.nodes[0] = make_node(3, 2);
  ctx.node_buffer_ids[0] = 0;
  ctx.leaf_buffer_ids[0] = 0;
  ctx.leaf_buffer_ids[1] = 0;

  CHECK(emel::buffer::planner::action::detail::register_graph_tensors(ctx));
  auto * view_src = emel::buffer::planner::action::detail::find_record(ctx, 1);
  REQUIRE(view_src != nullptr);
  view_src->allocatable = true;
  view_src->allocated = true;
  view_src->is_output = false;
  view_src->n_views = 2;
  view_src->n_children = 0;

  emel::buffer::planner::action::detail::reset_layouts(ctx);
  CHECK(emel::buffer::planner::action::detail::default_plan_nodes(ctx) == EMEL_OK);
}

TEST_CASE("planner_default_plan_nodes_skips_reuse_parent_count_mismatch") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.node_count = 1;
  ctx.leaf_count = 1;
  ctx.leafs[0] = make_leaf(1);
  ctx.nodes[0] = make_node(2, 1);
  ctx.node_buffer_ids[0] = 0;
  ctx.leaf_buffer_ids[0] = 0;

  CHECK(emel::buffer::planner::action::detail::register_graph_tensors(ctx));
  auto * parent = emel::buffer::planner::action::detail::find_record(ctx, 1);
  REQUIRE(parent != nullptr);
  parent->allocatable = true;
  parent->allocated = true;
  parent->is_output = false;
  parent->buffer_id = 0;
  parent->n_children = 2;
  parent->n_views = 0;

  emel::buffer::planner::action::detail::reset_layouts(ctx);
  CHECK(emel::buffer::planner::action::detail::default_plan_nodes(ctx) == EMEL_OK);
}

TEST_CASE("planner_default_plan_nodes_reports_free_bytes_failure") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.node_count = 1;
  ctx.leaf_count = 1;
  ctx.leafs[0] = make_leaf(1);
  ctx.nodes[0] = make_node(2, 1);
  ctx.nodes[0].alloc_size = 16;
  ctx.node_buffer_ids[0] = 0;
  ctx.leaf_buffer_ids[0] = 0;

  CHECK(emel::buffer::planner::action::detail::register_graph_tensors(ctx));
  auto * parent = emel::buffer::planner::action::detail::find_record(ctx, 1);
  REQUIRE(parent != nullptr);
  parent->allocatable = true;
  parent->allocated = true;
  parent->is_output = false;
  parent->buffer_id = 0;
  parent->alloc_offset = 0;
  parent->alloc_reserved = 64;
  parent->alloc_size = 64;
  parent->n_children = 1;
  parent->n_views = 0;
  ctx.buffer_layouts[0].free_block_count = emel::buffer::planner::action::k_max_free_blocks;

  CHECK(emel::buffer::planner::action::detail::default_plan_nodes(ctx) == EMEL_ERR_BACKEND);
}

TEST_CASE("planner_default_plan_nodes_rejects_view_release_without_source") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.node_count = 1;
  ctx.leaf_count = 1;
  ctx.leafs[0] = make_leaf(1);
  ctx.leafs[0].is_view = true;
  ctx.leafs[0].view_src_id = -1;
  ctx.nodes[0] = make_node(2, 1);
  ctx.nodes[0].is_input = true;
  ctx.node_buffer_ids[0] = 0;
  ctx.leaf_buffer_ids[0] = 0;

  CHECK(emel::buffer::planner::action::detail::register_graph_tensors(ctx));
  auto * parent = emel::buffer::planner::action::detail::find_record(ctx, 1);
  REQUIRE(parent != nullptr);
  parent->n_children = 1;
  parent->n_views = 0;

  CHECK(emel::buffer::planner::action::detail::default_plan_nodes(ctx) ==
        EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("planner_default_plan_nodes_rejects_view_release_missing_source") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.node_count = 1;
  ctx.leaf_count = 1;
  ctx.leafs[0] = make_leaf(1);
  ctx.leafs[0].is_view = true;
  ctx.leafs[0].view_src_id = 99;
  ctx.nodes[0] = make_node(2, 1);
  ctx.nodes[0].is_input = true;
  ctx.node_buffer_ids[0] = 0;
  ctx.leaf_buffer_ids[0] = 0;

  CHECK(emel::buffer::planner::action::detail::register_graph_tensors(ctx));
  auto * parent = emel::buffer::planner::action::detail::find_record(ctx, 1);
  REQUIRE(parent != nullptr);
  parent->n_children = 1;
  parent->n_views = 0;

  CHECK(emel::buffer::planner::action::detail::default_plan_nodes(ctx) ==
        EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("planner_default_plan_nodes_reports_free_record_failure_for_view_source") {
  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.node_count = 1;
  ctx.leaf_count = 2;
  ctx.leafs[0] = make_leaf(1);
  ctx.leafs[1] = make_leaf(2);
  ctx.leafs[1].is_view = true;
  ctx.leafs[1].view_src_id = 1;
  ctx.nodes[0] = make_node(3, 2);
  ctx.nodes[0].is_input = true;
  ctx.node_buffer_ids[0] = 0;
  ctx.leaf_buffer_ids[0] = 0;
  ctx.leaf_buffer_ids[1] = 0;

  CHECK(emel::buffer::planner::action::detail::register_graph_tensors(ctx));
  auto * parent = emel::buffer::planner::action::detail::find_record(ctx, 2);
  REQUIRE(parent != nullptr);
  parent->n_children = 1;
  parent->n_views = 0;

  auto * view_src = emel::buffer::planner::action::detail::find_record(ctx, 1);
  REQUIRE(view_src != nullptr);
  view_src->allocated = true;
  view_src->is_output = false;
  view_src->buffer_id = 2;
  view_src->n_views = 1;
  view_src->n_children = 0;

  CHECK(emel::buffer::planner::action::detail::default_plan_nodes(ctx) == EMEL_ERR_BACKEND);
}

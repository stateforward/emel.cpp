#include <array>
#include <cstdint>
#include <limits>

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
    .src_ids = emel::buffer::allocator::event::make_src_ids(),
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
    .src_ids = [src_id] {
      auto ids = emel::buffer::allocator::event::make_src_ids();
      ids[0] = src_id;
      return ids;
    }(),
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

TEST_CASE("planner_detail_helpers_cover_overflow_paths") {
  using namespace emel::buffer::planner::action::detail;

  int32_t out = 0;
  CHECK_FALSE(add_checked(std::numeric_limits<int32_t>::max(), 1, out));

  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;
  CHECK(alignment_for_buffer(ctx, -1) == emel::buffer::planner::action::k_default_alignment);
  CHECK(max_size_for_buffer(ctx, 2) == emel::buffer::planner::action::k_default_max_size);

  CHECK_FALSE(align_up_checked(std::numeric_limits<int32_t>::max(), 16, out));
}

TEST_CASE("planner_detail_align_up_saturates") {
  using emel::buffer::planner::action::detail::align_up;
  const int32_t value = std::numeric_limits<int32_t>::max();
  CHECK(align_up(value, 2) == std::numeric_limits<int32_t>::max());
}

TEST_CASE("planner_detail_alloc_and_free_alignment_failures") {
  using namespace emel::buffer::planner::action::detail;

  emel::buffer::planner::action::context ctx{};
  ctx.buffer_count = 1;

  int32_t offset = 0;
  CHECK_FALSE(alloc_bytes_from_layout(ctx, 0, std::numeric_limits<int32_t>::max(), offset));
  CHECK_FALSE(free_bytes_to_layout(ctx, 0, 0, std::numeric_limits<int32_t>::max()));

  emel::buffer::planner::action::tensor_record rec{};
  rec.tensor_id = 5;
  rec.alloc_size = std::numeric_limits<int32_t>::max();
  rec.allocatable = true;
  rec.allocated = false;
  CHECK_FALSE(allocate_record(ctx, rec, 0));
}

TEST_CASE("planner_default_alloc_explicit_inputs_paths") {
  {
    emel::buffer::planner::action::context ctx{};
    ctx.buffer_count = 1;
    ctx.node_count = 1;
    ctx.nodes[0] = make_node(10, -1);
    CHECK(emel::buffer::planner::action::detail::default_alloc_explicit_inputs(ctx) ==
          EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    emel::buffer::planner::action::context ctx{};
    ctx.buffer_count = 1;
    ctx.node_count = 1;
    ctx.nodes[0] = make_node(2, 1);
    ctx.node_buffer_ids[0] = 2;
    ctx.tensor_count = 2;
    ctx.tensors[0].tensor_id = 2;
    ctx.tensors[0].alloc_size = 16;
    ctx.tensors[0].allocatable = true;
    ctx.tensors[1].tensor_id = 1;
    ctx.tensors[1].alloc_size = 16;
    ctx.tensors[1].is_input = true;
    ctx.tensors[1].allocatable = true;
    CHECK(emel::buffer::planner::action::detail::default_alloc_explicit_inputs(ctx) ==
          EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    emel::buffer::planner::action::context ctx{};
    ctx.buffer_count = 1;
    ctx.node_count = 1;
    ctx.nodes[0] = make_node(2, 1);
    ctx.node_buffer_ids[0] = 0;
    ctx.tensor_count = 2;
    ctx.tensors[0].tensor_id = 2;
    ctx.tensors[0].alloc_size = 16;
    ctx.tensors[0].allocatable = true;
    ctx.tensors[1].tensor_id = 1;
    ctx.tensors[1].alloc_size = 16;
    ctx.tensors[1].is_input = true;
    ctx.tensors[1].allocatable = true;
    CHECK(emel::buffer::planner::action::detail::default_alloc_explicit_inputs(ctx) == EMEL_OK);
    CHECK(ctx.planned_nodes > 0);
  }
}

TEST_CASE("planner_default_finalize_error_paths") {
  {
    emel::buffer::planner::action::context ctx{};
    ctx.buffer_count = 1;
    ctx.bytes_by_buffer[0] = -1;
    CHECK(emel::buffer::planner::action::detail::default_finalize(ctx) ==
          EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    emel::buffer::planner::action::context ctx{};
    ctx.buffer_count = 2;
    ctx.bytes_by_buffer[0] = std::numeric_limits<int32_t>::max();
    ctx.bytes_by_buffer[1] = 1;
    CHECK(emel::buffer::planner::action::detail::default_finalize(ctx) ==
          EMEL_ERR_INVALID_ARGUMENT);
  }
}

TEST_CASE("planner_run_split_required_error_paths") {
  using emel::buffer::planner::action::detail::run_split_required;

  {
    emel::buffer::planner::action::context ctx{};
    ctx.buffer_count = 1;
    ctx.buffer_alignments[0] = 16;
    ctx.buffer_max_sizes[0] = 0;
    ctx.bytes_by_buffer[0] = std::numeric_limits<int32_t>::max();
    CHECK(run_split_required(ctx, nullptr) == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    emel::buffer::planner::action::context ctx{};
    ctx.buffer_count = 1;
    ctx.buffer_alignments[0] = 1;
    ctx.buffer_max_sizes[0] = 1;
    ctx.bytes_by_buffer[0] =
      (emel::buffer::planner::action::k_max_chunks_per_buffer + 1);
    CHECK(run_split_required(ctx, nullptr) == EMEL_OK);
    CHECK(ctx.chunk_counts[0] == emel::buffer::planner::action::k_max_chunks_per_buffer);
    CHECK(ctx.chunk_sizes[emel::buffer::planner::action::detail::chunk_plan_index(
      0, emel::buffer::planner::action::k_max_chunks_per_buffer - 1)] == 2);
  }

  {
    emel::buffer::planner::action::context ctx{};
    ctx.buffer_count = 1;
    ctx.buffer_alignments[0] = 16;
    ctx.buffer_max_sizes[0] = std::numeric_limits<int32_t>::max() - 1;
    ctx.bytes_by_buffer[0] = std::numeric_limits<int32_t>::max();
    CHECK(run_split_required(ctx, nullptr) == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    emel::buffer::planner::action::context ctx{};
    ctx.buffer_count = 1;
    ctx.buffer_alignments[0] = 8;
    ctx.buffer_max_sizes[0] = 10;
    ctx.bytes_by_buffer[0] = 20;
    CHECK(run_split_required(ctx, nullptr) == EMEL_ERR_INVALID_ARGUMENT);
  }
}

TEST_CASE("planner_on_unexpected_sets_error_out") {
  emel::buffer::planner::action::context ctx{};
  int32_t err = EMEL_OK;
  emel::buffer::planner::event::plan ev{};
  ev.error_out = &err;

  struct emel::buffer::planner::action::on_unexpected handler{};
  handler(ev, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}
TEST_CASE("planner_valid_plan_event_rejects_invalid_limits") {
  std::array<tensor_desc, 1> nodes = {{make_node(1, -1)}};
  std::array<tensor_desc, 1> leafs = {{make_leaf(2)}};
  std::array<int32_t, 1> sizes = {{0}};
  std::array<int32_t, 1> counts = {{0}};
  std::array<int32_t, 1> alignments = {{3}};
  std::array<int32_t, 1> max_sizes = {{-1}};

  emel::buffer::planner::event::plan ev{
    .graph = emel::buffer::allocator::event::graph_view{
      .nodes = nodes.data(),
      .n_nodes = 1,
      .leafs = leafs.data(),
      .n_leafs = 1,
    },
    .buffer_count = 1,
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
    .chunk_counts_out = counts.data(),
    .chunk_counts_out_count = 0,
    .owner_sm = reinterpret_cast<void *>(0x1),
    .dispatch_done = reinterpret_cast<bool (*)(void *, const emel::buffer::planner::events::plan_done &)>(0x1),
    .dispatch_error = reinterpret_cast<bool (*)(void *, const emel::buffer::planner::events::plan_error &)>(0x1),
  };

  CHECK_FALSE(emel::buffer::planner::action::detail::valid_plan_event(ev));

  ev.chunk_counts_out_count = 1;
  ev.buffer_alignments = alignments.data();
  CHECK_FALSE(emel::buffer::planner::action::detail::valid_plan_event(ev));

  ev.buffer_alignments = nullptr;
  ev.buffer_max_sizes = max_sizes.data();
  CHECK_FALSE(emel::buffer::planner::action::detail::valid_plan_event(ev));

  max_sizes[0] = 5;
  alignments[0] = 4;
  ev.buffer_alignments = alignments.data();
  CHECK_FALSE(emel::buffer::planner::action::detail::valid_plan_event(ev));

  max_sizes[0] = 2;
  CHECK_FALSE(emel::buffer::planner::action::detail::valid_plan_event(ev));

  max_sizes[0] = 6;
  CHECK_FALSE(emel::buffer::planner::action::detail::valid_plan_event(ev));
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

#include <array>
#include <doctest/doctest.h>

#include "emel/buffer/allocator/actions.hpp"
#include "emel/buffer/allocator/events.hpp"
#include "emel/buffer/allocator/guards.hpp"
#include "emel/buffer/allocator/sm.hpp"
#include "emel/buffer/chunk_allocator/sm.hpp"
#include "emel/buffer/planner/sm.hpp"
#include "emel/buffer/realloc_analyzer/sm.hpp"
#include "emel/emel.h"

namespace {

struct graph_storage {
  std::array<emel::buffer::allocator::event::tensor_desc, 1> nodes = {};
  std::array<emel::buffer::allocator::event::tensor_desc, 1> leafs = {};
  emel::buffer::allocator::event::graph_view view = {};
};

graph_storage make_valid_graph() {
  graph_storage g{};
  g.nodes[0].tensor_id = 0;
  g.nodes[0].alloc_size = 64;
  g.nodes[0].src_ids = {{-1, -1, -1, -1}};
  g.leafs[0].tensor_id = 1;
  g.leafs[0].alloc_size = 32;
  g.view = emel::buffer::allocator::event::graph_view{
    .nodes = g.nodes.data(),
    .n_nodes = 1,
    .leafs = g.leafs.data(),
    .n_leafs = 1,
  };
  return g;
}

}  // namespace

TEST_CASE("buffer_allocator_detail_valid_graph_tensors_rejects_invalid") {
  emel::buffer::allocator::event::graph_view bad_nodes{};
  bad_nodes.n_nodes = 1;
  CHECK_FALSE(emel::buffer::allocator::action::detail::valid_graph_tensors(bad_nodes));

  emel::buffer::allocator::event::graph_view too_many{};
  too_many.n_nodes = emel::buffer::allocator::action::k_max_graph_tensors + 1;
  CHECK_FALSE(emel::buffer::allocator::action::detail::valid_graph_tensors(too_many));

  emel::buffer::allocator::event::graph_view bad_leafs{};
  bad_leafs.n_leafs = 1;
  CHECK_FALSE(emel::buffer::allocator::action::detail::valid_graph_tensors(bad_leafs));

  graph_storage g = make_valid_graph();
  g.nodes[0].alloc_size = -1;
  CHECK_FALSE(emel::buffer::allocator::action::detail::valid_graph_tensors(g.view));

  graph_storage ok = make_valid_graph();
  CHECK(emel::buffer::allocator::action::detail::valid_graph_tensors(ok.view));
}

TEST_CASE("buffer_allocator_detail_build_tensor_alloc_validates_buffers") {
  emel::buffer::allocator::action::context ctx{};
  emel::buffer::allocator::action::tensor_alloc alloc{};
  emel::buffer::allocator::event::tensor_desc tensor{};
  tensor.tensor_id = 1;
  tensor.alloc_size = 16;

  CHECK_FALSE(emel::buffer::allocator::action::detail::build_tensor_alloc(alloc, ctx, tensor, 2, 1));
  CHECK_FALSE(emel::buffer::allocator::action::detail::build_tensor_alloc(alloc, ctx, tensor, -1, 1));

  tensor.has_external_data = true;
  CHECK(emel::buffer::allocator::action::detail::build_tensor_alloc(alloc, ctx, tensor, 0, 1));
  CHECK(alloc.buffer_id < 0);

  tensor.has_external_data = false;
  tensor.is_view = true;
  CHECK(emel::buffer::allocator::action::detail::build_tensor_alloc(alloc, ctx, tensor, 0, 1));
  CHECK(alloc.buffer_id < 0);

  tensor.has_external_data = false;
  tensor.is_view = false;
  CHECK(emel::buffer::allocator::action::detail::build_tensor_alloc(alloc, ctx, tensor, 0, 1));
  CHECK(alloc.buffer_id == 0);
}

TEST_CASE("buffer_allocator_detail_align_and_realloc_checks") {
  CHECK(emel::buffer::allocator::action::detail::align_up(0, 16) == 0);
  CHECK(emel::buffer::allocator::action::detail::align_up(1, 16) == 16);

  emel::buffer::allocator::action::context ctx{};
  emel::buffer::allocator::event::tensor_desc tensor{};
  tensor.tensor_id = 1;
  tensor.alloc_size = 32;

  emel::buffer::allocator::action::tensor_alloc alloc{};
  alloc.tensor_id = 1;
  alloc.buffer_id = -1;
  alloc.size_max = 0;

  tensor.has_external_data = true;
  CHECK_FALSE(emel::buffer::allocator::action::detail::tensor_needs_realloc(ctx, tensor, alloc));

  tensor.has_external_data = false;
  tensor.is_view = false;
  CHECK(emel::buffer::allocator::action::detail::tensor_needs_realloc(ctx, tensor, alloc));

  alloc.buffer_id = 0;
  alloc.size_max = 16;
  CHECK(emel::buffer::allocator::action::detail::tensor_needs_realloc(ctx, tensor, alloc));

  alloc.size_max = 32;
  CHECK_FALSE(emel::buffer::allocator::action::detail::tensor_needs_realloc(ctx, tensor, alloc));
}

TEST_CASE("buffer_allocator_detail_tensor_needs_realloc_handles_align_failure") {
  emel::buffer::allocator::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.buffer_alignments[0] = 16;

  emel::buffer::allocator::event::tensor_desc tensor{};
  tensor.tensor_id = 1;
  tensor.alloc_size = std::numeric_limits<int32_t>::max();

  emel::buffer::allocator::action::tensor_alloc alloc{};
  alloc.tensor_id = 1;
  alloc.buffer_id = 0;
  alloc.alignment = 16;

  CHECK(emel::buffer::allocator::action::detail::tensor_needs_realloc(ctx, tensor, alloc));
}

TEST_CASE("buffer_allocator_detail_graph_needs_realloc_paths") {
  emel::buffer::allocator::action::context ctx{};
  graph_storage g = make_valid_graph();

  ctx.has_reserve_snapshot = false;
  CHECK(emel::buffer::allocator::action::detail::graph_needs_realloc(g.view, ctx));

  ctx.has_reserve_snapshot = true;
  ctx.last_n_nodes = 0;
  ctx.last_n_leafs = 0;
  CHECK(emel::buffer::allocator::action::detail::graph_needs_realloc(g.view, ctx));

  ctx.last_n_nodes = 1;
  ctx.last_n_leafs = 1;
  ctx.buffer_count = 1;
  ctx.node_allocs[0].dst.tensor_id = g.nodes[0].tensor_id;
  ctx.node_allocs[0].dst.buffer_id = 0;
  ctx.node_allocs[0].dst.size_max =
    emel::buffer::allocator::action::detail::align_up(g.nodes[0].alloc_size, 16);
  ctx.node_allocs[0].src[0].tensor_id = g.nodes[0].src_ids[0];
  ctx.node_allocs[0].src[0].buffer_id = 0;
  ctx.node_allocs[0].src[0].size_max =
    emel::buffer::allocator::action::detail::align_up(g.leafs[0].alloc_size, 16);
  ctx.leaf_allocs[0].leaf.tensor_id = g.leafs[0].tensor_id;
  ctx.leaf_allocs[0].leaf.buffer_id = 0;
  ctx.leaf_allocs[0].leaf.size_max =
    emel::buffer::allocator::action::detail::align_up(g.leafs[0].alloc_size, 16);

  CHECK_FALSE(emel::buffer::allocator::action::detail::graph_needs_realloc(g.view, ctx));
}

TEST_CASE("buffer_allocator_detail_buffer_id_and_growth_checks") {
  graph_storage g = make_valid_graph();
  std::array<int32_t, 1> node_ids = {{0}};
  std::array<int32_t, 1> leaf_ids = {{0}};
  CHECK(emel::buffer::allocator::action::detail::valid_buffer_ids_for_graph(
    g.view, node_ids.data(), leaf_ids.data(), 1));
  node_ids[0] = 2;
  CHECK_FALSE(emel::buffer::allocator::action::detail::valid_buffer_ids_for_graph(
    g.view, node_ids.data(), leaf_ids.data(), 1));

  std::array<int32_t, emel::buffer::allocator::action::k_max_buffers> committed = {};
  std::array<int32_t, emel::buffer::allocator::action::k_max_buffers> required = {};
  committed[0] = 4;
  required[0] = 8;
  CHECK(emel::buffer::allocator::action::detail::requires_growth(committed, required, 1));
  required[0] = 4;
  CHECK_FALSE(emel::buffer::allocator::action::detail::requires_growth(committed, required, 1));
}

TEST_CASE("buffer_allocator_detail_chunk_binding_checks") {
  emel::buffer::allocator::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.committed_chunk_counts[0] = 0;
  ctx.committed_chunk_ids[0] = -1;
  ctx.committed_chunk_sizes[0] = 0;
  std::array<int32_t, emel::buffer::allocator::action::k_max_buffers> required = {};
  required[0] = 16;
  CHECK_FALSE(emel::buffer::allocator::action::detail::chunk_bindings_cover_required(
    ctx, required, 1));

  ctx.committed_chunk_counts[0] = 1;
  ctx.committed_chunk_ids[0] = 0;
  ctx.committed_chunk_sizes[0] = 32;
  CHECK(emel::buffer::allocator::action::detail::chunk_bindings_cover_required(
    ctx, required, 1));
}

TEST_CASE("buffer_allocator_detail_chunk_allocator_helpers") {
  emel::buffer::chunk_allocator::sm chunk_allocator{};
  int32_t err = EMEL_OK;

  CHECK(emel::buffer::allocator::action::detail::configure_chunk_allocator(&chunk_allocator, err));
  CHECK(err == EMEL_OK);

  emel::buffer::allocator::action::context ctx{};
  ctx.buffer_count = 1;
  std::array<int32_t, emel::buffer::allocator::action::k_max_buffers> required = {};
  required[0] = 16;
  err = EMEL_OK;
  CHECK(emel::buffer::allocator::action::detail::apply_required_sizes_to_chunks(
    ctx, required, nullptr, nullptr, &chunk_allocator, err));
  CHECK(err == EMEL_OK);

  err = EMEL_OK;
  CHECK(emel::buffer::allocator::action::detail::release_all_chunk_bindings(
    ctx, &chunk_allocator, err));
  CHECK(err == EMEL_OK);
}

TEST_CASE("buffer_allocator_detail_chunk_bindings_missing_ids") {
  emel::buffer::allocator::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.committed_chunk_counts[0] = 1;
  ctx.committed_chunk_ids[0] = -1;
  ctx.committed_chunk_sizes[0] = 32;

  std::array<int32_t, emel::buffer::allocator::action::k_max_buffers> required = {};
  required[0] = 16;
  CHECK_FALSE(emel::buffer::allocator::action::detail::chunk_bindings_cover_required(
    ctx, required, 1));
}

TEST_CASE("buffer_allocator_detail_apply_required_sizes_reports_invalid_plan") {
  namespace detail = emel::buffer::allocator::action::detail;

  emel::buffer::chunk_allocator::sm chunk_allocator{};
  std::array<int32_t, emel::buffer::allocator::action::k_max_buffers> required = {};
  required[0] = 64;
  int32_t err = EMEL_OK;

  {
    emel::buffer::allocator::action::context ctx{};
    ctx.buffer_count = 1;
    std::array<int32_t, 1> counts = {{-1}};
    std::array<int32_t, 1> sizes = {{64}};
    CHECK_FALSE(detail::apply_required_sizes_to_chunks(
      ctx, required, counts.data(), sizes.data(), &chunk_allocator, err));
    CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    emel::buffer::allocator::action::context ctx{};
    ctx.buffer_count = 1;
    std::array<int32_t, 1> counts = {{1}};
    err = EMEL_OK;
    CHECK_FALSE(detail::apply_required_sizes_to_chunks(
      ctx, required, counts.data(), nullptr, &chunk_allocator, err));
    CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    emel::buffer::allocator::action::context ctx{};
    ctx.buffer_count = 1;
    std::array<int32_t, 1> counts = {{1}};
    std::array<int32_t, 1> sizes = {{0}};
    err = EMEL_OK;
    CHECK_FALSE(detail::apply_required_sizes_to_chunks(
      ctx, required, counts.data(), sizes.data(), &chunk_allocator, err));
    CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    emel::buffer::allocator::action::context ctx{};
    ctx.buffer_count = 1;
    std::array<int32_t, 1> counts = {{1}};
    std::array<int32_t, 1> sizes = {{32}};
    err = EMEL_OK;
    CHECK_FALSE(detail::apply_required_sizes_to_chunks(
      ctx, required, counts.data(), sizes.data(), &chunk_allocator, err));
    CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  }
}

TEST_CASE("buffer_allocator_detail_apply_required_sizes_accepts_plans") {
  namespace detail = emel::buffer::allocator::action::detail;

  emel::buffer::allocator::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.buffer_alignments[0] = 16;
  ctx.buffer_max_sizes[0] = 16;

  emel::buffer::chunk_allocator::sm chunk_allocator{};
  int32_t err = EMEL_OK;
  CHECK(detail::configure_chunk_allocator(&chunk_allocator, err));

  std::array<int32_t, emel::buffer::allocator::action::k_max_buffers> required = {};
  required[0] = 48;
  std::array<int32_t, 1> counts = {{2}};
  std::array<int32_t, 2> sizes = {{16, 32}};

  err = EMEL_OK;
  CHECK(detail::apply_required_sizes_to_chunks(
    ctx, required, counts.data(), sizes.data(), &chunk_allocator, err));
  CHECK(err == EMEL_OK);
}

TEST_CASE("buffer_allocator_detail_release_all_chunk_bindings_reports_null") {
  emel::buffer::allocator::action::context ctx{};
  int32_t err = EMEL_OK;
  emel::buffer::chunk_allocator::sm chunk_allocator{};
  CHECK(emel::buffer::allocator::action::detail::release_all_chunk_bindings(
    ctx, &chunk_allocator, err));
  CHECK(err == EMEL_OK);
}

TEST_CASE("buffer_allocator_detail_release_all_chunk_bindings_skips_invalid") {
  emel::buffer::allocator::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.committed_chunk_counts[0] = 1;
  ctx.committed_chunk_ids[0] = -1;
  ctx.committed_chunk_sizes[0] = 0;
  emel::buffer::chunk_allocator::sm chunk_allocator{};
  int32_t err = EMEL_OK;
  CHECK(emel::buffer::allocator::action::detail::release_all_chunk_bindings(
    ctx, &chunk_allocator, err));
  CHECK(err == EMEL_OK);
}

TEST_CASE("buffer_allocator_begin_reserve_snapshot_failures") {
  emel::buffer::allocator::action::context ctx{};
  ctx.buffer_count = 1;
  emel::buffer::planner::sm planner{};
  emel::buffer::chunk_allocator::sm chunk_allocator{};

  graph_storage g = make_valid_graph();
  g.nodes[0].src_ids[0] = 999;
  std::array<int32_t, 1> node_ids = {{0}};
  std::array<int32_t, 1> leaf_ids = {{0}};
  std::array<int32_t, 1> sizes = {{0}};
  int32_t err = EMEL_OK;

  emel::buffer::allocator::action::begin_reserve_n_size(
    emel::buffer::allocator::event::reserve_n_size{
      .graph = g.view,
      .node_buffer_ids = node_ids.data(),
      .leaf_buffer_ids = leaf_ids.data(),
      .sizes_out = sizes.data(),
      .sizes_out_count = static_cast<int32_t>(sizes.size()),
      .error_out = &err,
    },
    ctx,
    planner);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  err = EMEL_OK;
  emel::buffer::allocator::action::begin_reserve_n(
    emel::buffer::allocator::event::reserve_n{
      .graph = g.view,
      .node_buffer_ids = node_ids.data(),
      .leaf_buffer_ids = leaf_ids.data(),
      .error_out = &err,
    },
    ctx,
    planner,
    chunk_allocator);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  err = EMEL_OK;
  emel::buffer::allocator::action::begin_reserve(
    emel::buffer::allocator::event::reserve{
      .graph = g.view,
      .error_out = &err,
    },
    ctx,
    planner,
    chunk_allocator);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_allocator_begin_reserve_reports_invalid_buffer_count") {
  emel::buffer::allocator::action::context ctx{};
  ctx.buffer_count = 0;
  graph_storage g = make_valid_graph();
  std::array<int32_t, 1> sizes = {{0}};
  CHECK_FALSE(emel::buffer::allocator::guard::can_reserve_n_size{}(
    emel::buffer::allocator::event::reserve_n_size{
      .graph = g.view,
      .sizes_out = sizes.data(),
      .sizes_out_count = static_cast<int32_t>(sizes.size()),
    },
    ctx));
  CHECK_FALSE(emel::buffer::allocator::guard::can_reserve_n{}(
    emel::buffer::allocator::event::reserve_n{
      .graph = g.view,
    },
    ctx));
  CHECK_FALSE(emel::buffer::allocator::guard::can_reserve{}(
    emel::buffer::allocator::event::reserve{
      .graph = g.view,
    },
    ctx));
  CHECK_FALSE(emel::buffer::allocator::guard::can_alloc_graph{}(
    emel::buffer::allocator::event::alloc_graph{
      .graph = g.view,
    },
    ctx));
}

TEST_CASE("buffer_allocator_detail_helpers_cover_edges") {
  using namespace emel::buffer::allocator::action::detail;

  CHECK(normalize_error(1, 2) == 1);
  CHECK(normalize_error(0, 2) == 2);
  CHECK(normalize_error(0, 0) == EMEL_ERR_BACKEND);

  CHECK(valid_alignment(16));
  CHECK_FALSE(valid_alignment(0));
  CHECK_FALSE(valid_alignment(3));

  CHECK(sanitize_alignment(16) == 16);
  CHECK(sanitize_alignment(3) == emel::buffer::allocator::action::k_default_alignment);

  CHECK(sanitize_max_size(0) == emel::buffer::allocator::action::k_default_max_size);
  CHECK(sanitize_max_size(64) == 64);

  emel::buffer::allocator::action::context ctx{};
  ctx.buffer_count = 1;
  ctx.buffer_alignments[0] = 3;
  ctx.buffer_max_sizes[0] = 0;
  CHECK(alignment_for_buffer(ctx, -1) == emel::buffer::allocator::action::k_default_alignment);
  CHECK(alignment_for_buffer(ctx, 0) == emel::buffer::allocator::action::k_default_alignment);
  CHECK(max_size_for_buffer(ctx, -1) == emel::buffer::allocator::action::k_default_max_size);
  CHECK(max_size_for_buffer(ctx, 0) == emel::buffer::allocator::action::k_default_max_size);

  int32_t out = -1;
  CHECK(align_up_checked(0, 16, out));
  CHECK(out == 0);
  CHECK(align_up_checked(-1, 16, out));
  CHECK(out == 0);
  CHECK_FALSE(align_up_checked(std::numeric_limits<int32_t>::max(), 16, out));

  CHECK(chunk_binding_index(2, 3) ==
        2 * emel::buffer::allocator::action::k_max_chunks_per_buffer + 3);
  CHECK(get_buffer_id(nullptr, 1) == 0);
}

TEST_CASE("buffer_allocator_detail_find_tensor_and_snapshot_failures") {
  graph_storage g = make_valid_graph();
  bool is_node = false;
  int32_t index = -1;

  CHECK(emel::buffer::allocator::action::detail::find_tensor(
    g.view, g.nodes[0].tensor_id, is_node, index) == &g.nodes[0]);
  CHECK(is_node);
  CHECK(index == 0);

  is_node = false;
  index = -1;
  CHECK(emel::buffer::allocator::action::detail::find_tensor(
    g.view, g.leafs[0].tensor_id, is_node, index) == &g.leafs[0]);
  CHECK_FALSE(is_node);
  CHECK(index == 0);

  is_node = false;
  index = -1;
  CHECK(emel::buffer::allocator::action::detail::find_tensor(
    g.view, 1234, is_node, index) == nullptr);

  emel::buffer::allocator::action::context ctx{};
  ctx.buffer_count = 1;

  emel::buffer::allocator::event::graph_view bad{};
  bad.n_nodes = 1;
  CHECK_FALSE(emel::buffer::allocator::action::detail::capture_alloc_snapshot(
    ctx, bad, nullptr, nullptr));

  graph_storage bad_src = make_valid_graph();
  bad_src.nodes[0].src_ids[0] = 999;
  CHECK_FALSE(emel::buffer::allocator::action::detail::capture_alloc_snapshot(
    ctx, bad_src.view, nullptr, nullptr));
}

TEST_CASE("buffer_allocator_begin_initialize_validates_buffer_count") {
  CHECK_FALSE(emel::buffer::allocator::guard::valid_initialize{}(
    emel::buffer::allocator::event::initialize{
      .buffer_count = 0,
    }));
  CHECK(emel::buffer::allocator::guard::valid_initialize{}(
    emel::buffer::allocator::event::initialize{
      .buffer_count = 1,
    }));
}

TEST_CASE("buffer_allocator_begin_reserve_n_size_validates_inputs") {
  emel::buffer::allocator::action::context ctx{};
  ctx.buffer_count = 1;
  graph_storage g = make_valid_graph();
  std::array<int32_t, 1> sizes = {{0}};

  CHECK_FALSE(emel::buffer::allocator::guard::can_reserve_n_size{}(
    emel::buffer::allocator::event::reserve_n_size{
      .graph = g.view,
      .sizes_out = nullptr,
      .sizes_out_count = 0,
    },
    ctx));
  CHECK(emel::buffer::allocator::guard::can_reserve_n_size{}(
    emel::buffer::allocator::event::reserve_n_size{
      .graph = g.view,
      .sizes_out = sizes.data(),
      .sizes_out_count = static_cast<int32_t>(sizes.size()),
    },
    ctx));
}

TEST_CASE("buffer_allocator_begin_reserve_n_rejects_invalid_buffer_ids") {
  emel::buffer::allocator::action::context ctx{};
  ctx.buffer_count = 1;
  graph_storage g = make_valid_graph();
  std::array<int32_t, 1> node_ids = {{1}};
  std::array<int32_t, 1> leaf_ids = {{0}};
  CHECK_FALSE(emel::buffer::allocator::guard::can_reserve_n{}(
    emel::buffer::allocator::event::reserve_n{
      .graph = g.view,
      .node_buffer_ids = node_ids.data(),
      .leaf_buffer_ids = leaf_ids.data(),
    },
    ctx));
}

TEST_CASE("buffer_allocator_begin_reserve_accepts_valid_graph") {
  emel::buffer::allocator::action::context ctx{};
  ctx.buffer_count = 1;
  emel::buffer::planner::sm planner{};
  emel::buffer::chunk_allocator::sm chunk_allocator{};
  graph_storage g = make_valid_graph();
  int32_t err = EMEL_OK;

  emel::buffer::allocator::action::begin_reserve(
    emel::buffer::allocator::event::reserve{
      .graph = g.view,
      .error_out = &err,
    },
    ctx,
    planner,
    chunk_allocator);
  CHECK(err == EMEL_OK);
}

TEST_CASE("buffer_allocator_begin_alloc_graph_reports_multi_buffer_realloc") {
  emel::buffer::allocator::action::context ctx{};
  ctx.buffer_count = 2;
  emel::buffer::planner::sm planner{};
  emel::buffer::chunk_allocator::sm chunk_allocator{};
  emel::buffer::realloc_analyzer::sm realloc_analyzer{};
  graph_storage g = make_valid_graph();
  int32_t err = EMEL_OK;

  emel::buffer::allocator::action::begin_alloc_graph(
    emel::buffer::allocator::event::alloc_graph{
      .graph = g.view,
      .error_out = &err,
    },
    ctx,
    planner,
    chunk_allocator,
    realloc_analyzer);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("buffer_allocator_begin_release_handles_empty_state") {
  emel::buffer::allocator::action::context ctx{};
  ctx.buffer_count = 1;
  emel::buffer::chunk_allocator::sm chunk_allocator{};
  int32_t err = EMEL_OK;

  emel::buffer::allocator::action::begin_release(
    emel::buffer::allocator::event::release{.error_out = &err},
    ctx,
    chunk_allocator);
  CHECK(err == EMEL_OK);
}

TEST_CASE("buffer_allocator_detail_capture_snapshot_success_path") {
  graph_storage g = make_valid_graph();
  g.nodes[0].src_ids[0] = g.leafs[0].tensor_id;

  emel::buffer::allocator::action::context ctx{};
  ctx.buffer_count = 1;
  std::array<int32_t, 1> node_ids = {{0}};
  std::array<int32_t, 1> leaf_ids = {{0}};

  CHECK(emel::buffer::allocator::action::detail::capture_alloc_snapshot(
    ctx, g.view, node_ids.data(), leaf_ids.data()));
  CHECK(ctx.has_reserve_snapshot);
}

TEST_CASE("buffer_allocator_sm_covers_error_and_success_paths") {
  emel::buffer::allocator::sm machine{};
  int32_t err = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::buffer::allocator::event::reserve{
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  emel::buffer::allocator::sm ok_machine{};
  std::array<int32_t, 1> alignments = {{16}};
  std::array<int32_t, 1> max_sizes = {{0}};
  err = EMEL_OK;
  CHECK(ok_machine.process_event(emel::buffer::allocator::event::initialize{
    .buffer_count = 1,
    .buffer_alignments = alignments.data(),
    .buffer_max_sizes = max_sizes.data(),
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  err = EMEL_OK;
  CHECK(ok_machine.process_event(emel::buffer::allocator::event::release{
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
}

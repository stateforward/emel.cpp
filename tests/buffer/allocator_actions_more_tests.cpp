#include <array>
#include <cstdint>

#include <doctest/doctest.h>

#include "emel/buffer/allocator/actions.hpp"
#include "emel/buffer/chunk_allocator/sm.hpp"
#include "emel/buffer/realloc_analyzer/sm.hpp"
#include "emel/emel.h"

namespace {

using tensor_desc = emel::buffer::allocator::event::tensor_desc;
using graph_view = emel::buffer::allocator::event::graph_view;

}  // namespace

TEST_CASE("buffer_allocator_actions_apply_required_sizes_replaces_existing_binding") {
  emel::buffer::allocator::action::context c{};
  c.buffer_count = 1;
  c.committed_chunk_counts.fill(0);
  c.committed_chunk_ids.fill(-1);
  c.committed_chunk_offsets.fill(0);
  c.committed_chunk_sizes.fill(0);
  c.committed_sizes.fill(0);

  emel::buffer::chunk_allocator::sm chunk_allocator{};
  int32_t err = EMEL_OK;
  CHECK(chunk_allocator.process_event(emel::buffer::chunk_allocator::event::configure{
    .alignment = 16,
    .max_chunk_size = 64,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  int32_t chunk = -1;
  uint64_t offset = 0;
  uint64_t aligned = 0;
  CHECK(chunk_allocator.process_event(emel::buffer::chunk_allocator::event::allocate{
    .size = 32,
    .chunk_out = &chunk,
    .offset_out = &offset,
    .aligned_size_out = &aligned,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  c.committed_chunk_ids[0] = chunk;
  c.committed_chunk_offsets[0] = offset;
  c.committed_chunk_sizes[0] = aligned;
  c.committed_chunk_counts[0] = 1;
  c.committed_sizes[0] = static_cast<int32_t>(aligned);

  std::array<int32_t, emel::buffer::allocator::action::k_max_buffers> required = {};
  required[0] = 64;
  CHECK(emel::buffer::allocator::action::detail::apply_required_sizes_to_chunks(
    c, required, nullptr, nullptr, &chunk_allocator, err));
  CHECK(err == EMEL_OK);
  CHECK(c.committed_chunk_sizes[0] >= 64);
}

TEST_CASE("buffer_allocator_actions_capture_snapshot_rejects_excessive_counts") {
  emel::buffer::allocator::action::context c{};
  c.buffer_count = 1;
  std::array<tensor_desc, 1> nodes = {{
    tensor_desc{
      .tensor_id = 1,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .is_input = false,
      .is_output = false,
      .has_external_data = false,
    },
  }};
  graph_view graph{
    .nodes = nodes.data(),
    .n_nodes = emel::buffer::allocator::action::k_max_graph_tensors + 1,
    .leafs = nullptr,
    .n_leafs = 0,
  };

  CHECK_FALSE(emel::buffer::allocator::action::detail::capture_alloc_snapshot(
    c, graph, nullptr, nullptr));
}

TEST_CASE("buffer_allocator_actions_valid_graph_rejects_negative_counts") {
  graph_view graph{
    .nodes = nullptr,
    .n_nodes = -1,
    .leafs = nullptr,
    .n_leafs = 0,
  };

  CHECK_FALSE(emel::buffer::allocator::action::detail::valid_graph_tensors(graph));
}

TEST_CASE("buffer_allocator_actions_realloc_analyzer_accepts_empty_graph") {
  emel::buffer::allocator::action::context c{};
  int32_t err = EMEL_OK;
  bool needs_realloc = false;

  graph_view graph{
    .nodes = nullptr,
    .n_nodes = 0,
    .leafs = nullptr,
    .n_leafs = 0,
  };

  emel::buffer::realloc_analyzer::sm analyzer{};
  CHECK(emel::buffer::allocator::action::detail::run_realloc_analyzer(
    &analyzer,
    graph,
    c,
    needs_realloc,
    err));
  CHECK(err == EMEL_OK);
}

#include <array>
#include <initializer_list>

#include "doctest/doctest.h"

#include "emel/buffer/allocator/sm.hpp"
#include "emel/emel.h"

namespace {

using tensor_desc = emel::buffer::allocator::event::tensor_desc;
using graph_view = emel::buffer::allocator::event::graph_view;

constexpr int32_t k_max_sources = emel::buffer::allocator::event::k_max_sources;

tensor_desc make_tensor(
    const int32_t tensor_id,
    const int32_t alloc_size,
    const bool is_input,
    const bool is_output,
    const bool is_view,
    const int32_t view_src_id,
    const bool can_inplace,
    const bool has_external_data,
    std::initializer_list<int32_t> srcs) {
  tensor_desc t{};
  t.tensor_id = tensor_id;
  t.alloc_size = alloc_size;
  t.is_input = is_input;
  t.is_output = is_output;
  t.is_view = is_view;
  t.view_src_id = view_src_id;
  t.can_inplace = can_inplace;
  t.has_external_data = has_external_data;
  t.src_ids = {{-1, -1, -1, -1}};
  int32_t idx = 0;
  for (const int32_t src : srcs) {
    if (idx >= k_max_sources) {
      break;
    }
    t.src_ids[idx++] = src;
  }
  return t;
}

}  // namespace

TEST_CASE("allocator_max_size_too_many_tensors") {
  emel::buffer::allocator::sm allocator;
  std::array<int32_t, 1> alignments = {{8}};
  std::array<int32_t, 1> max_sizes = {{16}};
  int32_t err = EMEL_OK;

  CHECK(allocator.process_event(emel::buffer::allocator::event::initialize{
    .buffer_count = 1,
    .buffer_alignments = alignments.data(),
    .buffer_max_sizes = max_sizes.data(),
    .error_out = &err,
  }));
  CHECK_EQ(err, EMEL_OK);

  std::array<tensor_desc, 3> leafs = {{
    make_tensor(0, 8, true, false, false, -1, false, false, {}),
    make_tensor(1, 8, true, false, false, -1, false, false, {}),
    make_tensor(2, 8, true, false, false, -1, false, false, {}),
  }};
  std::array<tensor_desc, 4> nodes = {{
    make_tensor(3, 8, false, false, false, -1, true, false, {0, 1}),
    make_tensor(4, 8, false, false, false, -1, true, false, {1, 2}),
    make_tensor(5, 8, false, false, false, -1, true, false, {3, 0}),
    make_tensor(6, 8, false, true, false, -1, true, false, {4, 5}),
  }};
  graph_view graph{
    .nodes = nodes.data(),
    .n_nodes = static_cast<int32_t>(nodes.size()),
    .leafs = leafs.data(),
    .n_leafs = static_cast<int32_t>(leafs.size()),
  };

  err = EMEL_OK;
  CHECK(allocator.process_event(emel::buffer::allocator::event::alloc_graph{
    .graph = graph,
    .error_out = &err,
  }));
  CHECK_EQ(err, EMEL_OK);
  CHECK(allocator.get_buffer_alloc_size(0) <= 32);
}

TEST_CASE("allocator_max_size_tensor_too_large") {
  emel::buffer::allocator::sm allocator;
  std::array<int32_t, 1> alignments = {{8}};
  std::array<int32_t, 1> max_sizes = {{16}};
  int32_t err = EMEL_OK;

  CHECK(allocator.process_event(emel::buffer::allocator::event::initialize{
    .buffer_count = 1,
    .buffer_alignments = alignments.data(),
    .buffer_max_sizes = max_sizes.data(),
    .error_out = &err,
  }));
  CHECK_EQ(err, EMEL_OK);

  std::array<tensor_desc, 1> leafs = {{
    make_tensor(0, 24, true, false, false, -1, false, false, {}),
  }};
  std::array<tensor_desc, 1> nodes = {{
    make_tensor(1, 24, false, true, false, -1, true, false, {0}),
  }};
  graph_view graph{
    .nodes = nodes.data(),
    .n_nodes = static_cast<int32_t>(nodes.size()),
    .leafs = leafs.data(),
    .n_leafs = static_cast<int32_t>(leafs.size()),
  };

  err = EMEL_OK;
  CHECK(allocator.process_event(emel::buffer::allocator::event::alloc_graph{
    .graph = graph,
    .error_out = &err,
  }));
  CHECK_EQ(err, EMEL_OK);
  CHECK_EQ(allocator.get_buffer_alloc_size(0), 24);
}

TEST_CASE("allocator_view_inplace") {
  emel::buffer::allocator::sm allocator;
  std::array<int32_t, 1> alignments = {{8}};
  std::array<int32_t, 1> max_sizes = {{32}};
  int32_t err = EMEL_OK;

  CHECK(allocator.process_event(emel::buffer::allocator::event::initialize{
    .buffer_count = 1,
    .buffer_alignments = alignments.data(),
    .buffer_max_sizes = max_sizes.data(),
    .error_out = &err,
  }));
  CHECK_EQ(err, EMEL_OK);

  std::array<tensor_desc, 2> leafs = {{
    make_tensor(0, 16, true, false, false, -1, false, false, {}),
    make_tensor(4, 8, true, false, false, -1, false, false, {}),
  }};
  std::array<tensor_desc, 4> nodes = {{
    make_tensor(1, 0, false, false, true, 0, false, false, {0}),
    make_tensor(2, 0, false, false, true, 0, false, false, {1}),
    make_tensor(3, 0, false, false, true, 0, false, false, {2}),
    make_tensor(5, 16, false, true, false, -1, true, false, {3, 4}),
  }};
  graph_view graph{
    .nodes = nodes.data(),
    .n_nodes = static_cast<int32_t>(nodes.size()),
    .leafs = leafs.data(),
    .n_leafs = static_cast<int32_t>(leafs.size()),
  };

  err = EMEL_OK;
  CHECK(allocator.process_event(emel::buffer::allocator::event::alloc_graph{
    .graph = graph,
    .error_out = &err,
  }));
  CHECK_EQ(err, EMEL_OK);
  CHECK(allocator.get_buffer_alloc_size(0) <= 24);
}

TEST_CASE("allocator_reuse_and_free") {
  emel::buffer::allocator::sm allocator;
  std::array<int32_t, 1> alignments = {{8}};
  std::array<int32_t, 1> max_sizes = {{40}};
  int32_t err = EMEL_OK;

  CHECK(allocator.process_event(emel::buffer::allocator::event::initialize{
    .buffer_count = 1,
    .buffer_alignments = alignments.data(),
    .buffer_max_sizes = max_sizes.data(),
    .error_out = &err,
  }));
  CHECK_EQ(err, EMEL_OK);

  std::array<tensor_desc, 3> leafs = {{
    make_tensor(0, 24, true, false, false, -1, false, false, {}),
    make_tensor(1, 8, true, false, false, -1, false, false, {}),
    make_tensor(2, 8, true, false, false, -1, false, false, {}),
  }};
  std::array<tensor_desc, 6> nodes = {{
    make_tensor(3, 8, false, false, false, -1, true, false, {1, 2}),
    make_tensor(4, 32, false, false, false, -1, false, false, {0}),
    make_tensor(5, 32, false, false, false, -1, true, false, {4}),
    make_tensor(6, 32, false, false, false, -1, true, false, {4, 5}),
    make_tensor(7, 0, false, false, true, 6, false, false, {6}),
    make_tensor(8, 8, false, true, false, -1, true, false, {3, 7}),
  }};
  graph_view graph{
    .nodes = nodes.data(),
    .n_nodes = static_cast<int32_t>(nodes.size()),
    .leafs = leafs.data(),
    .n_leafs = static_cast<int32_t>(leafs.size()),
  };

  err = EMEL_OK;
  CHECK(allocator.process_event(emel::buffer::allocator::event::alloc_graph{
    .graph = graph,
    .error_out = &err,
  }));
  CHECK_EQ(err, EMEL_OK);
  CHECK(allocator.get_buffer_alloc_size(0) <= 104);
}

TEST_CASE("allocator_merge_free_block") {
  emel::buffer::allocator::sm allocator;
  std::array<int32_t, 1> alignments = {{8}};
  std::array<int32_t, 1> max_sizes = {{32}};
  int32_t err = EMEL_OK;

  CHECK(allocator.process_event(emel::buffer::allocator::event::initialize{
    .buffer_count = 1,
    .buffer_alignments = alignments.data(),
    .buffer_max_sizes = max_sizes.data(),
    .error_out = &err,
  }));
  CHECK_EQ(err, EMEL_OK);

  std::array<tensor_desc, 3> leafs = {{
    make_tensor(0, 16, true, false, false, -1, false, false, {}),
    make_tensor(1, 16, true, false, false, -1, false, false, {}),
    make_tensor(2, 16, true, false, false, -1, false, false, {}),
  }};
  std::array<tensor_desc, 6> nodes = {{
    make_tensor(3, 4, false, false, false, -1, false, false, {0}),
    make_tensor(4, 4, false, false, false, -1, false, false, {1}),
    make_tensor(5, 24, false, false, false, -1, false, false, {2}),
    make_tensor(6, 4, false, false, false, -1, true, false, {3, 4}),
    make_tensor(7, 24, false, false, false, -1, false, false, {6}),
    make_tensor(8, 24, false, true, false, -1, true, false, {5, 7}),
  }};
  graph_view graph{
    .nodes = nodes.data(),
    .n_nodes = static_cast<int32_t>(nodes.size()),
    .leafs = leafs.data(),
    .n_leafs = static_cast<int32_t>(leafs.size()),
  };

  err = EMEL_OK;
  CHECK(allocator.process_event(emel::buffer::allocator::event::alloc_graph{
    .graph = graph,
    .error_out = &err,
  }));
  CHECK_EQ(err, EMEL_OK);
  CHECK(allocator.get_buffer_alloc_size(0) <= 88);
}

TEST_CASE("allocator_prefer_already_allocated_memory") {
  emel::buffer::allocator::sm allocator;
  std::array<int32_t, 1> alignments = {{4}};
  std::array<int32_t, 1> max_sizes = {{32}};
  int32_t err = EMEL_OK;

  CHECK(allocator.process_event(emel::buffer::allocator::event::initialize{
    .buffer_count = 1,
    .buffer_alignments = alignments.data(),
    .buffer_max_sizes = max_sizes.data(),
    .error_out = &err,
  }));
  CHECK_EQ(err, EMEL_OK);

  std::array<tensor_desc, 1> leafs = {{
    make_tensor(0, 24, true, false, false, -1, false, false, {}),
  }};
  std::array<tensor_desc, 2> nodes = {{
    make_tensor(1, 4, false, false, false, -1, false, false, {0}),
    make_tensor(2, 4, false, true, false, -1, false, false, {1}),
  }};
  graph_view graph{
    .nodes = nodes.data(),
    .n_nodes = static_cast<int32_t>(nodes.size()),
    .leafs = leafs.data(),
    .n_leafs = static_cast<int32_t>(leafs.size()),
  };

  err = EMEL_OK;
  CHECK(allocator.process_event(emel::buffer::allocator::event::alloc_graph{
    .graph = graph,
    .error_out = &err,
  }));
  CHECK_EQ(err, EMEL_OK);
  CHECK(allocator.get_buffer_alloc_size(0) <= 28);
}

TEST_CASE("allocator_multiple_buffers") {
  emel::buffer::allocator::sm allocator;
  std::array<int32_t, 2> alignments = {{8, 8}};
  std::array<int32_t, 2> max_sizes = {{32, 64}};
  int32_t err = EMEL_OK;

  CHECK(allocator.process_event(emel::buffer::allocator::event::initialize{
    .buffer_count = 2,
    .buffer_alignments = alignments.data(),
    .buffer_max_sizes = max_sizes.data(),
    .error_out = &err,
  }));
  CHECK_EQ(err, EMEL_OK);

  std::array<tensor_desc, 5> leafs = {{
    make_tensor(0, 16, true, false, false, -1, false, false, {}),
    make_tensor(1, 16, true, false, false, -1, false, false, {}),
    make_tensor(2, 24, true, false, false, -1, false, false, {}),
    make_tensor(3, 4, true, false, false, -1, false, false, {}),
    make_tensor(4, 16, true, false, false, -1, false, false, {}),
  }};
  std::array<tensor_desc, 8> nodes = {{
    make_tensor(5, 16, false, false, false, -1, true, false, {4, 0}),
    make_tensor(6, 24, false, false, false, -1, false, false, {5}),
    make_tensor(7, 24, false, false, false, -1, false, false, {6, 2}),
    make_tensor(8, 4, false, false, false, -1, false, false, {7}),
    make_tensor(9, 4, false, false, false, -1, false, false, {8, 3}),
    make_tensor(10, 24, false, false, false, -1, false, false, {9}),
    make_tensor(11, 16, false, false, false, -1, true, false, {10, 1}),
    make_tensor(12, 16, false, true, false, -1, true, false, {11}),
  }};
  graph_view graph{
    .nodes = nodes.data(),
    .n_nodes = static_cast<int32_t>(nodes.size()),
    .leafs = leafs.data(),
    .n_leafs = static_cast<int32_t>(leafs.size()),
  };

  int32_t leaf_buffer_ids[5] = {0, 0, 1, 1, 0};
  int32_t node_buffer_ids[8] = {0, 0, 1, 1, 1, 1, 0, 0};

  err = EMEL_OK;
  CHECK(allocator.process_event(emel::buffer::allocator::event::reserve_n{
    .graph = graph,
    .node_buffer_ids = node_buffer_ids,
    .leaf_buffer_ids = leaf_buffer_ids,
    .error_out = &err,
  }));
  CHECK_EQ(err, EMEL_OK);

  err = EMEL_OK;
  CHECK(allocator.process_event(emel::buffer::allocator::event::alloc_graph{
    .graph = graph,
    .error_out = &err,
  }));
  CHECK_EQ(err, EMEL_OK);

  CHECK(allocator.get_buffer_alloc_size(0) > 0);
  CHECK(allocator.get_buffer_alloc_size(1) > 0);
  CHECK(allocator.get_buffer_alloc_size(0) <= 96);
  CHECK(allocator.get_buffer_alloc_size(1) <= 64);
}

TEST_CASE("allocator_buffer_size_zero") {
  emel::buffer::allocator::sm allocator;
  std::array<int32_t, 2> alignments = {{8, 8}};
  std::array<int32_t, 2> max_sizes = {{32, 32}};
  int32_t err = EMEL_OK;

  CHECK(allocator.process_event(emel::buffer::allocator::event::initialize{
    .buffer_count = 2,
    .buffer_alignments = alignments.data(),
    .buffer_max_sizes = max_sizes.data(),
    .error_out = &err,
  }));
  CHECK_EQ(err, EMEL_OK);

  std::array<tensor_desc, 1> leafs = {{
    make_tensor(0, 16, true, false, false, -1, false, false, {}),
  }};
  std::array<tensor_desc, 1> nodes = {{
    make_tensor(1, 16, false, true, false, -1, true, false, {0}),
  }};
  graph_view graph{
    .nodes = nodes.data(),
    .n_nodes = static_cast<int32_t>(nodes.size()),
    .leafs = leafs.data(),
    .n_leafs = static_cast<int32_t>(leafs.size()),
  };

  int32_t leaf_buffer_ids[1] = {0};
  int32_t node_buffer_ids[1] = {0};

  err = EMEL_OK;
  CHECK(allocator.process_event(emel::buffer::allocator::event::reserve_n{
    .graph = graph,
    .node_buffer_ids = node_buffer_ids,
    .leaf_buffer_ids = leaf_buffer_ids,
    .error_out = &err,
  }));
  CHECK_EQ(err, EMEL_OK);

  err = EMEL_OK;
  CHECK(allocator.process_event(emel::buffer::allocator::event::alloc_graph{
    .graph = graph,
    .error_out = &err,
  }));
  CHECK_EQ(err, EMEL_OK);

  CHECK_EQ(allocator.get_buffer_alloc_size(1), 0);
}

TEST_CASE("allocator_reallocation") {
  emel::buffer::allocator::sm allocator;
  std::array<int32_t, 1> alignments = {{4}};
  std::array<int32_t, 1> max_sizes = {{32}};
  int32_t err = EMEL_OK;

  CHECK(allocator.process_event(emel::buffer::allocator::event::initialize{
    .buffer_count = 1,
    .buffer_alignments = alignments.data(),
    .buffer_max_sizes = max_sizes.data(),
    .error_out = &err,
  }));
  CHECK_EQ(err, EMEL_OK);

  {
    std::array<tensor_desc, 2> leafs = {{
      make_tensor(0, 24, true, false, false, -1, false, false, {}),
      make_tensor(1, 16, true, false, false, -1, false, false, {}),
    }};
    std::array<tensor_desc, 2> nodes = {{
      make_tensor(2, 0, false, false, true, 0, false, false, {0}),
      make_tensor(3, 16, false, true, false, -1, true, false, {2, 1}),
    }};
    graph_view graph{
      .nodes = nodes.data(),
      .n_nodes = static_cast<int32_t>(nodes.size()),
      .leafs = leafs.data(),
      .n_leafs = static_cast<int32_t>(leafs.size()),
    };

    err = EMEL_OK;
    CHECK(allocator.process_event(emel::buffer::allocator::event::alloc_graph{
      .graph = graph,
      .error_out = &err,
    }));
    CHECK_EQ(err, EMEL_OK);
  }

  const uint64_t first_size = allocator.get_buffer_alloc_size(0);

  {
    std::array<tensor_desc, 2> leafs = {{
      make_tensor(0, 20, true, false, false, -1, false, false, {}),
      make_tensor(1, 20, true, false, false, -1, false, false, {}),
    }};
    std::array<tensor_desc, 1> nodes = {{
      make_tensor(2, 20, false, true, false, -1, true, false, {0, 1}),
    }};
    graph_view graph{
      .nodes = nodes.data(),
      .n_nodes = static_cast<int32_t>(nodes.size()),
      .leafs = leafs.data(),
      .n_leafs = static_cast<int32_t>(leafs.size()),
    };

    err = EMEL_OK;
    CHECK(allocator.process_event(emel::buffer::allocator::event::alloc_graph{
      .graph = graph,
      .error_out = &err,
    }));
    CHECK_EQ(err, EMEL_OK);
  }

  CHECK(allocator.get_buffer_alloc_size(0) <= first_size);
}

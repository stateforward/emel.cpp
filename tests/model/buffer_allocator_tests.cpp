#include <array>
#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/buffer_allocator/sm.hpp"
#include "emel/emel.h"

namespace {

using tensor_desc = emel::buffer_allocator::event::tensor_desc;
using graph_view = emel::buffer_allocator::event::graph_view;

struct graph_storage {
  std::array<tensor_desc, 3> nodes = {};
  std::array<tensor_desc, 2> leafs = {};
  int32_t n_nodes = 0;
  int32_t n_leafs = 0;
};

graph_view as_view(const graph_storage & g) {
  return graph_view{
    .nodes = g.nodes.data(),
    .n_nodes = g.n_nodes,
    .leafs = g.leafs.data(),
    .n_leafs = g.n_leafs,
  };
}

void initialize_ready(emel::buffer_allocator::sm & machine, const int32_t buffers = 1) {
  CHECK(machine.process_event(emel::buffer_allocator::event::initialize{
    .buffer_count = buffers,
  }));
}

graph_storage make_chain_graph(
    const int32_t leaf_size, const int32_t node0_size, const int32_t node1_size) {
  graph_storage g{};
  g.n_leafs = 1;
  g.n_nodes = 2;
  g.leafs[0] = tensor_desc{
    .tensor_id = 10,
    .alloc_size = leaf_size,
    .src_ids = {{-1, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = true,
    .is_output = false,
    .has_external_data = false,
  };
  g.nodes[0] = tensor_desc{
    .tensor_id = 20,
    .alloc_size = node0_size,
    .src_ids = {{10, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = false,
    .has_external_data = false,
  };
  g.nodes[1] = tensor_desc{
    .tensor_id = 21,
    .alloc_size = node1_size,
    .src_ids = {{20, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = true,
    .has_external_data = false,
  };
  return g;
}

graph_storage make_view_graph() {
  graph_storage g{};
  g.n_leafs = 1;
  g.n_nodes = 2;
  g.leafs[0] = tensor_desc{
    .tensor_id = 30,
    .alloc_size = 256,
    .src_ids = {{-1, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = true,
    .is_output = false,
    .has_external_data = false,
  };
  g.nodes[0] = tensor_desc{
    .tensor_id = 31,
    .alloc_size = 0,
    .src_ids = {{30, -1, -1, -1}},
    .is_view = true,
    .view_src_id = 30,
    .is_input = false,
    .is_output = false,
    .has_external_data = false,
  };
  g.nodes[1] = tensor_desc{
    .tensor_id = 32,
    .alloc_size = 256,
    .src_ids = {{31, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = true,
    .has_external_data = false,
  };
  return g;
}

graph_storage make_direct_graph() {
  graph_storage g{};
  g.n_leafs = 1;
  g.n_nodes = 1;
  g.leafs[0] = tensor_desc{
    .tensor_id = 33,
    .alloc_size = 256,
    .src_ids = {{-1, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = true,
    .is_output = false,
    .has_external_data = false,
  };
  g.nodes[0] = tensor_desc{
    .tensor_id = 34,
    .alloc_size = 256,
    .src_ids = {{33, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = true,
    .has_external_data = false,
  };
  return g;
}

graph_storage make_invalid_source_graph() {
  graph_storage g{};
  g.n_leafs = 1;
  g.n_nodes = 1;
  g.leafs[0] = tensor_desc{
    .tensor_id = 40,
    .alloc_size = 64,
    .src_ids = {{-1, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = true,
    .is_output = false,
    .has_external_data = false,
  };
  g.nodes[0] = tensor_desc{
    .tensor_id = 41,
    .alloc_size = 64,
    .src_ids = {{999, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = true,
    .has_external_data = false,
  };
  return g;
}

graph_storage make_external_input_graph() {
  graph_storage g{};
  g.n_leafs = 1;
  g.n_nodes = 1;
  g.leafs[0] = tensor_desc{
    .tensor_id = 50,
    .alloc_size = 2048,
    .src_ids = {{-1, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = true,
    .is_output = false,
    .has_external_data = true,
  };
  g.nodes[0] = tensor_desc{
    .tensor_id = 51,
    .alloc_size = 128,
    .src_ids = {{50, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = true,
    .has_external_data = false,
  };
  return g;
}

}  // namespace

TEST_CASE("buffer_allocator_starts_uninitialized") {
  emel::buffer_allocator::sm machine{};
  int state_count = 0;
  machine.visit_current_states([&](auto) { state_count += 1; });
  CHECK(state_count == 1);
}

TEST_CASE("buffer_allocator_rejects_invalid_initialize") {
  emel::buffer_allocator::sm machine{};
  CHECK_FALSE(machine.process_event(emel::buffer_allocator::event::initialize{
    .buffer_count = 0,
  }));
}

TEST_CASE("buffer_allocator_reserve_n_size_and_commit_reserve") {
  emel::buffer_allocator::sm machine{};
  initialize_ready(machine, 1);
  const graph_storage g = make_chain_graph(256, 512, 768);

  std::array<int32_t, 1> sizes = {{0}};
  CHECK(machine.process_event(emel::buffer_allocator::event::reserve_n_size{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
  }));
  CHECK(sizes[0] > 0);
  CHECK(machine.get_buffer_size(0) == 0);

  CHECK(machine.process_event(emel::buffer_allocator::event::reserve{
    .graph = as_view(g),
  }));
  CHECK(machine.get_buffer_size(0) == sizes[0]);
}

TEST_CASE("buffer_allocator_reuses_parent_storage_for_inplace_chain") {
  emel::buffer_allocator::sm machine{};
  initialize_ready(machine, 1);
  const graph_storage g = make_chain_graph(512, 512, 256);

  std::array<int32_t, 1> sizes = {{0}};
  CHECK(machine.process_event(emel::buffer_allocator::event::reserve_n_size{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
  }));
  CHECK(sizes[0] >= 512);
}

TEST_CASE("buffer_allocator_keeps_view_source_live_until_view_consumed") {
  emel::buffer_allocator::sm machine{};
  initialize_ready(machine, 1);

  const graph_storage view_graph = make_view_graph();
  const graph_storage direct_graph = make_direct_graph();

  std::array<int32_t, 1> view_sizes = {{0}};
  std::array<int32_t, 1> direct_sizes = {{0}};

  CHECK(machine.process_event(emel::buffer_allocator::event::reserve_n_size{
    .graph = as_view(view_graph),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .sizes_out = view_sizes.data(),
    .sizes_out_count = static_cast<int32_t>(view_sizes.size()),
  }));
  CHECK(machine.process_event(emel::buffer_allocator::event::reserve_n_size{
    .graph = as_view(direct_graph),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .sizes_out = direct_sizes.data(),
    .sizes_out_count = static_cast<int32_t>(direct_sizes.size()),
  }));

  CHECK(view_sizes[0] >= direct_sizes[0]);
}

TEST_CASE("buffer_allocator_auto_reallocates_single_buffer_on_alloc_graph") {
  emel::buffer_allocator::sm machine{};
  initialize_ready(machine, 1);

  const graph_storage small = make_chain_graph(64, 128, 128);
  const graph_storage large = make_chain_graph(64, 1024, 2048);

  CHECK(machine.process_event(emel::buffer_allocator::event::reserve{
    .graph = as_view(small),
  }));
  const int32_t before = machine.get_buffer_size(0);
  CHECK(before > 0);

  CHECK(machine.process_event(emel::buffer_allocator::event::alloc_graph{
    .graph = as_view(small),
  }));
  CHECK(machine.process_event(emel::buffer_allocator::event::alloc_graph{
    .graph = as_view(large),
  }));
  CHECK(machine.get_buffer_size(0) > before);
}

TEST_CASE("buffer_allocator_multi_buffer_alloc_graph_requires_explicit_reserve") {
  emel::buffer_allocator::sm machine{};
  initialize_ready(machine, 2);

  const graph_storage small = make_chain_graph(64, 128, 128);
  const graph_storage large = make_chain_graph(64, 1024, 2048);
  const std::array<int32_t, 2> node_ids = {{0, 1}};
  const std::array<int32_t, 1> leaf_ids = {{0}};

  CHECK(machine.process_event(emel::buffer_allocator::event::reserve_n{
    .graph = as_view(small),
    .node_buffer_ids = node_ids.data(),
    .leaf_buffer_ids = leaf_ids.data(),
  }));
  CHECK(machine.process_event(emel::buffer_allocator::event::alloc_graph{
    .graph = as_view(small),
  }));

  CHECK_FALSE(machine.process_event(emel::buffer_allocator::event::alloc_graph{
    .graph = as_view(large),
  }));
  CHECK(machine.last_error() == EMEL_ERR_BACKEND);
}

TEST_CASE("buffer_allocator_multi_buffer_alloc_graph_rejects_unreserved_shape") {
  emel::buffer_allocator::sm machine{};
  initialize_ready(machine, 2);
  const graph_storage g = make_chain_graph(64, 128, 128);

  CHECK_FALSE(machine.process_event(emel::buffer_allocator::event::alloc_graph{
    .graph = as_view(g),
  }));
  CHECK(machine.last_error() == EMEL_ERR_BACKEND);
}

TEST_CASE("buffer_allocator_reserve_n_size_tracks_per_buffer_assignments") {
  emel::buffer_allocator::sm machine{};
  initialize_ready(machine, 2);

  const graph_storage g = make_chain_graph(128, 256, 512);
  const std::array<int32_t, 2> node_ids = {{0, 1}};
  const std::array<int32_t, 1> leaf_ids = {{1}};
  std::array<int32_t, 2> sizes = {{0, 0}};

  CHECK(machine.process_event(emel::buffer_allocator::event::reserve_n_size{
    .graph = as_view(g),
    .node_buffer_ids = node_ids.data(),
    .leaf_buffer_ids = leaf_ids.data(),
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
  }));

  CHECK(sizes[0] > 0);
  CHECK(sizes[1] > 0);
}

TEST_CASE("buffer_allocator_invalid_source_reports_error") {
  emel::buffer_allocator::sm machine{};
  initialize_ready(machine, 1);
  const graph_storage g = make_invalid_source_graph();

  CHECK_FALSE(machine.process_event(emel::buffer_allocator::event::reserve{
    .graph = as_view(g),
  }));
  CHECK(machine.last_error() == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_allocator_external_input_is_not_reserved") {
  emel::buffer_allocator::sm machine{};
  initialize_ready(machine, 1);
  const graph_storage g = make_external_input_graph();
  std::array<int32_t, 1> sizes = {{0}};

  CHECK(machine.process_event(emel::buffer_allocator::event::reserve_n_size{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
  }));
  CHECK(sizes[0] == 128);
}

TEST_CASE("buffer_allocator_release_resets_allocator") {
  emel::buffer_allocator::sm machine{};
  initialize_ready(machine, 1);
  const graph_storage g = make_chain_graph(128, 256, 256);

  CHECK(machine.process_event(emel::buffer_allocator::event::reserve{
    .graph = as_view(g),
  }));
  CHECK(machine.get_buffer_size(0) > 0);
  CHECK(machine.process_event(emel::buffer_allocator::event::release{}));
  CHECK(machine.get_buffer_size(0) == 0);
}

TEST_CASE("buffer_allocator_reserve_n_size_rejects_output_count_mismatch") {
  emel::buffer_allocator::sm machine{};
  initialize_ready(machine, 2);
  const graph_storage g = make_chain_graph(128, 256, 512);
  std::array<int32_t, 1> sizes = {{0}};

  CHECK_FALSE(machine.process_event(emel::buffer_allocator::event::reserve_n_size{
    .graph = as_view(g),
    .node_buffer_ids = nullptr,
    .leaf_buffer_ids = nullptr,
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
  }));
}

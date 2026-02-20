#include <array>
#include <cstdint>

#include "bench/bench_registry.hpp"
#include "emel/buffer/allocator/events.hpp"
#include "emel/buffer/allocator/sm.hpp"
#include "emel/emel.h"

namespace {

using tensor_desc = emel::buffer::allocator::event::tensor_desc;
using graph_view = emel::buffer::allocator::event::graph_view;

struct graph_storage {
  std::array<tensor_desc, 1> nodes = {};
  std::array<tensor_desc, 1> leafs = {};
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

graph_storage make_graph() {
  graph_storage g{};
  g.n_leafs = 1;
  g.n_nodes = 1;
  g.leafs[0] = tensor_desc{
    .tensor_id = 10,
    .alloc_size = 256,
    .src_ids = emel::buffer::allocator::event::make_src_ids(),
    .is_view = false,
    .view_src_id = -1,
    .is_input = true,
    .is_output = false,
    .has_external_data = false,
  };
  auto src_ids = emel::buffer::allocator::event::make_src_ids();
  src_ids[0] = 10;
  g.nodes[0] = tensor_desc{
    .tensor_id = 20,
    .alloc_size = 256,
    .src_ids = src_ids,
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = true,
    .has_external_data = false,
  };
  return g;
}

void run_allocator_reserve_only(std::uint64_t iterations) {
  emel::buffer::allocator::sm machine{};
  graph_storage g = make_graph();
  const std::array<int32_t, 1> node_ids = {{0}};
  const std::array<int32_t, 1> leaf_ids = {{0}};

  (void)machine.process_event(emel::buffer::allocator::event::initialize{
    .buffer_count = 1,
  });

  for (std::uint64_t i = 0; i < iterations; ++i) {
    (void)machine.process_event(emel::buffer::allocator::event::reserve_n{
      .graph = as_view(g),
      .node_buffer_ids = node_ids.data(),
      .leaf_buffer_ids = leaf_ids.data(),
    });
  }

  (void)machine.process_event(emel::buffer::allocator::event::release{});
}

void run_allocator_alloc_only(std::uint64_t iterations) {
  emel::buffer::allocator::sm machine{};
  graph_storage g = make_graph();
  const std::array<int32_t, 1> node_ids = {{0}};
  const std::array<int32_t, 1> leaf_ids = {{0}};

  (void)machine.process_event(emel::buffer::allocator::event::initialize{
    .buffer_count = 1,
  });
  (void)machine.process_event(emel::buffer::allocator::event::reserve_n{
    .graph = as_view(g),
    .node_buffer_ids = node_ids.data(),
    .leaf_buffer_ids = leaf_ids.data(),
  });

  for (std::uint64_t i = 0; i < iterations; ++i) {
    (void)machine.process_event(emel::buffer::allocator::event::alloc_graph{
      .graph = as_view(g),
    });
  }

  (void)machine.process_event(emel::buffer::allocator::event::release{});
}

void run_allocator_full(std::uint64_t iterations) {
  emel::buffer::allocator::sm machine{};
  graph_storage g = make_graph();
  const std::array<int32_t, 1> node_ids = {{0}};
  const std::array<int32_t, 1> leaf_ids = {{0}};

  (void)machine.process_event(emel::buffer::allocator::event::initialize{
    .buffer_count = 1,
  });

  for (std::uint64_t i = 0; i < iterations; ++i) {
    (void)machine.process_event(emel::buffer::allocator::event::reserve_n{
      .graph = as_view(g),
      .node_buffer_ids = node_ids.data(),
      .leaf_buffer_ids = leaf_ids.data(),
    });
    (void)machine.process_event(emel::buffer::allocator::event::alloc_graph{
      .graph = as_view(g),
    });
  }

  (void)machine.process_event(emel::buffer::allocator::event::release{});
}

const bool k_registered_reserve =
  emel::bench::register_case({"buffer/allocator_reserve_n", &run_allocator_reserve_only});
const bool k_registered_alloc =
  emel::bench::register_case({"buffer/allocator_alloc_graph", &run_allocator_alloc_only});
const bool k_registered_full =
  emel::bench::register_case({"buffer/allocator_full", &run_allocator_full});

}  // namespace

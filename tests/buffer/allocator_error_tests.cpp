#include <array>
#include <cstdint>

#include <doctest/doctest.h>

#include "emel/buffer/allocator/sm.hpp"
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

graph_view as_view(const graph_storage & g) {
  return graph_view{
    .nodes = g.nodes.data(),
    .n_nodes = g.n_nodes,
    .leafs = g.leafs.data(),
    .n_leafs = g.n_leafs,
  };
}

graph_storage make_direct_graph() {
  graph_storage g{};
  g.leafs[0] = tensor_desc{
    .tensor_id = 5,
    .alloc_size = 64,
    .src_ids = {{-1, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = true,
    .is_output = false,
    .has_external_data = false,
  };
  g.nodes[0] = tensor_desc{
    .tensor_id = 6,
    .alloc_size = 64,
    .src_ids = {{5, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = true,
    .has_external_data = false,
  };
  return g;
}

}  // namespace

TEST_CASE("buffer_allocator_initialize_rejects_invalid_buffer_count") {
  emel::buffer::allocator::sm machine{};
  int32_t error = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::buffer::allocator::event::initialize{
    .buffer_count = 0,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_allocator_reserve_before_initialize_fails") {
  emel::buffer::allocator::sm machine{};
  int32_t error = EMEL_OK;
  const graph_storage g = make_direct_graph();

  CHECK_FALSE(machine.process_event(emel::buffer::allocator::event::reserve{
    .graph = as_view(g),
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_allocator_alloc_graph_before_initialize_fails") {
  emel::buffer::allocator::sm machine{};
  int32_t error = EMEL_OK;
  const graph_storage g = make_direct_graph();

  CHECK_FALSE(machine.process_event(emel::buffer::allocator::event::alloc_graph{
    .graph = as_view(g),
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_allocator_initialize_twice_reports_backend_error") {
  emel::buffer::allocator::sm machine{};
  int32_t error = EMEL_OK;

  CHECK(machine.process_event(emel::buffer::allocator::event::initialize{
    .buffer_count = 1,
    .error_out = &error,
  }));
  CHECK(error == EMEL_OK);

  CHECK_FALSE(machine.process_event(emel::buffer::allocator::event::initialize{
    .buffer_count = 1,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_allocator_reserve_n_size_before_initialize_fails") {
  emel::buffer::allocator::sm machine{};
  int32_t error = EMEL_OK;
  const graph_storage g = make_direct_graph();
  std::array<int32_t, 1> sizes = {};

  CHECK_FALSE(machine.process_event(emel::buffer::allocator::event::reserve_n_size{
    .graph = as_view(g),
    .sizes_out = sizes.data(),
    .sizes_out_count = static_cast<int32_t>(sizes.size()),
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_allocator_reserve_n_before_initialize_fails") {
  emel::buffer::allocator::sm machine{};
  int32_t error = EMEL_OK;
  const graph_storage g = make_direct_graph();

  CHECK_FALSE(machine.process_event(emel::buffer::allocator::event::reserve_n{
    .graph = as_view(g),
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
}

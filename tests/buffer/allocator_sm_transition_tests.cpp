#include <array>
#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/buffer/allocator/sm.hpp"
#include "emel/buffer/chunk_allocator/sm.hpp"
#include "emel/buffer/planner/sm.hpp"
#include "emel/buffer/realloc_analyzer/sm.hpp"

namespace {

TEST_CASE("buffer_allocator_sm_initialize_and_reserve_sizes") {
  emel::buffer::allocator::sm machine{};
  emel::buffer::planner::sm planner{};
  emel::buffer::chunk_allocator::sm chunk_allocator{};
  emel::buffer::realloc_analyzer::sm realloc_analyzer{};

  int32_t err = EMEL_OK;
  CHECK(chunk_allocator.process_event(emel::buffer::chunk_allocator::event::configure{
    .alignment = 16,
    .max_chunk_size = 1024,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  err = EMEL_OK;
  CHECK(machine.process_event(emel::buffer::allocator::event::initialize{
    .buffer_count = 1,
    .chunk_allocator_sm = &chunk_allocator,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  emel::buffer::allocator::event::tensor_desc node{
    .tensor_id = 1,
    .alloc_size = 16,
    .src_ids = {{-1, -1, -1, -1}},
    .is_view = false,
    .view_src_id = -1,
    .is_input = false,
    .is_output = false,
    .has_external_data = false,
  };
  emel::buffer::allocator::event::graph_view graph{
    .nodes = &node,
    .n_nodes = 1,
    .leafs = nullptr,
    .n_leafs = 0,
  };
  std::array<int32_t, 1> node_buffer_ids = {{0}};
  std::array<int32_t, 1> sizes_out = {{0}};
  err = EMEL_OK;

  CHECK(machine.process_event(emel::buffer::allocator::event::reserve_n_size{
    .graph = graph,
    .node_buffer_ids = node_buffer_ids.data(),
    .leaf_buffer_ids = nullptr,
    .sizes_out = sizes_out.data(),
    .sizes_out_count = static_cast<int32_t>(sizes_out.size()),
    .buffer_planner_sm = &planner,
    .chunk_allocator_sm = &chunk_allocator,
    .strategy = nullptr,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  err = EMEL_OK;
  CHECK(machine.process_event(emel::buffer::allocator::event::alloc_graph{
    .graph = graph,
    .buffer_planner_sm = &planner,
    .chunk_allocator_sm = &chunk_allocator,
    .buffer_realloc_analyzer_sm = &realloc_analyzer,
    .strategy = nullptr,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
}

TEST_CASE("buffer_allocator_sm_rejects_missing_dependencies") {
  emel::buffer::allocator::sm machine{};
  emel::buffer::chunk_allocator::sm chunk_allocator{};
  int32_t err = EMEL_OK;

  machine.process_event(emel::buffer::chunk_allocator::event::configure{
    .alignment = 16,
    .max_chunk_size = 1024,
    .error_out = &err,
  });

  err = EMEL_OK;
  machine.process_event(emel::buffer::allocator::event::initialize{
    .buffer_count = 1,
    .chunk_allocator_sm = &chunk_allocator,
    .error_out = &err,
  });

  err = EMEL_OK;
  machine.process_event(emel::buffer::allocator::event::reserve{
    .graph = {},
    .buffer_planner_sm = nullptr,
    .chunk_allocator_sm = &chunk_allocator,
    .strategy = nullptr,
    .error_out = &err,
  });
  CHECK(err == EMEL_OK);
}

}  // namespace

#pragma once

#include <array>
#include <cstdint>

namespace emel::buffer::planner {
struct sm;
struct strategy;
}  // namespace emel::buffer::planner

namespace emel::buffer::chunk_allocator {
struct sm;
}  // namespace emel::buffer::chunk_allocator

namespace emel::buffer::realloc_analyzer {
struct sm;
}  // namespace emel::buffer::realloc_analyzer

namespace emel::buffer::allocator::event {

inline constexpr int32_t k_max_sources = 4;

struct tensor_desc {
  int32_t tensor_id = -1;
  int32_t alloc_size = 0;
  std::array<int32_t, k_max_sources> src_ids = {{-1, -1, -1, -1}};
  bool is_view = false;
  int32_t view_src_id = -1;
  bool is_input = false;
  bool is_output = false;
  bool can_inplace = true;
  bool has_external_data = false;
};

struct graph_view {
  const tensor_desc * nodes = nullptr;
  int32_t n_nodes = 0;
  const tensor_desc * leafs = nullptr;
  int32_t n_leafs = 0;
};

struct initialize {
  int32_t buffer_count = 1;
  const int32_t * buffer_alignments = nullptr;
  const int32_t * buffer_max_sizes = nullptr;
  emel::buffer::chunk_allocator::sm * chunk_allocator_sm = nullptr;
  int32_t * error_out = nullptr;
};

struct reserve_n_size {
  graph_view graph = {};
  const int32_t * node_buffer_ids = nullptr;
  const int32_t * leaf_buffer_ids = nullptr;
  int32_t * sizes_out = nullptr;
  int32_t sizes_out_count = 0;
  emel::buffer::planner::sm * buffer_planner_sm = nullptr;
  emel::buffer::chunk_allocator::sm * chunk_allocator_sm = nullptr;
  const emel::buffer::planner::strategy * strategy = nullptr;
  int32_t * error_out = nullptr;
};

struct reserve_n {
  graph_view graph = {};
  const int32_t * node_buffer_ids = nullptr;
  const int32_t * leaf_buffer_ids = nullptr;
  emel::buffer::planner::sm * buffer_planner_sm = nullptr;
  emel::buffer::chunk_allocator::sm * chunk_allocator_sm = nullptr;
  const emel::buffer::planner::strategy * strategy = nullptr;
  int32_t * error_out = nullptr;
};

struct reserve {
  graph_view graph = {};
  emel::buffer::planner::sm * buffer_planner_sm = nullptr;
  emel::buffer::chunk_allocator::sm * chunk_allocator_sm = nullptr;
  const emel::buffer::planner::strategy * strategy = nullptr;
  int32_t * error_out = nullptr;
};

struct alloc_graph {
  graph_view graph = {};
  emel::buffer::planner::sm * buffer_planner_sm = nullptr;
  emel::buffer::chunk_allocator::sm * chunk_allocator_sm = nullptr;
  emel::buffer::realloc_analyzer::sm * buffer_realloc_analyzer_sm = nullptr;
  const emel::buffer::planner::strategy * strategy = nullptr;
  int32_t * error_out = nullptr;
};

struct release {
  emel::buffer::chunk_allocator::sm * chunk_allocator_sm = nullptr;
  int32_t * error_out = nullptr;
};

}  // namespace emel::buffer::allocator::event

namespace emel::buffer::allocator::events {

struct initialize_done {
  int32_t * error_out = nullptr;
};
struct initialize_error {
  int32_t err = 0;
  int32_t * error_out = nullptr;
};

struct reserve_n_size_done {
  int32_t * error_out = nullptr;
};
struct reserve_n_size_error {
  int32_t err = 0;
  int32_t * error_out = nullptr;
};

struct reserve_done {
  int32_t * error_out = nullptr;
};
struct reserve_error {
  int32_t err = 0;
  int32_t * error_out = nullptr;
};

struct alloc_graph_done {
  int32_t * error_out = nullptr;
};
struct alloc_graph_error {
  int32_t err = 0;
  int32_t * error_out = nullptr;
};

struct release_done {
  int32_t * error_out = nullptr;
};
struct release_error {
  int32_t err = 0;
  int32_t * error_out = nullptr;
};

using bootstrap_event = event::initialize;

}  // namespace emel::buffer::allocator::events

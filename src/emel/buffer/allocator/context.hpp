#pragma once

#include <array>
#include <cstdint>
#include <limits>

#include "emel/buffer/realloc_analyzer/events.hpp"
#include "emel/emel.h"

namespace emel::buffer::allocator::action {

inline constexpr int32_t k_max_buffers = 16;
inline constexpr int32_t k_max_graph_tensors = 2048;
inline constexpr int32_t k_max_chunks_per_buffer = 16;
inline constexpr int32_t k_max_chunk_bindings = k_max_buffers * k_max_chunks_per_buffer;
inline constexpr int32_t k_default_alignment = 16;
inline constexpr int32_t k_default_max_size = std::numeric_limits<int32_t>::max();

using tensor_alloc = emel::buffer::realloc_analyzer::event::tensor_alloc;
using node_alloc = emel::buffer::realloc_analyzer::event::node_alloc;
using leaf_alloc = emel::buffer::realloc_analyzer::event::leaf_alloc;

struct context {
  int32_t buffer_count = 0;
  int32_t init_epoch = 0;
  int32_t reserve_epoch = 0;
  int32_t alloc_epoch = 0;
  int32_t release_epoch = 0;
  uint32_t step = 0;
  int32_t phase_error = EMEL_OK;
  int32_t last_n_nodes = 0;
  int32_t last_n_leafs = 0;
  bool has_reserve_snapshot = false;
  std::array<int32_t, k_max_buffers> buffer_alignments = {};
  std::array<int32_t, k_max_buffers> buffer_max_sizes = {};
  std::array<int32_t, k_max_buffers> committed_sizes = {};
  std::array<int32_t, k_max_buffers> committed_chunk_counts = {};
  std::array<int32_t, k_max_chunk_bindings> committed_chunk_ids = {};
  std::array<uint64_t, k_max_chunk_bindings> committed_chunk_offsets = {};
  std::array<uint64_t, k_max_chunk_bindings> committed_chunk_sizes = {};
  std::array<int32_t, k_max_graph_tensors> last_node_buffer_ids = {};
  std::array<int32_t, k_max_graph_tensors> last_leaf_buffer_ids = {};
  std::array<node_alloc, k_max_graph_tensors> node_allocs = {};
  std::array<leaf_alloc, k_max_graph_tensors> leaf_allocs = {};
};

}  // namespace emel::buffer::allocator::action

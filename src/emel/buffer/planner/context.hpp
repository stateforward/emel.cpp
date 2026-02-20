#pragma once

#include <array>
#include <cstdint>
#include <limits>

#include "emel/buffer/planner/events.hpp"
#include "emel/emel.h"

namespace emel::buffer::planner::action {
struct context;
}

namespace emel::buffer::planner {

struct strategy {
  int32_t (*seed_leafs)(action::context &) noexcept = nullptr;
  int32_t (*count_references)(action::context &) noexcept = nullptr;
  int32_t (*alloc_explicit_inputs)(action::context &) noexcept = nullptr;
  int32_t (*plan_nodes)(action::context &) noexcept = nullptr;
  int32_t (*release_expired)(action::context &) noexcept = nullptr;
  int32_t (*finalize)(action::context &) noexcept = nullptr;
};

}  // namespace emel::buffer::planner

namespace emel::buffer::planner::action {

inline constexpr int32_t k_max_buffers = 16;
inline constexpr int32_t k_max_tensors = 2048;
inline constexpr int32_t k_max_free_blocks = 256;
inline constexpr int32_t k_max_chunks_per_buffer = 16;
inline constexpr int32_t k_max_chunk_plan_entries = k_max_buffers * k_max_chunks_per_buffer;
inline constexpr int32_t k_default_alignment = 16;
inline constexpr int32_t k_default_max_size = std::numeric_limits<int32_t>::max();

struct free_block {
  int32_t offset = 0;
  int32_t size = 0;
};

struct buffer_layout {
  std::array<free_block, k_max_free_blocks> free_blocks = {};
  int32_t free_block_count = 0;
  int32_t high_watermark = 0;
};

struct tensor_record {
  int32_t tensor_id = -1;
  int32_t alloc_size = 0;
  int32_t buffer_id = 0;
  int32_t alloc_offset = -1;
  int32_t alloc_reserved = 0;
  int32_t n_children = 0;
  int32_t n_views = 0;
  int32_t view_src_id = -1;
  bool is_view = false;
  bool is_input = false;
  bool is_output = false;
  bool allocatable = false;
  bool allocated = false;
};

struct context {
  int32_t buffer_count = 0;
  int32_t node_count = 0;
  int32_t leaf_count = 0;
  std::array<emel::buffer::allocator::event::tensor_desc, k_max_tensors> nodes = {};
  std::array<emel::buffer::allocator::event::tensor_desc, k_max_tensors> leafs = {};
  std::array<int32_t, k_max_tensors> node_buffer_ids = {};
  std::array<int32_t, k_max_tensors> leaf_buffer_ids = {};
  emel::buffer::planner::strategy strategy = {};
  std::array<int32_t, k_max_buffers> buffer_alignments = {};
  std::array<int32_t, k_max_buffers> buffer_max_sizes = {};
  std::array<int32_t, k_max_buffers> current_bytes_by_buffer = {};
  std::array<int32_t, k_max_buffers> bytes_by_buffer = {};
  std::array<int32_t, k_max_buffers> max_alloc_by_buffer = {};
  std::array<buffer_layout, k_max_buffers> buffer_layouts = {};
  std::array<int32_t, k_max_buffers> chunk_counts = {};
  std::array<int32_t, k_max_chunk_plan_entries> chunk_sizes = {};
  int32_t total_chunk_count = 0;
  std::array<tensor_record, k_max_tensors> tensors = {};
  int32_t tensor_count = 0;
  int32_t total_bytes = 0;
  int32_t planned_nodes = 0;
  int32_t planned_leafs = 0;
  int32_t reference_edges = 0;
  int32_t phase_error = EMEL_OK;
};

}  // namespace emel::buffer::planner::action

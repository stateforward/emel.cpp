#pragma once

#include <array>
#include <cstdint>

#include "emel/emel.h"
#include "emel/tensor/allocator/events.hpp"

namespace emel::tensor::allocator::action {

inline constexpr int32_t k_max_tensors = 2048;
inline constexpr int32_t k_max_chunks = 64;

struct context {
  const event::tensor_desc * tensors = nullptr;
  int32_t tensor_count = 0;
  int32_t alignment = 16;
  int32_t max_buffer_size = 0;
  bool no_alloc = false;
  int32_t chunk_sizes_out_count = 0;
  bool has_chunk_sizes_out = false;

  std::array<int32_t, k_max_tensors> tensor_ids = {};
  std::array<int32_t, k_max_tensors> effective_sizes = {};
  std::array<int32_t, k_max_tensors> tensor_chunk_ids = {};
  std::array<int32_t, k_max_tensors> tensor_offsets = {};
  std::array<int32_t, k_max_chunks> chunk_sizes = {};
  std::array<void *, k_max_chunks> allocated_buffers = {};
  std::array<void *, k_max_chunks> assembled_buffers = {};
  void * result_buffer = nullptr;
  int32_t chunk_count = 0;
  int32_t total_bytes = 0;

  emel_error_detail detail = {};
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::tensor::allocator::action

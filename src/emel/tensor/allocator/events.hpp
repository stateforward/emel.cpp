#pragma once

#include <array>
#include <cstdint>

namespace emel::tensor::allocator::event {

inline constexpr int32_t k_max_sources = 4;

struct tensor_desc {
  int32_t tensor_id = -1;
  int32_t alloc_size = 0;
  std::array<int32_t, k_max_sources> src_ids = {{-1, -1, -1, -1}};
  bool is_view = false;
  int32_t view_src_id = -1;
  bool has_external_data = false;
};

using alloc_buffer_fn = void * (*)(void * backend_ctx, int32_t bytes) noexcept;
using free_buffer_fn = void (*)(void * backend_ctx, void * buffer) noexcept;
using init_tensor_fn = int32_t (*)(
    void * backend_ctx, const tensor_desc * tensor, void * buffer, int32_t offset_bytes) noexcept;
using init_view_tensor_fn =
  int32_t (*)(void * backend_ctx, const tensor_desc * tensor) noexcept;
using assemble_buffers_fn =
  void * (*)(void * backend_ctx, void * const * buffers, int32_t buffer_count) noexcept;

struct allocate_tensors {
  const tensor_desc * tensors = nullptr;
  int32_t tensor_count = 0;
  int32_t alignment = 16;
  int32_t max_buffer_size = 0x7fffffff;
  bool no_alloc = false;

  void * backend_ctx = nullptr;
  alloc_buffer_fn alloc_buffer = nullptr;
  free_buffer_fn free_buffer = nullptr;
  init_tensor_fn init_tensor = nullptr;
  init_view_tensor_fn init_view_tensor = nullptr;
  assemble_buffers_fn assemble_buffers = nullptr;

  void ** result_buffer_out = nullptr;
  int32_t * total_size_out = nullptr;
  int32_t * chunk_sizes_out = nullptr;
  int32_t chunk_sizes_out_count = 0;
  int32_t * chunk_count_out = nullptr;
  int32_t * error_out = nullptr;
};

struct validate {
  int32_t * error_out = nullptr;
};
struct scan_tensors {
  int32_t * error_out = nullptr;
};
struct partition_ranges {
  int32_t * error_out = nullptr;
};
struct allocate_ranges {
  int32_t * error_out = nullptr;
};
struct initialize_tensors {
  int32_t * error_out = nullptr;
};
struct assemble {
  int32_t * error_out = nullptr;
};
struct release {
  int32_t * error_out = nullptr;
};

}  // namespace emel::tensor::allocator::event

namespace emel::tensor::allocator::events {

struct validate_done {};
struct validate_error {
  int32_t err = 0;
};

struct scan_done {};
struct scan_error {
  int32_t err = 0;
};

struct partition_done {};
struct partition_error {
  int32_t err = 0;
};

struct allocate_ranges_done {};
struct allocate_ranges_error {
  int32_t err = 0;
};

struct initialize_tensors_done {};
struct initialize_tensors_error {
  int32_t err = 0;
};

struct assemble_done {};
struct assemble_error {
  int32_t err = 0;
};

struct allocate_done {
  int32_t total_bytes = 0;
  int32_t chunk_count = 0;
};

struct allocate_error {
  int32_t err = 0;
};

struct release_done {};
struct release_error {
  int32_t err = 0;
};

using bootstrap_event = event::allocate_tensors;

}  // namespace emel::tensor::allocator::events

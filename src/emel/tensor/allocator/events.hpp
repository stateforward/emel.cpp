#pragma once

#include <array>
#include <cstdint>

#include "emel/emel.h"

namespace emel::tensor::allocator::event {

inline constexpr int32_t k_max_sources = 4;

enum class error_phase : uint32_t {
  none = 0,
  validate = 1,
  scan_tensors = 2,
  partition_ranges = 3,
  allocate_ranges = 4,
  initialize_tensors = 5,
  assemble = 6,
  release = 7,
};

enum class error_reason : uint32_t {
  none = 0,
  invalid_argument = 1,
  duplicate_tensor_id = 2,
  invalid_view_source = 3,
  alignment_overflow = 4,
  chunk_overflow = 5,
  allocation_failed = 6,
  offset_out_of_range = 7,
  assemble_failed = 8,
  unknown = 9,
};

struct tensor_desc {
  int32_t tensor_id = -1;
  int32_t alloc_size = 0;
  std::array<int32_t, k_max_sources> src_ids = {{-1, -1, -1, -1}};
  bool is_view = false;
  int32_t view_src_id = -1;
  bool has_external_data = false;
};

struct allocate_tensors {
  const tensor_desc * tensors = nullptr;
  int32_t tensor_count = 0;
  int32_t alignment = 16;
  int32_t max_buffer_size = 0x7fffffff;
  bool no_alloc = false;

  void ** result_buffer_out = nullptr;
  int32_t * total_size_out = nullptr;
  int32_t * chunk_sizes_out = nullptr;
  int32_t chunk_sizes_out_count = 0;
  int32_t * chunk_count_out = nullptr;
  int32_t * error_out = nullptr;
  emel_error_detail * detail_out = nullptr;
};

struct validate {
  int32_t * error_out = nullptr;
  emel_error_detail * detail_out = nullptr;
  int32_t * chunk_sizes_out = nullptr;
  int32_t chunk_sizes_out_count = 0;
};
struct scan_tensors {
  int32_t * error_out = nullptr;
  emel_error_detail * detail_out = nullptr;
};
struct partition_ranges {
  int32_t * error_out = nullptr;
  emel_error_detail * detail_out = nullptr;
};
struct allocate_ranges {
  int32_t * error_out = nullptr;
  emel_error_detail * detail_out = nullptr;
};
struct initialize_tensors {
  int32_t * error_out = nullptr;
  emel_error_detail * detail_out = nullptr;
};
struct assemble {
  int32_t * error_out = nullptr;
  emel_error_detail * detail_out = nullptr;
  void ** result_buffer_out = nullptr;
  int32_t * total_size_out = nullptr;
  int32_t * chunk_sizes_out = nullptr;
  int32_t chunk_sizes_out_count = 0;
  int32_t * chunk_count_out = nullptr;
};
struct release {
  int32_t * error_out = nullptr;
  emel_error_detail * detail_out = nullptr;
};

}  // namespace emel::tensor::allocator::event

namespace emel::tensor::allocator::events {

struct validate_done {
  const event::allocate_tensors * request = nullptr;
};
struct validate_error {
  int32_t err = 0;
  const event::allocate_tensors * request = nullptr;
};

struct scan_done {
  const event::allocate_tensors * request = nullptr;
};
struct scan_error {
  int32_t err = 0;
  const event::allocate_tensors * request = nullptr;
};

struct partition_done {
  const event::allocate_tensors * request = nullptr;
};
struct partition_error {
  int32_t err = 0;
  const event::allocate_tensors * request = nullptr;
};

struct allocate_ranges_done {
  const event::allocate_tensors * request = nullptr;
};
struct allocate_ranges_error {
  int32_t err = 0;
  const event::allocate_tensors * request = nullptr;
};

struct initialize_tensors_done {
  const event::allocate_tensors * request = nullptr;
};
struct initialize_tensors_error {
  int32_t err = 0;
  const event::allocate_tensors * request = nullptr;
};

struct assemble_done {
  const event::allocate_tensors * request = nullptr;
};
struct assemble_error {
  int32_t err = 0;
  const event::allocate_tensors * request = nullptr;
};

struct allocate_done {
  int32_t total_bytes = 0;
  int32_t chunk_count = 0;
  const event::allocate_tensors * request = nullptr;
};

struct allocate_error {
  int32_t err = 0;
  const event::allocate_tensors * request = nullptr;
};

struct release_done {
  const event::release * request = nullptr;
};
struct release_error {
  int32_t err = 0;
  const event::release * request = nullptr;
};

using bootstrap_event = event::allocate_tensors;

}  // namespace emel::tensor::allocator::events

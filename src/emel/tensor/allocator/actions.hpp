#pragma once

#include <array>
#include <cstdint>
#include <limits>

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

  void * backend_ctx = nullptr;
  event::alloc_buffer_fn alloc_buffer = nullptr;
  event::free_buffer_fn free_buffer = nullptr;
  event::init_tensor_fn init_tensor = nullptr;
  event::init_view_tensor_fn init_view_tensor = nullptr;
  event::assemble_buffers_fn assemble_buffers = nullptr;

  void ** result_buffer_out = nullptr;
  int32_t * total_size_out = nullptr;
  int32_t * chunk_sizes_out = nullptr;
  int32_t chunk_sizes_out_count = 0;
  int32_t * chunk_count_out = nullptr;
  int32_t * error_out = nullptr;

  uint32_t step = 0;

  std::array<int32_t, k_max_tensors> tensor_ids = {};
  std::array<int32_t, k_max_tensors> effective_sizes = {};
  std::array<int32_t, k_max_tensors> tensor_chunk_ids = {};
  std::array<int32_t, k_max_tensors> tensor_offsets = {};
  std::array<int32_t, k_max_chunks> chunk_sizes = {};
  std::array<void *, k_max_chunks> allocated_buffers = {};
  void * result_buffer = nullptr;
  int32_t chunk_count = 0;
  int32_t total_bytes = 0;
};

namespace detail {

inline int32_t normalize_error(const int32_t err, const int32_t fallback) noexcept {
  if (err != 0) return err;
  if (fallback != 0) return fallback;
  return EMEL_ERR_BACKEND;
}

inline bool is_power_of_two(const int32_t v) noexcept {
  return v > 0 && (v & (v - 1)) == 0;
}

inline int32_t sat_add(const int32_t lhs, const int32_t rhs) noexcept {
  const int64_t sum = static_cast<int64_t>(lhs) + rhs;
  if (sum > std::numeric_limits<int32_t>::max()) return std::numeric_limits<int32_t>::max();
  if (sum < std::numeric_limits<int32_t>::min()) return std::numeric_limits<int32_t>::min();
  return static_cast<int32_t>(sum);
}

inline bool align_up(const int32_t value, const int32_t alignment, int32_t & out) noexcept {
  if (value <= 0) {
    out = 0;
    return true;
  }
  const int64_t v = static_cast<int64_t>(value);
  const int64_t a = static_cast<int64_t>(alignment);
  const int64_t aligned = ((v + a - 1) / a) * a;
  if (aligned > std::numeric_limits<int32_t>::max()) {
    return false;
  }
  out = static_cast<int32_t>(aligned);
  return true;
}

inline bool has_tensor_id(const context & c, const int32_t tensor_id) noexcept {
  for (int32_t i = 0; i < c.tensor_count; ++i) {
    if (c.tensor_ids[i] == tensor_id) return true;
  }
  return false;
}

inline bool release_allocated_buffers(context & c) noexcept {
  bool has_allocated = false;
  for (int32_t i = 0; i < k_max_chunks; ++i) {
    if (c.allocated_buffers[i] != nullptr) {
      has_allocated = true;
      break;
    }
  }
  if (!has_allocated) {
    return true;
  }
  if (c.free_buffer == nullptr) {
    return false;
  }
  for (int32_t i = 0; i < k_max_chunks; ++i) {
    if (c.allocated_buffers[i] != nullptr) {
      c.free_buffer(c.backend_ctx, c.allocated_buffers[i]);
      c.allocated_buffers[i] = nullptr;
    }
  }
  return true;
}

}  // namespace detail

struct begin_allocate_tensors {
  void operator()(const event::allocate_tensors & ev, context & c) const noexcept {
    c = {};
    c.tensors = ev.tensors;
    c.tensor_count = ev.tensor_count;
    c.alignment = ev.alignment;
    c.max_buffer_size = ev.max_buffer_size;
    c.no_alloc = ev.no_alloc;
    c.backend_ctx = ev.backend_ctx;
    c.alloc_buffer = ev.alloc_buffer;
    c.free_buffer = ev.free_buffer;
    c.init_tensor = ev.init_tensor;
    c.init_view_tensor = ev.init_view_tensor;
    c.assemble_buffers = ev.assemble_buffers;
    c.result_buffer_out = ev.result_buffer_out;
    c.total_size_out = ev.total_size_out;
    c.chunk_sizes_out = ev.chunk_sizes_out;
    c.chunk_sizes_out_count = ev.chunk_sizes_out_count;
    c.chunk_count_out = ev.chunk_count_out;
    c.error_out = ev.error_out;
    c.tensor_chunk_ids.fill(-1);
    c.tensor_offsets.fill(-1);
    c.allocated_buffers.fill(nullptr);
    c.result_buffer = nullptr;
    if (c.result_buffer_out != nullptr) *c.result_buffer_out = nullptr;
    if (c.error_out != nullptr) *c.error_out = EMEL_OK;
    c.step += 1;
  }
};

struct run_validate {
  void operator()(const event::validate & ev, context & c) const noexcept {
    int32_t err = EMEL_OK;
    if (c.tensor_count < 0 || c.tensor_count > k_max_tensors) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    } else if ((c.tensor_count > 0 && c.tensors == nullptr) || !detail::is_power_of_two(c.alignment)) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    } else if (c.max_buffer_size <= 0 || c.chunk_sizes_out_count < 0) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    } else if (c.chunk_sizes_out == nullptr && c.chunk_sizes_out_count != 0) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    } else if (
        !c.no_alloc &&
        (c.alloc_buffer == nullptr || c.free_buffer == nullptr || c.init_tensor == nullptr)) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    c.step += 1;
  }
};

struct run_scan_tensors {
  void operator()(const event::scan_tensors & ev, context & c) const noexcept {
    int32_t err = EMEL_OK;
    for (int32_t i = 0; i < c.tensor_count; ++i) {
      const auto & t = c.tensors[i];
      if (t.tensor_id < 0 || t.alloc_size < 0 || detail::has_tensor_id(c, t.tensor_id)) {
        err = EMEL_ERR_INVALID_ARGUMENT;
        break;
      }
      c.tensor_ids[i] = t.tensor_id;

      if (t.is_view && t.view_src_id < 0) {
        err = EMEL_ERR_INVALID_ARGUMENT;
        break;
      }
      if (t.has_external_data || t.is_view || t.alloc_size == 0) {
        c.effective_sizes[i] = 0;
        continue;
      }

      int32_t aligned = 0;
      if (!detail::align_up(t.alloc_size, c.alignment, aligned)) {
        err = EMEL_ERR_BACKEND;
        break;
      }
      c.effective_sizes[i] = aligned;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    c.step += 1;
  }
};

struct run_partition_ranges {
  void operator()(const event::partition_ranges & ev, context & c) const noexcept {
    int32_t err = EMEL_OK;
    c.chunk_sizes.fill(0);
    c.chunk_count = 0;
    c.total_bytes = 0;
    c.tensor_chunk_ids.fill(-1);
    c.tensor_offsets.fill(-1);

    auto begin_next_chunk = [&]() -> int32_t {
      if (c.chunk_count < k_max_chunks) {
        const int32_t idx = c.chunk_count;
        c.chunk_count += 1;
        c.chunk_sizes[idx] = 0;
        return idx;
      }
      return k_max_chunks - 1;
    };

    int32_t current_chunk = -1;
    for (int32_t i = 0; i < c.tensor_count && err == EMEL_OK; ++i) {
      const int32_t sz = c.effective_sizes[i];
      if (sz <= 0) continue;

      if (current_chunk < 0) {
        current_chunk = begin_next_chunk();
      }

      int32_t cur_bytes = c.chunk_sizes[current_chunk];
      const bool overflow_current = cur_bytes > 0 && detail::sat_add(cur_bytes, sz) > c.max_buffer_size;
      if (overflow_current) {
        current_chunk = begin_next_chunk();
        cur_bytes = c.chunk_sizes[current_chunk];
      }

      c.tensor_chunk_ids[i] = current_chunk;
      c.tensor_offsets[i] = cur_bytes;
      c.chunk_sizes[current_chunk] = detail::sat_add(cur_bytes, sz);
      c.total_bytes = detail::sat_add(c.total_bytes, sz);
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    c.step += 1;
  }
};

struct run_allocate_ranges {
  void operator()(const event::allocate_ranges & ev, context & c) const noexcept {
    int32_t err = EMEL_OK;
    c.allocated_buffers.fill(nullptr);
    c.result_buffer = nullptr;

    if (c.no_alloc || c.chunk_count == 0) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_OK;
      }
      c.step += 1;
      return;
    }

    for (int32_t i = 0; i < c.chunk_count; ++i) {
      if (c.chunk_sizes[i] <= 0) {
        err = EMEL_ERR_BACKEND;
        break;
      }
      void * buffer = c.alloc_buffer(c.backend_ctx, c.chunk_sizes[i]);
      if (buffer == nullptr) {
        err = EMEL_ERR_BACKEND;
        break;
      }
      c.allocated_buffers[i] = buffer;
    }

    if (err != EMEL_OK) {
      (void)detail::release_allocated_buffers(c);
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    c.step += 1;
  }
};

struct run_initialize_tensors {
  void operator()(const event::initialize_tensors & ev, context & c) const noexcept {
    int32_t err = EMEL_OK;
    if (c.no_alloc) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_OK;
      }
      c.step += 1;
      return;
    }

    for (int32_t i = 0; i < c.tensor_count; ++i) {
      const auto & t = c.tensors[i];
      if (t.is_view) {
        if (!detail::has_tensor_id(c, t.view_src_id)) {
          err = EMEL_ERR_INVALID_ARGUMENT;
          break;
        }
        if (c.init_view_tensor != nullptr) {
          const int32_t rc = c.init_view_tensor(c.backend_ctx, &t);
          if (rc != EMEL_OK) {
            err = detail::normalize_error(rc, EMEL_ERR_BACKEND);
            break;
          }
        }
        continue;
      }

      if (c.effective_sizes[i] <= 0) {
        continue;
      }
      const int32_t chunk_id = c.tensor_chunk_ids[i];
      const int32_t offset = c.tensor_offsets[i];
      if (chunk_id < 0 || chunk_id >= c.chunk_count || offset < 0) {
        err = EMEL_ERR_BACKEND;
        break;
      }
      void * buffer = c.allocated_buffers[chunk_id];
      if (buffer == nullptr) {
        err = EMEL_ERR_BACKEND;
        break;
      }
      const int32_t rc = c.init_tensor(c.backend_ctx, &t, buffer, offset);
      if (rc != EMEL_OK) {
        err = detail::normalize_error(rc, EMEL_ERR_BACKEND);
        break;
      }
    }
    if (err != EMEL_OK) {
      (void)detail::release_allocated_buffers(c);
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    c.step += 1;
  }
};

struct run_assemble {
  void operator()(const event::assemble & ev, context & c) const noexcept {
    int32_t err = EMEL_OK;
    if (c.chunk_sizes_out != nullptr && c.chunk_sizes_out_count < c.chunk_count) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    } else {
      if (c.total_size_out != nullptr) *c.total_size_out = c.total_bytes;
      if (c.chunk_count_out != nullptr) *c.chunk_count_out = c.chunk_count;
      for (int32_t i = 0; i < c.chunk_count && c.chunk_sizes_out != nullptr; ++i) {
        c.chunk_sizes_out[i] = c.chunk_sizes[i];
      }

      if (c.result_buffer_out != nullptr) {
        *c.result_buffer_out = nullptr;
        if (!c.no_alloc) {
          if (c.chunk_count == 1) {
            c.result_buffer = c.allocated_buffers[0];
            *c.result_buffer_out = c.result_buffer;
          } else if (c.chunk_count > 1) {
            if (c.assemble_buffers == nullptr) {
              err = EMEL_ERR_INVALID_ARGUMENT;
            } else {
              c.result_buffer = c.assemble_buffers(c.backend_ctx, c.allocated_buffers.data(), c.chunk_count);
              if (c.result_buffer == nullptr) {
                err = EMEL_ERR_BACKEND;
              } else {
                *c.result_buffer_out = c.result_buffer;
              }
            }
          }
        }
      }
    }
    if (err != EMEL_OK) {
      (void)detail::release_allocated_buffers(c);
      if (c.result_buffer_out != nullptr) *c.result_buffer_out = nullptr;
      c.result_buffer = nullptr;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    c.step += 1;
  }
};

struct begin_release {
  void operator()(const event::release & ev, context & c) const noexcept {
    const int32_t err = detail::release_allocated_buffers(c) ? EMEL_OK : EMEL_ERR_BACKEND;
    c.result_buffer = nullptr;
    if (c.result_buffer_out != nullptr) *c.result_buffer_out = nullptr;
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    c.step += 1;
  }
};

struct on_allocate_done {
  void operator()(const events::allocate_done &, context & c) const noexcept {
    if (c.error_out != nullptr) *c.error_out = EMEL_OK;
    c.step += 1;
  }
};

struct on_allocate_error {
  void operator()(const events::allocate_error & ev, context & c) const noexcept {
    if (c.error_out != nullptr) *c.error_out = detail::normalize_error(ev.err, EMEL_ERR_BACKEND);
    c.step += 1;
  }
};

struct on_release_done {
  void operator()(const events::release_done &, context & c) const noexcept {
    c = {};
    c.step = 1;
  }
};

struct on_release_error {
  void operator()(const events::release_error & ev, context & c) const noexcept {
    if (c.error_out != nullptr) *c.error_out = detail::normalize_error(ev.err, EMEL_ERR_BACKEND);
    c.step += 1;
  }
};

struct record_phase_error {
  template <class ErrorEvent>
  void operator()(const ErrorEvent & ev, context & c) const noexcept {
    if (c.error_out != nullptr) {
      *c.error_out = detail::normalize_error(ev.err, EMEL_ERR_BACKEND);
    }
  }
};

inline constexpr begin_allocate_tensors begin_allocate_tensors{};
inline constexpr run_validate run_validate{};
inline constexpr run_scan_tensors run_scan_tensors{};
inline constexpr run_partition_ranges run_partition_ranges{};
inline constexpr run_allocate_ranges run_allocate_ranges{};
inline constexpr run_initialize_tensors run_initialize_tensors{};
inline constexpr run_assemble run_assemble{};
inline constexpr begin_release begin_release{};
inline constexpr on_allocate_done on_allocate_done{};
inline constexpr on_allocate_error on_allocate_error{};
inline constexpr on_release_done on_release_done{};
inline constexpr on_release_error on_release_error{};
inline constexpr record_phase_error record_phase_error{};

}  // namespace emel::tensor::allocator::action

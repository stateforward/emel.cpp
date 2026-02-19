#pragma once

#include <array>
#include <cstdlib>
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

namespace detail {

inline int32_t normalize_error(const int32_t err, const int32_t fallback) noexcept {
  if (err != 0) return err;
  if (fallback != 0) return fallback;
  return EMEL_ERR_BACKEND;
}

inline void set_error_detail(
    emel_error_detail * detail,
    const int32_t status,
    const event::error_phase phase,
    const event::error_reason reason,
    const int32_t index,
    const int32_t aux) noexcept {
  if (detail == nullptr) return;
  detail->status = status;
  detail->domain = EMEL_ERROR_DOMAIN_TENSOR_ALLOCATOR;
  detail->phase = static_cast<uint32_t>(phase);
  detail->reason = static_cast<uint32_t>(reason);
  detail->index = index;
  detail->aux = aux;
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
  bool released_any = false;
  for (int32_t i = 0; i < k_max_chunks; ++i) {
    if (c.allocated_buffers[i] != nullptr) {
      std::free(c.allocated_buffers[i]);
      c.allocated_buffers[i] = nullptr;
      released_any = true;
    }
  }
  (void)released_any;
  return true;
}

}  // namespace detail

inline void reset_phase(context & c) noexcept {
  c.phase_error = EMEL_OK;
  c.last_error = EMEL_OK;
  detail::set_error_detail(
      &c.detail,
      EMEL_OK,
      event::error_phase::none,
      event::error_reason::none,
      -1,
      0);
}

inline void set_error(
    context & c,
    const int32_t err,
    const event::error_phase phase,
    const event::error_reason reason,
    const int32_t index,
    const int32_t aux) noexcept {
  c.phase_error = err;
  c.last_error = err;
  detail::set_error_detail(&c.detail, err, phase, reason, index, aux);
}

struct begin_allocate_tensors {
  void operator()(const event::allocate_tensors & ev, context & c) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    if (ev.result_buffer_out != nullptr) {
      *ev.result_buffer_out = nullptr;
    }
    if (ev.total_size_out != nullptr) {
      *ev.total_size_out = 0;
    }
    if (ev.chunk_count_out != nullptr) {
      *ev.chunk_count_out = 0;
    }
    if (ev.detail_out != nullptr) {
      ev.detail_out->status = EMEL_OK;
      ev.detail_out->domain = EMEL_ERROR_DOMAIN_TENSOR_ALLOCATOR;
      ev.detail_out->phase = static_cast<uint32_t>(event::error_phase::none);
      ev.detail_out->reason = static_cast<uint32_t>(event::error_reason::none);
      ev.detail_out->index = -1;
      ev.detail_out->aux = 0;
    }
    c = {};
    c.tensors = ev.tensors;
    c.tensor_count = ev.tensor_count;
    c.alignment = ev.alignment;
    c.max_buffer_size = ev.max_buffer_size;
    c.no_alloc = ev.no_alloc;
    c.chunk_sizes_out_count = ev.chunk_sizes_out_count;
    c.has_chunk_sizes_out = ev.chunk_sizes_out != nullptr;
    c.tensor_chunk_ids.fill(-1);
    c.tensor_offsets.fill(-1);
    c.tensor_ids.fill(-1);
    c.allocated_buffers.fill(nullptr);
    c.result_buffer = nullptr;
    reset_phase(c);
  }
};

struct run_validate {
  template <class Ev>
  void operator()(const Ev & ev, context & c) const noexcept {
    reset_phase(c);
    int32_t chunk_sizes_out_count = c.chunk_sizes_out_count;
    bool has_chunk_sizes_out = c.has_chunk_sizes_out;
    if constexpr (requires { ev.chunk_sizes_out_count; }) {
      chunk_sizes_out_count = ev.chunk_sizes_out_count;
      c.chunk_sizes_out_count = ev.chunk_sizes_out_count;
    }
    if constexpr (requires { ev.chunk_sizes_out; }) {
      has_chunk_sizes_out = ev.chunk_sizes_out != nullptr;
      c.has_chunk_sizes_out = has_chunk_sizes_out;
    }

    int32_t err = EMEL_OK;
    if (c.tensor_count < 0 || c.tensor_count > k_max_tensors) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    } else if ((c.tensor_count > 0 && c.tensors == nullptr) ||
               !detail::is_power_of_two(c.alignment)) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    } else if (c.max_buffer_size <= 0 || chunk_sizes_out_count < 0) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    } else if (!has_chunk_sizes_out && chunk_sizes_out_count != 0) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    }

    if (err != EMEL_OK) {
      set_error(
          c,
          err,
          event::error_phase::validate,
          event::error_reason::invalid_argument,
          -1,
          0);
    }
    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = c.phase_error;
      }
    }
    if constexpr (requires { ev.detail_out; }) {
      if (ev.detail_out != nullptr) {
        *ev.detail_out = c.detail;
      }
    }
  }
};

struct run_scan_tensors {
  template <class Ev>
  void operator()(const Ev & ev, context & c) const noexcept {
    reset_phase(c);
    c.tensor_ids.fill(-1);
    c.effective_sizes.fill(0);

    int32_t err = EMEL_OK;
    if (c.tensor_count < 0 || c.tensor_count > k_max_tensors || c.tensors == nullptr) {
      err = EMEL_ERR_INVALID_ARGUMENT;
      set_error(
          c,
          err,
          event::error_phase::scan_tensors,
          event::error_reason::invalid_argument,
          -1,
          0);
    }

    for (int32_t i = 0; i < c.tensor_count && err == EMEL_OK; ++i) {
      const auto & t = c.tensors[i];
      if (t.tensor_id < 0 || t.alloc_size < 0 || detail::has_tensor_id(c, t.tensor_id)) {
        err = EMEL_ERR_INVALID_ARGUMENT;
        const event::error_reason reason =
            detail::has_tensor_id(c, t.tensor_id)
                ? event::error_reason::duplicate_tensor_id
                : event::error_reason::invalid_argument;
        set_error(c, err, event::error_phase::scan_tensors, reason, i, t.tensor_id);
        break;
      }
      c.tensor_ids[i] = t.tensor_id;

      if (t.is_view && t.view_src_id < 0) {
        err = EMEL_ERR_INVALID_ARGUMENT;
        set_error(
            c,
            err,
            event::error_phase::scan_tensors,
            event::error_reason::invalid_view_source,
            i,
            t.view_src_id);
        break;
      }
      if (t.has_external_data || t.is_view || t.alloc_size == 0) {
        c.effective_sizes[i] = 0;
        continue;
      }

      int32_t aligned = 0;
      if (!detail::align_up(t.alloc_size, c.alignment, aligned)) {
        err = EMEL_ERR_BACKEND;
        set_error(
            c,
            err,
            event::error_phase::scan_tensors,
            event::error_reason::alignment_overflow,
            i,
            t.alloc_size);
        break;
      }
      c.effective_sizes[i] = aligned;
    }

    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = c.phase_error;
      }
    }
    if constexpr (requires { ev.detail_out; }) {
      if (ev.detail_out != nullptr) {
        *ev.detail_out = c.detail;
      }
    }
  }
};

struct run_partition_ranges {
  template <class Ev>
  void operator()(const Ev & ev, context & c) const noexcept {
    reset_phase(c);
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
    for (int32_t i = 0; i < c.tensor_count; ++i) {
      const int32_t sz = c.effective_sizes[i];
      if (sz <= 0) continue;

      if (current_chunk < 0) {
        current_chunk = begin_next_chunk();
      }

      int32_t cur_bytes = c.chunk_sizes[current_chunk];
      const bool overflow_current =
          cur_bytes > 0 && detail::sat_add(cur_bytes, sz) > c.max_buffer_size;
      if (overflow_current) {
        current_chunk = begin_next_chunk();
        cur_bytes = c.chunk_sizes[current_chunk];
      }

      c.tensor_chunk_ids[i] = current_chunk;
      c.tensor_offsets[i] = cur_bytes;
      c.chunk_sizes[current_chunk] = detail::sat_add(cur_bytes, sz);
      c.total_bytes = detail::sat_add(c.total_bytes, sz);
    }
    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = c.phase_error;
      }
    }
    if constexpr (requires { ev.detail_out; }) {
      if (ev.detail_out != nullptr) {
        *ev.detail_out = c.detail;
      }
    }
  }
};

struct run_allocate_ranges {
  template <class Ev>
  void operator()(const Ev & ev, context & c) const noexcept {
    reset_phase(c);
    c.allocated_buffers.fill(nullptr);
    c.result_buffer = nullptr;

    if (c.no_alloc || c.chunk_count == 0) {
      if constexpr (requires { ev.error_out; }) {
        if (ev.error_out != nullptr) {
          *ev.error_out = EMEL_OK;
        }
      }
      if constexpr (requires { ev.detail_out; }) {
        if (ev.detail_out != nullptr) {
          *ev.detail_out = c.detail;
        }
      }
      return;
    }

    int32_t err = EMEL_OK;
    for (int32_t i = 0; i < c.chunk_count; ++i) {
      if (c.chunk_sizes[i] <= 0) {
        err = EMEL_ERR_BACKEND;
        set_error(
            c,
            err,
            event::error_phase::allocate_ranges,
            event::error_reason::invalid_argument,
            i,
            c.chunk_sizes[i]);
        break;
      }
      void * buffer = std::malloc(static_cast<size_t>(c.chunk_sizes[i]));
      if (buffer == nullptr) {
        err = EMEL_ERR_BACKEND;
        set_error(
            c,
            err,
            event::error_phase::allocate_ranges,
            event::error_reason::allocation_failed,
            i,
            c.chunk_sizes[i]);
        break;
      }
      c.allocated_buffers[i] = buffer;
    }

    if (err != EMEL_OK) {
      (void)detail::release_allocated_buffers(c);
    }
    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = c.phase_error;
      }
    }
    if constexpr (requires { ev.detail_out; }) {
      if (ev.detail_out != nullptr) {
        *ev.detail_out = c.detail;
      }
    }
  }
};

struct run_initialize_tensors {
  template <class Ev>
  void operator()(const Ev & ev, context & c) const noexcept {
    reset_phase(c);
    int32_t err = EMEL_OK;
    if (c.no_alloc) {
      if constexpr (requires { ev.error_out; }) {
        if (ev.error_out != nullptr) {
          *ev.error_out = EMEL_OK;
        }
      }
      if constexpr (requires { ev.detail_out; }) {
        if (ev.detail_out != nullptr) {
          *ev.detail_out = c.detail;
        }
      }
      return;
    }

    if (c.tensors == nullptr || c.tensor_count < 0) {
      err = EMEL_ERR_INVALID_ARGUMENT;
      set_error(
          c,
          err,
          event::error_phase::initialize_tensors,
          event::error_reason::invalid_argument,
          -1,
          0);
      if constexpr (requires { ev.error_out; }) {
        if (ev.error_out != nullptr) {
          *ev.error_out = c.phase_error;
        }
      }
      if constexpr (requires { ev.detail_out; }) {
        if (ev.detail_out != nullptr) {
          *ev.detail_out = c.detail;
        }
      }
      return;
    }

    if (c.tensor_count > 0 && c.tensors != nullptr) {
      c.tensor_ids.fill(-1);
      for (int32_t i = 0; i < c.tensor_count; ++i) {
        c.tensor_ids[i] = c.tensors[i].tensor_id;
      }
    }

    for (int32_t i = 0; i < c.tensor_count; ++i) {
      const auto & t = c.tensors[i];
      if (t.is_view) {
        if (t.view_src_id < 0 || !detail::has_tensor_id(c, t.view_src_id)) {
          err = EMEL_ERR_INVALID_ARGUMENT;
          set_error(
              c,
              err,
              event::error_phase::initialize_tensors,
              event::error_reason::invalid_view_source,
              i,
              t.view_src_id);
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
        set_error(
            c,
            err,
            event::error_phase::initialize_tensors,
            event::error_reason::offset_out_of_range,
            i,
            offset);
        break;
      }
      void * buffer = c.allocated_buffers[chunk_id];
      if (buffer == nullptr) {
        err = EMEL_ERR_BACKEND;
        set_error(
            c,
            err,
            event::error_phase::initialize_tensors,
            event::error_reason::allocation_failed,
            i,
            chunk_id);
        break;
      }
      const int32_t end_offset = detail::sat_add(offset, c.effective_sizes[i]);
      if (end_offset <= 0 || end_offset > c.chunk_sizes[chunk_id]) {
        err = EMEL_ERR_BACKEND;
        set_error(
            c,
            err,
            event::error_phase::initialize_tensors,
            event::error_reason::offset_out_of_range,
            i,
            end_offset);
        break;
      }
    }
    if (err != EMEL_OK) {
      (void)detail::release_allocated_buffers(c);
    }
    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = c.phase_error;
      }
    }
    if constexpr (requires { ev.detail_out; }) {
      if (ev.detail_out != nullptr) {
        *ev.detail_out = c.detail;
      }
    }
  }
};

struct run_assemble {
  template <class Ev>
  void operator()(const Ev & ev, context & c) const noexcept {
    reset_phase(c);
    int32_t err = EMEL_OK;
    c.result_buffer = nullptr;
    int32_t chunk_sizes_out_count = c.chunk_sizes_out_count;
    bool has_chunk_sizes_out = c.has_chunk_sizes_out;
    if constexpr (requires { ev.chunk_sizes_out_count; }) {
      chunk_sizes_out_count = ev.chunk_sizes_out_count;
      c.chunk_sizes_out_count = ev.chunk_sizes_out_count;
    }
    if constexpr (requires { ev.chunk_sizes_out; }) {
      has_chunk_sizes_out = ev.chunk_sizes_out != nullptr;
      c.has_chunk_sizes_out = has_chunk_sizes_out;
    }
    if (has_chunk_sizes_out && chunk_sizes_out_count < c.chunk_count) {
      err = EMEL_ERR_INVALID_ARGUMENT;
      set_error(
          c,
          err,
          event::error_phase::assemble,
          event::error_reason::invalid_argument,
          -1,
          chunk_sizes_out_count);
    }
    if (err == EMEL_OK) {
      if constexpr (requires { ev.total_size_out; }) {
        if (ev.total_size_out != nullptr) {
          *ev.total_size_out = c.total_bytes;
        }
      }
      if constexpr (requires { ev.chunk_count_out; }) {
        if (ev.chunk_count_out != nullptr) {
          *ev.chunk_count_out = c.chunk_count;
        }
      }
      if constexpr (requires { ev.chunk_sizes_out; }) {
        for (int32_t i = 0; i < c.chunk_count && ev.chunk_sizes_out != nullptr; ++i) {
          ev.chunk_sizes_out[i] = c.chunk_sizes[i];
        }
      }
      if constexpr (requires { ev.result_buffer_out; }) {
        if (ev.result_buffer_out != nullptr) {
          *ev.result_buffer_out = nullptr;
        }
      }
    }
    if (err == EMEL_OK && !c.no_alloc) {
      if (c.chunk_count == 1) {
        c.result_buffer = c.allocated_buffers[0];
        if constexpr (requires { ev.result_buffer_out; }) {
          if (ev.result_buffer_out != nullptr) {
            *ev.result_buffer_out = c.result_buffer;
          }
        }
      } else if (c.chunk_count > 1) {
        c.assembled_buffers.fill(nullptr);
        for (int32_t i = 0; i < c.chunk_count; ++i) {
          c.assembled_buffers[i] = c.allocated_buffers[i];
        }
        c.result_buffer = static_cast<void *>(c.assembled_buffers.data());
        if (c.result_buffer == nullptr) {
          err = EMEL_ERR_BACKEND;
          set_error(
              c,
              err,
              event::error_phase::assemble,
              event::error_reason::assemble_failed,
              -1,
              c.chunk_count);
        } else if constexpr (requires { ev.result_buffer_out; }) {
          if (ev.result_buffer_out != nullptr) {
            *ev.result_buffer_out = c.result_buffer;
          }
        }
      }
    }
    if (err != EMEL_OK) {
      (void)detail::release_allocated_buffers(c);
      if constexpr (requires { ev.result_buffer_out; }) {
        if (ev.result_buffer_out != nullptr) {
          *ev.result_buffer_out = nullptr;
        }
      }
      c.result_buffer = nullptr;
    }
    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = c.phase_error;
      }
    }
    if constexpr (requires { ev.detail_out; }) {
      if (ev.detail_out != nullptr) {
        *ev.detail_out = c.detail;
      }
    }
  }
};

struct begin_release {
  template <class Ev>
  void operator()(const Ev & ev, context & c) const noexcept {
    reset_phase(c);
    const int32_t err = detail::release_allocated_buffers(c) ? EMEL_OK : EMEL_ERR_BACKEND;
    if (err != EMEL_OK) {
      set_error(
          c,
          err,
          event::error_phase::release,
          event::error_reason::unknown,
          -1,
          0);
    } else {
      c = {};
      c.tensor_ids.fill(-1);
      c.tensor_chunk_ids.fill(-1);
      c.tensor_offsets.fill(-1);
      reset_phase(c);
    }
    c.result_buffer = nullptr;
    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = c.phase_error;
      }
    }
    if constexpr (requires { ev.detail_out; }) {
      if (ev.detail_out != nullptr) {
        *ev.detail_out = c.detail;
      }
    }
  }
};

struct on_unexpected {
  template <class Ev>
  void operator()(const Ev & ev, context & c) const noexcept {
    set_error(
        c,
        EMEL_ERR_BACKEND,
        event::error_phase::none,
        event::error_reason::unknown,
        -1,
        0);
    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = c.phase_error;
      }
    }
    if constexpr (requires { ev.detail_out; }) {
      if (ev.detail_out != nullptr) {
        *ev.detail_out = c.detail;
      }
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
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::tensor::allocator::action

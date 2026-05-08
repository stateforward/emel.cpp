#pragma once

#include <cstdint>

#include "emel/io/staged_read/context.hpp"
#include "emel/io/staged_read/detail.hpp"
#include "emel/io/staged_read/errors.hpp"

namespace emel::io::staged_read::guard {

struct guard_staged_window_callbacks_present {
  bool operator()(const detail::staged_window_runtime &ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(ev.request.on_done) &&
           static_cast<bool>(ev.request.on_error);
  }
};

struct guard_staged_window_callbacks_missing {
  bool operator()(const detail::staged_window_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !guard_staged_window_callbacks_present{}(ev, ctx);
  }
};

struct guard_stg_source_contract_valid {
  bool operator()(const detail::staged_window_runtime &ev,
                  const action::context &) const noexcept {
    const auto &r = ev.request.request;
    if (r.logical_byte_length == 0u || r.stage_chunk_bytes == 0u) {
      return false;
    }
    if (r.stage_chunk_bytes > r.logical_byte_length) {
      return false;
    }
    if (r.file_offset > ~0ull - r.logical_byte_length) {
      return false;
    }
    return true;
  }
};

struct guard_stg_source_contract_invalid {
  bool operator()(const detail::staged_window_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !guard_stg_source_contract_valid{}(ev, ctx);
  }
};

struct guard_stg_target_window_valid {
  bool operator()(const detail::staged_window_runtime &ev,
                  const action::context &) const noexcept {
    const auto &r = ev.request.request;
    if (r.target_buffer == nullptr) {
      return false;
    }
    // Full-span copy commits `logical_byte_length` bytes starting at
    // `target_buffer`; callers must allocate at least that caller-owned window.
    return r.target_window_bytes >= r.logical_byte_length;
  }
};

struct guard_stg_target_window_invalid {
  bool operator()(const detail::staged_window_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !guard_stg_target_window_valid{}(ev, ctx);
  }
};

struct guard_platform_staged_read_supported {
  bool operator()(const detail::staged_window_runtime &,
                  const action::context &) const noexcept {
    if constexpr (EMEL_IO_STAGED_READ_PLATFORM_SUPPORTED != 0) {
      return true;
    } else {
      return false;
    }
  }
};

struct guard_platform_staged_read_unsupported {
  bool operator()(const detail::staged_window_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !guard_platform_staged_read_supported{}(ev, ctx);
  }
};

struct guard_stg_copy_span_valid {
  bool operator()(const detail::staged_window_runtime &ev,
                  const action::context &) const noexcept {
    const auto &r = ev.request.request;
    return r.source_span != nullptr &&
           r.source_span_bytes == r.logical_byte_length;
  }
};

struct guard_stg_source_span_present {
  bool operator()(const detail::staged_window_runtime &ev,
                  const action::context &) const noexcept {
    return ev.request.request.source_span != nullptr;
  }
};

struct guard_stg_source_span_missing {
  bool operator()(const detail::staged_window_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !guard_stg_source_span_present{}(ev, ctx);
  }
};

struct guard_stg_source_span_insufficient {
  bool operator()(const detail::staged_window_runtime &ev,
                  const action::context &) const noexcept {
    const auto &r = ev.request.request;
    return r.source_span != nullptr &&
           r.source_span_bytes < r.logical_byte_length;
  }
};

struct guard_stg_source_span_size_mismatch {
  bool operator()(const detail::staged_window_runtime &ev,
                  const action::context &) const noexcept {
    const auto &r = ev.request.request;
    return r.source_span != nullptr &&
           r.source_span_bytes > r.logical_byte_length;
  }
};

struct guard_stg_logical_chunk_aligned {
  bool operator()(const detail::staged_window_runtime &ev,
                  const action::context &) const noexcept {
    const auto &r = ev.request.request;
    return (r.logical_byte_length % r.stage_chunk_bytes) == 0u;
  }
};

struct guard_stg_logical_chunk_remainder {
  bool operator()(const detail::staged_window_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !guard_stg_logical_chunk_aligned{}(ev, ctx);
  }
};

struct error_callback_present {
  bool operator()(const detail::staged_window_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct error_callback_absent {
  bool operator()(const detail::staged_window_runtime &ev) const noexcept {
    return !error_callback_present{}(ev);
  }
};

struct guard_staged_window_batch_callbacks_present {
  bool operator()(const detail::staged_window_batch_runtime &ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(ev.request.on_done) &&
           static_cast<bool>(ev.request.on_error);
  }
};

struct guard_staged_window_batch_callbacks_missing {
  bool operator()(const detail::staged_window_batch_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !guard_staged_window_batch_callbacks_present{}(ev, ctx);
  }
};

struct guard_stg_batch_requests_valid {
  bool operator()(const detail::staged_window_batch_runtime &ev,
                  const action::context &) const noexcept {
    if (ev.request.tensors.empty()) {
      return false;
    }
    bool valid = true;
    for (uint32_t index = 0u;
         index < static_cast<uint32_t>(ev.request.tensors.size()); ++index) {
      const auto &tensor = ev.request.tensors[index];
      valid = valid && tensor.byte_size > 0u &&
              tensor.file_offset <= ~0ull - tensor.byte_size &&
              tensor.target != nullptr &&
              tensor.target_bytes >= tensor.byte_size &&
              tensor.source_buffer != nullptr &&
              tensor.source_buffer_bytes >= tensor.file_offset &&
              (tensor.source_buffer_bytes - tensor.file_offset) >=
                  tensor.byte_size;
    }
    return valid;
  }
};

struct guard_stg_batch_requests_invalid {
  bool operator()(const detail::staged_window_batch_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !guard_stg_batch_requests_valid{}(ev, ctx);
  }
};

struct guard_platform_staged_read_batch_supported {
  bool operator()(const detail::staged_window_batch_runtime &,
                  const action::context &) const noexcept {
    if constexpr (EMEL_IO_STAGED_READ_PLATFORM_SUPPORTED != 0) {
      return true;
    } else {
      return false;
    }
  }
};

struct guard_platform_staged_read_batch_unsupported {
  bool operator()(const detail::staged_window_batch_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !guard_platform_staged_read_batch_supported{}(ev, ctx);
  }
};

struct batch_error_callback_present {
  bool operator()(const detail::staged_window_batch_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct batch_error_callback_absent {
  bool operator()(const detail::staged_window_batch_runtime &ev) const noexcept {
    return !batch_error_callback_present{}(ev);
  }
};

} // namespace emel::io::staged_read::guard

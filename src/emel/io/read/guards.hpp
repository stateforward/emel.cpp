#pragma once

#include <string_view>

#include "emel/io/read/context.hpp"
#include "emel/io/read/detail.hpp"

namespace emel::io::read::guard {

struct request_span_valid {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &) const noexcept {
    return ev.request.request.byte_size > 0u &&
           static_cast<bool>(ev.request.on_done);
  }
};

struct request_span_invalid {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !request_span_valid{}(ev, ctx);
  }
};

struct file_path_valid {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &) const noexcept {
    const auto path = ev.request.request.file_path;
    return !path.empty() && path.size() <= k_max_file_path_bytes &&
           path.find('\0') == std::string_view::npos;
  }
};

struct file_path_invalid {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !file_path_valid{}(ev, ctx);
  }
};

struct file_index_valid {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &) const noexcept {
    return ev.request.request.file_index <= k_max_file_index;
  }
};

struct file_index_invalid {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !file_index_valid{}(ev, ctx);
  }
};

struct length_within_bounds {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &) const noexcept {
    return ev.request.request.byte_size <= k_max_read_bytes;
  }
};

struct length_overflow {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !length_within_bounds{}(ev, ctx);
  }
};

struct layout_supported {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &) const noexcept {
    constexpr uint64_t addr_max = static_cast<uint64_t>(-1);
    const uint64_t offset = ev.request.request.file_offset;
    const uint64_t size = ev.request.request.byte_size;
    return offset <= (addr_max - size);
  }
};

struct layout_unsupported {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !layout_supported{}(ev, ctx);
  }
};

struct target_buffer_valid {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &) const noexcept {
    return ev.request.request.target_buffer != nullptr &&
           ev.request.request.target_buffer_bytes >= ev.request.request.byte_size;
  }
};

struct target_buffer_invalid {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !target_buffer_valid{}(ev, ctx);
  }
};

struct platform_read_supported {
  bool operator()(const detail::read_tensor_runtime &,
                  const action::context &) const noexcept {
    if constexpr (EMEL_IO_READ_PLATFORM_SUPPORTED != 0) {
      return true;
    } else {
      return false;
    }
  }
};

struct platform_read_unsupported {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !platform_read_supported{}(ev, ctx);
  }
};

struct file_open_succeeded {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &) const noexcept {
    return ev.status.file_open_ok;
  }
};

struct file_open_failed {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !file_open_succeeded{}(ev, ctx);
  }
};

struct file_seek_succeeded {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &) const noexcept {
    return ev.status.file_seek_ok;
  }
};

struct file_seek_failed {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !file_seek_succeeded{}(ev, ctx);
  }
};

struct file_read_succeeded {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &) const noexcept {
    return ev.status.file_read_ok &&
           ev.status.bytes_copied == ev.request.request.byte_size;
  }
};

struct file_read_failed {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &) const noexcept {
    return !ev.status.file_read_ok;
  }
};

struct file_read_short {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &) const noexcept {
    return ev.status.file_read_ok &&
           ev.status.bytes_copied != ev.request.request.byte_size;
  }
};

struct error_callback_present {
  bool operator()(const detail::read_tensor_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct error_callback_absent {
  bool operator()(const detail::read_tensor_runtime &ev) const noexcept {
    return !error_callback_present{}(ev);
  }
};

} // namespace emel::io::read::guard

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

#include "emel/io/read/context.hpp"
#include "emel/io/read/detail.hpp"

namespace emel::io::read::guard {

inline bool batch_span_request_valid(
    const emel::io::event::tensor_load_span &span) noexcept {
  return span.byte_size > 0u && span.target != nullptr &&
         span.target_bytes >= span.byte_size;
}

inline bool batch_span_resource_supported(
    const emel::io::event::tensor_load_span &span) noexcept {
  constexpr uint64_t size_max =
      static_cast<uint64_t>(static_cast<std::size_t>(-1));
  constexpr uint64_t addr_max = static_cast<uint64_t>(-1);
  const auto path = span.file_path;
  return !path.empty() && path.size() <= k_max_file_path_bytes &&
         path.find('\0') == std::string_view::npos &&
         span.file_index <= k_max_file_index &&
         span.byte_size <= k_max_read_bytes && span.byte_size <= size_max &&
         span.file_offset <= (addr_max - span.byte_size);
}

inline bool batch_span_source_open_succeeded(
    const emel::io::event::tensor_load_span &span) noexcept {
  const auto source_error = span.source_error;
  return source_error != emel::error::cast(error::file_open_failed) &&
         (source_error != emel::error::cast(error::none) ||
          span.source_buffer != nullptr);
}

inline bool batch_span_source_seek_succeeded(
    const emel::io::event::tensor_load_span &span) noexcept {
  return span.source_error != emel::error::cast(error::file_seek_failed) &&
         (span.source_error != emel::error::cast(error::none) ||
          span.file_offset <= span.source_buffer_bytes);
}

inline bool batch_span_file_read_failed(
    const emel::io::event::tensor_load_span &span) noexcept {
  const auto source_error = span.source_error;
  return source_error != emel::error::cast(error::none) &&
         source_error != emel::error::cast(error::short_read);
}

inline bool
batch_span_short_read(const emel::io::event::tensor_load_span &span) noexcept {
  return span.source_error == emel::error::cast(error::short_read) ||
         (span.source_error == emel::error::cast(error::none) &&
          span.file_offset <= span.source_buffer_bytes &&
          span.byte_size > span.source_buffer_bytes - span.file_offset);
}

inline bool batch_span_file_read_succeeded(
    const emel::io::event::tensor_load_span &span) noexcept {
  return span.source_error == emel::error::cast(error::none) &&
         span.file_offset <= span.source_buffer_bytes &&
         span.byte_size <= span.source_buffer_bytes - span.file_offset;
}

inline void update_first_batch_failure_index(uint32_t &failed_index,
                                             uint32_t &found,
                                             const uint32_t index,
                                             const bool failed) noexcept {
  const uint32_t failed_value = static_cast<uint32_t>(failed);
  const uint32_t take = (1u - found) * failed_value;
  failed_index = (failed_index * (1u - take)) + (index * take);
  found = found | failed_value;
}

inline uint32_t first_batch_invalid_request_index(
    const detail::read_tensor_batch_runtime &ev) noexcept {
  uint32_t failed_index = 0u;
  uint32_t found = 0u;
  for (uint32_t index = 0u;
       index < static_cast<uint32_t>(ev.request.tensors.size()); ++index) {
    update_first_batch_failure_index(
        failed_index, found, index,
        !batch_span_request_valid(ev.request.tensors[index]));
  }
  return failed_index;
}

inline uint32_t first_batch_unsupported_resource_index(
    const detail::read_tensor_batch_runtime &ev) noexcept {
  uint32_t failed_index = 0u;
  uint32_t found = 0u;
  for (uint32_t index = 0u;
       index < static_cast<uint32_t>(ev.request.tensors.size()); ++index) {
    update_first_batch_failure_index(
        failed_index, found, index,
        !batch_span_resource_supported(ev.request.tensors[index]));
  }
  return failed_index;
}

inline uint32_t first_batch_source_open_failed_index(
    const detail::read_tensor_batch_runtime &ev) noexcept {
  uint32_t failed_index = 0u;
  uint32_t found = 0u;
  for (uint32_t index = 0u;
       index < static_cast<uint32_t>(ev.request.tensors.size()); ++index) {
    update_first_batch_failure_index(
        failed_index, found, index,
        !batch_span_source_open_succeeded(ev.request.tensors[index]));
  }
  return failed_index;
}

inline uint32_t first_batch_source_seek_failed_index(
    const detail::read_tensor_batch_runtime &ev) noexcept {
  uint32_t failed_index = 0u;
  uint32_t found = 0u;
  for (uint32_t index = 0u;
       index < static_cast<uint32_t>(ev.request.tensors.size()); ++index) {
    update_first_batch_failure_index(
        failed_index, found, index,
        !batch_span_source_seek_succeeded(ev.request.tensors[index]));
  }
  return failed_index;
}

inline uint32_t first_batch_file_read_failed_index(
    const detail::read_tensor_batch_runtime &ev) noexcept {
  uint32_t failed_index = 0u;
  uint32_t found = 0u;
  for (uint32_t index = 0u;
       index < static_cast<uint32_t>(ev.request.tensors.size()); ++index) {
    update_first_batch_failure_index(
        failed_index, found, index,
        batch_span_file_read_failed(ev.request.tensors[index]));
  }
  return failed_index;
}

inline uint32_t first_batch_short_read_index(
    const detail::read_tensor_batch_runtime &ev) noexcept {
  uint32_t failed_index = 0u;
  uint32_t found = 0u;
  for (uint32_t index = 0u;
       index < static_cast<uint32_t>(ev.request.tensors.size()); ++index) {
    update_first_batch_failure_index(
        failed_index, found, index,
        batch_span_short_read(ev.request.tensors[index]));
  }
  return failed_index;
}

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
    constexpr uint64_t size_max =
        static_cast<uint64_t>(static_cast<std::size_t>(-1));
    return ev.request.request.byte_size <= k_max_read_bytes &&
           ev.request.request.byte_size <= size_max;
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
           ev.request.request.target_buffer_bytes >=
               ev.request.request.byte_size;
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
    const auto source_error = ev.request.request.source_error;
    return ev.request.request.source_error !=
               emel::error::cast(error::file_open_failed) &&
           (source_error != emel::error::cast(error::none) ||
            ev.request.request.source_buffer != nullptr);
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
    const auto &request = ev.request.request;
    return request.source_error != emel::error::cast(error::file_seek_failed) &&
           (request.source_error != emel::error::cast(error::none) ||
            request.file_offset <= request.source_buffer_bytes);
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
    const auto &request = ev.request.request;
    return request.source_error == emel::error::cast(error::none) &&
           request.file_offset <= request.source_buffer_bytes &&
           request.byte_size <=
               request.source_buffer_bytes - request.file_offset;
  }
};

struct file_read_failed {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &) const noexcept {
    const auto source_error = ev.request.request.source_error;
    return source_error != emel::error::cast(error::none) &&
           source_error != emel::error::cast(error::short_read);
  }
};

struct file_read_short {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &) const noexcept {
    const auto &request = ev.request.request;
    return request.source_error == emel::error::cast(error::short_read) ||
           (request.source_error == emel::error::cast(error::none) &&
            request.file_offset <= request.source_buffer_bytes &&
            request.byte_size >
                request.source_buffer_bytes - request.file_offset);
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

struct batch_count_valid {
  bool operator()(const detail::read_tensor_batch_runtime &ev,
                  const action::context &) const noexcept {
    return !ev.request.tensors.empty() &&
           ev.request.tensors.size() <= k_max_read_batch_tensors &&
           ev.request.tensors.size() <= std::numeric_limits<uint32_t>::max();
  }
};

struct batch_count_invalid {
  bool operator()(const detail::read_tensor_batch_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !batch_count_valid{}(ev, ctx);
  }
};

struct batch_request_valid {
  bool operator()(const detail::read_tensor_batch_runtime &ev,
                  const action::context &) const noexcept {
    bool valid = static_cast<bool>(ev.request.on_done);
    for (uint32_t index = 0u;
         valid && index < static_cast<uint32_t>(ev.request.tensors.size());
         ++index) {
      valid = batch_span_request_valid(ev.request.tensors[index]);
    }
    return valid;
  }
};

struct batch_request_invalid {
  bool operator()(const detail::read_tensor_batch_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !batch_request_valid{}(ev, ctx);
  }
};

struct batch_resource_supported {
  bool operator()(const detail::read_tensor_batch_runtime &ev,
                  const action::context &) const noexcept {
    bool supported = true;
    for (uint32_t index = 0u;
         supported && index < static_cast<uint32_t>(ev.request.tensors.size());
         ++index) {
      supported = batch_span_resource_supported(ev.request.tensors[index]);
    }
    return supported;
  }
};

struct batch_resource_unsupported {
  bool operator()(const detail::read_tensor_batch_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !batch_resource_supported{}(ev, ctx);
  }
};

struct batch_source_open_succeeded {
  bool operator()(const detail::read_tensor_batch_runtime &ev,
                  const action::context &) const noexcept {
    bool succeeded = true;
    for (uint32_t index = 0u;
         succeeded && index < static_cast<uint32_t>(ev.request.tensors.size());
         ++index) {
      succeeded = batch_span_source_open_succeeded(ev.request.tensors[index]);
    }
    return succeeded;
  }
};

struct batch_source_open_failed {
  bool operator()(const detail::read_tensor_batch_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !batch_source_open_succeeded{}(ev, ctx);
  }
};

struct batch_source_seek_succeeded {
  bool operator()(const detail::read_tensor_batch_runtime &ev,
                  const action::context &) const noexcept {
    bool succeeded = true;
    for (uint32_t index = 0u;
         succeeded && index < static_cast<uint32_t>(ev.request.tensors.size());
         ++index) {
      succeeded = batch_span_source_seek_succeeded(ev.request.tensors[index]);
    }
    return succeeded;
  }
};

struct batch_source_seek_failed {
  bool operator()(const detail::read_tensor_batch_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !batch_source_seek_succeeded{}(ev, ctx);
  }
};

struct batch_file_read_failed {
  bool operator()(const detail::read_tensor_batch_runtime &ev,
                  const action::context &) const noexcept {
    bool failed = false;
    for (uint32_t index = 0u;
         !failed && index < static_cast<uint32_t>(ev.request.tensors.size());
         ++index) {
      failed = batch_span_file_read_failed(ev.request.tensors[index]);
    }
    return failed;
  }
};

struct batch_file_read_short {
  bool operator()(const detail::read_tensor_batch_runtime &ev,
                  const action::context &) const noexcept {
    bool short_read = false;
    for (uint32_t index = 0u;
         !short_read &&
         index < static_cast<uint32_t>(ev.request.tensors.size());
         ++index) {
      short_read = batch_span_short_read(ev.request.tensors[index]);
    }
    return short_read;
  }
};

struct batch_file_read_succeeded {
  bool operator()(const detail::read_tensor_batch_runtime &ev,
                  const action::context &) const noexcept {
    bool succeeded = true;
    for (uint32_t index = 0u;
         succeeded && index < static_cast<uint32_t>(ev.request.tensors.size());
         ++index) {
      succeeded = batch_span_file_read_succeeded(ev.request.tensors[index]);
    }
    return succeeded;
  }
};

struct batch_done_callback_present {
  bool operator()(const detail::read_tensor_batch_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_done);
  }
};

struct batch_error_callback_present {
  bool operator()(const detail::read_tensor_batch_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct batch_error_callback_absent {
  bool operator()(const detail::read_tensor_batch_runtime &ev) const noexcept {
    return !batch_error_callback_present{}(ev);
  }
};

} // namespace emel::io::read::guard

#pragma once

#include <cstdint>

#include "emel/io/mmap/context.hpp"
#include "emel/io/mmap/detail.hpp"
#include "emel/io/mmap/errors.hpp"
#include "emel/io/mmap/events.hpp"

namespace emel::io::mmap::guard {

struct request_span_valid {
  bool operator()(const detail::map_tensor_runtime &ev,
                  const action::context &) const noexcept {
    return ev.request.request.byte_size > 0u &&
           static_cast<bool>(ev.request.on_done);
  }
};

struct request_span_invalid {
  bool operator()(const detail::map_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !request_span_valid{}(ev, ctx);
  }
};

struct file_path_valid {
  bool operator()(const detail::map_tensor_runtime &ev,
                  const action::context &) const noexcept {
    const auto path = ev.request.request.file_path;
    return !path.empty() && path.size() <= k_max_file_path_bytes &&
           path.find('\0') == std::string_view::npos;
  }
};

struct file_path_invalid {
  bool operator()(const detail::map_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !file_path_valid{}(ev, ctx);
  }
};

struct file_index_valid {
  bool operator()(const detail::map_tensor_runtime &ev,
                  const action::context &) const noexcept {
    return ev.request.request.file_index <= k_max_file_index;
  }
};

struct file_index_invalid {
  bool operator()(const detail::map_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !file_index_valid{}(ev, ctx);
  }
};

struct offset_aligned {
  bool operator()(const detail::map_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return ctx.required_offset_alignment != 0u &&
           (ev.request.request.file_offset % ctx.required_offset_alignment) ==
               0u;
  }
};

struct offset_unaligned {
  bool operator()(const detail::map_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !offset_aligned{}(ev, ctx);
  }
};

struct length_within_bounds {
  bool operator()(const detail::map_tensor_runtime &ev,
                  const action::context &) const noexcept {
    return ev.request.request.byte_size <= k_max_mapping_bytes;
  }
};

struct length_overflow {
  bool operator()(const detail::map_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !length_within_bounds{}(ev, ctx);
  }
};

struct layout_supported {
  bool operator()(const detail::map_tensor_runtime &ev,
                  const action::context &) const noexcept {
    constexpr uint64_t addr_max = static_cast<uint64_t>(-1);
    const uint64_t offset = ev.request.request.file_offset;
    const uint64_t size = ev.request.request.byte_size;
    return offset <= (addr_max - size);
  }
};

struct layout_unsupported {
  bool operator()(const detail::map_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !layout_supported{}(ev, ctx);
  }
};

struct platform_mmap_supported {
  bool operator()(const detail::map_tensor_runtime &,
                  const action::context &) const noexcept {
    if constexpr (EMEL_IO_MMAP_PLATFORM_SUPPORTED != 0) {
      return true;
    } else {
      return false;
    }
  }
};

struct platform_mmap_unsupported {
  bool operator()(const detail::map_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !platform_mmap_supported{}(ev, ctx);
  }
};

struct slot_capacity_available {
  bool operator()(const detail::map_tensor_runtime &,
                  const action::context &ctx) const noexcept {
    return ctx.free_count > 0u;
  }
};

struct slot_pool_exhausted {
  bool operator()(const detail::map_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !slot_capacity_available{}(ev, ctx);
  }
};

struct file_open_succeeded {
  bool operator()(const detail::map_tensor_runtime &ev,
                  const action::context &) const noexcept {
    return ev.status.file_open_ok;
  }
};

struct file_open_failed {
  bool operator()(const detail::map_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !file_open_succeeded{}(ev, ctx);
  }
};

struct file_span_within_file {
  bool operator()(const detail::map_tensor_runtime &ev,
                  const action::context &) const noexcept {
    const uint64_t offset = ev.request.request.file_offset;
    const uint64_t size = ev.request.request.byte_size;
    return ev.status.file_size_ok && offset <= ev.status.file_size_bytes &&
           size <= (ev.status.file_size_bytes - offset);
  }
};

struct file_span_exceeds_file {
  bool operator()(const detail::map_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !file_span_within_file{}(ev, ctx);
  }
};

struct mapping_succeeded {
  bool operator()(const detail::map_tensor_runtime &ev,
                  const action::context &) const noexcept {
    return ev.status.mapping_ok;
  }
};

struct mapping_failed {
  bool operator()(const detail::map_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !mapping_succeeded{}(ev, ctx);
  }
};

struct done_callback_present {
  bool operator()(const detail::map_tensor_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_done);
  }
};

struct error_callback_present {
  bool operator()(const detail::map_tensor_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct error_callback_absent {
  bool operator()(const detail::map_tensor_runtime &ev) const noexcept {
    return !error_callback_present{}(ev);
  }
};

struct release_handle_in_range {
  bool operator()(const detail::release_mapping_runtime &ev,
                  const action::context &) const noexcept {
    return ev.request.handle < k_max_mappings;
  }
};

struct release_handle_out_of_range {
  bool operator()(const detail::release_mapping_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !release_handle_in_range{}(ev, ctx);
  }
};

struct release_slot_in_use {
  bool operator()(const detail::release_mapping_runtime &ev,
                  const action::context &ctx) const noexcept {
    return ctx.slots[ev.request.handle].in_use;
  }
};

struct release_slot_not_in_use {
  bool operator()(const detail::release_mapping_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !release_slot_in_use{}(ev, ctx);
  }
};

struct release_slot_owned_by_tensor {
  bool operator()(const detail::release_mapping_runtime &ev,
                  const action::context &ctx) const noexcept {
    return ctx.slots[ev.request.handle].tensor_id == ev.request.tensor_id;
  }
};

struct release_slot_not_owned_by_tensor {
  bool operator()(const detail::release_mapping_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !release_slot_owned_by_tensor{}(ev, ctx);
  }
};

struct release_slot_in_use_owned_by_tensor {
  bool operator()(const detail::release_mapping_runtime &ev,
                  const action::context &ctx) const noexcept {
    return release_slot_in_use{}(ev, ctx) &&
           release_slot_owned_by_tensor{}(ev, ctx);
  }
};

struct release_slot_in_use_not_owned_by_tensor {
  bool operator()(const detail::release_mapping_runtime &ev,
                  const action::context &ctx) const noexcept {
    return release_slot_in_use{}(ev, ctx) &&
           release_slot_not_owned_by_tensor{}(ev, ctx);
  }
};

struct unmap_succeeded {
  bool operator()(const detail::release_mapping_runtime &ev,
                  const action::context &) const noexcept {
    return ev.status.unmap_ok;
  }
};

struct unmap_failed {
  bool operator()(const detail::release_mapping_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !unmap_succeeded{}(ev, ctx);
  }
};

struct release_done_callback_present {
  bool operator()(const detail::release_mapping_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_done);
  }
};

struct release_done_callback_absent {
  bool operator()(const detail::release_mapping_runtime &ev) const noexcept {
    return !release_done_callback_present{}(ev);
  }
};

struct release_error_callback_present {
  bool operator()(const detail::release_mapping_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct release_error_callback_absent {
  bool operator()(const detail::release_mapping_runtime &ev) const noexcept {
    return !release_error_callback_present{}(ev);
  }
};

} // namespace emel::io::mmap::guard

#pragma once

#include "emel/io/mmap/context.hpp"
#include "emel/io/mmap/detail.hpp"
#include "emel/io/mmap/errors.hpp"
#include "emel/io/mmap/events.hpp"

namespace emel::io::mmap::action {

struct effect_begin_map_tensor {
  void operator()(const detail::map_tensor_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::none);
    ev.status.ok = false;
    ev.status.reserved_slot = k_invalid_mapping_handle;
    ev.status.os_resource = -1;
    ev.status.mapped_base = nullptr;
    ev.status.mapped_bytes = 0u;
    ev.status.file_open_ok = false;
    ev.status.mapping_ok = false;
  }
};

struct effect_mark_invalid_request {
  void operator()(const detail::map_tensor_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::invalid_request);
    ev.status.ok = false;
  }
};

struct effect_mark_unsupported_file {
  void operator()(const detail::map_tensor_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::unsupported_resource);
    ev.status.ok = false;
  }
};

struct effect_mark_unsupported_offset {
  void operator()(const detail::map_tensor_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::unsupported_resource);
    ev.status.ok = false;
  }
};

struct effect_mark_unsupported_length {
  void operator()(const detail::map_tensor_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::unsupported_resource);
    ev.status.ok = false;
  }
};

struct effect_mark_unsupported_layout {
  void operator()(const detail::map_tensor_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::unsupported_resource);
    ev.status.ok = false;
  }
};

struct effect_mark_unsupported_platform {
  void operator()(const detail::map_tensor_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::unsupported_platform);
    ev.status.ok = false;
  }
};

struct effect_mark_resource_exhausted {
  void operator()(const detail::map_tensor_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::resource_exhausted);
    ev.status.ok = false;
  }
};

struct effect_reserve_top_free_slot_then_attempt_open {
  void operator()(const detail::map_tensor_runtime &ev,
                  context &ctx) const noexcept;
};

struct effect_attempt_mapping {
  void operator()(const detail::map_tensor_runtime &ev,
                  context &ctx) const noexcept;
};

struct effect_commit_mapping {
  void operator()(const detail::map_tensor_runtime &ev,
                  context &ctx) const noexcept {
    auto &slot_ref = ctx.slots[ev.status.reserved_slot];
    slot_ref.base = ev.status.mapped_base;
    slot_ref.mapped_bytes = ev.status.mapped_bytes;
    slot_ref.os_resource = ev.status.os_resource;
    slot_ref.file_offset = ev.request.request.file_offset;
    slot_ref.requested_bytes = ev.request.request.byte_size;
    ev.status.ok = true;
  }
};

struct effect_release_reserved_slot_on_open_failure {
  void operator()(const detail::map_tensor_runtime &ev,
                  context &ctx) const noexcept {
    auto &slot_ref = ctx.slots[ev.status.reserved_slot];
    slot_ref.in_use = false;
    ctx.free_stack[ctx.free_count] = ev.status.reserved_slot;
    ctx.free_count += 1u;
    ev.status.err = emel::error::cast(error::file_open_failed);
    ev.status.ok = false;
  }
};

struct effect_close_open_resource_and_release_slot_on_mapping_failure {
  void operator()(const detail::map_tensor_runtime &ev,
                  context &ctx) const noexcept;
};

struct effect_publish_map_tensor_done {
  void operator()(const detail::map_tensor_runtime &ev,
                  context &) const noexcept {
    ev.request.on_done(events::map_tensor_done{
        .request = ev.request,
        .handle = ev.status.reserved_slot,
        .buffer = ev.status.mapped_base,
        .buffer_bytes = ev.status.mapped_bytes,
    });
  }
};

struct effect_record_map_tensor_done {
  void operator()(const detail::map_tensor_runtime &,
                  context &) const noexcept {}
};

struct effect_publish_map_tensor_error {
  void operator()(const detail::map_tensor_runtime &ev,
                  context &) const noexcept {
    ev.request.on_error(events::map_tensor_error{
        .request = ev.request,
        .err = ev.status.err,
    });
  }
};

struct effect_record_map_tensor_error {
  void operator()(const detail::map_tensor_runtime &,
                  context &) const noexcept {}
};

struct effect_begin_release {
  void operator()(const detail::release_mapping_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::none);
    ev.status.ok = false;
    ev.status.target_slot = ev.request.handle;
    ev.status.unmap_base = nullptr;
    ev.status.unmap_bytes = 0u;
    ev.status.os_resource = -1;
    ev.status.unmap_ok = false;
  }
};

struct effect_mark_release_invalid_handle {
  void operator()(const detail::release_mapping_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::invalid_request);
    ev.status.ok = false;
  }
};

struct effect_attempt_unmap {
  void operator()(const detail::release_mapping_runtime &ev,
                  context &ctx) const noexcept;
};

struct effect_release_slot_after_unmap {
  void operator()(const detail::release_mapping_runtime &ev,
                  context &ctx) const noexcept {
    auto &slot_ref = ctx.slots[ev.status.target_slot];
    slot_ref.in_use = false;
    slot_ref.base = nullptr;
    slot_ref.mapped_bytes = 0u;
    slot_ref.os_resource = -1;
    slot_ref.file_offset = 0u;
    slot_ref.requested_bytes = 0u;
    ctx.free_stack[ctx.free_count] = ev.status.target_slot;
    ctx.free_count += 1u;
    ev.status.ok = true;
  }
};

struct effect_mark_unmap_failed_and_release_slot {
  void operator()(const detail::release_mapping_runtime &ev,
                  context &ctx) const noexcept {
    auto &slot_ref = ctx.slots[ev.status.target_slot];
    slot_ref.in_use = false;
    slot_ref.base = nullptr;
    slot_ref.mapped_bytes = 0u;
    slot_ref.os_resource = -1;
    slot_ref.file_offset = 0u;
    slot_ref.requested_bytes = 0u;
    ctx.free_stack[ctx.free_count] = ev.status.target_slot;
    ctx.free_count += 1u;
    ev.status.err = emel::error::cast(error::unmap_failed);
    ev.status.ok = false;
  }
};

struct effect_publish_release_mapping_done {
  void operator()(const detail::release_mapping_runtime &ev,
                  context &) const noexcept {
    ev.request.on_done(events::release_mapping_done{
        .request = ev.request,
    });
  }
};

struct effect_record_release_mapping_done {
  void operator()(const detail::release_mapping_runtime &,
                  context &) const noexcept {}
};

struct effect_publish_release_mapping_error {
  void operator()(const detail::release_mapping_runtime &ev,
                  context &) const noexcept {
    ev.request.on_error(events::release_mapping_error{
        .request = ev.request,
        .err = ev.status.err,
    });
  }
};

struct effect_record_release_mapping_error {
  void operator()(const detail::release_mapping_runtime &,
                  context &) const noexcept {}
};

struct effect_on_unexpected {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    if constexpr (requires { ev.status.err; }) {
      ev.status.err = emel::error::cast(error::internal_error);
      ev.status.ok = false;
    }
  }
};

inline constexpr effect_begin_map_tensor effect_begin_map_tensor{};
inline constexpr effect_mark_invalid_request effect_mark_invalid_request{};
inline constexpr effect_mark_unsupported_file effect_mark_unsupported_file{};
inline constexpr effect_mark_unsupported_offset
    effect_mark_unsupported_offset{};
inline constexpr effect_mark_unsupported_length
    effect_mark_unsupported_length{};
inline constexpr effect_mark_unsupported_layout
    effect_mark_unsupported_layout{};
inline constexpr effect_mark_unsupported_platform
    effect_mark_unsupported_platform{};
inline constexpr effect_mark_resource_exhausted
    effect_mark_resource_exhausted{};
inline constexpr effect_reserve_top_free_slot_then_attempt_open
    effect_reserve_top_free_slot_then_attempt_open{};
inline constexpr effect_attempt_mapping effect_attempt_mapping{};
inline constexpr effect_commit_mapping effect_commit_mapping{};
inline constexpr effect_release_reserved_slot_on_open_failure
    effect_release_reserved_slot_on_open_failure{};
inline constexpr effect_close_open_resource_and_release_slot_on_mapping_failure
    effect_close_open_resource_and_release_slot_on_mapping_failure{};
inline constexpr effect_publish_map_tensor_done
    effect_publish_map_tensor_done{};
inline constexpr effect_record_map_tensor_done effect_record_map_tensor_done{};
inline constexpr effect_publish_map_tensor_error
    effect_publish_map_tensor_error{};
inline constexpr effect_record_map_tensor_error
    effect_record_map_tensor_error{};
inline constexpr effect_begin_release effect_begin_release{};
inline constexpr effect_mark_release_invalid_handle
    effect_mark_release_invalid_handle{};
inline constexpr effect_attempt_unmap effect_attempt_unmap{};
inline constexpr effect_release_slot_after_unmap
    effect_release_slot_after_unmap{};
inline constexpr effect_mark_unmap_failed_and_release_slot
    effect_mark_unmap_failed_and_release_slot{};
inline constexpr effect_publish_release_mapping_done
    effect_publish_release_mapping_done{};
inline constexpr effect_record_release_mapping_done
    effect_record_release_mapping_done{};
inline constexpr effect_publish_release_mapping_error
    effect_publish_release_mapping_error{};
inline constexpr effect_record_release_mapping_error
    effect_record_release_mapping_error{};
inline constexpr effect_on_unexpected effect_on_unexpected{};

} // namespace emel::io::mmap::action

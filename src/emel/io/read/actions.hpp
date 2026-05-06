#pragma once

#include <cstddef>
#include <cstring>

#include "emel/io/read/context.hpp"
#include "emel/io/read/detail.hpp"
#include "emel/io/read/errors.hpp"
#include "emel/io/read/events.hpp"
#include "emel/io/read/guards.hpp"

namespace emel::io::read::action {

struct effect_begin_read_tensor {
  void operator()(const detail::read_tensor_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::none);
    ev.status.ok = false;
    ev.status.bytes_copied = 0u;
  }
};

struct effect_begin_read_tensor_batch {
  void operator()(const detail::read_tensor_batch_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::none);
    ev.status.ok = false;
    ev.status.done_count = 0u;
    ev.status.bytes_copied = 0u;
    ev.status.failed_index = 0u;
  }
};

struct effect_mark_unsupported_platform {
  void operator()(const detail::read_tensor_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::unsupported_platform);
    ev.status.ok = false;
  }
};

struct effect_mark_invalid_request {
  void operator()(const detail::read_tensor_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::invalid_request);
    ev.status.ok = false;
  }
};

struct effect_mark_unsupported_resource {
  void operator()(const detail::read_tensor_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::unsupported_resource);
    ev.status.ok = false;
  }
};

struct effect_mark_read_tensor_batch_invalid_request {
  void operator()(const detail::read_tensor_batch_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::invalid_request);
    ev.status.ok = false;
    ev.status.failed_index = guard::first_batch_invalid_request_index(ev);
  }
};

struct effect_mark_read_tensor_batch_count_invalid {
  void operator()(const detail::read_tensor_batch_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::invalid_request);
    ev.status.ok = false;
    ev.status.failed_index = 0u;
  }
};

struct effect_mark_read_tensor_batch_unsupported_resource {
  void operator()(const detail::read_tensor_batch_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::unsupported_resource);
    ev.status.ok = false;
    ev.status.failed_index = guard::first_batch_unsupported_resource_index(ev);
  }
};

struct effect_prepare_read_attempt {
  void operator()(const detail::read_tensor_runtime &ev,
                  context &) const noexcept {
    ev.status.bytes_copied = 0u;
  }
};

struct effect_prepare_read_copy {
  void operator()(const detail::read_tensor_runtime &ev,
                  context &) const noexcept {
    ev.status.bytes_copied = 0u;
  }
};

struct effect_mark_read_tensor_batch_file_open_failed {
  void operator()(const detail::read_tensor_batch_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::file_open_failed);
    ev.status.ok = false;
    ev.status.failed_index = guard::first_batch_source_open_failed_index(ev);
  }
};

struct effect_mark_read_tensor_batch_file_seek_failed {
  void operator()(const detail::read_tensor_batch_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::file_seek_failed);
    ev.status.ok = false;
    ev.status.failed_index = guard::first_batch_source_seek_failed_index(ev);
  }
};

struct effect_mark_read_tensor_batch_file_read_failed {
  void operator()(const detail::read_tensor_batch_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::file_read_failed);
    ev.status.ok = false;
    ev.status.failed_index = guard::first_batch_file_read_failed_index(ev);
  }
};

struct effect_mark_read_tensor_batch_short_read {
  void operator()(const detail::read_tensor_batch_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::short_read);
    ev.status.ok = false;
    ev.status.failed_index = guard::first_batch_short_read_index(ev);
  }
};

struct effect_mark_file_open_failed {
  void operator()(const detail::read_tensor_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::file_open_failed);
    ev.status.ok = false;
  }
};

struct effect_mark_file_seek_failed {
  void operator()(const detail::read_tensor_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::file_seek_failed);
    ev.status.ok = false;
  }
};

struct effect_mark_file_read_failed {
  void operator()(const detail::read_tensor_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::file_read_failed);
    ev.status.ok = false;
  }
};

struct effect_mark_short_read {
  void operator()(const detail::read_tensor_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::short_read);
    ev.status.ok = false;
  }
};

struct effect_mark_read_tensor_done {
  void operator()(const detail::read_tensor_runtime &ev,
                  context &) const noexcept {
    const auto &request = ev.request.request;
    const auto *source =
        static_cast<const unsigned char *>(request.source_buffer);
    auto *target = static_cast<unsigned char *>(request.target_buffer);
    std::memcpy(target, source + request.file_offset,
                static_cast<std::size_t>(request.byte_size));
    ev.status.err = emel::error::cast(error::none);
    ev.status.bytes_copied = ev.request.request.byte_size;
    ev.status.ok = true;
  }
};

struct effect_mark_read_tensor_batch_done {
  void operator()(const detail::read_tensor_batch_runtime &ev,
                  context &) const noexcept {
    uint64_t bytes_copied = 0u;
    for (uint32_t index = 0u;
         index < static_cast<uint32_t>(ev.request.tensors.size()); ++index) {
      const auto &span = ev.request.tensors[index];
      const auto *source =
          static_cast<const unsigned char *>(span.source_buffer);
      auto *target = static_cast<unsigned char *>(span.target);
      std::memcpy(target, source + span.file_offset,
                  static_cast<std::size_t>(span.byte_size));
      bytes_copied += span.byte_size;
    }
    ev.status.err = emel::error::cast(error::none);
    ev.status.done_count = static_cast<uint32_t>(ev.request.tensors.size());
    ev.status.bytes_copied = bytes_copied;
    ev.status.ok = true;
  }
};

struct effect_publish_read_tensor_done {
  void operator()(const detail::read_tensor_runtime &ev,
                  context &) const noexcept {
    ev.request.on_done(events::read_tensor_done{
        .request = ev.request,
        .bytes_copied = ev.status.bytes_copied,
        .target_buffer = ev.request.request.target_buffer,
    });
  }
};

struct effect_publish_read_tensor_batch_done {
  void operator()(const detail::read_tensor_batch_runtime &ev,
                  context &) const noexcept {
    ev.request.on_done(events::read_tensor_batch_done{
        .request = ev.request,
        .done_count = ev.status.done_count,
        .bytes_copied = ev.status.bytes_copied,
    });
  }
};

struct effect_publish_read_tensor_error {
  void operator()(const detail::read_tensor_runtime &ev,
                  context &) const noexcept {
    ev.request.on_error(events::read_tensor_error{
        .request = ev.request,
        .err = ev.status.err,
    });
  }
};

struct effect_publish_read_tensor_batch_error {
  void operator()(const detail::read_tensor_batch_runtime &ev,
                  context &) const noexcept {
    ev.request.on_error(events::read_tensor_batch_error{
        .request = ev.request,
        .err = ev.status.err,
        .failed_index = ev.status.failed_index,
    });
  }
};

struct effect_record_read_tensor_error {
  void operator()(const detail::read_tensor_runtime &,
                  context &) const noexcept {}
};

struct effect_record_read_tensor_batch_error {
  void operator()(const detail::read_tensor_batch_runtime &,
                  context &) const noexcept {}
};

struct effect_on_unexpected {
  // Boundary unexpected handler is a deterministic no-op: foreign events drop
  // and the actor returns to `state_ready` without observable side effects.
  template <class event_type>
  void operator()(const event_type &, context &) const noexcept {}
};

inline constexpr effect_begin_read_tensor effect_begin_read_tensor{};
inline constexpr effect_begin_read_tensor_batch
    effect_begin_read_tensor_batch{};
inline constexpr effect_mark_unsupported_platform
    effect_mark_unsupported_platform{};
inline constexpr effect_mark_invalid_request effect_mark_invalid_request{};
inline constexpr effect_mark_unsupported_resource
    effect_mark_unsupported_resource{};
inline constexpr effect_mark_read_tensor_batch_invalid_request
    effect_mark_read_tensor_batch_invalid_request{};
inline constexpr effect_mark_read_tensor_batch_count_invalid
    effect_mark_read_tensor_batch_count_invalid{};
inline constexpr effect_mark_read_tensor_batch_unsupported_resource
    effect_mark_read_tensor_batch_unsupported_resource{};
inline constexpr effect_prepare_read_attempt effect_prepare_read_attempt{};
inline constexpr effect_prepare_read_copy effect_prepare_read_copy{};
inline constexpr effect_mark_file_open_failed effect_mark_file_open_failed{};
inline constexpr effect_mark_file_seek_failed effect_mark_file_seek_failed{};
inline constexpr effect_mark_file_read_failed effect_mark_file_read_failed{};
inline constexpr effect_mark_short_read effect_mark_short_read{};
inline constexpr effect_mark_read_tensor_batch_file_open_failed
    effect_mark_read_tensor_batch_file_open_failed{};
inline constexpr effect_mark_read_tensor_batch_file_seek_failed
    effect_mark_read_tensor_batch_file_seek_failed{};
inline constexpr effect_mark_read_tensor_batch_file_read_failed
    effect_mark_read_tensor_batch_file_read_failed{};
inline constexpr effect_mark_read_tensor_batch_short_read
    effect_mark_read_tensor_batch_short_read{};
inline constexpr effect_mark_read_tensor_done effect_mark_read_tensor_done{};
inline constexpr effect_mark_read_tensor_batch_done
    effect_mark_read_tensor_batch_done{};
inline constexpr effect_publish_read_tensor_done
    effect_publish_read_tensor_done{};
inline constexpr effect_publish_read_tensor_batch_done
    effect_publish_read_tensor_batch_done{};
inline constexpr effect_publish_read_tensor_error
    effect_publish_read_tensor_error{};
inline constexpr effect_publish_read_tensor_batch_error
    effect_publish_read_tensor_batch_error{};
inline constexpr effect_record_read_tensor_error
    effect_record_read_tensor_error{};
inline constexpr effect_record_read_tensor_batch_error
    effect_record_read_tensor_batch_error{};
inline constexpr effect_on_unexpected effect_on_unexpected{};

} // namespace emel::io::read::action

#pragma once

#include "emel/io/read/context.hpp"
#include "emel/io/read/detail.hpp"
#include "emel/io/read/errors.hpp"
#include "emel/io/read/events.hpp"

namespace emel::io::read::action {

struct effect_begin_read_tensor {
  void operator()(const detail::read_tensor_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::none);
    ev.status.ok = false;
    ev.status.os_resource = -1;
    ev.status.bytes_copied = 0u;
    ev.status.file_open_ok = false;
    ev.status.file_seek_ok = false;
    ev.status.file_read_ok = false;
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

struct effect_attempt_open {
  void operator()(const detail::read_tensor_runtime &ev,
                  context &ctx) const noexcept;
};

struct effect_attempt_seek {
  void operator()(const detail::read_tensor_runtime &ev,
                  context &ctx) const noexcept;
};

struct effect_attempt_read_and_close {
  void operator()(const detail::read_tensor_runtime &ev,
                  context &ctx) const noexcept;
};

struct effect_mark_file_open_failed {
  void operator()(const detail::read_tensor_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::file_open_failed);
    ev.status.ok = false;
  }
};

struct effect_mark_file_seek_failed_and_close {
  void operator()(const detail::read_tensor_runtime &ev,
                  context &ctx) const noexcept;
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
    ev.status.err = emel::error::cast(error::none);
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

struct effect_publish_read_tensor_error {
  void operator()(const detail::read_tensor_runtime &ev,
                  context &) const noexcept {
    ev.request.on_error(events::read_tensor_error{
        .request = ev.request,
        .err = ev.status.err,
    });
  }
};

struct effect_record_read_tensor_error {
  void operator()(const detail::read_tensor_runtime &,
                  context &) const noexcept {}
};

struct effect_on_unexpected {
  // Boundary unexpected handler is a deterministic no-op: foreign events drop
  // and the actor returns to `state_ready` without observable side effects.
  template <class event_type>
  void operator()(const event_type &, context &) const noexcept {}
};

inline constexpr effect_begin_read_tensor effect_begin_read_tensor{};
inline constexpr effect_mark_unsupported_platform
    effect_mark_unsupported_platform{};
inline constexpr effect_mark_invalid_request effect_mark_invalid_request{};
inline constexpr effect_mark_unsupported_resource
    effect_mark_unsupported_resource{};
inline constexpr effect_attempt_open effect_attempt_open{};
inline constexpr effect_attempt_seek effect_attempt_seek{};
inline constexpr effect_attempt_read_and_close effect_attempt_read_and_close{};
inline constexpr effect_mark_file_open_failed effect_mark_file_open_failed{};
inline constexpr effect_mark_file_seek_failed_and_close
    effect_mark_file_seek_failed_and_close{};
inline constexpr effect_mark_file_read_failed effect_mark_file_read_failed{};
inline constexpr effect_mark_short_read effect_mark_short_read{};
inline constexpr effect_mark_read_tensor_done effect_mark_read_tensor_done{};
inline constexpr effect_publish_read_tensor_done
    effect_publish_read_tensor_done{};
inline constexpr effect_publish_read_tensor_error
    effect_publish_read_tensor_error{};
inline constexpr effect_record_read_tensor_error
    effect_record_read_tensor_error{};
inline constexpr effect_on_unexpected effect_on_unexpected{};

} // namespace emel::io::read::action

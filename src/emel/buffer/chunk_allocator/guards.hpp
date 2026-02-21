#pragma once

#include "emel/buffer/chunk_allocator/actions.hpp"

namespace emel::buffer::chunk_allocator::guard {

inline constexpr auto phase_ok = [](const action::context & c) {
  return c.phase_error == EMEL_OK;
};

inline constexpr auto phase_failed = [](const action::context & c) {
  return c.phase_error != EMEL_OK;
};

inline constexpr auto always = [](const action::context &) {
  return true;
};

struct no_error {
  template <class event>
  bool operator()(const event & ev, const action::context &) const noexcept {
    if constexpr (requires { ev.err; }) {
      return ev.err == EMEL_OK;
    }
    return true;
  }
};

struct has_error {
  template <class event>
  bool operator()(const event & ev, const action::context &) const noexcept {
    if constexpr (requires { ev.err; }) {
      return ev.err != EMEL_OK;
    }
    return false;
  }
};

struct valid_configure {
  bool operator()(const event::validate_configure &, const action::context & c) const noexcept {
    return action::detail::valid_alignment(c.pending_alignment) && c.pending_max_chunk_size > 0;
  }
};

struct can_configure {
  bool operator()(const event::configure & ev) const noexcept {
    return action::detail::valid_alignment(ev.alignment) && ev.max_chunk_size > 0;
  }
};

struct invalid_configure {
  bool operator()(const event::validate_configure & ev, const action::context & c) const noexcept {
    return !valid_configure{}(ev, c);
  }
};

struct valid_allocate_request {
  bool operator()(const event::validate_allocate & ev, const action::context & c) const noexcept {
    if (ev.request == nullptr) {
      return false;
    }
    if (ev.request->chunk_out == nullptr || ev.request->offset_out == nullptr) {
      return false;
    }
    if (c.request_size == 0) {
      return false;
    }
    if (!action::detail::valid_alignment(c.request_alignment) || c.request_max_chunk_size == 0) {
      return false;
    }
    uint64_t aligned = 0;
    if (!action::detail::align_up(c.request_size, c.request_alignment, aligned)) {
      return false;
    }
    return true;
  }
};

struct can_allocate {
  bool operator()(const event::allocate & ev, const action::context & c) const noexcept {
    if (ev.chunk_out == nullptr || ev.offset_out == nullptr) {
      return false;
    }
    if (ev.size == 0) {
      return false;
    }
    const uint64_t alignment = ev.alignment != 0 ? ev.alignment : c.alignment;
    const uint64_t max_chunk_size = ev.max_chunk_size != 0
      ? action::detail::clamp_chunk_size_limit(ev.max_chunk_size)
      : c.max_chunk_size;
    if (!action::detail::valid_alignment(alignment) || max_chunk_size == 0) {
      return false;
    }
    uint64_t aligned = 0;
    if (!action::detail::align_up(ev.size, alignment, aligned)) {
      return false;
    }
    return true;
  }
};

struct invalid_allocate_request {
  bool operator()(const event::validate_allocate & ev, const action::context & c) const noexcept {
    return !valid_allocate_request{}(ev, c);
  }
};

struct valid_release_request {
  bool operator()(const event::validate_release & ev, const action::context & c) const noexcept {
    if (ev.request == nullptr) {
      return false;
    }
    if (c.request_chunk < 0 || c.request_chunk >= c.chunk_count || c.request_size == 0) {
      return false;
    }
    if (!action::detail::valid_alignment(c.request_alignment)) {
      return false;
    }
    uint64_t aligned = 0;
    if (!action::detail::align_up(c.request_size, c.request_alignment, aligned)) {
      return false;
    }
    uint64_t end = 0;
    if (action::detail::add_overflow(c.request_offset, aligned, end)) {
      return false;
    }
    if (end > c.chunks[c.request_chunk].max_size) {
      return false;
    }
    return true;
  }
};

struct can_release {
  bool operator()(const event::release & ev, const action::context & c) const noexcept {
    if (ev.chunk < 0 || ev.chunk >= c.chunk_count || ev.size == 0) {
      return false;
    }
    const uint64_t alignment = ev.alignment != 0 ? ev.alignment : c.alignment;
    if (!action::detail::valid_alignment(alignment)) {
      return false;
    }
    uint64_t aligned = 0;
    if (!action::detail::align_up(ev.size, alignment, aligned)) {
      return false;
    }
    uint64_t end = 0;
    if (action::detail::add_overflow(ev.offset, aligned, end)) {
      return false;
    }
    if (end > c.chunks[ev.chunk].max_size) {
      return false;
    }
    return true;
  }
};

struct invalid_release_request {
  bool operator()(const event::validate_release & ev, const action::context & c) const noexcept {
    return !valid_release_request{}(ev, c);
  }
};

}  // namespace emel::buffer::chunk_allocator::guard

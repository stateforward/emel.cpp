#pragma once

#include "emel/buffer/chunk_allocator/actions.hpp"

namespace emel::buffer::chunk_allocator::guard {

struct no_error {
  template <class Event>
  bool operator()(const Event & ev, const action::context &) const noexcept {
    if constexpr (requires { ev.err; }) {
      return ev.err == EMEL_OK;
    }
    return true;
  }
};

struct has_error {
  template <class Event>
  bool operator()(const Event & ev, const action::context &) const noexcept {
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

struct invalid_release_request {
  bool operator()(const event::validate_release & ev, const action::context & c) const noexcept {
    return !valid_release_request{}(ev, c);
  }
};

}  // namespace emel::buffer::chunk_allocator::guard

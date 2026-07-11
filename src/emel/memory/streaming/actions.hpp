#pragma once

#include <cstdint>

#include "emel/error/error.hpp"
#include "emel/memory/streaming/context.hpp"
#include "emel/memory/streaming/errors.hpp"
#include "emel/memory/streaming/events.hpp"

namespace emel::memory::streaming::action {

enum class window_mode : uint8_t {
  empty,
  filling,
  full,
};

enum class cursor_mode : uint8_t {
  advance,
  wrap,
};

inline int32_t error_code(const error value) noexcept {
  return static_cast<int32_t>(emel::error::cast(value));
}

struct effect_initialize {
  void operator()(const event::initialize &ev, context &ctx) const noexcept {
    ctx.next_logical_position = 0;
    ctx.next_physical_position = 0;
    ev.error_out = error_code(error::none);
  }
};

template <window_mode resulting_window, cursor_mode resulting_cursor>
struct effect_advance {
  void operator()(const event::advance &ev, context &ctx) const noexcept {
    const int64_t logical_position = ctx.next_logical_position;
    const int32_t physical_position = ctx.next_physical_position;
    const int64_t logical_end = logical_position + 1;
    int32_t next_physical_position = physical_position + 1;

    if constexpr (resulting_cursor == cursor_mode::wrap) {
      next_physical_position = 0;
    }

    ev.result.logical_position = logical_position;
    ev.result.physical_position = physical_position;
    ev.result.window.logical_end = logical_end;
    ev.result.window.next_physical_position = next_physical_position;
    ev.result.window.capacity = ctx.capacity;

    if constexpr (resulting_window == window_mode::full) {
      ev.result.window.logical_begin =
          logical_end - static_cast<int64_t>(ctx.capacity);
      ev.result.window.physical_begin = next_physical_position;
      ev.result.window.valid_positions = ctx.capacity;
    } else {
      ev.result.window.logical_begin = 0;
      ev.result.window.physical_begin = 0;
      ev.result.window.valid_positions = static_cast<int32_t>(logical_end);
    }

    ctx.next_logical_position = logical_end;
    ctx.next_physical_position = next_physical_position;
    ev.error_out = error_code(error::none);
  }
};

struct effect_reset {
  void operator()(const event::reset &ev, context &ctx) const noexcept {
    ctx.next_logical_position = 0;
    ctx.next_physical_position = 0;
    ev.error_out = error_code(error::none);
  }
};

template <window_mode current_window> struct effect_capture_view {
  void operator()(const event::capture_view &ev,
                  const context &ctx) const noexcept {
    ev.view_out.logical_end = ctx.next_logical_position;
    ev.view_out.next_physical_position = ctx.next_physical_position;
    ev.view_out.capacity = ctx.capacity;

    if constexpr (current_window == window_mode::empty) {
      ev.view_out.logical_begin = 0;
      ev.view_out.physical_begin = 0;
      ev.view_out.valid_positions = 0;
    } else if constexpr (current_window == window_mode::filling) {
      ev.view_out.logical_begin = 0;
      ev.view_out.physical_begin = 0;
      ev.view_out.valid_positions =
          static_cast<int32_t>(ctx.next_logical_position);
    } else {
      ev.view_out.logical_begin =
          ctx.next_logical_position - static_cast<int64_t>(ctx.capacity);
      ev.view_out.physical_begin = ctx.next_physical_position;
      ev.view_out.valid_positions = ctx.capacity;
    }

    ev.error_out = error_code(error::none);
  }
};

template <error error_value> struct effect_reject {
  template <class event_type>
  void operator()(const event_type &ev, const context &) const noexcept {
    ev.error_out = error_code(error_value);
  }
};

struct effect_unexpected {
  template <class event_type>
  void operator()(const event_type &, const context &) const noexcept {}
};

} // namespace emel::memory::streaming::action

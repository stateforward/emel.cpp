#pragma once

#include <cstdint>
#include <limits>

#include "emel/memory/streaming/context.hpp"

namespace emel::memory::streaming::guard {

struct guard_configuration_valid {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.capacity > 0;
  }
};

struct guard_configuration_invalid {
  bool operator()(const action::context &ctx) const noexcept {
    return !guard_configuration_valid{}(ctx);
  }
};

struct guard_capacity_one {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.capacity == 1;
  }
};

struct guard_capacity_many {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.capacity > 1;
  }
};

struct guard_filling_remains_partial {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.next_logical_position + 1 < static_cast<int64_t>(ctx.capacity);
  }
};

struct guard_filling_becomes_full {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.next_logical_position + 1 == static_cast<int64_t>(ctx.capacity);
  }
};

struct guard_full_position_available_before_wrap {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.next_logical_position < std::numeric_limits<int64_t>::max() &&
           ctx.next_physical_position >= 0 &&
           ctx.next_physical_position < ctx.capacity - 1;
  }
};

struct guard_full_position_available_at_wrap {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.next_logical_position < std::numeric_limits<int64_t>::max() &&
           ctx.next_physical_position >= 0 &&
           ctx.next_physical_position == ctx.capacity - 1;
  }
};

struct guard_full_position_overflow {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.next_logical_position == std::numeric_limits<int64_t>::max();
  }
};

struct guard_full_cursor_invalid {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.next_logical_position < std::numeric_limits<int64_t>::max() &&
           (ctx.next_physical_position < 0 ||
            ctx.next_physical_position >= ctx.capacity);
  }
};

} // namespace emel::memory::streaming::guard

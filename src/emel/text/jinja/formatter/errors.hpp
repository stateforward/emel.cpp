#pragma once

#include <cstdint>

namespace emel::text::jinja::formatter {

enum class error : int32_t {
  none = 0,
  invalid_request = 1,
};

constexpr bool is_ok(const error value) noexcept {
  return value == error::none;
}

}  // namespace emel::text::jinja::formatter

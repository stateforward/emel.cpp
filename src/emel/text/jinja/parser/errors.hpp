#pragma once

#include <cstdint>

#include "emel/error/error.hpp"

namespace emel::text::jinja::parser {

enum class error : emel::error::type {
  none = 0u,
  invalid_request = (1u << 0),
  parse_failed = (1u << 1),
  internal_error = (1u << 2),
  untracked = (1u << 3),
};

constexpr bool is_ok(const error value) noexcept {
  return value == error::none;
}

constexpr int32_t to_error_code(const error value) noexcept {
  return static_cast<int32_t>(emel::error::cast(value));
}

}  // namespace emel::text::jinja::parser

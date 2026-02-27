#pragma once

#include "emel/error/error.hpp"

namespace emel::text::detokenizer {

enum class error : emel::error::type {
  none = 0u,
  invalid_request = (1u << 0),
  model_invalid = (1u << 1),
  backend_error = (1u << 2),
  internal_error = (1u << 3),
  untracked = (1u << 16)
};

inline constexpr int32_t error_code(const error err) noexcept {
  return static_cast<int32_t>(emel::error::cast(err));
}

}  // namespace emel::text::detokenizer

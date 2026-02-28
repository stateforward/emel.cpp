#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "emel/emel.h"

namespace emel::text::tokenizer::preprocessor {

enum class error : uint8_t {
  none = 0,
  invalid_request = 1,
  backend_error = 2,
};

inline constexpr bool is_ok(const error err) noexcept { return err == error::none; }

inline constexpr int32_t error_code(const error err) noexcept {
  constexpr std::array<int32_t, 3> k_error_codes = {
      EMEL_OK,
      EMEL_ERR_INVALID_ARGUMENT,
      EMEL_ERR_BACKEND,
  };
  return k_error_codes[static_cast<size_t>(err)];
}

inline constexpr error resolve_failure_error(const error runtime_error) noexcept {
  const std::array<error, 2> errors = {error::backend_error, runtime_error};
  return errors[static_cast<size_t>(runtime_error != error::none)];
}

inline constexpr error select_result_error(const bool ok,
                                           const error runtime_error) noexcept {
  const error failure_error = resolve_failure_error(runtime_error);
  const std::array<error, 2> errors = {failure_error, error::none};
  return errors[static_cast<size_t>(ok)];
}

}  // namespace emel::text::tokenizer::preprocessor

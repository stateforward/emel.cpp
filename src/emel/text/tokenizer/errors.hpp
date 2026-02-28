#pragma once

#include <cstdint>

#include "emel/emel.h"

namespace emel::text::tokenizer {

enum class error : int32_t {
  none = EMEL_OK,
  invalid_request = EMEL_ERR_INVALID_ARGUMENT,
  model_invalid = EMEL_ERR_MODEL_INVALID,
  backend_error = EMEL_ERR_BACKEND,
};

constexpr int32_t error_code(const error err) noexcept {
  return static_cast<int32_t>(err);
}

} // namespace emel::text::tokenizer

#pragma once

#include <cstdint>

namespace emel::text::conditioner {

enum class error : int32_t {
  none = 0u,
  invalid_argument = (1u << 0),
  model_invalid = (1u << 1),
  capacity = (1u << 2),
  backend = (1u << 3),
  untracked = (1u << 4),
};

}  // namespace emel::text::conditioner

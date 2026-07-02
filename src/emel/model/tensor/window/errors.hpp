#pragma once

#include <cstdint>

#include "emel/error/error.hpp"

namespace emel::model::tensor::window {

enum class error : emel::error::type {
  none = 0u,
  invalid_request = (1u << 0),
  not_bound = (1u << 1),
  already_bound = (1u << 2),
  source_map_failed = (1u << 3),
  budget_too_small = (1u << 4),
  layer_out_of_range = (1u << 5),
  slot_copy_failed = (1u << 6),
  not_streaming = (1u << 7),
  internal_error = (1u << 8),
};

} // namespace emel::model::tensor::window

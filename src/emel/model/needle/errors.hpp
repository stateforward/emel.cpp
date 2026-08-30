#pragma once

#include "emel/error/error.hpp"

namespace emel::model::needle {

enum class error : emel::error::type {
  none = 0u,
  invalid_request = (1u << 0),
  geometry_invalid = (1u << 1),
  tensor_count_mismatch = (1u << 2),
  tensor_dtype_mismatch = (1u << 3),
  tensor_shape_mismatch = (1u << 4),
  head_manifest_invalid = (1u << 5),
  internal_error = (1u << 6),
};

} // namespace emel::model::needle

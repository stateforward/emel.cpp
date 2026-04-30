#pragma once

#include "emel/error/error.hpp"

namespace emel::text::generator {

enum class error : emel::error::type {
  none = 0u,
  invalid_request = (1u << 0),
  backend = (1u << 1),
};

}  // namespace emel::text::generator


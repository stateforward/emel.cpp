#pragma once

#include "emel/error/error.hpp"

namespace emel::text::generator::decode_wavefront {

enum class error : emel::error::type {
  none = 0u,
  invalid_request = (1u << 0),
  incompatible_lanes = (1u << 1),
  lane_rejected = (1u << 2),
  backend = (1u << 3),
};

}  // namespace emel::text::generator::decode_wavefront

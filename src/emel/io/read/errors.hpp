#pragma once

#include "emel/error/error.hpp"

namespace emel::io::read {

// Boundary error taxonomy for Phase 212. The component routes every accepted
// request to a fail-closed `unsupported_platform` outcome. Phase 213
// introduces real validation/platform guards, and Phase 214 extends this enum
// with the read execution-error taxonomy required by ERR-01.
enum class error : emel::error::type {
  none = 0u,
  invalid_request = (1u << 0),
  unsupported_platform = (1u << 1),
};

} // namespace emel::io::read

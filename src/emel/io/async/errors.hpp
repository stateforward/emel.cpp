#pragma once

#include <cstdint>

#include "emel/error/error.hpp"

namespace emel::io::async {

enum class error : emel::error::type {
  none = 0u,
  unsupported_strategy = (1u << 0),
  internal_error = (1u << 1),
  invalid_callbacks = (1u << 2),
  invalid_source_contract = (1u << 3),
  invalid_target_window = (1u << 4),
  invalid_progress_contract = (1u << 5),
  invalid_scheduler_contract = (1u << 6),
  cancelled = (1u << 7),
};

} // namespace emel::io::async

#pragma once

#include <cstdint>

#include "emel/error/error.hpp"

namespace emel::io::staged_read {

#ifndef EMEL_IO_STAGED_READ_PLATFORM_SUPPORTED
#if defined(__APPLE__) || defined(__linux__) || defined(__unix__) ||           \
    defined(_WIN32)
#define EMEL_IO_STAGED_READ_PLATFORM_SUPPORTED 1
#else
#define EMEL_IO_STAGED_READ_PLATFORM_SUPPORTED 0
#endif
#endif

// Validation-phase taxonomy ahead of staged copy execution (Phase 229+).
enum class error : emel::error::type {
  none = 0u,
  invalid_callbacks = (1u << 0),
  invalid_stage_contract = (1u << 1),
  invalid_target_window = (1u << 2),
  unsupported_platform = (1u << 3),
  null_source_span = (1u << 4),
  source_span_size_mismatch = (1u << 5),
  insufficient_source_span = (1u << 6),
};

} // namespace emel::io::staged_read

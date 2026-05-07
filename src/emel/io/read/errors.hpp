#pragma once

#include <cstdint>

#include "emel/error/error.hpp"

namespace emel::io::read {

#ifndef EMEL_IO_READ_PLATFORM_SUPPORTED
#if defined(__APPLE__) || defined(__linux__) || defined(__unix__) ||           \
    defined(_WIN32)
#define EMEL_IO_READ_PLATFORM_SUPPORTED 1
#else
#define EMEL_IO_READ_PLATFORM_SUPPORTED 0
#endif
#endif

// Validation, platform-gating, and externally supplied source-result taxonomy.
enum class error : emel::error::type {
  none = 0u,
  invalid_request = (1u << 0),
  unsupported_platform = (1u << 1),
  unsupported_resource = (1u << 2),
  file_open_failed = (1u << 3),
  file_seek_failed = (1u << 4),
  file_read_failed = (1u << 5),
  short_read = (1u << 6),
  internal_error = (1u << 7),
};

inline constexpr uint16_t k_max_file_index = 65534u;
inline constexpr uint64_t k_max_file_path_bytes = 4095u;
inline constexpr uint64_t k_max_read_bytes = (1ULL << 40);
inline constexpr uint32_t k_max_read_batch_tensors = 65536u;

} // namespace emel::io::read

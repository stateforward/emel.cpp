#pragma once

#include <cstdint>

#include "emel/error/error.hpp"

#ifndef EMEL_IO_MMAP_PLATFORM_SUPPORTED
#if defined(__APPLE__) || defined(__linux__) || defined(__unix__) ||           \
    defined(_WIN32)
#define EMEL_IO_MMAP_PLATFORM_SUPPORTED 1
#else
#define EMEL_IO_MMAP_PLATFORM_SUPPORTED 0
#endif
#endif

namespace emel::io::mmap {

enum class error : emel::error::type {
  none = 0u,
  invalid_request = (1u << 0),
  unsupported_platform = (1u << 1),
  unsupported_resource = (1u << 2),
  resource_exhausted = (1u << 3),
  file_open_failed = (1u << 4),
  mapping_failed = (1u << 5),
  unmap_failed = (1u << 6),
  internal_error = (1u << 7),
  invalid_advise_range = (1u << 8),
  advise_failed = (1u << 9),
};

inline constexpr uint16_t k_max_file_index = 65534u;
#if defined(_WIN32)
inline constexpr uint64_t k_required_offset_alignment = 65536u;
#elif defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
inline constexpr uint64_t k_required_offset_alignment = 16384u;
#else
inline constexpr uint64_t k_required_offset_alignment = 4096u;
#endif
inline constexpr uint64_t k_max_file_path_bytes = 4095u;
inline constexpr uint64_t k_max_mapping_bytes = (1ULL << 40);

#ifndef EMEL_IO_MMAP_MAX_MAPPINGS
#define EMEL_IO_MMAP_MAX_MAPPINGS 256u
#endif

inline constexpr uint32_t k_max_mappings = EMEL_IO_MMAP_MAX_MAPPINGS;
inline constexpr uint32_t k_invalid_mapping_handle = static_cast<uint32_t>(-1);

} // namespace emel::io::mmap

#pragma once

#include <cstdint>
#include <span>

#include "emel/error/error.hpp"
#include "emel/gguf/loader/errors.hpp"
#include "emel/gguf/loader/events.hpp"

namespace emel::gguf::loader::detail {

inline emel::error::type probe_requirements(const std::span<const uint8_t> &,
                                            requirements & requirements_out) noexcept {
  requirements_out = {};
  return emel::error::cast(error::none);
}

inline emel::error::type parse_bound_storage(const std::span<const uint8_t> &) noexcept {
  return emel::error::cast(error::none);
}

inline uint64_t required_kv_arena_bytes(const requirements & requirements_in) noexcept {
  const uint64_t entry_bytes = static_cast<uint64_t>(requirements_in.max_key_bytes) +
                               static_cast<uint64_t>(requirements_in.max_value_bytes);
  return static_cast<uint64_t>(requirements_in.kv_count) * entry_bytes;
}

}  // namespace emel::gguf::loader::detail

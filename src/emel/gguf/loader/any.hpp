#pragma once

#include <cstdint>
#include <string_view>

#include "emel/gguf/loader/detail.hpp"
#include "emel/gguf/loader/events.hpp"

namespace emel::gguf::loader {

namespace constants {

inline constexpr std::string_view magic = detail::constants::magic;
inline constexpr std::string_view general_alignment = detail::constants::general_alignment;
inline constexpr uint32_t version = detail::constants::version;
inline constexpr uint32_t default_alignment = detail::constants::default_alignment;
inline constexpr uint32_t max_tensor_dims = detail::constants::max_tensor_dims;

inline constexpr uint32_t gguf_type_uint8 = detail::constants::gguf_type_uint8;
inline constexpr uint32_t gguf_type_int8 = detail::constants::gguf_type_int8;
inline constexpr uint32_t gguf_type_uint16 = detail::constants::gguf_type_uint16;
inline constexpr uint32_t gguf_type_int16 = detail::constants::gguf_type_int16;
inline constexpr uint32_t gguf_type_uint32 = detail::constants::gguf_type_uint32;
inline constexpr uint32_t gguf_type_int32 = detail::constants::gguf_type_int32;
inline constexpr uint32_t gguf_type_float32 = detail::constants::gguf_type_float32;
inline constexpr uint32_t gguf_type_bool = detail::constants::gguf_type_bool;
inline constexpr uint32_t gguf_type_string = detail::constants::gguf_type_string;
inline constexpr uint32_t gguf_type_array = detail::constants::gguf_type_array;
inline constexpr uint32_t gguf_type_uint64 = detail::constants::gguf_type_uint64;
inline constexpr uint32_t gguf_type_int64 = detail::constants::gguf_type_int64;
inline constexpr uint32_t gguf_type_float64 = detail::constants::gguf_type_float64;
inline constexpr uint32_t gguf_type_count = detail::constants::gguf_type_count;

}  // namespace constants

inline uint64_t required_kv_arena_bytes(const requirements & requirements_in) noexcept {
  return detail::required_kv_arena_bytes(requirements_in);
}

}  // namespace emel::gguf::loader

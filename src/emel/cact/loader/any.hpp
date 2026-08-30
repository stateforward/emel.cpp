#pragma once

#include <cstddef>
#include <cstdint>

#include "emel/cact/loader/detail.hpp"
#include "emel/cact/loader/events.hpp"

namespace emel::cact::loader {

namespace constants {

inline constexpr uint32_t tag = detail::constants::tag;
inline constexpr uint32_t alignment = detail::constants::alignment;
inline constexpr size_t header_bytes = detail::constants::header_bytes;
inline constexpr size_t record_bytes = detail::constants::record_bytes;
inline constexpr uint32_t max_engram_orders = detail::constants::max_engram_orders;
inline constexpr uint32_t max_engram_sites = detail::constants::max_engram_sites;
inline constexpr uint32_t max_tensor_dims = detail::constants::max_tensor_dims;

inline constexpr uint32_t dtype_fp16 = detail::constants::dtype_fp16;
inline constexpr uint32_t dtype_fp32 = detail::constants::dtype_fp32;
inline constexpr uint32_t dtype_cq = detail::constants::dtype_cq;
inline constexpr uint32_t dtype_raw = detail::constants::dtype_raw;
inline constexpr uint32_t ternary_record_bits = detail::constants::ternary_record_bits;

}  // namespace constants

}  // namespace emel::cact::loader

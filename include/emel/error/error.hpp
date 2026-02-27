#pragma once

#include <cstdint>
#include <type_traits>

namespace emel::error {

using type = std::uint32_t;

template <typename ErrorCode>
inline constexpr type cast(const ErrorCode code) noexcept {
  static_assert(std::is_enum_v<ErrorCode> || std::is_integral_v<ErrorCode>,
                "ErrorCode must be an enum or integral value");
  return static_cast<type>(code);
}

template <typename ErrorCode>
inline constexpr bool has(const type value, const ErrorCode code) noexcept {
  return (value & cast(code)) != type{0};
}

template <typename... ErrorCodes>
inline constexpr type set(const type current, const ErrorCodes... codes) noexcept {
  return current | (type{0} | ... | cast(codes));
}

template <typename ErrorCode>
inline constexpr type clear(const type current, const ErrorCode code) noexcept {
  return current & static_cast<type>(~cast(code));
}

}  // namespace emel::error

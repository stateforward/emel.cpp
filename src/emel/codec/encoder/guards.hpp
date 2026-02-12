#pragma once

namespace emel::codec::encoder::guard {

inline constexpr auto no_error = [] { return true; };
inline constexpr auto error = [] { return false; };

}  // namespace emel::codec::encoder::guard

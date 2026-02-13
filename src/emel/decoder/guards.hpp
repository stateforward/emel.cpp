#pragma once

namespace emel::decoder::guard {

inline constexpr auto no_error = [] { return true; };
inline constexpr auto error = [] { return false; };

}  // namespace emel::decoder::guard

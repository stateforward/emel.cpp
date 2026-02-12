#pragma once

namespace emel::generator::guard {

inline constexpr auto no_error = [] { return true; };
inline constexpr auto error = [] { return false; };
inline constexpr auto has_error = [] { return error(); };
inline constexpr auto should_continue_decode = [] { return true; };

}  // namespace emel::generator::guard

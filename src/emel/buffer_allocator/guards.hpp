#pragma once

namespace emel::buffer_allocator::guard {

inline constexpr auto no_error = [] { return true; };
inline constexpr auto error = [] { return false; };

}  // namespace emel::buffer_allocator::guard

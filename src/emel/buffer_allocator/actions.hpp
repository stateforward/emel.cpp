#pragma once

namespace emel::buffer_allocator::action {

inline constexpr auto allocate = [](auto&&...) {};
inline constexpr auto upload = [](auto&&...) {};

}  // namespace emel::buffer_allocator::action

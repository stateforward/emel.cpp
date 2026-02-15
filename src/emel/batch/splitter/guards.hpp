#pragma once

#include "emel/batch/splitter/actions.hpp"

namespace emel::batch::splitter::guard {

inline constexpr auto has_error = [](const action::context &) { return false; };
inline constexpr auto no_error = [](const action::context &) { return true; };

}  // namespace emel::batch::splitter::guard

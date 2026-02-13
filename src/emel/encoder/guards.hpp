#pragma once

#include "emel/encoder/events.hpp"

namespace emel::encoder::guard {

inline constexpr auto no_error = [] { return true; };
inline constexpr auto error = [] { return false; };
inline constexpr auto use_merging = [](const event::algorithm_selected & ev) {
  return ev.backend == event::backend_type::merging;
};
inline constexpr auto use_searching = [](const event::algorithm_selected & ev) {
  return ev.backend == event::backend_type::searching;
};
inline constexpr auto use_scanning = [](const event::algorithm_selected & ev) {
  return ev.backend == event::backend_type::scanning;
};

}  // namespace emel::encoder::guard

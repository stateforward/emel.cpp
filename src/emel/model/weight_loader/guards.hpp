#pragma once

#include "emel/model/weight_loader/events.hpp"

namespace emel::model::weight_loader::guard {

inline constexpr auto use_mmap = [](const event::load_weights & ev) {
  return ev.request_mmap && ev.mmap_supported;
};
inline constexpr auto not_use_mmap = [](const event::load_weights & ev) {
  return !use_mmap(ev);
};
inline constexpr auto no_error = [] { return true; };
inline constexpr auto error = [] { return false; };
inline constexpr auto has_error = [] { return error(); };

}  // namespace emel::model::weight_loader::guard

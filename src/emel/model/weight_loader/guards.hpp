#pragma once

#include "emel/model/weight_loader/events.hpp"

namespace emel::model::weight_loader::guard {

inline constexpr auto use_mmap = [](const event::load_weights & ev) {
  if (!ev.request_mmap || !ev.mmap_supported) {
    return false;
  }

  // If direct I/O is requested and supported together with mmap, mmap must be disabled.
  if (ev.request_direct_io && ev.direct_io_supported) {
    return false;
  }

  return true;
};
inline constexpr auto not_use_mmap = [](const event::load_weights & ev) {
  return !use_mmap(ev);
};

inline constexpr auto no_error = [](const event::weights_loaded & ev) {
  return ev.success && ev.status_code == 0;
};
inline constexpr auto has_error = [](const event::weights_loaded & ev) {
  return !no_error(ev);
};

}  // namespace emel::model::weight_loader::guard

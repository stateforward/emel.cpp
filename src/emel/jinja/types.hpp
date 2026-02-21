#pragma once

#include <cstdint>

namespace emel::jinja {

struct program {
  uint32_t node_count = 0;
  uint32_t error_count = 0;

  void reset() noexcept {
    node_count = 0;
    error_count = 0;
  }
};

} // namespace emel::jinja

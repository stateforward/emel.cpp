#pragma once

#include <cstdint>

namespace emel::graph::allocator::placement_pass::events {

enum class phase_outcome : uint8_t {
  unknown = 0,
  done = 1,
  failed = 2,
};

}  // namespace emel::graph::allocator::placement_pass::events

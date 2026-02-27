#pragma once

#include <cstdint>

namespace emel::graph::processor::extract_step::events {

enum class phase_outcome : uint8_t {
  unknown = 0,
  done = 1,
  failed = 2,
};

}  // namespace emel::graph::processor::extract_step::events

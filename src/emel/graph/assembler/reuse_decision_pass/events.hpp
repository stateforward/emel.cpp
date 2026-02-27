#pragma once

#include <cstdint>

namespace emel::graph::assembler::reuse_decision_pass::events {

enum class phase_outcome : uint8_t {
  unknown = 0,
  reused = 1,
  rebuild = 2,
  failed = 3,
};

}  // namespace emel::graph::assembler::reuse_decision_pass::events

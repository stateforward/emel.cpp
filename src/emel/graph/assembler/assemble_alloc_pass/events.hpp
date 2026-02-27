#pragma once

#include <cstdint>

namespace emel::graph::assembler::assemble_alloc_pass::events {

enum class phase_outcome : uint8_t {
  unknown = 0,
  done = 1,
  failed = 2,
};

}  // namespace emel::graph::assembler::assemble_alloc_pass::events

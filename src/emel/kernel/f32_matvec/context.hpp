#pragma once

#include <cstdint>

namespace emel::kernel::f32_matvec::action {

struct context {
  uint64_t prepare_calls = 0u;
  uint64_t prepared_floats = 0u;
  uint64_t reference_calls = 0u;
  uint64_t exact_x4_calls = 0u;
};

} // namespace emel::kernel::f32_matvec::action

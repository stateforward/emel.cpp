#pragma once

#include <cstdint>

#include "emel/kernel/sm.hpp"

namespace emel::speech::decoder::whisper::action {

struct context {
  emel::kernel::sm kernel{emel::kernel::detect_host_kind()};
  uint64_t q8_0_dispatch_count = 0;
  uint64_t q4_0_dispatch_count = 0;
  uint64_t q4_1_dispatch_count = 0;
};

}  // namespace emel::speech::decoder::whisper::action

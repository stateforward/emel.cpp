#pragma once

#include <cstdint>

namespace emel::speech::encoder::whisper::action {

struct context {
  uint64_t q8_0_dispatch_count = 0;
  uint64_t q4_0_dispatch_count = 0;
  uint64_t q4_1_dispatch_count = 0;
};

}  // namespace emel::speech::encoder::whisper::action

#pragma once

#include <cstdint>

namespace emel::speech::codec::mimi::quantizer::action {

struct context {
  uint64_t frames_quantized = 0;
  uint64_t frames_dequantized = 0;
};

} // namespace emel::speech::codec::mimi::quantizer::action

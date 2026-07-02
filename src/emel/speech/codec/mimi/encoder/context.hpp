#pragma once

#include <cstdint>

namespace emel::speech::codec::mimi::encoder::action {

struct context {
  uint64_t frames_encoded = 0;
};

} // namespace emel::speech::codec::mimi::encoder::action

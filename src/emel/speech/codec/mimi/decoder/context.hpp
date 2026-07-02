#pragma once

#include <cstdint>

namespace emel::speech::codec::mimi::decoder::action {

struct context {
  uint64_t frames_decoded = 0;
};

} // namespace emel::speech::codec::mimi::decoder::action

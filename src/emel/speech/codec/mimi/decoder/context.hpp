#pragma once

#include <cstdint>

namespace emel::speech::codec::mimi::decoder::action {

// The actors keep no persistent state across dispatches; per-dispatch data
// travels in the typed runtime-event ctx.
struct context {};

} // namespace emel::speech::codec::mimi::decoder::action

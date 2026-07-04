#pragma once

#include <cstdint>

namespace emel::speech::codec::mimi::quantizer::action {

// The actor keeps no persistent state across dispatches; per-dispatch data
// travels in the typed runtime-event ctx.
struct context {};

} // namespace emel::speech::codec::mimi::quantizer::action

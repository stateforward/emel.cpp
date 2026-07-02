#pragma once

// Public facade for the Mimi streaming codec (Kyutai Moshi / NVIDIA
// PersonaPlex): initialize once against an enriched mimi-component GGUF,
// then stream 80 ms frames through encode_frame / decode_frame.
#include "emel/speech/codec/mimi/sm.hpp"

namespace emel::speech::codec::mimi {

using MimiCodec = sm;

} // namespace emel::speech::codec::mimi

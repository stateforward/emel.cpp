#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "emel/speech/codec/mimi/decoder/sm.hpp"
#include "emel/speech/codec/mimi/detail.hpp"
#include "emel/speech/codec/mimi/encoder/sm.hpp"
#include "emel/speech/codec/mimi/quantizer/sm.hpp"

namespace emel::speech::codec::mimi::action {

inline constexpr int32_t k_max_latent_floats = 512;

// Facade-owned persistent runtime: the bound codec, its streaming state, the
// caller-provided arenas, the latent staging column shared by the child
// dispatches of one frame, and the child machines (child-machine data owned
// by the parent per the composition rules).
struct context {
  detail::codec_runtime runtime = {};
  detail::codec_streaming_state streaming = {};
  std::span<float> workspace = {};
  std::span<float> frame = {};
  std::array<float, k_max_latent_floats> latent = {};
  encoder::sm frontend = {};
  quantizer::sm quantizer_machine = {};
  decoder::sm backend = {};
};

} // namespace emel::speech::codec::mimi::action

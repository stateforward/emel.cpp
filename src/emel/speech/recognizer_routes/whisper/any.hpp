#pragma once

#include <cstdint>

#include "emel/speech/recognizer_routes/whisper/actions.hpp"
#include "emel/speech/recognizer_routes/whisper/guards.hpp"

namespace emel::speech::recognizer_routes::whisper {

uint64_t required_encoder_workspace_floats(uint64_t sample_count) noexcept;
uint64_t required_encoder_state_floats(uint64_t sample_count) noexcept;
uint64_t required_decoder_workspace_floats(uint64_t sample_count) noexcept;
int32_t logits_size() noexcept;
int32_t max_generated_token_count() noexcept;

struct route {
  using guard_tokenizer_supported = guard::guard_tokenizer_supported;
  using guard_model_supported = guard::guard_model_supported;
  using guard_recognition_ready = guard::guard_recognition_ready;
  using effect_encode = action::effect_encode;
  using effect_decode = action::effect_decode;
  using effect_detokenize = action::effect_detokenize;
};

} // namespace emel::speech::recognizer_routes::whisper

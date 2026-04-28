#include "emel/speech/recognizer_routes/whisper/any.hpp"

#include <cstdint>

#include "emel/speech/decoder/whisper/any.hpp"
#include "emel/speech/encoder/whisper/any.hpp"

namespace emel::speech::recognizer_routes::whisper {

namespace {

namespace whisper_decoder = emel::speech::decoder::whisper;
namespace whisper_encoder = emel::speech::encoder::whisper;

} // namespace

uint64_t
required_encoder_workspace_floats(const uint64_t sample_count) noexcept {
  return whisper_encoder::required_workspace_floats(sample_count);
}

uint64_t required_encoder_state_floats(const uint64_t sample_count) noexcept {
  return whisper_encoder::required_encoder_output_floats(sample_count);
}

uint64_t
required_decoder_workspace_floats(const uint64_t sample_count) noexcept {
  const uint64_t mel_frames =
      whisper_encoder::detail::mel_frame_count_for_samples(sample_count);
  const uint64_t encoder_frames =
      whisper_encoder::detail::encoder_frame_count_for_mel_frames(mel_frames);
  return whisper_decoder::required_workspace_floats(encoder_frames);
}

int32_t logits_size() noexcept { return whisper_decoder::vocab_size(); }

int32_t max_generated_token_count() noexcept {
  return whisper_decoder::max_generated_token_count();
}

} // namespace emel::speech::recognizer_routes::whisper

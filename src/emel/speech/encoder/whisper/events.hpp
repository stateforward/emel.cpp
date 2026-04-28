#pragma once

#include <cstdint>
#include <span>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/speech/encoder/whisper/errors.hpp"

namespace emel::speech::encoder::whisper::detail {

struct execution_contract;

} // namespace emel::speech::encoder::whisper::detail

namespace emel::speech::encoder::whisper::events {

struct encode_done;
struct encode_error;

} // namespace emel::speech::encoder::whisper::events

namespace emel::speech::encoder::whisper::event {

struct encode {
  encode(const detail::execution_contract &contract_ref,
         std::span<const float> pcm_ref, const int32_t sample_rate_ref,
         const int32_t channel_count_ref, std::span<float> workspace_ref,
         std::span<float> encoder_state_ref, int32_t &frame_count_out_ref,
         int32_t &width_out_ref, uint64_t &digest_out_ref) noexcept
      : contract(contract_ref), pcm(pcm_ref), sample_rate(sample_rate_ref),
        channel_count(channel_count_ref), workspace(workspace_ref),
        encoder_state(encoder_state_ref), frame_count_out(frame_count_out_ref),
        width_out(width_out_ref), digest_out(digest_out_ref) {}

  const detail::execution_contract &contract;
  std::span<const float> pcm = {};
  int32_t sample_rate = 0;
  int32_t channel_count = 0;
  std::span<float> workspace = {};
  std::span<float> encoder_state = {};
  int32_t &frame_count_out;
  int32_t &width_out;
  uint64_t &digest_out;
  emel::error::type *error_out = nullptr;
  emel::callback<void(const events::encode_done &)> on_done = {};
  emel::callback<void(const events::encode_error &)> on_error = {};
};

struct encode_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct encode_run {
  const encode &request;
  encode_ctx &ctx;
};

} // namespace emel::speech::encoder::whisper::event

namespace emel::speech::encoder::whisper::events {

struct encode_done {
  const event::encode *request = nullptr;
  int32_t frame_count = 0;
  int32_t width = 0;
  uint64_t digest = 0;
};

struct encode_error {
  const event::encode *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

} // namespace emel::speech::encoder::whisper::events

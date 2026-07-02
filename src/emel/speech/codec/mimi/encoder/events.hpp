#pragma once

#include <cstdint>
#include <span>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/speech/codec/mimi/detail.hpp"
#include "emel/speech/codec/mimi/encoder/errors.hpp"

namespace emel::speech::codec::mimi::encoder::events {

struct encode_done;
struct encode_error;

} // namespace emel::speech::codec::mimi::encoder::events

namespace emel::speech::codec::mimi::encoder::event {

// One 80 ms frame of mono 24 kHz PCM -> one 12.5 Hz latent column. The
// caller owns every buffer; the actor allocates nothing during dispatch.
struct encode {
  encode(mimi::detail::codec_runtime &runtime_ref,
         mimi::detail::codec_streaming_state &streaming_ref,
         const std::span<const float> pcm_ref, const std::span<float> frame_ref,
         const std::span<float> workspace_ref,
         const std::span<float> latent_out_ref) noexcept
      : runtime(runtime_ref), streaming(streaming_ref), pcm(pcm_ref),
        frame(frame_ref), workspace(workspace_ref), latent_out(latent_out_ref) {
  }

  mimi::detail::codec_runtime &runtime;
  mimi::detail::codec_streaming_state &streaming;
  std::span<const float> pcm = {};
  std::span<float> frame = {};
  std::span<float> workspace = {};
  std::span<float> latent_out = {};
  emel::error::type *error_out = nullptr;
  emel::callback<void(const events::encode_done &)> on_done = {};
  emel::callback<void(const events::encode_error &)> on_error = {};
};

struct encode_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool stage_ok = true;
  mimi::detail::frame_buffer io = {};
};

struct encode_run {
  const encode &request;
  encode_ctx &ctx;
};

} // namespace emel::speech::codec::mimi::encoder::event

namespace emel::speech::codec::mimi::encoder::events {

struct encode_done {
  const event::encode *request = nullptr;
};

struct encode_error {
  const event::encode *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

} // namespace emel::speech::codec::mimi::encoder::events

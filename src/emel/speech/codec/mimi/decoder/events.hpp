#pragma once

#include <cstdint>
#include <span>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/speech/codec/mimi/decoder/errors.hpp"
#include "emel/speech/codec/mimi/detail.hpp"

namespace emel::speech::codec::mimi::decoder::events {

struct decode_done;
struct decode_error;

} // namespace emel::speech::codec::mimi::decoder::events

namespace emel::speech::codec::mimi::decoder::event {

// One 12.5 Hz latent column -> one 80 ms frame of mono 24 kHz PCM. The
// caller owns every buffer; the actor allocates nothing during dispatch.
struct decode {
  decode(mimi::detail::codec_runtime &runtime_ref,
         mimi::detail::codec_streaming_state &streaming_ref,
         const std::span<const float> latent_ref,
         const std::span<float> frame_ref, const std::span<float> workspace_ref,
         const std::span<float> pcm_out_ref) noexcept
      : runtime(runtime_ref), streaming(streaming_ref), latent(latent_ref),
        frame(frame_ref), workspace(workspace_ref), pcm_out(pcm_out_ref) {}

  mimi::detail::codec_runtime &runtime;
  mimi::detail::codec_streaming_state &streaming;
  std::span<const float> latent = {};
  std::span<float> frame = {};
  std::span<float> workspace = {};
  std::span<float> pcm_out = {};
  emel::error::type *error_out = nullptr;
  emel::callback<void(const events::decode_done &)> on_done = {};
  emel::callback<void(const events::decode_error &)> on_error = {};
};

struct decode_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool stage_ok = true;
  mimi::detail::frame_buffer io = {};
};

struct decode_run {
  const decode &request;
  decode_ctx &ctx;
};

} // namespace emel::speech::codec::mimi::decoder::event

namespace emel::speech::codec::mimi::decoder::events {

struct decode_done {
  const event::decode *request = nullptr;
};

struct decode_error {
  const event::decode *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

} // namespace emel::speech::codec::mimi::decoder::events

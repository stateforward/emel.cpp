#pragma once

#include <cstdint>
#include <span>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/speech/codec/mimi/detail.hpp"
#include "emel/speech/codec/mimi/quantizer/errors.hpp"

namespace emel::speech::codec::mimi::quantizer::events {

struct encode_done;
struct encode_error;
struct decode_done;
struct decode_error;

} // namespace emel::speech::codec::mimi::quantizer::events

namespace emel::speech::codec::mimi::quantizer::event {

// One 12.5 Hz latent column -> n_q codebook indexes.
struct encode {
  encode(mimi::detail::codec_runtime &runtime_ref,
         const std::span<float> latent_ref,
         const std::span<int32_t> codes_out_ref,
         const std::span<float> workspace_ref) noexcept
      : runtime(runtime_ref), latent(latent_ref), codes_out(codes_out_ref),
        workspace(workspace_ref) {}

  mimi::detail::codec_runtime &runtime;
  std::span<float> latent = {};
  std::span<int32_t> codes_out = {};
  std::span<float> workspace = {};
  emel::error::type *error_out = nullptr;
  emel::callback<void(const events::encode_done &)> on_done = {};
  emel::callback<void(const events::encode_error &)> on_error = {};
};

// n_q codebook indexes -> one 12.5 Hz latent column.
struct decode {
  decode(mimi::detail::codec_runtime &runtime_ref,
         const std::span<const int32_t> codes_ref,
         const std::span<float> latent_out_ref,
         const std::span<float> workspace_ref) noexcept
      : runtime(runtime_ref), codes(codes_ref), latent_out(latent_out_ref),
        workspace(workspace_ref) {}

  mimi::detail::codec_runtime &runtime;
  std::span<const int32_t> codes = {};
  std::span<float> latent_out = {};
  std::span<float> workspace = {};
  emel::error::type *error_out = nullptr;
  emel::callback<void(const events::decode_done &)> on_done = {};
  emel::callback<void(const events::decode_error &)> on_error = {};
};

struct encode_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool stage_ok = true;
};

struct decode_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool stage_ok = true;
};

struct encode_run {
  const encode &request;
  encode_ctx &ctx;
};

struct decode_run {
  const decode &request;
  decode_ctx &ctx;
};

} // namespace emel::speech::codec::mimi::quantizer::event

namespace emel::speech::codec::mimi::quantizer::events {

struct encode_done {
  const event::encode *request = nullptr;
};

struct encode_error {
  const event::encode *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

struct decode_done {
  const event::decode *request = nullptr;
};

struct decode_error {
  const event::decode *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

} // namespace emel::speech::codec::mimi::quantizer::events

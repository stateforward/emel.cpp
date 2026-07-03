#pragma once

#include <cstdint>
#include <span>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/model/data.hpp"
#include "emel/speech/codec/mimi/errors.hpp"

namespace emel::speech::codec::mimi::events {

struct initialize_done;
struct initialize_error;
struct encode_frame_done;
struct encode_frame_error;
struct decode_frame_done;
struct decode_frame_error;

} // namespace emel::speech::codec::mimi::events

namespace emel::speech::codec::mimi::event {

// One-time bind. The caller owns all four arenas (sized via the public
// arena-sizing contract in any.hpp) and must keep them alive for the codec's
// lifetime; the actor performs no allocation.
struct initialize {
  initialize(const emel::model::data &model_ref,
             const std::span<float> prepared_ref,
             const std::span<float> state_arena_ref,
             const std::span<float> workspace_ref,
             const std::span<float> frame_ref) noexcept
      : model(model_ref), prepared(prepared_ref), state_arena(state_arena_ref),
        workspace(workspace_ref), frame(frame_ref) {}

  const emel::model::data &model;
  std::span<float> prepared = {};
  std::span<float> state_arena = {};
  std::span<float> workspace = {};
  std::span<float> frame = {};
  emel::error::type *error_out = nullptr;
  emel::callback<void(const events::initialize_done &)> on_done = {};
  emel::callback<void(const events::initialize_error &)> on_error = {};
};

// One 80 ms PCM frame (frame_samples mono 24 kHz floats) -> n_q codes.
struct encode_frame {
  encode_frame(const std::span<const float> pcm_ref,
               const std::span<int32_t> codes_out_ref) noexcept
      : pcm(pcm_ref), codes_out(codes_out_ref) {}

  std::span<const float> pcm = {};
  std::span<int32_t> codes_out = {};
  emel::error::type *error_out = nullptr;
  emel::callback<void(const events::encode_frame_done &)> on_done = {};
  emel::callback<void(const events::encode_frame_error &)> on_error = {};
};

// n_q codes -> one 80 ms PCM frame.
struct decode_frame {
  decode_frame(const std::span<const int32_t> codes_ref,
               const std::span<float> pcm_out_ref) noexcept
      : codes(codes_ref), pcm_out(pcm_out_ref) {}

  std::span<const int32_t> codes = {};
  std::span<float> pcm_out = {};
  emel::error::type *error_out = nullptr;
  emel::callback<void(const events::decode_frame_done &)> on_done = {};
  emel::callback<void(const events::decode_frame_error &)> on_error = {};
};

// Rewind both streaming directions to the first frame.
struct reset_stream {};

// Per-dispatch runtime ctx: carries the typed error of one top-level
// dispatch. Guards never read it - every success/error route is decided by
// pure validation guards over the request and the bound runtime BEFORE the
// corresponding action runs - so `err` is written only on already-selected
// error transitions and consumed by the publish effects. Never stored in
// machine context; never outlives the dispatch.
struct initialize_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct encode_frame_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct decode_frame_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct initialize_run {
  const initialize &request;
  initialize_ctx &ctx;
};

struct encode_frame_run {
  const encode_frame &request;
  encode_frame_ctx &ctx;
};

struct decode_frame_run {
  const decode_frame &request;
  decode_frame_ctx &ctx;
};

} // namespace emel::speech::codec::mimi::event

namespace emel::speech::codec::mimi::events {

struct initialize_done {
  const event::initialize *request = nullptr;
  int32_t frame_samples = 0;
  int32_t n_q = 0;
};

struct initialize_error {
  const event::initialize *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

struct encode_frame_done {
  const event::encode_frame *request = nullptr;
};

struct encode_frame_error {
  const event::encode_frame *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

struct decode_frame_done {
  const event::decode_frame *request = nullptr;
};

struct decode_frame_error {
  const event::decode_frame *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

} // namespace emel::speech::codec::mimi::events

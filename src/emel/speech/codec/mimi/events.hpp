#pragma once

#include <cstdint>
#include <span>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/model/data.hpp"
#include "emel/speech/codec/mimi/detail.hpp"
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

class bind_contract;

} // namespace emel::speech::codec::mimi::event

namespace emel::speech::codec::mimi::guard {

event::bind_contract
make_bind_contract(const emel::model::data &model) noexcept;

} // namespace emel::speech::codec::mimi::guard

namespace emel::speech::codec::mimi::event {

// Opaque pre-dispatch validation facts. Only the authoritative factory can
// construct this value, so callers cannot forge route validity or arena sizes.
// The model address is retained and rechecked by the facade guards before any
// capacity fact is trusted.
class bind_contract {
public:
  bind_contract(const bind_contract &) noexcept = default;
  bind_contract(bind_contract &&) noexcept = default;
  ~bind_contract() noexcept {}

  const emel::model::data *model() const noexcept { return model_; }
  uint64_t model_identity_low() const noexcept { return model_identity_low_; }
  uint64_t model_identity_high() const noexcept { return model_identity_high_; }
  uint64_t prepared_floats() const noexcept { return prepared_floats_; }
  uint64_t state_floats() const noexcept { return state_floats_; }
  uint64_t workspace_floats() const noexcept { return workspace_floats_; }
  uint64_t frame_floats() const noexcept { return frame_floats_; }
  bool f32_valid() const noexcept { return f32_valid_; }
  bool native_valid() const noexcept { return native_valid_; }
  bool q8_valid() const noexcept { return q8_valid_; }

private:
  friend bind_contract
  guard::make_bind_contract(const emel::model::data &model) noexcept;

  bind_contract(const emel::model::data &model_ref,
                const uint64_t model_identity_low_ref,
                const uint64_t model_identity_high_ref,
                const uint64_t prepared_floats_ref,
                const uint64_t state_floats_ref,
                const uint64_t workspace_floats_ref,
                const uint64_t frame_floats_ref, const bool f32_valid_ref,
                const bool native_valid_ref, const bool q8_valid_ref) noexcept
      : model_(&model_ref), model_identity_low_(model_identity_low_ref),
        model_identity_high_(model_identity_high_ref),
        prepared_floats_(prepared_floats_ref), state_floats_(state_floats_ref),
        workspace_floats_(workspace_floats_ref),
        frame_floats_(frame_floats_ref), f32_valid_(f32_valid_ref),
        native_valid_(native_valid_ref), q8_valid_(q8_valid_ref) {}

  const emel::model::data *model_;
  uint64_t model_identity_low_;
  uint64_t model_identity_high_;
  uint64_t prepared_floats_;
  uint64_t state_floats_;
  uint64_t workspace_floats_;
  uint64_t frame_floats_;
  bool f32_valid_;
  bool native_valid_;
  bool q8_valid_;
};

// One-time bind. The caller owns all four arenas (sized via the public
// arena-sizing contract in any.hpp) and must keep them alive for the codec's
// lifetime; the actor performs no allocation.
struct initialize {
  initialize(const emel::model::data &model_ref,
             const bind_contract &contract_ref,
             const std::span<float> prepared_ref,
             const std::span<float> state_arena_ref,
             const std::span<float> workspace_ref,
             const std::span<float> frame_ref) noexcept
      : model(model_ref), contract(contract_ref), prepared(prepared_ref),
        state_arena(state_arena_ref), workspace(workspace_ref),
        frame(frame_ref) {}

  const emel::model::data &model;
  bind_contract contract;
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

struct diagnostics {
  uint64_t projection_prepare_calls = 0u;
  uint64_t projection_prepared_floats = 0u;
  uint64_t projection_reference_calls = 0u;
  uint64_t projection_exact_x4_calls = 0u;
  uint64_t legacy_f32_projection_calls = 0u;
};

struct capture_diagnostics {
  explicit capture_diagnostics(diagnostics &out_ref) noexcept : out(out_ref) {}

  diagnostics &out;
};

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

struct reset_stream_ctx {
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

struct reset_stream_run {
  const reset_stream &request;
  reset_stream_ctx &ctx;
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

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/memory/view.hpp"
#include "emel/model/data.hpp"
#include "emel/speech/predictor/moshi/errors.hpp"

namespace emel::speech::predictor::moshi::events {

struct initialize_done;
struct initialize_error;
struct load_voice_done;
struct load_voice_error;
struct begin_personaplex_prompt_done;
struct begin_personaplex_prompt_error;
struct prefill_voice_done;
struct prefill_voice_error;
struct prefill_personaplex_prompt_done;
struct prefill_personaplex_prompt_error;
struct step_done;
struct step_error;

} // namespace emel::speech::predictor::moshi::events

namespace emel::speech::predictor::moshi::event {

inline constexpr std::size_t k_max_codebooks = 64;
inline constexpr std::size_t k_max_voice_embedding_dim = 8192;

struct initialize {
  explicit initialize(const emel::model::data &model_ref) noexcept
      : model(model_ref) {}

  const emel::model::data &model;
  int32_t max_sequences = 1;
  int32_t max_blocks = 4096;
  int32_t block_tokens = emel::memory::view::DEFAULT_BLOCK_TOKENS;
  int32_t sequence_id = 0;
  emel::error::type *error_out = nullptr;
  emel::callback<void(const events::initialize_done &)> on_done = {};
  emel::callback<void(const events::initialize_error &)> on_error = {};
};

struct load_voice {
  explicit load_voice(const emel::model::data &voice_model_ref) noexcept
      : voice_model(voice_model_ref) {}

  const emel::model::data &voice_model;
  emel::error::type *error_out = nullptr;
  emel::callback<void(const events::load_voice_done &)> on_done = {};
  emel::callback<void(const events::load_voice_error &)> on_error = {};
};

struct begin_personaplex_prompt {
  int32_t text_token_count = 0;
  int32_t pre_text_silence_frames = 0;
  int32_t post_text_silence_frames = 0;
  emel::error::type *error_out = nullptr;
  emel::callback<void(const events::begin_personaplex_prompt_done &)> on_done =
      {};
  emel::callback<void(const events::begin_personaplex_prompt_error &)>
      on_error = {};
};

struct prefill_voice {
  emel::error::type *error_out = nullptr;
  emel::error::type *graph_error_out = nullptr;
  bool *complete_out = nullptr;
  int32_t *remaining_frames_out = nullptr;
  emel::callback<void(const events::prefill_voice_done &)> on_done = {};
  emel::callback<void(const events::prefill_voice_error &)> on_error = {};
};

struct prefill_personaplex_prompt {
  int32_t text_token = -1;
  emel::error::type *error_out = nullptr;
  emel::error::type *graph_error_out = nullptr;
  bool *complete_out = nullptr;
  int32_t *remaining_frames_out = nullptr;
  emel::callback<void(const events::prefill_personaplex_prompt_done &)>
      on_done = {};
  emel::callback<void(const events::prefill_personaplex_prompt_error &)>
      on_error = {};
};

struct step {
  step(std::span<const int32_t> audio_tokens_ref,
       std::span<int32_t> audio_tokens_out_ref,
       int32_t &text_token_out_ref) noexcept
      : audio_tokens(audio_tokens_ref), audio_tokens_out(audio_tokens_out_ref),
        text_token_out(text_token_out_ref) {}

  std::span<const int32_t> audio_tokens = {};
  std::span<int32_t> audio_tokens_out = {};
  int32_t &text_token_out;
  emel::error::type *error_out = nullptr;
  emel::error::type *graph_error_out = nullptr;
  bool *produced_out = nullptr;
  emel::callback<void(const events::step_done &)> on_done = {};
  emel::callback<void(const events::step_error &)> on_error = {};
};

struct reset {};

struct graph_step {
  graph_step(const emel::model::data &model_ref,
             const emel::memory::view::snapshot &memory_snapshot_ref,
             std::span<const int32_t> input_sequence_ref,
             std::span<int32_t> audio_tokens_out_ref,
             int32_t &text_token_out_ref) noexcept
      : model(model_ref), memory_snapshot(memory_snapshot_ref),
        input_sequence(input_sequence_ref),
        audio_tokens_out(audio_tokens_out_ref),
        text_token_out(text_token_out_ref) {}

  const emel::model::data &model;
  const emel::memory::view::snapshot &memory_snapshot;
  int32_t sequence_id = 0;
  std::span<const int32_t> input_sequence = {};
  std::span<const float> input_embedding = {};
  int32_t forced_text_token = -1;
  std::span<int32_t> audio_tokens_out = {};
  int32_t &text_token_out;
  emel::error::type *error_out = nullptr;
};

struct initialize_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool memory_accepted = false;
  int32_t memory_error = static_cast<int32_t>(emel::error::cast(error::none));
};

struct load_voice_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct begin_personaplex_prompt_ctx {
  emel::error::type err = emel::error::cast(error::none);
  int32_t remaining_frames = 0;
};

struct prefill_voice_ctx {
  explicit prefill_voice_ctx(
      emel::memory::view::snapshot &memory_snapshot_ref) noexcept
      : memory_snapshot(memory_snapshot_ref) {}

  emel::error::type err = emel::error::cast(error::none);
  bool memory_accepted = false;
  int32_t memory_error = static_cast<int32_t>(emel::error::cast(error::none));
  int32_t memory_block_count = 0;
  bool embedding_frame_ok = false;
  bool graph_accepted = false;
  emel::error::type graph_error = emel::error::cast(error::none);
  bool complete = false;
  int32_t remaining_frames = 0;
  int32_t text_token = 0;
  std::array<float, k_max_voice_embedding_dim> embedding_frame = {};
  std::array<int32_t, k_max_codebooks> input_sequence = {};
  std::array<int32_t, k_max_codebooks> audio_tokens = {};
  emel::memory::view::snapshot &memory_snapshot;
};

struct prefill_personaplex_prompt_ctx {
  explicit prefill_personaplex_prompt_ctx(
      emel::memory::view::snapshot &memory_snapshot_ref) noexcept
      : memory_snapshot(memory_snapshot_ref) {}

  emel::error::type err = emel::error::cast(error::none);
  bool memory_accepted = false;
  int32_t memory_error = static_cast<int32_t>(emel::error::cast(error::none));
  int32_t memory_block_count = 0;
  bool graph_accepted = false;
  emel::error::type graph_error = emel::error::cast(error::none);
  bool complete = false;
  int32_t remaining_frames = 0;
  int32_t text_token = 0;
  std::array<int32_t, k_max_codebooks> prompt_sequence = {};
  std::array<int32_t, k_max_codebooks> input_sequence = {};
  std::array<int32_t, k_max_codebooks> audio_tokens = {};
  emel::memory::view::snapshot &memory_snapshot;
};

struct step_ctx {
  explicit step_ctx(emel::memory::view::snapshot &memory_snapshot_ref) noexcept
      : memory_snapshot(memory_snapshot_ref) {}

  emel::error::type err = emel::error::cast(error::none);
  bool memory_accepted = false;
  int32_t memory_error = static_cast<int32_t>(emel::error::cast(error::none));
  int32_t memory_block_count = 0;
  bool graph_accepted = false;
  emel::error::type graph_error = emel::error::cast(error::none);
  bool provided_input = false;
  bool produced = false;
  int32_t text_token = 0;
  int32_t generated_dep_q = 0;
  int32_t delayed_dep_q = 0;
  int32_t needed_tokens = 0;
  std::array<int32_t, k_max_codebooks> input_sequence = {};
  std::array<int32_t, k_max_codebooks> audio_tokens = {};
  emel::memory::view::snapshot &memory_snapshot;
};

struct initialize_run {
  const initialize &request;
  initialize_ctx &ctx;
};

struct load_voice_run {
  const load_voice &request;
  load_voice_ctx &ctx;
};

struct begin_personaplex_prompt_run {
  const begin_personaplex_prompt &request;
  begin_personaplex_prompt_ctx &ctx;
};

struct prefill_voice_run {
  const prefill_voice &request;
  prefill_voice_ctx &ctx;
};

struct prefill_personaplex_prompt_run {
  const prefill_personaplex_prompt &request;
  prefill_personaplex_prompt_ctx &ctx;
};

struct step_run {
  const step &request;
  step_ctx &ctx;
};

struct reset_run {
  const reset &request;
};

} // namespace emel::speech::predictor::moshi::event

namespace emel::speech::predictor::moshi::events {

struct initialize_done {
  const event::initialize *request = nullptr;
  int32_t n_q = 0;
  int32_t dep_q = 0;
};

struct initialize_error {
  const event::initialize *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

struct load_voice_done {
  const event::load_voice *request = nullptr;
  int32_t prompt_frames = 0;
};

struct load_voice_error {
  const event::load_voice *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

struct begin_personaplex_prompt_done {
  const event::begin_personaplex_prompt *request = nullptr;
  int32_t remaining_frames = 0;
};

struct begin_personaplex_prompt_error {
  const event::begin_personaplex_prompt *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

struct prefill_voice_done {
  const event::prefill_voice *request = nullptr;
  bool complete = false;
  int32_t remaining_frames = 0;
};

struct prefill_voice_error {
  const event::prefill_voice *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

struct prefill_personaplex_prompt_done {
  const event::prefill_personaplex_prompt *request = nullptr;
  bool complete = false;
  int32_t remaining_frames = 0;
};

struct prefill_personaplex_prompt_error {
  const event::prefill_personaplex_prompt *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

struct step_done {
  const event::step *request = nullptr;
  bool produced = false;
};

struct step_error {
  const event::step *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

} // namespace emel::speech::predictor::moshi::events

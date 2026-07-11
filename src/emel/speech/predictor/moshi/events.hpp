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

} // namespace emel::speech::predictor::moshi::events

namespace emel::speech::predictor::moshi::event {

inline constexpr std::size_t k_max_codebooks = 64;
inline constexpr std::size_t k_max_voice_embedding_dim = 8192;

struct initialize {
  explicit initialize(const emel::model::data &model_ref) noexcept
      : model(model_ref) {}

  const emel::model::data &model;
  int32_t max_sequences = 0;
  int32_t max_blocks = 0;
  int32_t block_tokens = 0;
  int32_t sequence_id = -1;
  int32_t codebook_capacity = 0;
  int32_t delay_cache_row_capacity = 0;
  bool sampling_enabled = false;
  bool sampling_consume_forced_text = false;
  float sampling_audio_temperature = 0.0f;
  float sampling_text_temperature = 0.0f;
  int32_t sampling_audio_top_k = 0;
  int32_t sampling_text_top_k = 0;
  uint32_t sampling_seed = 0u;
  emel::error::type *error_out = nullptr;
  emel::callback<void(const events::initialize_done &)> on_done = {};
  emel::callback<void(const events::initialize_error &)> on_error = {};
};

struct initialize_graph {
  explicit initialize_graph(const initialize &request_ref) noexcept
      : request(request_ref) {}

  const initialize &request;
  emel::error::type *error_out = nullptr;
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

struct predict {
  struct workspace {
    emel::memory::view::snapshot memory = {};
  };

  predict(std::span<const int32_t> model_tokens_ref, workspace &workspace_ref,
          const int32_t planned_step_size_ref,
          const int32_t planned_output_count_ref) noexcept
      : model_tokens(model_tokens_ref), prediction_workspace(workspace_ref),
        planned_step_size(planned_step_size_ref),
        planned_output_count(planned_output_count_ref) {}

  std::span<const int32_t> model_tokens = {};
  workspace &prediction_workspace;
  int32_t planned_step_size = 0;
  int32_t planned_output_count = 0;
  emel::error::type *error_out = nullptr;
};

struct sample {
  sample(predict::workspace &workspace_ref,
         std::span<const int32_t> model_tokens_ref,
         std::span<int32_t> audio_tokens_out_ref,
         int32_t &text_token_out_ref) noexcept
      : prediction_workspace(workspace_ref), model_tokens(model_tokens_ref),
        audio_tokens_out(audio_tokens_out_ref),
        text_token_out(text_token_out_ref) {}

  predict::workspace &prediction_workspace;
  std::span<const int32_t> model_tokens = {};
  std::span<int32_t> audio_tokens_out = {};
  int32_t &text_token_out;
  emel::error::type *error_out = nullptr;
  emel::error::type *graph_error_out = nullptr;
};

struct capture_tokenizer_state {
  capture_tokenizer_state(std::span<int32_t> cache_out_ref,
                          int64_t &offset_out_ref,
                          emel::error::type &error_out_ref) noexcept
      : cache_out(cache_out_ref), offset_out(offset_out_ref),
        error_out(error_out_ref) {}

  std::span<int32_t> cache_out = {};
  int64_t &offset_out;
  emel::error::type &error_out;
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
  emel::error::type graph_error = emel::error::cast(error::none);
  bool graph_accepted = false;
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

struct predict_ctx {
  explicit predict_ctx(
      emel::memory::view::snapshot &memory_snapshot_ref) noexcept
      : memory_snapshot(memory_snapshot_ref) {}

  emel::error::type err = emel::error::cast(error::none);
  bool memory_accepted = false;
  int32_t memory_error = static_cast<int32_t>(emel::error::cast(error::none));
  int32_t memory_block_count = 0;
  emel::memory::view::snapshot &memory_snapshot;
};

struct sample_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool graph_accepted = false;
  emel::error::type graph_error = emel::error::cast(error::none);
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

struct predict_run {
  const predict &request;
  predict_ctx &ctx;
};

struct sample_run {
  const sample &request;
  sample_ctx &ctx;
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

} // namespace emel::speech::predictor::moshi::events

#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/model/data.hpp"
#include "emel/speech/recognizer/errors.hpp"

namespace emel::speech::recognizer::events {

struct initialize_done;
struct initialize_error;
struct recognition_done;
struct recognition_error;

} // namespace emel::speech::recognizer::events

namespace emel::speech::recognizer::event {

struct tokenizer_assets {
  std::string_view model_json = {};
  std::string_view sha256 = {};
};

struct runtime_storage {
  std::span<float> encoder_workspace = {};
  std::span<float> encoder_state = {};
  std::span<float> decoder_workspace = {};
  std::span<float> logits = {};
  std::span<int32_t> generated_tokens = {};
};

struct initialize {
  initialize(const emel::model::data &model_ref,
             const tokenizer_assets &tokenizer_assets_ref) noexcept
      : model(model_ref), tokenizer(tokenizer_assets_ref) {}

  initialize(const emel::model::data &model_ref,
             const std::string_view tokenizer_model_json_ref) noexcept
      : model(model_ref),
        tokenizer(tokenizer_assets{.model_json = tokenizer_model_json_ref}) {}

  const emel::model::data &model;
  tokenizer_assets tokenizer = {};
  emel::error::type *error_out = nullptr;
  emel::callback<void(const events::initialize_done &)> on_done = {};
  emel::callback<void(const events::initialize_error &)> on_error = {};
};

struct initialize_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct initialize_run {
  const initialize &request;
  initialize_ctx &ctx;
};

struct recognize {
  recognize(const emel::model::data &model_ref,
            const tokenizer_assets &tokenizer_assets_ref,
            std::span<const float> pcm_ref, const int32_t sample_rate_ref,
            std::span<char> transcript_ref, int32_t &transcript_size_out_ref,
            int32_t &selected_token_out_ref, float &confidence_out_ref,
            int32_t &encoder_frame_count_out_ref,
            int32_t &encoder_width_out_ref, uint64_t &encoder_digest_out_ref,
            uint64_t &decoder_digest_out_ref) noexcept
      : model(model_ref), tokenizer(tokenizer_assets_ref), pcm(pcm_ref),
        sample_rate(sample_rate_ref), transcript(transcript_ref),
        transcript_size_out(transcript_size_out_ref),
        selected_token_out(selected_token_out_ref),
        confidence_out(confidence_out_ref),
        encoder_frame_count_out(encoder_frame_count_out_ref),
        encoder_width_out(encoder_width_out_ref),
        encoder_digest_out(encoder_digest_out_ref),
        decoder_digest_out(decoder_digest_out_ref),
        generated_token_count_storage(0),
        generated_token_count_out(generated_token_count_storage) {}

  recognize(const emel::model::data &model_ref,
            const tokenizer_assets &tokenizer_assets_ref,
            std::span<const float> pcm_ref, const int32_t sample_rate_ref,
            std::span<char> transcript_ref, int32_t &transcript_size_out_ref,
            int32_t &selected_token_out_ref, float &confidence_out_ref,
            int32_t &encoder_frame_count_out_ref,
            int32_t &encoder_width_out_ref, uint64_t &encoder_digest_out_ref,
            uint64_t &decoder_digest_out_ref,
            int32_t &generated_token_count_out_ref) noexcept
      : model(model_ref), tokenizer(tokenizer_assets_ref), pcm(pcm_ref),
        sample_rate(sample_rate_ref), transcript(transcript_ref),
        transcript_size_out(transcript_size_out_ref),
        selected_token_out(selected_token_out_ref),
        confidence_out(confidence_out_ref),
        encoder_frame_count_out(encoder_frame_count_out_ref),
        encoder_width_out(encoder_width_out_ref),
        encoder_digest_out(encoder_digest_out_ref),
        decoder_digest_out(decoder_digest_out_ref),
        generated_token_count_storage(0),
        generated_token_count_out(generated_token_count_out_ref) {}

  recognize(const emel::model::data &model_ref,
            const std::string_view tokenizer_model_json_ref,
            std::span<const float> pcm_ref, const int32_t sample_rate_ref,
            std::span<char> transcript_ref, int32_t &transcript_size_out_ref,
            int32_t &selected_token_out_ref, float &confidence_out_ref,
            int32_t &encoder_frame_count_out_ref,
            int32_t &encoder_width_out_ref, uint64_t &encoder_digest_out_ref,
            uint64_t &decoder_digest_out_ref) noexcept
      : model(model_ref),
        tokenizer(tokenizer_assets{.model_json = tokenizer_model_json_ref}),
        pcm(pcm_ref), sample_rate(sample_rate_ref), transcript(transcript_ref),
        transcript_size_out(transcript_size_out_ref),
        selected_token_out(selected_token_out_ref),
        confidence_out(confidence_out_ref),
        encoder_frame_count_out(encoder_frame_count_out_ref),
        encoder_width_out(encoder_width_out_ref),
        encoder_digest_out(encoder_digest_out_ref),
        decoder_digest_out(decoder_digest_out_ref),
        generated_token_count_storage(0),
        generated_token_count_out(generated_token_count_storage) {}

  recognize(const emel::model::data &model_ref,
            const std::string_view tokenizer_model_json_ref,
            std::span<const float> pcm_ref, const int32_t sample_rate_ref,
            std::span<char> transcript_ref, int32_t &transcript_size_out_ref,
            int32_t &selected_token_out_ref, float &confidence_out_ref,
            int32_t &encoder_frame_count_out_ref,
            int32_t &encoder_width_out_ref, uint64_t &encoder_digest_out_ref,
            uint64_t &decoder_digest_out_ref,
            int32_t &generated_token_count_out_ref) noexcept
      : model(model_ref),
        tokenizer(tokenizer_assets{.model_json = tokenizer_model_json_ref}),
        pcm(pcm_ref), sample_rate(sample_rate_ref), transcript(transcript_ref),
        transcript_size_out(transcript_size_out_ref),
        selected_token_out(selected_token_out_ref),
        confidence_out(confidence_out_ref),
        encoder_frame_count_out(encoder_frame_count_out_ref),
        encoder_width_out(encoder_width_out_ref),
        encoder_digest_out(encoder_digest_out_ref),
        decoder_digest_out(decoder_digest_out_ref),
        generated_token_count_storage(0),
        generated_token_count_out(generated_token_count_out_ref) {}

  const emel::model::data &model;
  tokenizer_assets tokenizer = {};
  runtime_storage storage = {};
  std::span<const float> pcm = {};
  int32_t sample_rate = 0;
  int32_t channel_count = 1;
  std::span<char> transcript = {};
  int32_t &transcript_size_out;
  int32_t &selected_token_out;
  float &confidence_out;
  int32_t &encoder_frame_count_out;
  int32_t &encoder_width_out;
  uint64_t &encoder_digest_out;
  uint64_t &decoder_digest_out;
  int32_t generated_token_count_storage = 0;
  int32_t &generated_token_count_out;
  emel::error::type *error_out = nullptr;
  emel::callback<void(const events::recognition_done &)> on_done = {};
  emel::callback<void(const events::recognition_error &)> on_error = {};
};

struct recognize_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool encoder_accepted = false;
  bool decoder_accepted = false;
  int32_t encoder_frame_count = 0;
  int32_t encoder_width = 0;
  int32_t generated_token_count = 0;
  int32_t selected_token = 0;
  float confidence = 0.0f;
  int32_t transcript_size = 0;
  uint64_t encoder_digest = 0u;
  uint64_t decoder_digest = 0u;
};

struct recognize_run {
  const recognize &request;
  recognize_ctx &ctx;
};

} // namespace emel::speech::recognizer::event

namespace emel::speech::recognizer::events {

struct initialize_done {
  const event::initialize *request = nullptr;
};

struct initialize_error {
  const event::initialize *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

struct recognition_done {
  const event::recognize *request = nullptr;
  int32_t transcript_size = 0;
  int32_t selected_token = 0;
  float confidence = 0.0f;
  int32_t encoder_frame_count = 0;
  int32_t encoder_width = 0;
  int32_t generated_token_count = 0;
  uint64_t encoder_digest = 0u;
  uint64_t decoder_digest = 0u;
};

struct recognition_error {
  const event::recognize *request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

} // namespace emel::speech::recognizer::events

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/embeddings/generator/errors.hpp"
#include "emel/text/formatter/format.hpp"
#include "emel/text/tokenizer/events.hpp"

namespace emel::embeddings::generator::events {

struct initialize_done;
struct initialize_error;
struct text_embedding_done;
struct text_embedding_error;
struct image_embedding_done;
struct image_embedding_error;
struct audio_embedding_done;
struct audio_embedding_error;

}  // namespace emel::embeddings::generator::events

namespace emel::embeddings::generator {

using tokenizer_bind_dispatch_fn =
    bool(void * tokenizer_sm, const emel::text::tokenizer::event::bind &);
using tokenizer_tokenize_dispatch_fn =
    bool(void * tokenizer_sm, const emel::text::tokenizer::event::tokenize &);

}  // namespace emel::embeddings::generator

namespace emel::embeddings::generator::event {

using benchmark_timestamp_now_fn = std::uint64_t (*)() noexcept;

struct benchmark_stage_timings {
  std::uint64_t prepare_ns = 0u;
  std::uint64_t encode_ns = 0u;
  std::uint64_t publish_ns = 0u;
  std::uint64_t total_ns = 0u;
};

inline std::uint64_t benchmark_timestamp_zero() noexcept {
  return 0u;
}

inline benchmark_stage_timings & default_benchmark_stage_timings() noexcept {
  static thread_local benchmark_stage_timings sink = {};
  sink = {};
  return sink;
}

struct initialize {
  initialize(void * tokenizer_sm_ref,
             emel::embeddings::generator::tokenizer_bind_dispatch_fn & dispatch_tokenizer_bind_ref,
             emel::embeddings::generator::tokenizer_tokenize_dispatch_fn &
                 dispatch_tokenizer_tokenize_ref) noexcept
      : tokenizer_sm(tokenizer_sm_ref),
        dispatch_tokenizer_bind(dispatch_tokenizer_bind_ref),
        dispatch_tokenizer_tokenize(dispatch_tokenizer_tokenize_ref) {}

  void * tokenizer_sm = nullptr;
  emel::embeddings::generator::tokenizer_bind_dispatch_fn & dispatch_tokenizer_bind;
  emel::embeddings::generator::tokenizer_tokenize_dispatch_fn & dispatch_tokenizer_tokenize;
  emel::text::tokenizer::preprocessor::preprocessor_kind preprocessor_variant =
      emel::text::tokenizer::preprocessor::preprocessor_kind::fallback;
  emel::text::encoders::encoder_kind encoder_variant =
      emel::text::encoders::encoder_kind::fallback;
  bool add_special = true;
  bool parse_special = false;
  emel::error::type * error_out = nullptr;
  emel::callback<void(const events::initialize_done &)> on_done = {};
  emel::callback<void(const events::initialize_error &)> on_error = {};
};

struct initialize_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool bind_accepted = false;
  int32_t bind_err_code = 0;
};

struct initialize_run {
  const initialize & request;
  initialize_ctx & ctx;
};

struct embed_text {
  embed_text(std::span<const emel::text::formatter::chat_message> messages_ref,
             std::span<float> output_ref,
             int32_t & output_dimension_out_ref) noexcept
      : messages(messages_ref),
        output(output_ref),
        output_dimension_out(output_dimension_out_ref),
        benchmark_timings_out(default_benchmark_stage_timings()) {}

  embed_text(std::span<const emel::text::formatter::chat_message> messages_ref,
             std::span<float> output_ref,
             int32_t & output_dimension_out_ref,
             benchmark_stage_timings & benchmark_timings_out_ref) noexcept
      : messages(messages_ref),
        output(output_ref),
        output_dimension_out(output_dimension_out_ref),
        benchmark_timings_out(benchmark_timings_out_ref) {}

  std::span<const emel::text::formatter::chat_message> messages = {};
  bool add_generation_prompt = false;
  bool enable_thinking = false;
  int32_t truncate_dimension = 0;
  std::span<float> output = {};
  int32_t & output_dimension_out;
  emel::error::type * error_out = nullptr;
  benchmark_timestamp_now_fn benchmark_time_now = benchmark_timestamp_zero;
  benchmark_stage_timings & benchmark_timings_out;
  emel::callback<void(const events::text_embedding_done &)> on_done = {};
  emel::callback<void(const events::text_embedding_error &)> on_error = {};
};

struct embed_text_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool prepare_accepted = false;
  int32_t prepare_err_code = 0;
  int32_t token_count = 0;
  int32_t output_dimension = 0;
  std::uint64_t total_start_ns = 0u;
  std::uint64_t prepare_start_ns = 0u;
  std::uint64_t prepare_end_ns = 0u;
  std::uint64_t encode_start_ns = 0u;
  std::uint64_t encode_end_ns = 0u;
  std::uint64_t publish_start_ns = 0u;
  std::uint64_t publish_end_ns = 0u;
};

struct embed_text_run {
  const embed_text & request;
  embed_text_ctx & ctx;
};

struct embed_image {
  embed_image(std::span<const uint8_t> rgba_ref,
              const int32_t width_ref,
              const int32_t height_ref,
              std::span<float> output_ref,
              int32_t & output_dimension_out_ref) noexcept
      : rgba(rgba_ref),
        width(width_ref),
        height(height_ref),
        output(output_ref),
        output_dimension_out(output_dimension_out_ref),
        benchmark_timings_out(default_benchmark_stage_timings()) {}

  embed_image(std::span<const uint8_t> rgba_ref,
              const int32_t width_ref,
              const int32_t height_ref,
              std::span<float> output_ref,
              int32_t & output_dimension_out_ref,
              benchmark_stage_timings & benchmark_timings_out_ref) noexcept
      : rgba(rgba_ref),
        width(width_ref),
        height(height_ref),
        output(output_ref),
        output_dimension_out(output_dimension_out_ref),
        benchmark_timings_out(benchmark_timings_out_ref) {}

  std::span<const uint8_t> rgba = {};
  int32_t width = 0;
  int32_t height = 0;
  int32_t truncate_dimension = 0;
  std::span<float> output = {};
  int32_t & output_dimension_out;
  emel::error::type * error_out = nullptr;
  benchmark_timestamp_now_fn benchmark_time_now = benchmark_timestamp_zero;
  benchmark_stage_timings & benchmark_timings_out;
  emel::callback<void(const events::image_embedding_done &)> on_done = {};
  emel::callback<void(const events::image_embedding_error &)> on_error = {};
};

struct embed_image_ctx {
  emel::error::type err = emel::error::cast(error::none);
  int32_t output_dimension = 0;
  std::uint64_t total_start_ns = 0u;
  std::uint64_t prepare_start_ns = 0u;
  std::uint64_t prepare_end_ns = 0u;
  std::uint64_t encode_start_ns = 0u;
  std::uint64_t encode_end_ns = 0u;
  std::uint64_t publish_start_ns = 0u;
  std::uint64_t publish_end_ns = 0u;
};

struct embed_image_run {
  const embed_image & request;
  embed_image_ctx & ctx;
};

struct embed_audio {
  embed_audio(std::span<const float> pcm_ref,
              const int32_t sample_rate_ref,
              std::span<float> output_ref,
              int32_t & output_dimension_out_ref) noexcept
      : pcm(pcm_ref),
        sample_rate(sample_rate_ref),
        output(output_ref),
        output_dimension_out(output_dimension_out_ref),
        benchmark_timings_out(default_benchmark_stage_timings()) {}

  embed_audio(std::span<const float> pcm_ref,
              const int32_t sample_rate_ref,
              std::span<float> output_ref,
              int32_t & output_dimension_out_ref,
              benchmark_stage_timings & benchmark_timings_out_ref) noexcept
      : pcm(pcm_ref),
        sample_rate(sample_rate_ref),
        output(output_ref),
        output_dimension_out(output_dimension_out_ref),
        benchmark_timings_out(benchmark_timings_out_ref) {}

  std::span<const float> pcm = {};
  int32_t sample_rate = 0;
  int32_t truncate_dimension = 0;
  std::span<float> output = {};
  int32_t & output_dimension_out;
  emel::error::type * error_out = nullptr;
  benchmark_timestamp_now_fn benchmark_time_now = benchmark_timestamp_zero;
  benchmark_stage_timings & benchmark_timings_out;
  emel::callback<void(const events::audio_embedding_done &)> on_done = {};
  emel::callback<void(const events::audio_embedding_error &)> on_error = {};
};

struct embed_audio_ctx {
  emel::error::type err = emel::error::cast(error::none);
  int32_t output_dimension = 0;
  std::uint64_t total_start_ns = 0u;
  std::uint64_t prepare_start_ns = 0u;
  std::uint64_t prepare_end_ns = 0u;
  std::uint64_t encode_start_ns = 0u;
  std::uint64_t encode_end_ns = 0u;
  std::uint64_t publish_start_ns = 0u;
  std::uint64_t publish_end_ns = 0u;
};

struct embed_audio_run {
  const embed_audio & request;
  embed_audio_ctx & ctx;
};

}  // namespace emel::embeddings::generator::event

namespace emel::embeddings::generator::events {

struct initialize_done {
  const event::initialize * request = nullptr;
};

struct initialize_error {
  const event::initialize * request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

struct text_embedding_done {
  const event::embed_text * request = nullptr;
  int32_t output_dimension = 0;
};

struct text_embedding_error {
  const event::embed_text * request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

struct image_embedding_done {
  const event::embed_image * request = nullptr;
  int32_t output_dimension = 0;
};

struct image_embedding_error {
  const event::embed_image * request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

struct audio_embedding_done {
  const event::embed_audio * request = nullptr;
  int32_t output_dimension = 0;
};

struct audio_embedding_error {
  const event::embed_audio * request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

}  // namespace emel::embeddings::generator::events

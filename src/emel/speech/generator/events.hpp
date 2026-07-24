#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"

namespace emel::speech::generator::events {

struct initialize_done;
struct initialize_error;
struct condition_done;
struct condition_error;
struct generation_done;
struct generation_error;
struct stream_frame_done;
struct stream_frame_error;
struct flush_done;
struct flush_error;
struct wavefront_frame_done;
struct wavefront_frame_error;
struct wavefront_flush_done;
struct wavefront_flush_error;

} // namespace emel::speech::generator::events

namespace emel::speech::generator::event {

struct initialize {
  explicit initialize(emel::error::type &error_out_ref) noexcept
      : error_out(error_out_ref) {}

  emel::error::type &error_out;
  emel::callback<void(const events::initialize_done &)> on_done = {};
  emel::callback<void(const events::initialize_error &)> on_error = {};
};

struct condition {
  condition(const int32_t token_ref, bool &complete_out_ref,
            int32_t &remaining_out_ref,
            emel::error::type &error_out_ref) noexcept
      : token(token_ref), complete_out(complete_out_ref),
        remaining_out(remaining_out_ref), error_out(error_out_ref) {}

  int32_t token = -1;
  std::string_view text = {};
  std::span<const float> reference_pcm = {};
  bool &complete_out;
  int32_t &remaining_out;
  emel::error::type &error_out;
  emel::callback<void(const events::condition_done &)> on_done = {};
  emel::callback<void(const events::condition_error &)> on_error = {};
};

struct generate {
  generate(const std::string_view text_ref, std::span<float> pcm_out_ref,
           int32_t &sample_count_out_ref,
           emel::error::type &error_out_ref) noexcept
      : text(text_ref), pcm_out(pcm_out_ref),
        sample_count_out(sample_count_out_ref), error_out(error_out_ref) {}

  std::string_view text = {};
  std::span<const float> reference_pcm = {};
  std::span<float> pcm_out = {};
  int32_t &sample_count_out;
  emel::error::type &error_out;
  emel::callback<void(const events::generation_done &)> on_done = {};
  emel::callback<void(const events::generation_error &)> on_error = {};
};

struct stream_frame {
  stream_frame(std::span<const float> pcm_in_ref, std::span<float> pcm_out_ref,
               std::span<int32_t> encoded_tokens_out_ref,
               std::span<int32_t> generated_tokens_out_ref,
               int32_t &text_token_out_ref, int32_t &sample_count_out_ref,
               bool &produced_out_ref,
               emel::error::type &error_out_ref) noexcept
      : pcm_in(pcm_in_ref), pcm_out(pcm_out_ref),
        encoded_tokens_out(encoded_tokens_out_ref),
        generated_tokens_out(generated_tokens_out_ref),
        text_token_out(text_token_out_ref),
        sample_count_out(sample_count_out_ref), produced_out(produced_out_ref),
        error_out(error_out_ref) {}

  std::span<const float> pcm_in = {};
  std::span<float> pcm_out = {};
  std::span<int32_t> encoded_tokens_out = {};
  std::span<int32_t> generated_tokens_out = {};
  int32_t &text_token_out;
  int32_t &sample_count_out;
  bool &produced_out;
  emel::error::type &error_out;
  emel::callback<void(const events::stream_frame_done &)> on_done = {};
  emel::callback<void(const events::stream_frame_error &)> on_error = {};
};

struct flush {
  flush(std::span<float> pcm_out_ref, std::span<int32_t> encoded_tokens_out_ref,
        std::span<int32_t> generated_tokens_out_ref,
        int32_t &text_token_out_ref, int32_t &sample_count_out_ref,
        bool &complete_out_ref, emel::error::type &error_out_ref) noexcept
      : pcm_out(pcm_out_ref), encoded_tokens_out(encoded_tokens_out_ref),
        generated_tokens_out(generated_tokens_out_ref),
        text_token_out(text_token_out_ref),
        sample_count_out(sample_count_out_ref), complete_out(complete_out_ref),
        error_out(error_out_ref) {}

  std::span<float> pcm_out = {};
  std::span<int32_t> encoded_tokens_out = {};
  std::span<int32_t> generated_tokens_out = {};
  int32_t &text_token_out;
  int32_t &sample_count_out;
  bool &complete_out;
  emel::error::type &error_out;
  emel::callback<void(const events::flush_done &)> on_done = {};
  emel::callback<void(const events::flush_error &)> on_error = {};
};

struct reset {
  emel::error::type &error_out;
};

inline constexpr uint64_t k_invalid_wavefront_sequence = UINT64_MAX;

struct wavefront_attribution {
  uint64_t sequence = k_invalid_wavefront_sequence;
  uint64_t source = 0u;
};

struct wavefront_frame {
  wavefront_frame(std::span<const float> pcm_in_ref,
                  std::span<float> pcm_out_ref,
                  std::span<int32_t> encoded_tokens_out_ref,
                  std::span<int32_t> generated_tokens_out_ref,
                  const wavefront_attribution input_attribution_ref,
                  wavefront_attribution &output_attribution_ref,
                  int32_t &text_token_out_ref, int32_t &sample_count_out_ref,
                  bool &produced_out_ref,
                  emel::error::type &error_out_ref) noexcept
      : pcm_in(pcm_in_ref), pcm_out(pcm_out_ref),
        encoded_tokens_out(encoded_tokens_out_ref),
        generated_tokens_out(generated_tokens_out_ref),
        input_attribution(input_attribution_ref),
        output_attribution(output_attribution_ref),
        text_token_out(text_token_out_ref),
        sample_count_out(sample_count_out_ref), produced_out(produced_out_ref),
        error_out(error_out_ref) {}

  const std::span<const float> pcm_in;
  const std::span<float> pcm_out;
  const std::span<int32_t> encoded_tokens_out;
  const std::span<int32_t> generated_tokens_out;
  const wavefront_attribution input_attribution;
  wavefront_attribution &output_attribution;
  int32_t &text_token_out;
  int32_t &sample_count_out;
  bool &produced_out;
  emel::error::type &error_out;
  emel::callback<void(const events::wavefront_frame_done &)> on_done = {};
  emel::callback<void(const events::wavefront_frame_error &)> on_error = {};
};

struct wavefront_flush {
  wavefront_flush(std::span<float> pcm_out_ref,
                  std::span<int32_t> generated_tokens_out_ref,
                  wavefront_attribution &output_attribution_ref,
                  int32_t &text_token_out_ref, int32_t &sample_count_out_ref,
                  bool &produced_out_ref, bool &complete_out_ref,
                  emel::error::type &error_out_ref) noexcept
      : pcm_out(pcm_out_ref), generated_tokens_out(generated_tokens_out_ref),
        output_attribution(output_attribution_ref),
        text_token_out(text_token_out_ref),
        sample_count_out(sample_count_out_ref), produced_out(produced_out_ref),
        complete_out(complete_out_ref), error_out(error_out_ref) {}

  const std::span<float> pcm_out;
  const std::span<int32_t> generated_tokens_out;
  wavefront_attribution &output_attribution;
  int32_t &text_token_out;
  int32_t &sample_count_out;
  bool &produced_out;
  bool &complete_out;
  emel::error::type &error_out;
  emel::callback<void(const events::wavefront_flush_done &)> on_done = {};
  emel::callback<void(const events::wavefront_flush_error &)> on_error = {};
};

struct wavefront_reset {
  emel::error::type &error_out;
};

struct initialize_ctx {
  emel::error::type err = {};
  emel::error::type child_err = {};
  bool child_accepted = false;
  int32_t tokenizer_err = 0;
};

struct condition_ctx {
  emel::error::type err = {};
  emel::error::type child_err = {};
  emel::error::type graph_err = {};
  bool child_accepted = false;
  bool complete = false;
  int32_t remaining = -1;
  int32_t tokenizer_err = 0;
  int64_t tokenizer_offset = 0;
};

struct generate_ctx {
  emel::error::type err = {};
  emel::error::type child_err = {};
  bool child_accepted = false;
  int32_t sample_count = 0;
};

struct frame_ctx {
  emel::error::type err = {};
  emel::error::type child_err = {};
  emel::error::type graph_err = {};
  bool child_accepted = false;
  bool produced = false;
  int32_t tokenizer_err = 0;
  int32_t predicted_text_token = -1;
  int32_t text_token = -1;
  int32_t plan_step_size = 0;
  int32_t plan_output_count = 0;
};

struct flush_ctx : frame_ctx {};

struct initialize_run {
  const initialize &request;
  initialize_ctx &ctx;
};

struct condition_run {
  const condition &request;
  condition_ctx &ctx;
};

struct generate_run {
  const generate &request;
  generate_ctx &ctx;
};

struct stream_frame_run {
  const stream_frame &request;
  frame_ctx &ctx;
};

struct flush_run {
  const flush &request;
  flush_ctx &ctx;
};

} // namespace emel::speech::generator::event

namespace emel::speech::generator::detail {

struct wavefront_run_ctx {
  emel::error::type err = {};
  emel::error::type encode_err = {};
  emel::error::type middle_err = {};
  emel::error::type decode_err = {};
  bool encode_accepted = true;
  bool middle_accepted = true;
  bool decode_accepted = true;
  bool all_submitted = true;
  bool joined = true;
  event::wavefront_attribution decoded_attribution = {};
  int32_t decoded_text_token = -1;
};

struct wavefront_frame_run {
  const event::wavefront_frame &request;
  wavefront_run_ctx &ctx;
};

struct wavefront_flush_run {
  const event::wavefront_flush &request;
  wavefront_run_ctx &ctx;
};

struct wavefront_reset_ctx {
  emel::error::type err = {};
  emel::error::type encode_err = {};
  emel::error::type middle_err = {};
  emel::error::type decode_err = {};
  bool encode_accepted = false;
  bool middle_accepted = false;
  bool decode_accepted = false;
  bool all_submitted = true;
  bool joined = true;
};

struct event_wavefront_reset_run {
  const event::wavefront_reset &request;
  wavefront_reset_ctx &ctx;
};

} // namespace emel::speech::generator::detail

namespace emel::speech::generator::events {

struct initialize_done {
  const event::initialize &request;
  int32_t frame_samples = 0;
  int32_t codebook_count = 0;
};

struct initialize_error {
  const event::initialize &request;
  emel::error::type err = {};
};

struct condition_done {
  const event::condition &request;
  bool complete = false;
  int32_t remaining = -1;
};

struct condition_error {
  const event::condition &request;
  emel::error::type err = {};
};

struct generation_done {
  const event::generate &request;
  int32_t sample_count = 0;
};

struct generation_error {
  const event::generate &request;
  emel::error::type err = {};
};

struct stream_frame_done {
  const event::stream_frame &request;
  int32_t sample_count = 0;
  bool produced = false;
};

struct stream_frame_error {
  const event::stream_frame &request;
  emel::error::type err = {};
};

struct flush_done {
  const event::flush &request;
  int32_t sample_count = 0;
  bool complete = false;
};

struct flush_error {
  const event::flush &request;
  emel::error::type err = {};
};

struct wavefront_frame_done {
  const event::wavefront_frame &request;
  event::wavefront_attribution attribution = {};
  int32_t sample_count = 0;
  bool produced = false;
};

struct wavefront_frame_error {
  const event::wavefront_frame &request;
  emel::error::type err = {};
};

struct wavefront_flush_done {
  const event::wavefront_flush &request;
  event::wavefront_attribution attribution = {};
  int32_t sample_count = 0;
  bool produced = false;
  bool complete = false;
};

struct wavefront_flush_error {
  const event::wavefront_flush &request;
  emel::error::type err = {};
};

} // namespace emel::speech::generator::events

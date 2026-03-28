#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "emel/batch/planner/errors.hpp"
#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/generator/errors.hpp"
#include "emel/graph/events.hpp"
#include "emel/logits/sampler/events.hpp"
#include "emel/text/conditioner/errors.hpp"
#include "emel/text/formatter/format.hpp"
#include "emel/text/renderer/events.hpp"
#include "emel/text/tokenizer/events.hpp"

namespace emel::generator::events {

struct initialize_done;
struct initialize_error;
struct generation_done;
struct generation_error;

}  // namespace emel::generator::events

namespace emel::generator {

enum class attention_mode : uint8_t {
  flash,
  nonflash,
};

enum class selection_mode : uint8_t {
  sample_logits,
  preselected_argmax,
};

using tokenizer_bind_dispatch_fn =
    bool(void * tokenizer_sm, const emel::text::tokenizer::event::bind &);
using tokenizer_tokenize_dispatch_fn =
    bool(void * tokenizer_sm, const emel::text::tokenizer::event::tokenize &);

struct compute_io {
  void * backend_ctx = nullptr;
  const int32_t * token_ids = nullptr;
  int32_t token_count = 0;
  float * logits = nullptr;
  int32_t logits_capacity = 0;
  int32_t * selected_token_out = nullptr;
  float * selected_score_out = nullptr;
};

}  // namespace emel::generator

namespace emel::generator::event {

struct initialize {
  initialize(void * tokenizer_sm_ref,
             emel::generator::tokenizer_bind_dispatch_fn & dispatch_tokenizer_bind_ref,
             emel::generator::tokenizer_tokenize_dispatch_fn & dispatch_tokenizer_tokenize_ref,
             std::span<emel::logits::sampler::fn> sampler_fns_ref) noexcept
    : tokenizer_sm(tokenizer_sm_ref),
      dispatch_tokenizer_bind(dispatch_tokenizer_bind_ref),
      dispatch_tokenizer_tokenize(dispatch_tokenizer_tokenize_ref),
      sampler_fns(sampler_fns_ref) {}

  void * tokenizer_sm = nullptr;
  emel::generator::tokenizer_bind_dispatch_fn & dispatch_tokenizer_bind;
  emel::generator::tokenizer_tokenize_dispatch_fn & dispatch_tokenizer_tokenize;
  emel::text::tokenizer::preprocessor::preprocessor_kind preprocessor_variant =
      emel::text::tokenizer::preprocessor::preprocessor_kind::fallback;
  emel::text::encoders::encoder_kind encoder_variant =
      emel::text::encoders::encoder_kind::fallback;
  bool add_special = true;
  bool parse_special = false;
  std::span<emel::logits::sampler::fn> sampler_fns = {};
  emel::generator::selection_mode selection_mode =
      emel::generator::selection_mode::sample_logits;
  int32_t max_prompt_tokens = 0;
  int32_t max_generated_tokens = 0;
  int32_t max_blocks = 0;
  int32_t block_tokens = 0;
  bool strip_leading_space = false;
  std::span<const std::string_view> stop_sequences = {};
  emel::error::type * error_out = nullptr;
  emel::callback<void(const events::initialize_done &)> on_done = {};
  emel::callback<void(const events::initialize_error &)> on_error = {};
};

struct initialize_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool phase_accepted = false;
  int32_t phase_code = 0;
  bool buffers_ready = false;
};

struct initialize_run {
  const initialize & request;
  initialize_ctx & ctx;
};

struct generate {
  generate(std::span<const emel::text::formatter::chat_message> messages_ref,
           int32_t max_tokens_value,
           std::span<char> output_ref,
           size_t & output_length_out_ref) noexcept
    : messages(messages_ref),
      max_tokens(max_tokens_value),
      output(output_ref),
      output_length_out(output_length_out_ref) {}

  std::span<const emel::text::formatter::chat_message> messages = {};
  bool add_generation_prompt = false;
  bool enable_thinking = false;
  int32_t max_tokens = 0;
  std::span<char> output = {};
  size_t & output_length_out;
  emel::error::type * error_out = nullptr;
  emel::callback<void(const events::generation_done &)> on_done = {};
  emel::callback<void(const events::generation_error &)> on_error = {};
};

struct generate_ctx {
  emel::error::type err = emel::error::cast(error::none);
  bool phase_accepted = false;
  int32_t phase_code = 0;
  int32_t tokens_generated = 0;
  int32_t target_tokens = 0;
  int32_t prompt_token_count = 0;
  int32_t prefill_step_size = 0;
  int32_t plan_step_count = 0;
  int32_t plan_outputs = 0;
  int32_t kv_tokens = 0;
  int32_t selected_token = -1;
  float selected_score = 0.0f;
  size_t output_length = 0;
  size_t phase_output_length = 0;
  emel::text::renderer::sequence_status render_status =
      emel::text::renderer::sequence_status::running;
  emel::graph::event::compute_output graph_output = {};
  emel::generator::compute_io io = {};
};

// Internal event used by generator::sm wrapper; not part of public API.
struct generate_run {
  const generate & request;
  generate_ctx & ctx;
};

}  // namespace emel::generator::event

namespace emel::generator::events {

struct initialize_done {
  const event::initialize * request = nullptr;
};

struct initialize_error {
  const event::initialize * request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
};

struct generation_done {
  const event::generate * request = nullptr;
  int32_t tokens_generated = 0;
  size_t output_length = 0;
};

struct generation_error {
  const event::generate * request = nullptr;
  emel::error::type err = emel::error::cast(error::none);
  int32_t tokens_generated = 0;
  size_t output_length = 0;
};

}  // namespace emel::generator::events

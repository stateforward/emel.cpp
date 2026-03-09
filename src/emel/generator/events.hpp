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

using tokenizer_bind_dispatch_fn =
    bool(void * tokenizer_sm, const emel::text::tokenizer::event::bind &);
using tokenizer_tokenize_dispatch_fn =
    bool(void * tokenizer_sm, const emel::text::tokenizer::event::tokenize &);
using validate_dispatch_fn =
    bool(const emel::graph::processor::event::execute & request, int32_t * err_out);
using prepare_graph_dispatch_fn =
    bool(const emel::graph::processor::event::execute & request,
         bool * reused_out,
         int32_t * err_out);
using alloc_graph_dispatch_fn =
    bool(const emel::graph::processor::event::execute & request, int32_t * err_out);
using bind_inputs_dispatch_fn =
    bool(const emel::graph::processor::event::execute & request, int32_t * err_out);
using run_kernel_dispatch_fn =
    bool(const emel::graph::processor::event::execute & request, int32_t * err_out);
using extract_outputs_dispatch_fn =
    bool(const emel::graph::processor::event::execute & request,
         int32_t * outputs_out,
         int32_t * err_out);

struct compute_io {
  void * backend_ctx = nullptr;
  const int32_t * token_ids = nullptr;
  int32_t token_count = 0;
  float * logits = nullptr;
  int32_t logits_capacity = 0;
};

}  // namespace emel::generator

namespace emel::generator::event {

struct initialize {
  initialize(const void * model_topology_ref,
             const void * prefill_plan_ref,
             const void * decode_plan_ref,
             void * tokenizer_sm_ref,
             emel::generator::tokenizer_bind_dispatch_fn & dispatch_tokenizer_bind_ref,
             emel::generator::tokenizer_tokenize_dispatch_fn & dispatch_tokenizer_tokenize_ref,
             void * backend_ctx_ref,
             emel::generator::validate_dispatch_fn & validate_ref,
             emel::generator::prepare_graph_dispatch_fn & prepare_graph_ref,
             emel::generator::alloc_graph_dispatch_fn & alloc_graph_ref,
             emel::generator::bind_inputs_dispatch_fn & bind_inputs_ref,
             emel::generator::run_kernel_dispatch_fn & run_kernel_ref,
             emel::generator::extract_outputs_dispatch_fn & extract_outputs_ref,
             std::span<emel::logits::sampler::fn> sampler_fns_ref) noexcept
    : model_topology(model_topology_ref),
      prefill_plan(prefill_plan_ref),
      decode_plan(decode_plan_ref),
      tokenizer_sm(tokenizer_sm_ref),
      dispatch_tokenizer_bind(dispatch_tokenizer_bind_ref),
      dispatch_tokenizer_tokenize(dispatch_tokenizer_tokenize_ref),
      backend_ctx(backend_ctx_ref),
      validate(validate_ref),
      prepare_graph(prepare_graph_ref),
      alloc_graph(alloc_graph_ref),
      bind_inputs(bind_inputs_ref),
      run_kernel(run_kernel_ref),
      extract_outputs(extract_outputs_ref),
      sampler_fns(sampler_fns_ref) {}

  const void * model_topology = nullptr;
  const void * prefill_plan = nullptr;
  const void * decode_plan = nullptr;
  void * tokenizer_sm = nullptr;
  emel::generator::tokenizer_bind_dispatch_fn & dispatch_tokenizer_bind;
  emel::generator::tokenizer_tokenize_dispatch_fn & dispatch_tokenizer_tokenize;
  emel::text::tokenizer::preprocessor::preprocessor_kind preprocessor_variant =
      emel::text::tokenizer::preprocessor::preprocessor_kind::fallback;
  emel::text::encoders::encoder_kind encoder_variant =
      emel::text::encoders::encoder_kind::fallback;
  bool add_special = true;
  bool parse_special = false;
  void * backend_ctx = nullptr;
  emel::generator::validate_dispatch_fn & validate;
  emel::generator::prepare_graph_dispatch_fn & prepare_graph;
  emel::generator::alloc_graph_dispatch_fn & alloc_graph;
  emel::generator::bind_inputs_dispatch_fn & bind_inputs;
  emel::generator::run_kernel_dispatch_fn & run_kernel;
  emel::generator::extract_outputs_dispatch_fn & extract_outputs;
  std::span<emel::logits::sampler::fn> sampler_fns = {};
  uint32_t max_node_count = 0;
  uint32_t max_tensor_count = 0;
  uint64_t bytes_per_tensor = 0;
  uint64_t workspace_capacity_bytes = 0;
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
  generate(std::string_view prompt_ref,
           int32_t max_tokens_value,
           std::span<char> output_ref,
           size_t & output_length_out_ref) noexcept
    : prompt(prompt_ref),
      max_tokens(max_tokens_value),
      output(output_ref),
      output_length_out(output_length_out_ref) {}

  std::string_view prompt = {};
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

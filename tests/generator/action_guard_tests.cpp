#include <array>
#include <doctest/doctest.h>
#include <span>
#include <vector>

#include "emel/generator/actions.hpp"
#include "emel/generator/guards.hpp"
#include "emel/model/data.hpp"

namespace {

struct callback_tracker {
  bool initialize_done_called = false;
  bool initialize_error_called = false;
  bool generate_done_called = false;
  bool generate_error_called = false;
  emel::error::type err = emel::error::cast(emel::generator::error::none);
  int32_t tokens_generated = -1;
  size_t output_length = 0;
};

void on_initialize_done(void * owner, const emel::generator::events::initialize_done &) {
  static_cast<callback_tracker *>(owner)->initialize_done_called = true;
}

void on_initialize_error(void * owner, const emel::generator::events::initialize_error & ev) {
  auto * tracker = static_cast<callback_tracker *>(owner);
  tracker->initialize_error_called = true;
  tracker->err = ev.err;
}

void on_generate_done(void * owner, const emel::generator::events::generation_done & ev) {
  auto * tracker = static_cast<callback_tracker *>(owner);
  tracker->generate_done_called = true;
  tracker->tokens_generated = ev.tokens_generated;
  tracker->output_length = ev.output_length;
}

void on_generate_error(void * owner, const emel::generator::events::generation_error & ev) {
  auto * tracker = static_cast<callback_tracker *>(owner);
  tracker->generate_error_called = true;
  tracker->err = ev.err;
  tracker->tokens_generated = ev.tokens_generated;
  tracker->output_length = ev.output_length;
}

bool tokenizer_bind_dispatch(void *, const emel::text::tokenizer::event::bind &) {
  return true;
}

bool tokenizer_tokenize_dispatch(void *, const emel::text::tokenizer::event::tokenize &) {
  return true;
}

emel::error::type sampler_passthrough(int32_t &, float &, int32_t &, int32_t &) {
  return emel::error::cast(emel::logits::sampler::error::none);
}

int dummy_tokenizer_actor = 4;
std::array<emel::logits::sampler::fn, 1> dummy_samplers = {
    emel::logits::sampler::fn::from<sampler_passthrough>(),
};
constexpr std::array<emel::text::formatter::chat_message, 1> k_generate_messages = {
    emel::text::formatter::chat_message{
        .role = "user",
        .content = "hello",
    },
};

emel::model::data & test_model() {
  static auto * model = []() {
    auto * created = new emel::model::data{};
    created->vocab_data.eos_id = 7;
    created->vocab_data.eot_id = 8;
    return created;
  }();
  return *model;
}

emel::generator::event::initialize make_initialize_request(
    callback_tracker * tracker,
    emel::error::type * error_out,
    const emel::generator::selection_mode selection_mode =
        emel::generator::selection_mode::sample_logits) {
  const std::span<emel::logits::sampler::fn> sampler_span =
      selection_mode == emel::generator::selection_mode::sample_logits
          ? std::span<emel::logits::sampler::fn>{dummy_samplers}
          : std::span<emel::logits::sampler::fn>{};
  emel::generator::event::initialize request{
    &dummy_tokenizer_actor,
    tokenizer_bind_dispatch,
    tokenizer_tokenize_dispatch,
    sampler_span,
  };
  request.selection_mode = selection_mode;
  request.max_prompt_tokens = 8;
  request.max_generated_tokens = 4;
  request.max_blocks = 8;
  request.block_tokens = 4;
  request.error_out = error_out;
  request.on_done = tracker == nullptr
                        ? emel::callback<void(const emel::generator::events::initialize_done &)>{}
                        : emel::callback<void(const emel::generator::events::initialize_done &)>(
                              tracker, on_initialize_done);
  request.on_error = tracker == nullptr
                         ? emel::callback<void(const emel::generator::events::initialize_error &)>{}
                         : emel::callback<void(const emel::generator::events::initialize_error &)>(
                               tracker, on_initialize_error);
  return request;
}

emel::generator::event::generate make_generate_request(
    callback_tracker * tracker,
    emel::error::type * error_out,
    size_t & output_length_out) {
  static char output_storage[16] = {};
  emel::generator::event::generate request{
    std::span<const emel::text::formatter::chat_message>{k_generate_messages},
    2,
    std::span<char>{output_storage},
    output_length_out,
  };
  request.add_generation_prompt = true;
  request.enable_thinking = false;
  request.error_out = error_out;
  request.on_done = tracker == nullptr
                        ? emel::callback<void(const emel::generator::events::generation_done &)>{}
                        : emel::callback<void(const emel::generator::events::generation_done &)>(
                              tracker, on_generate_done);
  request.on_error = tracker == nullptr
                         ? emel::callback<void(const emel::generator::events::generation_error &)>{}
                         : emel::callback<void(const emel::generator::events::generation_error &)>(
                               tracker, on_generate_error);
  return request;
}

}  // namespace

TEST_CASE("generator initialize dispatch actions cover channel variants") {
  emel::generator::action::context context{};
  callback_tracker tracker{};
  emel::error::type error_out = emel::error::cast(emel::generator::error::backend);

  auto with_both = make_initialize_request(&tracker, &error_out);
  emel::generator::event::initialize_ctx init_ctx{};
  emel::generator::event::initialize_run init_run{with_both, init_ctx};
  emel::generator::action::dispatch_initialize_done_with_callback_and_error_out(
      init_run, context);
  CHECK(tracker.initialize_done_called);
  CHECK(error_out == emel::error::cast(emel::generator::error::none));

  tracker = {};
  error_out = emel::error::cast(emel::generator::error::none);
  auto callback_only = make_initialize_request(&tracker, nullptr);
  emel::generator::event::initialize_run callback_only_run{callback_only, init_ctx};
  emel::generator::action::dispatch_initialize_done_with_callback_only(callback_only_run,
                                                                       context);
  CHECK(tracker.initialize_done_called);

  tracker = {};
  error_out = emel::error::cast(emel::generator::error::backend);
  auto error_out_only = make_initialize_request(nullptr, &error_out);
  emel::generator::event::initialize_run error_out_only_run{error_out_only, init_ctx};
  emel::generator::action::dispatch_initialize_done_with_error_out_only(error_out_only_run,
                                                                        context);
  CHECK(error_out == emel::error::cast(emel::generator::error::none));

  emel::generator::action::dispatch_initialize_done_without_channels(error_out_only_run,
                                                                     context);

  tracker = {};
  error_out = emel::error::cast(emel::generator::error::none);
  init_ctx.err = emel::error::cast(emel::generator::error::backend);
  emel::generator::action::dispatch_initialize_error_with_callback_and_error_out(
      init_run, context);
  CHECK(tracker.initialize_error_called);
  CHECK(error_out == emel::error::cast(emel::generator::error::backend));

  tracker = {};
  emel::generator::action::dispatch_initialize_error_with_callback_only(callback_only_run,
                                                                        context);
  CHECK(tracker.initialize_error_called);

  error_out = emel::error::cast(emel::generator::error::none);
  emel::generator::action::dispatch_initialize_error_with_error_out_only(error_out_only_run,
                                                                         context);
  CHECK(error_out == emel::error::cast(emel::generator::error::backend));

  emel::generator::action::dispatch_initialize_error_without_channels(error_out_only_run,
                                                                      context);
}

TEST_CASE("generator generate dispatch actions cover channel variants") {
  emel::generator::action::context context{};
  callback_tracker tracker{};
  emel::error::type error_out = emel::error::cast(emel::generator::error::backend);
  size_t output_length_out = 0;

  auto with_both = make_generate_request(&tracker, &error_out, output_length_out);
  emel::generator::event::generate_ctx gen_ctx{};
  gen_ctx.tokens_generated = 2;
  gen_ctx.output_length = 10;
  gen_ctx.err = emel::error::cast(emel::generator::error::backend);
  emel::generator::event::generate_run gen_run{with_both, gen_ctx};
  emel::generator::action::dispatch_generate_done_with_callback_and_error_out(gen_run,
                                                                              context);
  CHECK(tracker.generate_done_called);
  CHECK(tracker.tokens_generated == 2);
  CHECK(tracker.output_length == 10);
  CHECK(error_out == emel::error::cast(emel::generator::error::none));

  tracker = {};
  auto callback_only = make_generate_request(&tracker, nullptr, output_length_out);
  emel::generator::event::generate_run callback_only_run{callback_only, gen_ctx};
  emel::generator::action::dispatch_generate_done_with_callback_only(callback_only_run,
                                                                     context);
  CHECK(tracker.generate_done_called);

  tracker = {};
  error_out = emel::error::cast(emel::generator::error::backend);
  auto error_out_only = make_generate_request(nullptr, &error_out, output_length_out);
  emel::generator::event::generate_run error_out_only_run{error_out_only, gen_ctx};
  emel::generator::action::dispatch_generate_done_with_error_out_only(error_out_only_run,
                                                                      context);
  CHECK(error_out == emel::error::cast(emel::generator::error::none));

  emel::generator::action::dispatch_generate_done_without_channels(error_out_only_run,
                                                                   context);

  tracker = {};
  error_out = emel::error::cast(emel::generator::error::none);
  gen_ctx.err = emel::error::cast(emel::generator::error::backend);
  emel::generator::action::dispatch_generate_error_with_callback_and_error_out(gen_run,
                                                                               context);
  CHECK(tracker.generate_error_called);
  CHECK(tracker.tokens_generated == 2);
  CHECK(tracker.output_length == 10);
  CHECK(error_out == emel::error::cast(emel::generator::error::backend));

  tracker = {};
  emel::generator::action::dispatch_generate_error_with_callback_only(callback_only_run,
                                                                      context);
  CHECK(tracker.generate_error_called);

  error_out = emel::error::cast(emel::generator::error::none);
  emel::generator::action::dispatch_generate_error_with_error_out_only(error_out_only_run,
                                                                       context);
  CHECK(error_out == emel::error::cast(emel::generator::error::backend));

  emel::generator::action::dispatch_generate_error_without_channels(error_out_only_run,
                                                                    context);
}

TEST_CASE(
    "generator request_planning accepts multi-token single-sequence prompt metadata") {
  emel::generator::action::context context{};
  callback_tracker tracker{};
  emel::error::type error_out = emel::error::cast(emel::generator::error::backend);
  size_t output_length_out = 0;

  auto generate = make_generate_request(&tracker, &error_out, output_length_out);
  emel::generator::event::generate_ctx generate_ctx{};
  generate_ctx.prompt_token_count = 2;
  emel::generator::event::generate_run generate_run{generate, generate_ctx};
  context.buffers.prompt_tokens[0] = 11;
  context.buffers.prompt_tokens[1] = 13;

  emel::generator::action::request_planning(generate_run, context);

  CHECK(generate_ctx.phase_accepted);
  CHECK(generate_ctx.phase_code == 0);
  CHECK(generate_ctx.prefill_step_size == 1);
  CHECK(generate_ctx.plan_step_count == 2);
  CHECK(generate_ctx.plan_outputs >= 0);
}

TEST_CASE("generator structured message request and channel guards classify callback variants") {
  emel::generator::action::context context{};
  emel::text::conditioner::sm conditioner{};
  context.model = &test_model();
  context.conditioner = &conditioner;
  context.format_prompt = emel::text::formatter::format_raw;
  context.limits.decode_capacity = 4;
  context.state.sequence_live = true;

  callback_tracker tracker{};
  emel::error::type error_out = emel::error::cast(emel::generator::error::none);
  size_t output_length_out = 0;
  auto generate = make_generate_request(&tracker, &error_out, output_length_out);
  emel::generator::event::generate_ctx generate_ctx{};
  generate_ctx.target_tokens = 2;
  generate_ctx.tokens_generated = 0;
  generate_ctx.render_status = emel::text::renderer::sequence_status::running;
  generate_ctx.selected_token = 1;
  emel::generator::event::generate_run generate_run{generate, generate_ctx};

  CHECK(generate.messages.size() == 1u);
  CHECK(generate.messages[0].role == "user");
  CHECK(generate.messages[0].content == "hello");
  CHECK(generate.add_generation_prompt);
  CHECK_FALSE(generate.enable_thinking);
  CHECK(emel::generator::guard::valid_generate{}(generate_run, context));
  CHECK(emel::generator::guard::valid_generate_with_reset{}(generate_run, context));
  CHECK_FALSE(emel::generator::guard::valid_generate_without_reset{}(generate_run, context));
  CHECK(emel::generator::guard::sequence_needs_reset{}(generate_run, context));
  CHECK_FALSE(emel::generator::guard::sequence_is_clear{}(generate_run, context));
  CHECK(emel::generator::guard::generate_done_callback_with_error_out{}(generate_run, context));
  CHECK_FALSE(emel::generator::guard::generate_no_done_callback_with_error_out{}(
      generate_run, context));
  CHECK_FALSE(emel::generator::guard::generate_done_callback_without_error_out{}(
      generate_run, context));
  CHECK(emel::generator::guard::generate_error_callback_with_error_out{}(generate_run, context));
  CHECK(emel::generator::guard::decode_should_continue{}(generate_run, context));

  context.state.sequence_live = false;
  CHECK(emel::generator::guard::valid_generate_without_reset{}(generate_run, context));
  CHECK_FALSE(emel::generator::guard::valid_generate_with_reset{}(generate_run, context));

  generate.on_done = {};
  generate.on_error = {};
  generate.error_out = nullptr;
  CHECK(emel::generator::guard::generate_no_done_callback_without_error_out{}(
      generate_run, context));
  CHECK(emel::generator::guard::generate_no_error_callback_without_error_out{}(
      generate_run, context));

  generate.max_tokens = 0;
  CHECK(emel::generator::guard::invalid_generate{}(generate_run, context));
  generate.max_tokens = 2;
  std::array<char, 16> output_buffer = {};
  generate.messages = {};
  CHECK(emel::generator::guard::invalid_generate{}(generate_run, context));
  generate.messages = std::span<const emel::text::formatter::chat_message>{
      static_cast<const emel::text::formatter::chat_message *>(nullptr), 1u};
  CHECK(emel::generator::guard::invalid_generate{}(generate_run, context));
  generate.messages = std::span<const emel::text::formatter::chat_message>{k_generate_messages};
  generate.output = std::span<char>{static_cast<char *>(nullptr), 1u};
  CHECK(emel::generator::guard::invalid_generate{}(generate_run, context));
  generate.output = std::span<char>{output_buffer};
  generate_ctx.tokens_generated = 2;
  CHECK(emel::generator::guard::decode_complete{}(generate_run, context));
  generate_ctx.tokens_generated = 0;
  generate_ctx.selected_token = test_model().vocab_data.eos_id;
  CHECK(emel::generator::guard::decode_complete{}(generate_run, context));

  auto initialize = make_initialize_request(&tracker, &error_out);
  emel::generator::event::initialize_ctx initialize_ctx{};
  emel::generator::event::initialize_run initialize_run{initialize, initialize_ctx};
  CHECK(emel::generator::guard::initialize_done_callback_with_error_out{}(
      initialize_run, context));
  CHECK(emel::generator::guard::initialize_error_callback_with_error_out{}(
      initialize_run, context));
  initialize.on_done = {};
  initialize.on_error = {};
  initialize.error_out = nullptr;
  CHECK(emel::generator::guard::initialize_no_done_callback_without_error_out{}(
      initialize_run, context));
  CHECK(emel::generator::guard::initialize_no_error_callback_without_error_out{}(
      initialize_run, context));

  auto preselected_initialize = make_initialize_request(
      &tracker, &error_out, emel::generator::selection_mode::preselected_argmax);
  emel::generator::event::initialize_ctx preselected_ctx{};
  emel::generator::event::initialize_run preselected_run{
      preselected_initialize, preselected_ctx};
  context.state.selection_mode = emel::generator::selection_mode::preselected_argmax;
  CHECK(emel::generator::guard::valid_initialize{}(preselected_run, context));
  CHECK(emel::generator::guard::initialize_uses_preselected_argmax{}(preselected_run, context));
  CHECK_FALSE(
      emel::generator::guard::initialize_uses_materialized_logits{}(preselected_run, context));
}

TEST_CASE("generator phase guards classify invalid and backend errors") {
  emel::generator::action::context context{};
  context.model = &test_model();
  context.limits.decode_capacity = 4;

  callback_tracker tracker{};
  emel::error::type error_out = emel::error::cast(emel::generator::error::none);
  size_t output_length_out = 0;

  auto initialize = make_initialize_request(&tracker, &error_out);
  emel::generator::event::initialize_ctx initialize_ctx{};
  emel::generator::event::initialize_run initialize_run{initialize, initialize_ctx};

  initialize_ctx.phase_accepted = false;
  initialize_ctx.phase_code =
      static_cast<int32_t>(emel::text::conditioner::error::invalid_argument);
  CHECK(emel::generator::guard::conditioner_bind_invalid_request{}(initialize_run, context));
  initialize_ctx.phase_code = static_cast<int32_t>(emel::text::conditioner::error::backend);
  CHECK(emel::generator::guard::conditioner_bind_backend_error{}(initialize_run, context));
  initialize_ctx.phase_code =
      static_cast<int32_t>(emel::error::cast(emel::text::renderer::error::invalid_request));
  CHECK(emel::generator::guard::renderer_initialize_invalid_request{}(initialize_run, context));
  initialize_ctx.phase_code =
      static_cast<int32_t>(emel::error::cast(emel::text::renderer::error::backend_error));
  CHECK(emel::generator::guard::renderer_initialize_backend_error{}(initialize_run, context));
  initialize_ctx.phase_code =
      static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::invalid_request));
  CHECK(emel::generator::guard::memory_reserve_invalid_request{}(initialize_run, context));
  initialize_ctx.phase_code =
      static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::backend_error));
  CHECK(emel::generator::guard::memory_reserve_backend_error{}(initialize_run, context));
  initialize_ctx.phase_code =
      static_cast<int32_t>(emel::error::cast(emel::graph::error::invalid_request));
  CHECK(emel::generator::guard::graph_reserve_invalid_request{}(initialize_run, context));
  initialize_ctx.phase_code =
      static_cast<int32_t>(emel::error::cast(emel::graph::error::assembler_failed));
  CHECK(emel::generator::guard::graph_reserve_backend_error{}(initialize_run, context));
  initialize_ctx.buffers_ready = true;
  CHECK(emel::generator::guard::sampler_configured{}(initialize_run, context));
  initialize_ctx.buffers_ready = false;
  CHECK(emel::generator::guard::sampler_config_failed{}(initialize_run, context));

  auto generate = make_generate_request(&tracker, &error_out, output_length_out);
  emel::generator::event::generate_ctx generate_ctx{};
  generate_ctx.err = emel::error::cast(emel::generator::error::none);
  emel::generator::event::generate_run generate_run{generate, generate_ctx};

  generate_ctx.phase_accepted = false;
  generate_ctx.phase_code =
      static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::invalid_request));
  CHECK(emel::generator::guard::reset_sequence_invalid_request{}(generate_run, context));
  generate_ctx.phase_code =
      static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::backend_error));
  CHECK(emel::generator::guard::reset_sequence_backend_error{}(generate_run, context));
  generate_ctx.phase_code = static_cast<int32_t>(emel::text::conditioner::error::capacity);
  CHECK(emel::generator::guard::conditioning_invalid_request{}(generate_run, context));
  generate_ctx.phase_accepted = true;
  generate_ctx.phase_code = 0;
  generate_ctx.prompt_token_count = 0;
  CHECK(emel::generator::guard::conditioning_backend_error{}(generate_run, context));
  generate_ctx.phase_accepted = false;
  generate_ctx.phase_code =
      static_cast<int32_t>(emel::error::cast(emel::batch::planner::error::invalid_step_size));
  CHECK(emel::generator::guard::planning_invalid_request{}(generate_run, context));
  generate_ctx.phase_code = static_cast<int32_t>(emel::error::set(
      emel::error::cast(emel::batch::planner::error::invalid_request),
      emel::batch::planner::error::invalid_sequence_metadata));
  CHECK(emel::generator::guard::planning_invalid_request{}(generate_run, context));
  CHECK_FALSE(emel::generator::guard::planning_backend_error{}(generate_run, context));
  generate_ctx.phase_accepted = true;
  generate_ctx.phase_code = 0;
  generate_ctx.plan_step_count = 0;
  CHECK(emel::generator::guard::planning_backend_error{}(generate_run, context));
  generate_ctx.phase_accepted = false;
  generate_ctx.phase_code =
      static_cast<int32_t>(emel::error::cast(emel::graph::error::invalid_request));
  CHECK(emel::generator::guard::prefill_compute_invalid_request{}(generate_run, context));
  generate_ctx.phase_code =
      static_cast<int32_t>(emel::error::cast(emel::graph::error::processor_failed));
  CHECK(emel::generator::guard::prefill_compute_backend_error{}(generate_run, context));
  CHECK(emel::generator::guard::decode_compute_backend_error{}(generate_run, context));
  generate_ctx.phase_code =
      static_cast<int32_t>(emel::error::cast(emel::logits::sampler::error::invalid_request));
  CHECK(emel::generator::guard::decode_sample_invalid_request{}(generate_run, context));
  generate_ctx.phase_code =
      static_cast<int32_t>(emel::error::cast(emel::logits::sampler::error::backend_error));
  CHECK(emel::generator::guard::decode_sample_backend_error{}(generate_run, context));
  generate_ctx.phase_code =
      static_cast<int32_t>(emel::error::cast(emel::text::renderer::error::invalid_request));
  CHECK(emel::generator::guard::decode_render_invalid_request{}(generate_run, context));
  CHECK(emel::generator::guard::flush_invalid_request{}(generate_run, context));
  generate_ctx.phase_code =
      static_cast<int32_t>(emel::error::cast(emel::text::renderer::error::backend_error));
  CHECK(emel::generator::guard::decode_render_backend_error{}(generate_run, context));
  CHECK(emel::generator::guard::flush_backend_error{}(generate_run, context));

  generate_ctx.err = emel::error::cast(emel::generator::error::none);
  CHECK(emel::generator::guard::generate_result_none{}(generate_run, context));
  initialize_ctx.err = emel::error::cast(emel::generator::error::invalid_request);
  CHECK(emel::generator::guard::initialize_result_invalid_request{}(initialize_run, context));
  generate_ctx.err = emel::error::cast(emel::generator::error::backend);
  CHECK(emel::generator::guard::generate_result_backend{}(generate_run, context));
}

TEST_CASE("generator runtime guards model explicit flash and nonflash compute selection") {
  emel::generator::action::context context{};
  auto & backend = context.compute.backend;
  backend.n_layer = 1;
  backend.n_head = 2;
  backend.n_head_kv = 2;
  backend.head_dim = 2;
  backend.head_dim_kv = 2;
  backend.n_ctx = 4;
  backend.q_attn.resize(4, 0.0f);
  backend.key_cache.resize(16, 0.0f);
  backend.value_cache.resize(16, 0.0f);
  backend.flash_key_cache.resize(16, 0.0f);
  backend.flash_value_cache.resize(16, 0.0f);
  backend.attn_ctx.resize(4, 0.0f);

  callback_tracker tracker{};
  emel::error::type error_out = emel::error::cast(emel::generator::error::none);
  size_t output_length_out = 0;
  auto generate = make_generate_request(&tracker, &error_out, output_length_out);
  emel::generator::event::generate_ctx generate_ctx{};
  generate_ctx.prompt_token_count = 2;
  generate_ctx.kv_tokens = 1;
  emel::generator::event::generate_run generate_run{generate, generate_ctx};

  CHECK(emel::generator::guard::prefill_flash_runtime_supported{}(generate_run, context));
  CHECK_FALSE(emel::generator::guard::prefill_nonflash_runtime_required{}(generate_run, context));
  CHECK(emel::generator::guard::decode_flash_runtime_supported{}(generate_run, context));
  CHECK_FALSE(emel::generator::guard::decode_nonflash_runtime_required{}(generate_run, context));

  backend.head_dim_kv = 1;
  backend.key_cache.resize(8, 0.0f);
  backend.value_cache.resize(8, 0.0f);
  backend.flash_key_cache.resize(8, 0.0f);
  backend.flash_value_cache.resize(8, 0.0f);

  CHECK_FALSE(emel::generator::guard::prefill_flash_runtime_supported{}(generate_run, context));
  CHECK(emel::generator::guard::prefill_nonflash_runtime_required{}(generate_run, context));
  CHECK_FALSE(emel::generator::guard::decode_flash_runtime_supported{}(generate_run, context));
  CHECK(emel::generator::guard::decode_nonflash_runtime_required{}(generate_run, context));
}

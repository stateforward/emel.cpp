#include <array>
#include <boost/sml.hpp>
#include <cstdint>
#include <cstring>
#include <doctest/doctest.h>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "emel/docs/detail.hpp"
#include "emel/emel.h"
#include "emel/generator/errors.hpp"
#include "emel/generator/sm.hpp"
#include "emel/model/data.hpp"
#include "emel/text/formatter/format.hpp"
#include "emel/text/tokenizer/sm.hpp"

namespace {

struct callback_tracker {
  bool initialize_done_called = false;
  bool initialize_error_called = false;
  bool generate_done_called = false;
  bool generate_error_called = false;
  const emel::generator::event::initialize * initialize_request = nullptr;
  const emel::generator::event::generate * generate_request = nullptr;
  int32_t tokens_generated = -1;
  size_t output_length = 0;
  emel::error::type err = emel::error::cast(emel::generator::error::none);
};

void on_initialize_done(void * owner, const emel::generator::events::initialize_done & ev) {
  auto * tracker = static_cast<callback_tracker *>(owner);
  tracker->initialize_done_called = true;
  tracker->initialize_request = ev.request;
}

void on_initialize_error(void * owner, const emel::generator::events::initialize_error & ev) {
  auto * tracker = static_cast<callback_tracker *>(owner);
  tracker->initialize_error_called = true;
  tracker->initialize_request = ev.request;
  tracker->err = ev.err;
}

void on_generate_done(void * owner, const emel::generator::events::generation_done & ev) {
  auto * tracker = static_cast<callback_tracker *>(owner);
  tracker->generate_done_called = true;
  tracker->generate_request = ev.request;
  tracker->tokens_generated = ev.tokens_generated;
  tracker->output_length = ev.output_length;
}

void on_generate_error(void * owner, const emel::generator::events::generation_error & ev) {
  auto * tracker = static_cast<callback_tracker *>(owner);
  tracker->generate_error_called = true;
  tracker->generate_request = ev.request;
  tracker->tokens_generated = ev.tokens_generated;
  tracker->output_length = ev.output_length;
  tracker->err = ev.err;
}

int32_t add_token(emel::model::data::vocab & vocab, const char * text,
                  const int32_t type = 0) {
  const uint32_t length = static_cast<uint32_t>(std::strlen(text));
  const uint32_t offset = vocab.token_bytes_used;
  std::memcpy(vocab.token_storage.data() + offset, text, length);
  const uint32_t id = vocab.n_tokens;
  vocab.entries[id].text_offset = offset;
  vocab.entries[id].text_length = length;
  vocab.entries[id].score = 0.0f;
  vocab.entries[id].type = type;
  vocab.token_bytes_used += length;
  vocab.n_tokens = id + 1;
  return static_cast<int32_t>(id);
}

bool tokenizer_bind_dispatch(void * tokenizer_sm,
                             const emel::text::tokenizer::event::bind & ev) {
  return static_cast<emel::text::tokenizer::sm *>(tokenizer_sm)->process_event(ev);
}

bool tokenizer_tokenize_dispatch(
    void * tokenizer_sm, const emel::text::tokenizer::event::tokenize & ev) {
  return static_cast<emel::text::tokenizer::sm *>(tokenizer_sm)->process_event(ev);
}

emel::error::type sampler_select_argmax(int32_t & candidate_ids,
                                        float & candidate_scores,
                                        int32_t & candidate_count,
                                        int32_t & selected_token_out) {
  int32_t best_index = 0;
  float best_score = (&candidate_scores)[0];
  for (int32_t idx = 1; idx < candidate_count; ++idx) {
    if ((&candidate_scores)[idx] > best_score) {
      best_score = (&candidate_scores)[idx];
      best_index = idx;
    }
  }
  selected_token_out = (&candidate_ids)[best_index];
  return emel::error::cast(emel::logits::sampler::error::none);
}

struct fake_backend {
  int32_t token_id = -1;
};

bool backend_validate(const emel::graph::processor::event::execute & request, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return request.compute_ctx != nullptr;
}

bool backend_prepare_graph(const emel::graph::processor::event::execute &,
                           bool * reused_out,
                           int32_t * err_out) {
  if (reused_out != nullptr) {
    *reused_out = false;
  }
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool backend_alloc_graph(const emel::graph::processor::event::execute &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool backend_bind_inputs(const emel::graph::processor::event::execute &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool backend_run_kernel(const emel::graph::processor::event::execute &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool backend_extract_outputs(const emel::graph::processor::event::execute & request,
                             int32_t * outputs_out,
                             int32_t * err_out) {
  auto * io = static_cast<emel::generator::compute_io *>(request.compute_ctx);
  auto * backend = static_cast<fake_backend *>(io->backend_ctx);
  for (int32_t idx = 0; idx < io->logits_capacity; ++idx) {
    io->logits[idx] = -1.0f;
  }
  io->logits[backend->token_id] = 8.0f;
  if (outputs_out != nullptr) {
    *outputs_out = 1;
  }
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

template <class... Ts, class fn>
constexpr void for_each_type(boost::sml::aux::type_list<Ts...>, fn && visitor) {
  (visitor.template operator()<Ts>(), ...);
}

struct prepared_model {
  emel::model::data data = {};
  int32_t hello_id = -1;
  int32_t world_id = -1;
};

static_assert(std::is_reference_v<
              decltype(std::declval<const emel::generator::event::initialize &>()
                           .dispatch_tokenizer_bind)>);
static_assert(std::is_reference_v<
              decltype(std::declval<const emel::generator::event::initialize &>().validate)>);
static_assert(std::is_same_v<
              std::remove_cvref_t<
                  decltype(std::declval<const emel::generator::event::initialize &>()
                               .sampler_fns)>,
              std::span<emel::logits::sampler::fn>>);
static_assert(std::is_same_v<
              std::remove_cvref_t<
                  decltype(std::declval<const emel::generator::event::generate &>().output)>,
              std::span<char>>);
static_assert(std::is_reference_v<
              decltype(std::declval<const emel::generator::event::generate &>()
                           .output_length_out)>);

prepared_model make_prepared_model() {
  prepared_model prepared{};
  prepared.data.vocab_data.tokenizer_model_id = emel::model::data::tokenizer_model::BPE;
  prepared.data.vocab_data.tokenizer_pre_id = emel::model::data::tokenizer_pre::GPT2;
  prepared.data.vocab_data.ignore_merges = true;
  prepared.hello_id = add_token(prepared.data.vocab_data, "hello");
  prepared.world_id = add_token(prepared.data.vocab_data, "world");
  prepared.data.params.n_vocab = static_cast<int32_t>(prepared.data.vocab_data.n_tokens);
  return prepared;
}

struct generator_fixture {
  static constexpr std::string_view k_phase_4_prompt = "hello";
  static constexpr int32_t k_phase_4_max_tokens = 1;

  prepared_model prepared = make_prepared_model();
  emel::text::tokenizer::sm tokenizer{};
  emel::text::conditioner::sm conditioner{};
  emel::generator::sm generator;
  fake_backend backend{};
  std::array<emel::logits::sampler::fn, 1> samplers = {
      emel::logits::sampler::fn::from<sampler_select_argmax>(),
  };
  int model_topology = 1;
  int prefill_plan = 2;
  int decode_plan = 3;
  int32_t hello_id = -1;
  int32_t world_id = -1;

  generator_fixture()
      : generator(prepared.data, conditioner, nullptr, emel::text::formatter::format_raw),
        hello_id(prepared.hello_id),
        world_id(prepared.world_id) {
    backend.token_id = world_id;
  }

  emel::generator::event::initialize make_initialize(
      callback_tracker & tracker,
      emel::error::type * error_out = nullptr) {
    emel::generator::event::initialize request{
      &model_topology,
      &prefill_plan,
      &decode_plan,
      &tokenizer,
      tokenizer_bind_dispatch,
      tokenizer_tokenize_dispatch,
      &backend,
      backend_validate,
      backend_prepare_graph,
      backend_alloc_graph,
      backend_bind_inputs,
      backend_run_kernel,
      backend_extract_outputs,
      std::span<emel::logits::sampler::fn>{samplers},
    };
    request.preprocessor_variant = emel::text::tokenizer::preprocessor::preprocessor_kind::bpe;
    request.encoder_variant = emel::text::encoders::encoder_kind::bpe;
    request.add_special = false;
    request.parse_special = false;
    request.max_node_count = 8;
    request.max_tensor_count = 8;
    request.bytes_per_tensor = 4;
    request.workspace_capacity_bytes = 4096;
    request.max_prompt_tokens = 8;
    request.max_generated_tokens = 4;
    request.max_blocks = 8;
    request.block_tokens = 4;
    request.strip_leading_space = false;
    request.error_out = error_out;
    request.on_done = emel::callback<void(const emel::generator::events::initialize_done &)>(
        &tracker, on_initialize_done);
    request.on_error = emel::callback<void(const emel::generator::events::initialize_error &)>(
        &tracker, on_initialize_error);
    return request;
  }

  emel::generator::event::generate make_generate(callback_tracker & tracker,
                                                 char * output,
                                                 size_t output_capacity,
                                                 size_t & output_length_out,
                                                 emel::error::type * error_out = nullptr) {
    emel::generator::event::generate request{
      k_phase_4_prompt,
      k_phase_4_max_tokens,
      std::span<char>{output, output_capacity},
      output_length_out,
    };
    request.error_out = error_out;
    request.on_done = emel::callback<void(const emel::generator::events::generation_done &)>(
        &tracker, on_generate_done);
    request.on_error = emel::callback<void(const emel::generator::events::generation_error &)>(
        &tracker, on_generate_error);
    return request;
  }

  emel::model::data & model() noexcept { return prepared.data; }
};

}  // namespace

TEST_CASE("generator_starts_uninitialized") {
  auto fixture = std::make_unique<generator_fixture>();
  CHECK(fixture->generator.is(boost::sml::state<emel::generator::uninitialized>));
}

TEST_CASE("generator_rejects_generate_before_initialize") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker tracker{};
  std::array<char, 16> output = {};
  size_t output_length = 7;
  emel::error::type error = emel::error::cast(emel::generator::error::none);
  const auto request =
      fixture->make_generate(tracker, output.data(), output.size(), output_length, &error);

  CHECK_FALSE(fixture->generator.process_event(request));
  CHECK(fixture->generator.is(boost::sml::state<emel::generator::uninitialized>));
  CHECK_FALSE(tracker.generate_done_called);
  CHECK(tracker.generate_error_called);
  CHECK(error == emel::error::cast(emel::generator::error::invalid_request));
  CHECK(tracker.err == emel::error::cast(emel::generator::error::invalid_request));
}

TEST_CASE("generator_initialize_succeeds_and_enters_ready") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker tracker{};
  emel::error::type error = emel::error::cast(emel::generator::error::backend);
  const auto request = fixture->make_initialize(tracker, &error);

  CHECK(fixture->generator.process_event(request));
  CHECK(fixture->generator.is(boost::sml::state<emel::generator::ready>));
  CHECK(tracker.initialize_done_called);
  CHECK_FALSE(tracker.initialize_error_called);
  CHECK(error == emel::error::cast(emel::generator::error::none));
}

TEST_CASE("generator_rejects_invalid_initialize_request") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker tracker{};
  emel::error::type error = emel::error::cast(emel::generator::error::none);
  auto request = fixture->make_initialize(tracker, &error);
  request.model_topology = nullptr;

  CHECK_FALSE(fixture->generator.process_event(request));
  CHECK(fixture->generator.is(boost::sml::state<emel::generator::uninitialized>));
  CHECK_FALSE(tracker.initialize_done_called);
  CHECK(tracker.initialize_error_called);
  CHECK(error == emel::error::cast(emel::generator::error::invalid_request));
  CHECK(tracker.err == emel::error::cast(emel::generator::error::invalid_request));
}

TEST_CASE("generator_initialize_reports_original_request_without_generation_callbacks") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker tracker{};
  emel::error::type error = emel::error::cast(emel::generator::error::backend);
  const auto request = fixture->make_initialize(tracker, &error);

  REQUIRE(fixture->generator.process_event(request));
  CHECK(fixture->generator.is(boost::sml::state<emel::generator::ready>));
  CHECK(tracker.initialize_done_called);
  CHECK_FALSE(tracker.initialize_error_called);
  CHECK(tracker.initialize_request == &request);
  CHECK_FALSE(tracker.generate_done_called);
  CHECK_FALSE(tracker.generate_error_called);
  CHECK(tracker.generate_request == nullptr);
  CHECK(error == emel::error::cast(emel::generator::error::none));
}

TEST_CASE("generator_initialize_can_rebind_ready_session_without_re_reserving_graph") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker first_tracker{};
  emel::error::type first_error = emel::error::cast(emel::generator::error::backend);
  const auto first_request = fixture->make_initialize(first_tracker, &first_error);

  REQUIRE(fixture->generator.process_event(first_request));
  CHECK(fixture->generator.is(boost::sml::state<emel::generator::ready>));
  CHECK(first_tracker.initialize_done_called);
  CHECK(first_error == emel::error::cast(emel::generator::error::none));

  callback_tracker second_tracker{};
  emel::error::type second_error = emel::error::cast(emel::generator::error::backend);
  const auto second_request = fixture->make_initialize(second_tracker, &second_error);

  CHECK(fixture->generator.process_event(second_request));
  CHECK(fixture->generator.is(boost::sml::state<emel::generator::ready>));
  CHECK(second_tracker.initialize_done_called);
  CHECK_FALSE(second_tracker.initialize_error_called);
  CHECK(second_error == emel::error::cast(emel::generator::error::none));
}

TEST_CASE("generator_generate_runs_real_inference_and_writes_output") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker initialize_tracker{};
  emel::error::type initialize_error = emel::error::cast(emel::generator::error::backend);
  const auto initialize_request = fixture->make_initialize(initialize_tracker, &initialize_error);
  REQUIRE(fixture->generator.process_event(initialize_request));

  callback_tracker generate_tracker{};
  std::array<char, 32> output = {};
  size_t output_length = 99;
  emel::error::type generate_error = emel::error::cast(emel::generator::error::backend);
  const auto generate_request =
      fixture->make_generate(generate_tracker, output.data(), output.size(), output_length,
                             &generate_error);

  CHECK(fixture->generator.process_event(generate_request));
  CHECK(fixture->generator.is(boost::sml::state<emel::generator::ready>));
  CHECK_FALSE(generate_tracker.generate_error_called);
  CHECK(generate_tracker.generate_done_called);
  CHECK(generate_error == emel::error::cast(emel::generator::error::none));
  CHECK(generate_tracker.tokens_generated == 1);
  CHECK(output_length == 5);
  CHECK(generate_tracker.output_length == 5);
  CHECK(std::string_view(output.data(), output_length) == "world");
}

TEST_CASE("generator_generate_pins_the_phase_4_request_contract") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker initialize_tracker{};
  emel::error::type initialize_error = emel::error::cast(emel::generator::error::backend);
  const auto initialize_request = fixture->make_initialize(initialize_tracker, &initialize_error);
  REQUIRE(fixture->generator.process_event(initialize_request));

  callback_tracker generate_tracker{};
  std::array<char, 32> output = {};
  size_t output_length = 0;
  emel::error::type generate_error = emel::error::cast(emel::generator::error::backend);
  const auto generate_request =
      fixture->make_generate(generate_tracker, output.data(), output.size(), output_length,
                             &generate_error);

  CHECK(generate_request.prompt == generator_fixture::k_phase_4_prompt);
  CHECK(generate_request.max_tokens == generator_fixture::k_phase_4_max_tokens);
  CHECK(generate_request.output.data() == output.data());
  CHECK(generate_request.output.size() == output.size());
  CHECK(fixture->generator.process_event(generate_request));
  CHECK_FALSE(generate_tracker.generate_error_called);
  CHECK(generate_tracker.generate_done_called);
  CHECK(generate_tracker.generate_request == &generate_request);
  CHECK(generate_tracker.tokens_generated == generator_fixture::k_phase_4_max_tokens);
  CHECK(generate_tracker.output_length == 5);
  CHECK(output_length == generate_tracker.output_length);
  CHECK(std::string_view(output.data(), output_length) == "world");
}

TEST_CASE("generator_generate_reports_bounded_output_buffer_errors") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker initialize_tracker{};
  emel::error::type initialize_error = emel::error::cast(emel::generator::error::backend);
  const auto initialize_request = fixture->make_initialize(initialize_tracker, &initialize_error);
  REQUIRE(fixture->generator.process_event(initialize_request));

  callback_tracker generate_tracker{};
  std::array<char, 4> output = {};
  size_t output_length = 17;
  emel::error::type generate_error = emel::error::cast(emel::generator::error::none);
  const auto generate_request =
      fixture->make_generate(generate_tracker, output.data(), output.size(), output_length,
                             &generate_error);

  CHECK_FALSE(fixture->generator.process_event(generate_request));
  CHECK(fixture->generator.is(boost::sml::state<emel::generator::ready>));
  CHECK_FALSE(generate_tracker.generate_done_called);
  CHECK(generate_tracker.generate_error_called);
  CHECK(generate_tracker.generate_request == &generate_request);
  CHECK(generate_tracker.tokens_generated == 0);
  CHECK(generate_tracker.output_length == 0);
  CHECK(output_length == 0);
  CHECK(generate_error == emel::error::cast(emel::generator::error::invalid_request));
  CHECK(generate_tracker.err == emel::error::cast(emel::generator::error::invalid_request));
}

TEST_CASE("generator_generate_multiple_tokens_and_resets_sequence_on_reuse") {
  auto fixture = std::make_unique<generator_fixture>();
  callback_tracker initialize_tracker{};
  emel::error::type initialize_error = emel::error::cast(emel::generator::error::backend);
  const auto initialize_request = fixture->make_initialize(initialize_tracker, &initialize_error);
  REQUIRE(fixture->generator.process_event(initialize_request));

  callback_tracker first_tracker{};
  std::array<char, 32> first_output = {};
  size_t first_output_length = 0;
  emel::error::type first_error = emel::error::cast(emel::generator::error::backend);
  auto first_request =
      fixture->make_generate(first_tracker, first_output.data(), first_output.size(),
                             first_output_length, &first_error);
  first_request.max_tokens = 2;

  CHECK(fixture->generator.process_event(first_request));
  CHECK(first_error == emel::error::cast(emel::generator::error::none));
  CHECK(first_tracker.tokens_generated == 2);
  CHECK(std::string_view(first_output.data(), first_output_length) == "worldworld");

  callback_tracker second_tracker{};
  std::array<char, 16> second_output = {};
  size_t second_output_length = 0;
  emel::error::type second_error = emel::error::cast(emel::generator::error::backend);
  const auto second_request =
      fixture->make_generate(second_tracker, second_output.data(), second_output.size(),
                             second_output_length, &second_error);

  CHECK(fixture->generator.process_event(second_request));
  CHECK(second_error == emel::error::cast(emel::generator::error::none));
  CHECK(second_tracker.tokens_generated == 1);
  CHECK(std::string_view(second_output.data(), second_output_length) == "world");
}

TEST_CASE("generator_docs_table_uses_typed_completion_event_names") {
  using machine_t = boost::sml::sm<emel::generator::model>;
  using transitions = typename machine_t::transitions;

  bool has_initialize_completion = false;
  bool has_generate_completion = false;

  for_each_type(transitions{}, [&]<class transition_t>() {
    using event = typename transition_t::event;
    const std::string event_name = emel::docs::detail::table_event_name<event>();
    if (event_name == "completion<initialize_run>") {
      has_initialize_completion = true;
    }
    if (event_name == "completion<generate_run>") {
      has_generate_completion = true;
    }
  });

  CHECK(has_initialize_completion);
  CHECK(has_generate_completion);
}

TEST_CASE("docs_detail_shortens_lambda_type_names_for_mermaid") {
  using emel::docs::detail::shorten_type_name;

  CHECK(shorten_type_name("lambda at /tmp/path/my_action.cpp:42:7>") == "lambda_my_action_42_7");
  CHECK(shorten_type_name("lambda at my_action.cpp:42>") == "lambda_my_action_42");
  CHECK(shorten_type_name("lambda at my_action.cpp>") == "lambda_my_action");
}

TEST_CASE("docs_detail_table_event_name_supports_non_completion_event") {
  const auto event_name =
      emel::docs::detail::table_event_name<emel::generator::event::generate_run>();
  CHECK(event_name == "generate_run");
}

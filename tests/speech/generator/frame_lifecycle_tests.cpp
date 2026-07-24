#include "doctest/doctest.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <span>

#include "../../allocation_tracker.hpp"
#include "emel/batch/planner/events.hpp"
#include "emel/error/error.hpp"
#include "emel/speech/generator/frame/sm.hpp"

namespace {

namespace frame = emel::speech::generator::frame;
namespace sml = stateforward::sml;

constexpr size_t PUBLIC_CODEBOOK_COUNT = 2u;
constexpr size_t MODEL_TOKEN_COUNT = 4u;
constexpr size_t PREDICTED_CODEBOOK_COUNT = 3u;
constexpr int32_t MAX_FRAME_TRANSITIONS = 8;
constexpr int32_t MAX_CROSS_ACTOR_CALLS = 6;
constexpr auto FAKE_DISPATCH_BUDGET = std::chrono::milliseconds{10};

enum class phase_id : int32_t {
  tokenize = 1,
  plan = 2,
  predict = 3,
  graph = 4,
  sample = 5,
  detokenize = 6,
};

struct phase_trace {
  std::array<phase_id, 16> entries{};
  size_t size = 0;
  bool overflow = false;

  void record(const phase_id phase) noexcept {
    if (size < entries.size()) {
      entries[size++] = phase;
    } else {
      overflow = true;
    }
  }

  void reset() noexcept {
    entries.fill({});
    size = 0;
    overflow = false;
  }
};

struct sml_trace {
  std::array<uint8_t, 128> entries{};
  size_t size = 0;
  int32_t transitions = 0;
  bool overflow = false;

  void record(const uint8_t value) noexcept {
    if (size < entries.size()) {
      entries[size++] = value;
    } else {
      overflow = true;
    }
  }

  void reset() noexcept {
    entries.fill(0);
    size = 0;
    transitions = 0;
    overflow = false;
  }

  template <class SM, class event_type>
  void log_process_event(const event_type &) noexcept {
    record(1);
  }

  template <class SM, class guard_type, class event_type>
  void log_guard(const guard_type &, const event_type &,
                 const bool accepted) noexcept {
    record(accepted ? 3 : 2);
  }

  template <class SM, class action_type, class event_type>
  void log_action(const action_type &, const event_type &) noexcept {
    record(4);
  }

  template <class SM, class source_state, class destination_state>
  void log_state_change(const source_state &,
                        const destination_state &) noexcept {
    ++transitions;
    record(5);
  }
};

struct fake_workspace {};

struct fake_tokenize {
  std::span<const int32_t> audio_tokens = {};
  std::span<int32_t> model_tokens_out = {};
  int32_t &error_out;
};

struct fake_predict {
  fake_predict(std::span<const int32_t> tokens_ref,
               fake_workspace &workspace_ref,
               const int32_t planned_step_size_ref,
               const int32_t planned_output_count_ref) noexcept
      : tokens(tokens_ref), workspace(workspace_ref),
        planned_step_size(planned_step_size_ref),
        planned_output_count(planned_output_count_ref) {}

  std::span<const int32_t> tokens;
  fake_workspace &workspace;
  int32_t planned_step_size = 0;
  int32_t planned_output_count = 0;
  emel::error::type *error_out = nullptr;
};

struct fake_graph {
  fake_graph(fake_workspace &workspace_ref,
             std::span<const int32_t> tokens_ref) noexcept
      : workspace(workspace_ref), tokens(tokens_ref) {}

  fake_workspace &workspace;
  std::span<const int32_t> tokens;
  emel::error::type *error_out = nullptr;
  emel::error::type *graph_error_out = nullptr;
};

struct fake_sample {
  fake_sample(fake_workspace &workspace_ref,
              std::span<const int32_t> tokens_ref,
              std::span<int32_t> tokens_out_ref,
              int32_t &text_token_out_ref) noexcept
      : workspace(workspace_ref), tokens(tokens_ref),
        tokens_out(tokens_out_ref), text_token_out(text_token_out_ref) {}

  fake_workspace &workspace;
  std::span<const int32_t> tokens;
  std::span<int32_t> tokens_out;
  int32_t &text_token_out;
  emel::error::type *error_out = nullptr;
  emel::error::type *graph_error_out = nullptr;
};

struct fake_detokenize {
  int32_t text_token = -1;
  std::span<const int32_t> audio_tokens = {};
  int32_t &text_token_out;
  std::span<int32_t> audio_tokens_out = {};
  bool &produced_out;
  int32_t &error_out;
};

struct fake_tokenizer_actor {
  phase_trace *trace = nullptr;
  int32_t tokenize_calls = 0;
  int32_t detokenize_calls = 0;
  bool tokenize_accepted = true;
  bool detokenize_accepted = true;
  bool produce = true;
  int32_t tokenize_error = 0;
  int32_t detokenize_error = 0;

  bool process_event(const fake_tokenize &request) noexcept {
    trace->record(phase_id::tokenize);
    ++tokenize_calls;
    std::copy(request.audio_tokens.begin(), request.audio_tokens.end(),
              request.model_tokens_out.begin());
    request.error_out = tokenize_error;
    return tokenize_accepted;
  }

  bool process_event(const fake_detokenize &request) noexcept {
    trace->record(phase_id::detokenize);
    ++detokenize_calls;
    for (size_t index = 0; index < request.audio_tokens_out.size(); ++index) {
      request.audio_tokens_out[index] = request.audio_tokens[index];
    }
    request.text_token_out = request.text_token;
    request.produced_out = produce;
    request.error_out = detokenize_error;
    return detokenize_accepted;
  }
};

struct fake_planner_actor {
  phase_trace *trace = nullptr;
  int32_t calls = 0;
  bool accepted = true;
  bool report_error = false;

  bool process_event(
      const emel::batch::planner::event::plan_request &request) noexcept {
    trace->record(phase_id::plan);
    ++calls;
    if (accepted) {
      if (report_error) {
        request.on_error(emel::batch::planner::events::plan_error{
            .err =
                emel::error::cast(emel::batch::planner::error::invalid_request),
            .request = &request,
        });
      } else {
        const std::array<int32_t, 1> step_sizes{1};
        request.on_done(emel::batch::planner::events::plan_done{
            .request = &request,
            .step_sizes = step_sizes.data(),
            .step_count = 1,
            .total_outputs = 1,
        });
      }
    }
    return accepted;
  }
};

struct fake_predictor_actor {
  phase_trace *trace = nullptr;
  int32_t calls = 0;
  int32_t last_plan_step_size = 0;
  int32_t last_plan_output_count = 0;
  bool accepted = true;
  emel::error::type error = 0;

  bool process_event(const fake_predict &request) noexcept {
    trace->record(phase_id::predict);
    ++calls;
    last_plan_step_size = request.planned_step_size;
    last_plan_output_count = request.planned_output_count;
    *request.error_out = error;
    return accepted;
  }
};

struct fake_graph_actor {
  phase_trace *trace = nullptr;
  int32_t calls = 0;
  bool accepted = true;
  emel::error::type error = 0;
  emel::error::type graph_error = 0;

  bool process_event(const fake_graph &request) noexcept {
    trace->record(phase_id::graph);
    ++calls;
    *request.error_out = error;
    *request.graph_error_out = graph_error;
    return accepted;
  }
};

struct fake_sampler_actor {
  phase_trace *trace = nullptr;
  int32_t calls = 0;
  bool accepted = true;
  emel::error::type error = 0;
  emel::error::type graph_error = 0;

  bool process_event(const fake_sample &request) noexcept {
    trace->record(phase_id::sample);
    ++calls;
    for (size_t index = 0; index < request.tokens_out.size(); ++index) {
      request.tokens_out[index] = 111 + static_cast<int32_t>(index);
    }
    request.text_token_out = 42;
    *request.error_out = error;
    *request.graph_error_out = graph_error;
    return accepted;
  }
};

struct fake_dependencies {
  using tokenize_event = fake_tokenize;
  using predict_event = fake_predict;
  using graph_event = fake_graph;
  using sample_event = fake_sample;
  using detokenize_event = fake_detokenize;

  fake_planner_actor &planner;
  fake_tokenizer_actor &tokenizer;
  fake_predictor_actor &predictor;
  fake_graph_actor &graph;
  fake_sampler_actor &sampler;
  fake_workspace &prediction_workspace;
  std::span<int32_t> model_codes = {};
  std::span<int32_t> predicted_codes = {};
  int32_t codebook_count = 0;
  emel::batch::planner::event::plan_mode frame_plan_mode =
      emel::batch::planner::event::plan_mode::simple;
  int32_t frame_plan_steps = 0;
  int32_t frame_plan_token_count = 0;
  bool frame_plan_output_all = false;
};

template <class logger_type> struct fixture_machine {
  using type = frame::traced_sm<fake_dependencies, logger_type>;
};

template <> struct fixture_machine<void> {
  using type = frame::sm<fake_dependencies>;
};

template <class logger_type = void> struct fixture_base {
  phase_trace phases{};
  std::array<int32_t, MODEL_TOKEN_COUNT> model_codes{};
  std::array<int32_t, PREDICTED_CODEBOOK_COUNT> predicted_codes{};
  fake_planner_actor planner{.trace = &phases};
  fake_tokenizer_actor tokenizer{.trace = &phases};
  fake_predictor_actor predictor{.trace = &phases};
  fake_graph_actor graph{.trace = &phases};
  fake_sampler_actor sampler{.trace = &phases};
  fake_workspace prediction_workspace{};
  fake_dependencies dependencies{
      .planner = planner,
      .tokenizer = tokenizer,
      .predictor = predictor,
      .graph = graph,
      .sampler = sampler,
      .prediction_workspace = prediction_workspace,
      .model_codes = std::span<int32_t>{model_codes},
      .predicted_codes = std::span<int32_t>{predicted_codes},
      .codebook_count = static_cast<int32_t>(PUBLIC_CODEBOOK_COUNT),
      .frame_plan_mode = emel::batch::planner::event::plan_mode::simple,
      .frame_plan_steps = 1,
      .frame_plan_token_count = 1,
      .frame_plan_output_all = true,
  };
  typename fixture_machine<logger_type>::type machine;

  template <class... args>
  explicit fixture_base(args &...args_in) : machine{dependencies, args_in...} {}
};

using fixture = fixture_base<>;

struct callback_probe {
  int32_t done_calls = 0;
  int32_t error_calls = 0;
  int32_t text_token = -1;
  emel::error::type last_error = 0;

  void done(const frame::events::run_done &ev) noexcept {
    ++done_calls;
    text_token = ev.text_token;
  }

  void error(const frame::events::run_error &ev) noexcept {
    ++error_calls;
    last_error = ev.err;
  }
};

struct unexpected_request {
  emel::error::type &error_out;
};

template <class machine_type>
concept accepts_internal_run = requires(
    machine_type &machine, const frame::detail::run_frame &runtime_ev) {
  machine.process_event(runtime_ev);
};

static_assert(!accepts_internal_run<frame::sm<fake_dependencies>>);

template <class...> struct forbidden_queue {};
struct forbidden_lock {};

template <class dependencies_type, class policy_type>
concept accepts_actor_policy =
    requires(dependencies_type &dependencies, policy_type &policy) {
      frame::sm<dependencies_type>{dependencies, policy};
    };

static_assert(!accepts_actor_policy<fake_dependencies,
                                    sml::process_queue<forbidden_queue>>);
static_assert(!accepts_actor_policy<fake_dependencies,
                                    sml::defer_queue<forbidden_queue>>);
static_assert(
    !accepts_actor_policy<fake_dependencies, sml::thread_safe<forbidden_lock>>);
static_assert(!accepts_actor_policy<fake_dependencies, sml::logger<sml_trace>>);

emel::error::type error_code(const frame::error value) noexcept {
  return frame::action::error_code(value);
}

} // namespace

TEST_CASE("speech_generator_frame_runs_the_middle_pipeline_synchronously") {
  fixture test{};
  callback_probe probe{};
  const std::array<int32_t, 2> encoded_tokens{11, 12};
  std::array<int32_t, 2> generated_tokens{};
  int32_t text_token = -1;
  emel::error::type err = -1;
  const auto on_done =
      emel::callback<void(const frame::events::run_done &)>::from<
          callback_probe, &callback_probe::done>(&probe);
  const frame::event::run request{encoded_tokens, generated_tokens, text_token,
                                  err, on_done};

  REQUIRE(test.machine.process_event(request));
  CHECK(test.machine.is(sml::state<frame::state_idle>));
  CHECK(test.tokenizer.tokenize_calls == 1);
  CHECK(test.planner.calls == 1);
  CHECK(test.predictor.calls == 1);
  CHECK(test.graph.calls == 1);
  CHECK(test.sampler.calls == 1);
  CHECK(test.tokenizer.detokenize_calls == 1);
  CHECK(test.predictor.last_plan_step_size == 1);
  CHECK(test.predictor.last_plan_output_count == 1);
  CHECK(generated_tokens == std::array<int32_t, 2>{111, 112});
  CHECK(text_token == 42);
  CHECK(err == error_code(frame::error::none));
  CHECK(probe.done_calls == 1);
  CHECK(probe.error_calls == 0);
  CHECK(probe.text_token == 42);
}

TEST_CASE("speech_generator_frame_rejects_invalid_spans_and_restores_idle") {
  SUBCASE("encoded input is narrower than the public codebook contract") {
    fixture test{};
    callback_probe probe{};
    const std::array<int32_t, 1> encoded_tokens{11};
    std::array<int32_t, PUBLIC_CODEBOOK_COUNT> generated_tokens{};
    int32_t text_token = 7;
    emel::error::type err = 0;
    const auto on_error =
        emel::callback<void(const frame::events::run_error &)>::from<
            callback_probe, &callback_probe::error>(&probe);
    const frame::event::run request{
        encoded_tokens, generated_tokens, text_token, err, {}, on_error};

    CHECK_FALSE(test.machine.process_event(request));
    CHECK(test.machine.is(sml::state<frame::state_idle>));
    CHECK(err == error_code(frame::error::invalid_request));
    CHECK(text_token == -1);
    CHECK(probe.error_calls == 1);
    CHECK(test.tokenizer.tokenize_calls == 0);
  }

  SUBCASE("generated output capacity must equal the public codebook contract") {
    fixture test{};
    const std::array<int32_t, PUBLIC_CODEBOOK_COUNT> encoded_tokens{11, 12};
    std::array<int32_t, PUBLIC_CODEBOOK_COUNT + 1> generated_tokens{31, 32,
                                                                    991};
    int32_t text_token = 7;
    emel::error::type err = 0;
    const frame::event::run request{encoded_tokens, generated_tokens,
                                    text_token, err};

    CHECK_FALSE(test.machine.process_event(request));
    CHECK(test.machine.is(sml::state<frame::state_idle>));
    CHECK(err == error_code(frame::error::invalid_request));
    CHECK(text_token == -1);
    CHECK(generated_tokens ==
          std::array<int32_t, PUBLIC_CODEBOOK_COUNT + 1>{31, 32, 991});
    CHECK(test.tokenizer.tokenize_calls == 0);
  }
}

TEST_CASE("speech_generator_frame_reports_each_child_phase_failure") {
  const auto run_failure = [](fixture &test, const frame::error expected) {
    callback_probe probe{};
    const std::array<int32_t, 2> encoded_tokens{11, 12};
    std::array<int32_t, 2> generated_tokens{};
    int32_t text_token = 7;
    emel::error::type err = 0;
    const auto on_error =
        emel::callback<void(const frame::events::run_error &)>::from<
            callback_probe, &callback_probe::error>(&probe);
    const frame::event::run request{
        encoded_tokens, generated_tokens, text_token, err, {}, on_error};

    CHECK_FALSE(test.machine.process_event(request));
    CHECK(test.machine.is(sml::state<frame::state_idle>));
    CHECK(err == error_code(expected));
    CHECK(text_token == -1);
    CHECK(probe.error_calls == 1);
    CHECK(probe.last_error == error_code(expected));
  };

  SUBCASE("tokenize") {
    fixture test{};
    test.tokenizer.tokenize_accepted = false;
    run_failure(test, frame::error::tokenize_failed);
  }
  SUBCASE("plan") {
    fixture test{};
    test.planner.accepted = false;
    run_failure(test, frame::error::planning_failed);
  }
  SUBCASE("predict") {
    fixture test{};
    test.predictor.accepted = false;
    run_failure(test, frame::error::predict_failed);
  }
  SUBCASE("graph") {
    fixture test{};
    test.graph.accepted = false;
    run_failure(test, frame::error::graph_failed);
  }
  SUBCASE("sample") {
    fixture test{};
    test.sampler.accepted = false;
    run_failure(test, frame::error::sample_failed);
  }
  SUBCASE("detokenize") {
    fixture test{};
    test.tokenizer.detokenize_accepted = false;
    run_failure(test, frame::error::detokenize_failed);
  }
}

TEST_CASE("speech_generator_frame_maps_accepted_child_error_channels") {
  const auto run_failure = [](fixture &test, const frame::error expected) {
    const std::array<int32_t, PUBLIC_CODEBOOK_COUNT> encoded_tokens{11, 12};
    std::array<int32_t, PUBLIC_CODEBOOK_COUNT> generated_tokens{};
    int32_t text_token = 7;
    emel::error::type err = 0;
    const frame::event::run request{encoded_tokens, generated_tokens,
                                    text_token, err};

    CHECK_FALSE(test.machine.process_event(request));
    CHECK(test.machine.is(sml::state<frame::state_idle>));
    CHECK(err == error_code(expected));
    CHECK(text_token == -1);
  };

  SUBCASE("tokenize error with accepted dispatch") {
    fixture test{};
    test.tokenizer.tokenize_error = 9;
    run_failure(test, frame::error::tokenize_failed);
  }
  SUBCASE("planner error callback with accepted dispatch") {
    fixture test{};
    test.planner.report_error = true;
    run_failure(test, frame::error::planning_failed);
  }
  SUBCASE("predict error with accepted dispatch") {
    fixture test{};
    test.predictor.error = 9;
    run_failure(test, frame::error::predict_failed);
  }
  SUBCASE("graph error channel with accepted dispatch") {
    fixture test{};
    test.graph.graph_error = 9;
    run_failure(test, frame::error::graph_failed);
  }
  SUBCASE("graph child error channel with accepted dispatch") {
    fixture test{};
    test.graph.error = 9;
    run_failure(test, frame::error::graph_failed);
  }
  SUBCASE("sample graph error channel with accepted dispatch") {
    fixture test{};
    test.sampler.graph_error = 9;
    run_failure(test, frame::error::sample_failed);
  }
  SUBCASE("sample child error channel with accepted dispatch") {
    fixture test{};
    test.sampler.error = 9;
    run_failure(test, frame::error::sample_failed);
  }
  SUBCASE("detokenize error with accepted dispatch") {
    fixture test{};
    test.tokenizer.detokenize_error = 9;
    run_failure(test, frame::error::detokenize_failed);
  }
}

TEST_CASE("speech_generator_frame_treats_missing_output_as_pending_error") {
  fixture test{};
  test.tokenizer.produce = false;
  callback_probe probe{};
  const std::array<int32_t, 2> encoded_tokens{11, 12};
  std::array<int32_t, 2> generated_tokens{};
  int32_t text_token = 7;
  emel::error::type err = 0;
  const auto on_error =
      emel::callback<void(const frame::events::run_error &)>::from<
          callback_probe, &callback_probe::error>(&probe);
  const frame::event::run request{
      encoded_tokens, generated_tokens, text_token, err, {}, on_error};

  CHECK_FALSE(test.machine.process_event(request));
  CHECK(test.machine.is(sml::state<frame::state_idle>));
  CHECK(err == error_code(frame::error::frame_pending));
  CHECK(text_token == -1);
  CHECK(probe.error_calls == 1);
}

TEST_CASE("speech_generator_frame_handles_unexpected_external_events") {
  fixture test{};
  emel::error::type err = 0;

  REQUIRE(test.machine.process_event(unexpected_request{err}));
  CHECK(err == error_code(frame::error::unexpected_event));
  CHECK(test.machine.is(sml::state<frame::state_idle>));
}

TEST_CASE(
    "speech_generator_frame_trace_is_deterministic_and_statically_bounded") {
  sml_trace logger{};
  fixture_base<sml_trace> test{logger};
  const std::array<int32_t, PUBLIC_CODEBOOK_COUNT> encoded_tokens{11, 12};
  std::array<int32_t, PUBLIC_CODEBOOK_COUNT> generated_tokens{};
  int32_t text_token = -1;
  emel::error::type err = 0;
  const frame::event::run request{encoded_tokens, generated_tokens, text_token,
                                  err};
  const std::array<phase_id, 16> expected_phase_trace{
      phase_id::tokenize, phase_id::plan,   phase_id::predict,
      phase_id::graph,    phase_id::sample, phase_id::detokenize};

  logger.reset();
  test.phases.reset();
  REQUIRE(test.machine.process_event(request));
  const auto first_sml_trace = logger.entries;
  const size_t first_sml_trace_size = logger.size;
  const auto first_phase_trace = test.phases.entries;
  const size_t first_phase_trace_size = test.phases.size;

  CHECK(logger.transitions == MAX_FRAME_TRANSITIONS);
  CHECK(logger.transitions <= MAX_FRAME_TRANSITIONS);
  CHECK_FALSE(logger.overflow);
  CHECK(test.phases.size == static_cast<size_t>(MAX_CROSS_ACTOR_CALLS));
  CHECK_FALSE(test.phases.overflow);
  CHECK(test.phases.entries == expected_phase_trace);

  logger.reset();
  test.phases.reset();
  REQUIRE(test.machine.process_event(request));

  CHECK(logger.entries == first_sml_trace);
  CHECK(logger.size == first_sml_trace_size);
  CHECK(logger.transitions == MAX_FRAME_TRANSITIONS);
  CHECK_FALSE(logger.overflow);
  CHECK(test.phases.entries == first_phase_trace);
  CHECK(test.phases.size == first_phase_trace_size);
  CHECK_FALSE(test.phases.overflow);
  CHECK(test.machine.is(sml::state<frame::state_idle>));
}

TEST_CASE("speech_generator_frame_meets_the_fake_dispatch_time_budget") {
  fixture test{};
  const std::array<int32_t, PUBLIC_CODEBOOK_COUNT> encoded_tokens{11, 12};
  std::array<int32_t, PUBLIC_CODEBOOK_COUNT> generated_tokens{};
  int32_t text_token = -1;
  emel::error::type err = 0;
  const frame::event::run request{encoded_tokens, generated_tokens, text_token,
                                  err};
  std::chrono::steady_clock::duration maximum_elapsed{};

  for (int32_t iteration = 0; iteration < 64; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    const bool accepted = test.machine.process_event(request);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    REQUIRE(accepted);
    maximum_elapsed = std::max(maximum_elapsed, elapsed);
  }

  CHECK(maximum_elapsed < FAKE_DISPATCH_BUDGET);
  CHECK(test.machine.is(sml::state<frame::state_idle>));
}

TEST_CASE("speech_generator_frame_repeated_dispatch_is_allocation_free") {
  fixture test{};
  const std::array<int32_t, 2> encoded_tokens{11, 12};
  std::array<int32_t, 2> generated_tokens{};
  int32_t text_token = -1;
  emel::error::type err = 0;
  const frame::event::run request{encoded_tokens, generated_tokens, text_token,
                                  err};
  REQUIRE(test.machine.process_event(request));

  bool accepted = true;
  size_t allocation_count = 0;
  {
    emel::test::allocation::allocation_scope allocations;
    for (int32_t iteration = 0; iteration < 64; ++iteration) {
      accepted = test.machine.process_event(request) && accepted;
    }
    allocation_count = allocations.allocations();
  }

  CHECK(accepted);
  CHECK(allocation_count == 0u);
  CHECK(test.machine.is(sml::state<frame::state_idle>));
  CHECK(text_token == 42);
}
